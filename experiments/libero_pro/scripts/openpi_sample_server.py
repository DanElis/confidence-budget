#!/usr/bin/env python
"""openpi's websocket server for pi05_libero with N samples per request, for AAC. Runs in the venv of third_party/openpi.

A request is an ordinary LIBERO observation plus "noise" (N, H, A). The reply has "actions" (N, H, 7) in the environment's
units and "normalized_action" (N, H, A), the model output before Unnormalize, as the AAC authors' client expects. The
transforms are openpi's own. N = 1 goes through openpi's own sampler. N > 1 goes through sample_shared: the same steps,
but the observation prefix is encoded once and its KV cache repeated over the samples, which leaves only the denoising
per sample. The same noise gives the same first sample, so conditions share it (eval_openpi.py checks this at start).

RTC: a request may add "rtc_target" (H, A), the old plan in model units, "rtc_weights" (H,) or (H, A), "rtc_max_weight"
and "rtc_jacobian". True is Eq. 2 of the RTC paper (Black et al., 2025), a vector-Jacobian product through the network;
false is LeRobot's RTCProcessor, without it. The weights come from the client, and the request goes through sample_shared
whatever N is.

    cd third_party/openpi && .venv/bin/python ../../scripts/openpi_sample_server.py --port 8765
"""

from __future__ import annotations

import argparse
import logging
import time
import types

import einops
import jax
import jax.numpy as jnp
import numpy as np

CONFIG = "pi05_libero"
CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
LOG_EVERY = 500  # requests between timing lines in the server log


def guidance_weight(t, max_weight):
    """Clipped PiGDM weight min(beta, (1 - tau) / (tau * r_tau^2)), tau = 1 - t, as h1.guidance.guidance_weight."""
    tau = 1.0 - t
    one_minus = (1.0 - tau) ** 2
    weight = (1.0 - tau) / tau * (one_minus + tau**2) / one_minus
    return jnp.minimum(weight, max_weight)  # tau = 0 gives inf; t = 0 (nan) is never reached by the sampler


def guided_velocity(v_t, x_t, t, target, weights, max_weight):
    """RTC without the network's Jacobian. openpi and LeRobot share the convention: t goes 1 -> 0, x_0 = x_t - t * v_t."""
    correction = (target - (x_t - t * v_t)) * expand_weights(weights)
    return v_t - guidance_weight(t, max_weight) * correction


def paper_velocity(denoise, x_t, t, target, weights, max_weight):
    """Eq. 2 of the RTC paper. The paper's tau is 1 - t and its velocity is -v, hence the sign, as in the authors' code."""

    def denoiser(x):
        v = denoise(x)
        return x - t * v, v

    x_1, vjp_fn, v_t = jax.vjp(denoiser, x_t, has_aux=True)
    correction = vjp_fn(((target - x_1) * expand_weights(weights)).astype(x_1.dtype))[0]
    return v_t - guidance_weight(t, max_weight) * correction


def expand_weights(weights):
    """(H,) or (H, A) -> (1, H, 1) or (1, H, A)."""
    return weights.reshape((1, weights.shape[0], -1))


def sample_shared(self, rng, observation, *, noise, num_steps: int = 10, target=None, weights=None, max_weight=10.0,
                  jacobian: bool = False):
    """Pi0.sample_actions at openpi@215abfb for one observation (batch 1) and N rows of noise, as a method of Pi0.

    With target (H, A) and weights the velocity is guided toward target. The KV cache has axes (layer, batch, ...).
    """
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = noise.shape[0]
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    kv_cache = jax.tree.map(lambda x: jnp.repeat(x, batch_size, axis=1), kv_cache)
    prefix_mask = jnp.repeat(prefix_mask, batch_size, axis=0)
    observation = jax.tree.map(lambda x: jnp.repeat(x, batch_size, axis=0), observation)

    def denoise(x_t, t):
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(t, batch_size)
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_to_suffix = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_to_suffix, suffix_attn_mask], axis=-1)
        suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=suffix_positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

    def step(carry):
        x_t, t = carry
        if target is None:
            v_t = denoise(x_t, t)
        elif jacobian:
            v_t = paper_velocity(lambda x: denoise(x, t), x_t, t, target, weights, max_weight)
        else:
            v_t = guided_velocity(denoise(x_t, t), x_t, t, target, weights, max_weight)
        return x_t + dt * v_t, t + dt

    def cond(carry):
        return carry[1] >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0


def sample_paper(self, rng, observation, *, noise, target, weights, max_weight):
    """sample_shared with the Jacobian. A method of its own: a flag would be traced by jit, and the branch needs its value."""
    return sample_shared(self, rng, observation, noise=noise, target=target, weights=weights, max_weight=max_weight,
                         jacobian=True)


class SamplePolicy:
    """Policy.infer of openpi, with N samples from one observation. sampler, paper_sampler: jitted sample_shared, sample_paper."""

    def __init__(self, policy, horizon: int, action_dim: int, sampler=None, paper_sampler=None):
        self._policy = policy
        self._shape = (horizon, action_dim)
        if sampler is None or paper_sampler is None:
            from openpi.shared import nnx_utils

            sampler = sampler or nnx_utils.module_jit(types.MethodType(sample_shared, policy._model))
            paper_sampler = paper_sampler or nnx_utils.module_jit(types.MethodType(sample_paper, policy._model))
        self._sampler = sampler
        self._paper_sampler = paper_sampler
        self._calls, self._ms = 0, 0.0

    @property
    def metadata(self) -> dict:
        return self._policy.metadata

    def _guidance(self, obs: dict) -> dict:
        """Takes the RTC fields out of the request: the sampler's keyword arguments, empty without RTC."""
        names = ("rtc_target", "rtc_weights", "rtc_max_weight")
        given = [obs.pop(name, None) for name in names]
        self._jacobian = bool(obs.pop("rtc_jacobian", False))
        if all(v is None for v in given):
            return {}
        if any(v is None for v in given):
            raise ValueError(f"RTC needs all of {names}")
        target, weights, max_weight = given
        target, weights = np.asarray(target, dtype=np.float32), np.asarray(weights, dtype=np.float32)
        if target.shape != self._shape or weights.shape not in (self._shape[:1], self._shape):
            raise ValueError(f"need rtc_target {self._shape} and rtc_weights (H,) or (H, A), got {target.shape}, {weights.shape}")
        return {"target": jnp.asarray(target), "weights": jnp.asarray(weights), "max_weight": float(max_weight)}

    def infer(self, obs: dict) -> dict:
        from openpi.models import model as _model

        p = self._policy
        obs = dict(obs)
        noise = np.asarray(obs.pop("noise"), dtype=np.float32)
        if noise.ndim != 3 or noise.shape[1:] != self._shape:
            raise ValueError(f"noise must be (N, {self._shape[0]}, {self._shape[1]}), got {noise.shape}")
        guidance = self._guidance(obs)
        inputs = p._input_transform(jax.tree.map(lambda x: x, obs))
        batch = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        p._rng, rng = jax.random.split(p._rng)  # unused: the noise is given
        if guidance and self._jacobian:
            sampler = self._paper_sampler
        else:
            sampler = p._sample_actions if len(noise) == 1 and not guidance else self._sampler
        started = time.monotonic()
        normalized = np.asarray(sampler(rng, _model.Observation.from_dict(batch), noise=jnp.asarray(noise), **guidance))
        infer_ms = (time.monotonic() - started) * 1000
        self._calls, self._ms = self._calls + 1, self._ms + infer_ms
        if self._calls % LOG_EVERY == 0:
            logging.info("%d requests, N = %d: %.0f ms per request", self._calls, len(noise), self._ms / LOG_EVERY)
            self._ms = 0.0
        state = np.asarray(inputs["state"])
        actions = np.stack([np.asarray(p._output_transform({"state": state, "actions": row})["actions"]) for row in normalized])
        return {"actions": actions, "normalized_action": normalized, "policy_timing": {"infer_ms": infer_ms}}


def main() -> None:
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    config = _config.get_config(CONFIG)
    policy = policy_config.create_trained_policy(config, CHECKPOINT)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=SamplePolicy(policy, config.model.action_horizon, config.model.action_dim),
        host="127.0.0.1", port=args.port, metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
