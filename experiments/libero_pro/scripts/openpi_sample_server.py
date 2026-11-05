#!/usr/bin/env python
"""openpi's websocket server for pi05_libero with N samples per request, for AAC. Runs in the venv of third_party/openpi.

A request is an ordinary LIBERO observation plus "noise" (N, H, A). The reply has "actions" (N, H, 7) in the environment's
units and "normalized_action" (N, H, A), the model output before Unnormalize, as the AAC authors' client expects. The
transforms are openpi's own. N = 1 goes through openpi's own sampler. N > 1 goes through sample_shared: the same steps,
but the observation prefix is encoded once and its KV cache repeated over the samples, which leaves only the denoising
per sample. The same noise gives the same first sample, so conditions share it (eval_openpi.py checks this at start).

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


def sample_shared(self, rng, observation, *, noise, num_steps: int = 10):
    """Pi0.sample_actions at openpi@215abfb for one observation (batch 1) and N rows of noise, as a method of Pi0.

    The KV cache has axes (layer, batch, ...).
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

    def step(carry):
        x_t, t = carry
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
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return x_t + dt * v_t, t + dt

    def cond(carry):
        return carry[1] >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0


class SamplePolicy:
    """Policy.infer of openpi, with N samples from one observation. sampler: jitted sample_shared."""

    def __init__(self, policy, horizon: int, action_dim: int, sampler=None):
        self._policy = policy
        self._shape = (horizon, action_dim)
        if sampler is None:
            from openpi.shared import nnx_utils

            sampler = nnx_utils.module_jit(types.MethodType(sample_shared, policy._model))
        self._sampler = sampler
        self._calls, self._ms = 0, 0.0

    @property
    def metadata(self) -> dict:
        return self._policy.metadata

    def infer(self, obs: dict) -> dict:
        from openpi.models import model as _model

        p = self._policy
        obs = dict(obs)
        noise = np.asarray(obs.pop("noise"), dtype=np.float32)
        if noise.ndim != 3 or noise.shape[1:] != self._shape:
            raise ValueError(f"noise must be (N, {self._shape[0]}, {self._shape[1]}), got {noise.shape}")
        inputs = p._input_transform(jax.tree.map(lambda x: x, obs))
        batch = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        p._rng, rng = jax.random.split(p._rng)  # unused: the noise is given
        sampler = p._sample_actions if len(noise) == 1 else self._sampler
        started = time.monotonic()
        normalized = np.asarray(sampler(rng, _model.Observation.from_dict(batch), noise=jnp.asarray(noise)))
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
