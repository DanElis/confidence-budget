#!/usr/bin/env python
"""openpi's websocket server for pi05_libero with N samples per request, for AAC. Runs in the venv of third_party/openpi.

A request is an ordinary LIBERO observation plus "noise" (N, H, A). The reply has "actions" (N, H, 7) in the environment's
units and "normalized_action" (N, H, A), the model output before Unnormalize, as the AAC authors' client expects. The
transforms and the sampler are openpi's own: the observation is repeated N times and sampled as one batch. N = 1 with
the same noise gives the same first sample, so conditions share it (eval_openpi.py checks this at start).

    cd third_party/openpi && .venv/bin/python ../../scripts/openpi_sample_server.py --port 8765
"""

from __future__ import annotations

import argparse
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np

CONFIG = "pi05_libero"
CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"


class SamplePolicy:
    """Policy.infer of openpi, with a batch of N samples from one observation."""

    def __init__(self, policy, horizon: int, action_dim: int):
        self._policy = policy
        self._shape = (horizon, action_dim)

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
        n = len(noise)
        inputs = p._input_transform(jax.tree.map(lambda x: x, obs))
        batch = jax.tree.map(lambda x: jnp.repeat(jnp.asarray(x)[np.newaxis, ...], n, axis=0), inputs)
        p._rng, rng = jax.random.split(p._rng)  # unused: the noise is given
        started = time.monotonic()
        normalized = np.asarray(p._sample_actions(rng, _model.Observation.from_dict(batch), noise=jnp.asarray(noise)))
        infer_ms = (time.monotonic() - started) * 1000
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
