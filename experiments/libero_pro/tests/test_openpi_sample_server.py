"""N samples of openpi_sample_server.py on a stand-in for openpi's Policy, and sample_shared on a small real model.

Needs openpi's venv, runs on the CPU:  third_party/openpi/.venv/bin/python -m pytest -q tests/test_openpi_sample_server.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # a jax process on the shared GPU would preallocate most of its memory

import numpy as np  # noqa: E402
import pytest  # noqa: E402

jax = pytest.importorskip("jax")
pytest.importorskip("openpi")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from openpi_sample_server import SamplePolicy, sample_shared  # noqa: E402

H, A = 10, 32


class Policy:
    """As openpi's: an input transform, a sampler, an output transform that needs the state."""

    metadata = {"m": 1}

    def __init__(self):
        self._rng = jax.random.key(0)
        self.batch_sizes = []

    def _input_transform(self, obs):
        assert "noise" not in obs
        size = 8
        return {
            "state": np.arange(8.0),
            "image": {"base_0_rgb": np.zeros((size, size, 3), dtype=np.uint8)},
            "image_mask": {"base_0_rgb": np.True_},
            "tokenized_prompt": np.zeros(4, dtype=np.int32),
            "tokenized_prompt_mask": np.ones(4, dtype=bool),
        }

    def _sample_actions(self, rng, observation, *, noise):
        assert len(noise) == 1 and observation.state.shape == (1, 8)
        self.batch_sizes.append(("stock", len(noise)))
        return noise * 2

    def _output_transform(self, data):
        assert data["state"].shape == (8,) and data["actions"].shape == (H, A)
        return {"actions": data["actions"][..., :7] + 1}


def make(policy):
    def sampler(rng, observation, *, noise):
        assert observation.state.shape == (1, 8)  # the observation is encoded once
        policy.batch_sizes.append(("shared", len(noise)))
        return noise * 2

    return SamplePolicy(policy, H, A, sampler=sampler)


@pytest.mark.parametrize(("n", "used"), [(3, "shared"), (1, "stock")])
def test_samples(n, used):
    policy = Policy()
    noise = np.random.default_rng(0).standard_normal((n, H, A), dtype=np.float32)
    obs = {"prompt": "x", "noise": noise}
    reply = make(policy).infer(obs)
    assert policy.batch_sizes == [(used, n)] and "noise" in obs  # one call; the request is left as it was
    np.testing.assert_allclose(reply["normalized_action"], noise * 2)
    np.testing.assert_allclose(reply["actions"], noise[..., :7] * 2 + 1)
    assert reply["policy_timing"]["infer_ms"] >= 0


@pytest.mark.parametrize("shape", [(H, A), (1, H, 7), (1, 50, A)])
def test_wrong_noise_is_refused(shape):
    with pytest.raises(ValueError, match="noise"):
        make(Policy()).infer({"noise": np.zeros(shape, dtype=np.float32)})


def test_sample_shared_equals_openpis_sampler_on_a_repeated_observation():
    from openpi.models import pi0_config

    config = pi0_config.Pi0Config(pi05=True, action_horizon=H, discrete_state_input=False, paligemma_variant="dummy",
                                  action_expert_variant="dummy")
    key = jax.random.key(0)
    model = config.create(key)
    n = 3
    noise = jax.random.normal(jax.random.key(1), (n, H, config.action_dim))
    stock = model.sample_actions(key, config.fake_obs(n), num_steps=2, noise=noise)
    shared = sample_shared(model, key, config.fake_obs(1), noise=noise, num_steps=2)
    np.testing.assert_allclose(np.asarray(shared), np.asarray(stock), atol=1e-2)
