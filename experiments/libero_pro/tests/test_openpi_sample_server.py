"""N samples of openpi_sample_server.py on a stand-in for openpi's Policy.

Needs openpi's venv:  third_party/openpi/.venv/bin/python -m pytest -q tests/test_openpi_sample_server.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("openpi")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from openpi_sample_server import SamplePolicy  # noqa: E402

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
        self.batch_sizes.append(observation.state.shape[0])
        assert observation.state.shape == (len(noise), 8)
        return noise * 2

    def _output_transform(self, data):
        assert data["state"].shape == (8,) and data["actions"].shape == (H, A)
        return {"actions": data["actions"][..., :7] + 1}


def test_batch_of_samples():
    policy = Policy()
    noise = np.random.default_rng(0).standard_normal((3, H, A), dtype=np.float32)
    obs = {"prompt": "x", "noise": noise}
    reply = SamplePolicy(policy, H, A).infer(obs)
    assert policy.batch_sizes == [3] and "noise" in obs  # one call; the request is left as it was
    np.testing.assert_allclose(reply["normalized_action"], noise * 2)
    np.testing.assert_allclose(reply["actions"], noise[..., :7] * 2 + 1)
    assert reply["policy_timing"]["infer_ms"] >= 0


@pytest.mark.parametrize("shape", [(H, A), (1, H, 7), (1, 50, A)])
def test_wrong_noise_is_refused(shape):
    with pytest.raises(ValueError, match="noise"):
        SamplePolicy(Policy(), H, A).infer({"noise": np.zeros(shape, dtype=np.float32)})
