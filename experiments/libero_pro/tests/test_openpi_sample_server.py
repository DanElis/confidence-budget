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
from openpi_sample_server import (  # noqa: E402
    SamplePolicy,
    guidance_weight,
    guided_velocity,
    paper_velocity,
    sample_paper,
    sample_shared,
)

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

    return SamplePolicy(policy, H, A, sampler=sampler, paper_sampler=lambda *a, **k: pytest.fail("paper sampler"))


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


@pytest.mark.parametrize(("t", "weight"), [(1.0, 10.0), (0.5, 2.0), (0.1, 9.1111)])
def test_guidance_weight_table_shared_with_the_client(t, weight):
    """The same table is in test_eval_openpi.py, against h1.guidance.guidance_weight of the LeRobot runs."""
    assert float(guidance_weight(jax.numpy.float32(t), 10.0)) == pytest.approx(weight, abs=1e-3)


def test_guided_velocity_formula():
    x = np.full((1, H, A), 2.0, dtype=np.float32)
    v = np.full((1, H, A), 1.0, dtype=np.float32)
    target = np.full((H, A), 4.0, dtype=np.float32)
    weights = np.zeros(H, dtype=np.float32)
    weights[0], weights[1] = 1.0, 0.5
    out = np.asarray(guided_velocity(v, x, 0.5, target, weights, 10.0))  # x_0 = 2 - 0.5 = 1.5, error 2.5, weight 2
    np.testing.assert_allclose(out[0, 0], np.full(A, 1.0 - 2 * 2.5))
    np.testing.assert_allclose(out[0, 1], np.full(A, 1.0 - 2 * 2.5 * 0.5))
    np.testing.assert_allclose(out[0, 2:], np.ones((H - 2, A)))


def test_paper_velocity_is_the_vjp_on_a_linear_network():
    """v = a x: x_0 = (1 - t a) x, and the Jacobian multiplies the weighted error by (1 - t a). Without it the factor is 1."""
    a, t = 0.5, 0.5
    x = np.full((1, H, A), 2.0, dtype=np.float32)
    target = np.full((H, A), 4.0, dtype=np.float32)
    weights = np.zeros((H, A), dtype=np.float32)
    weights[0, :7] = 1.0
    out = np.asarray(paper_velocity(lambda z: a * z, x, t, target, weights, 10.0))
    x_0, error = 2.0 * (1 - t * a), 4.0 - 2.0 * (1 - t * a)
    assert x_0 == 1.5
    np.testing.assert_allclose(out[0, 0, :7], np.full(7, a * 2.0 - 2.0 * error * (1 - t * a)), atol=1e-5)  # weight 2 at t = 0.5
    np.testing.assert_allclose(out[0, 0, 7:], np.full(A - 7, a * 2.0), atol=1e-5)  # zero weight: the plain velocity
    np.testing.assert_allclose(out[0, 1:], np.full((H - 1, A), a * 2.0), atol=1e-5)
    plain = np.asarray(guided_velocity(a * x, x, t, target, weights, 10.0))
    np.testing.assert_allclose(plain[0, 0, 0], a * 2.0 - 2.0 * error, atol=1e-5)  # the two modes differ


def guided_policy(calls, paper_calls=None):
    policy = Policy()

    def sampler(rng, observation, *, noise, **guidance):
        calls.append(guidance)
        return noise * 2

    def paper_sampler(rng, observation, *, noise, **guidance):
        paper_calls.append(guidance)
        return noise * 2

    return SamplePolicy(policy, H, A, sampler=sampler, paper_sampler=paper_sampler), policy


def test_rtc_jacobian_flag_selects_the_paper_sampler():
    calls, paper_calls = [], []
    sample_policy, _ = guided_policy(calls, paper_calls)
    obs = {"noise": np.zeros((1, H, A), dtype=np.float32), "rtc_target": np.ones((H, A), dtype=np.float32),
           "rtc_weights": np.ones((H, A), dtype=np.float32), "rtc_max_weight": 5.0, "rtc_jacobian": True}
    sample_policy.infer(obs)
    assert not calls and paper_calls[0]["max_weight"] == 5.0 and paper_calls[0]["weights"].shape == (H, A)
    sample_policy.infer(obs | {"rtc_jacobian": False})
    assert len(calls) == 1 and len(paper_calls) == 1


def test_paper_rtc_on_a_small_real_model():
    config, model, key = small_model()
    noise = jax.random.normal(jax.random.key(1), (1, H, config.action_dim))
    obs = config.fake_obs(1)
    free = np.asarray(sample_shared(model, key, obs, noise=noise))[0]
    target = free + 1.0
    zero = np.zeros((H, A), dtype=np.float32)
    same = np.asarray(sample_paper(model, key, obs, noise=noise, target=target, weights=zero, max_weight=5.0))[0]
    np.testing.assert_allclose(same, free, atol=1e-3)  # zero weights: the same as no RTC

    weights = zero.copy()
    weights[:4, :7] = 1.0
    guided = np.asarray(sample_paper(model, key, obs, noise=noise, target=target, weights=weights, max_weight=5.0))[0]
    assert np.abs(guided[:4, :7] - target[:4, :7]).mean() < 0.9  # toward the target, 1.0 away without RTC
    plain = np.asarray(sample_shared(model, key, obs, noise=noise, target=target, weights=weights, max_weight=5.0))[0]
    assert np.abs(guided - plain).max() > 1e-4  # the Jacobian changes the result


def test_rtc_fields_reach_the_sampler_and_stay_out_of_the_transform():
    calls = []
    sample_policy, policy = guided_policy(calls)
    target, weights = np.ones((H, A), dtype=np.float32), np.ones(H, dtype=np.float32)
    obs = {"noise": np.zeros((1, H, A), dtype=np.float32), "rtc_target": target, "rtc_weights": weights, "rtc_max_weight": 7.0}
    sample_policy.infer(obs)  # Policy._sample_actions would fail the assertion below: N = 1 with RTC is not the stock path
    assert policy.batch_sizes == [] and "rtc_target" in obs
    np.testing.assert_array_equal(calls[0]["target"], target)
    np.testing.assert_array_equal(calls[0]["weights"], weights)
    assert calls[0]["max_weight"] == 7.0


@pytest.mark.parametrize("fields", [
    {"rtc_target": np.zeros((H, A))},
    {"rtc_target": np.zeros((H, A)), "rtc_weights": np.zeros(H)},
    {"rtc_target": np.zeros((H, 7)), "rtc_weights": np.zeros(H), "rtc_max_weight": 10.0},
    {"rtc_target": np.zeros((H, A)), "rtc_weights": np.zeros(5), "rtc_max_weight": 10.0},
])
def test_incomplete_or_wrong_rtc_is_refused(fields):
    sample_policy, _ = guided_policy([])
    with pytest.raises(ValueError, match="rtc"):
        sample_policy.infer({"noise": np.zeros((1, H, A), dtype=np.float32)} | fields)


def small_model():
    from openpi.models import pi0_config

    config = pi0_config.Pi0Config(pi05=True, action_horizon=H, discrete_state_input=False, paligemma_variant="dummy",
                                  action_expert_variant="dummy")
    key = jax.random.key(0)
    return config, config.create(key), key


def test_rtc_on_a_small_real_model():
    config, model, key = small_model()
    noise = jax.random.normal(jax.random.key(1), (1, H, config.action_dim))
    obs = config.fake_obs(1)
    free = np.asarray(sample_shared(model, key, obs, noise=noise))[0]
    target = free + 1.0

    zero = np.asarray(sample_shared(model, key, obs, noise=noise, target=target, weights=np.zeros(H, dtype=np.float32)))[0]
    np.testing.assert_allclose(zero, free, atol=1e-3)  # zero weights: the same as no RTC

    weights = (np.arange(H) < 4).astype(np.float32)
    guided = np.asarray(sample_shared(model, key, obs, noise=noise, target=target, weights=weights))[0]
    assert np.abs(guided[:4] - target[:4]).max() < 0.5  # the frozen actions reach the target, 1.0 away without RTC


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
