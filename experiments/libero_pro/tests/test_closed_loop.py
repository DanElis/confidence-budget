"""Chunk schedule of the closed loop with a fake policy and fake environments."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from closed_loop import Chunk, LoopConfig, chunk_noise, run_batch, stack_observations, switch_jumps  # noqa: E402

H, A = 50, 2


class FakeEnv:
    """Records actions; succeeds after ``succeed_at`` steps (or never)."""

    def __init__(self, succeed_at=None):
        self.succeed_at, self.t, self.received = succeed_at, 0, []

    def reset(self):
        self.t = 0
        return {"step": np.array([0.0]), "nested": {"x": np.zeros(3)}}, {}

    def step(self, action):
        self.received.append(np.array(action))
        self.t += 1
        ok = self.succeed_at is not None and self.t >= self.succeed_at
        return {"step": np.array([float(self.t)]), "nested": {"x": np.full(3, self.t)}}, 0.0, ok, False, {"is_success": ok}


def fake_policy(calls):
    """Chunk generated at step t has action[j] = (t, j); model space holds the same numbers."""
    def infer(observations, noise, leftover, delay, mask_end):
        t = int(observations[0]["step"][0])
        calls.append({"t": t, "leftover": None if leftover is None else leftover.clone(), "delay": delay,
                      "mask_end": mask_end, "noise": noise.clone(), "n": len(observations)})
        j = torch.arange(H, dtype=torch.float32)
        model = torch.stack([torch.full((H,), float(t)), j], dim=-1).expand(len(observations), H, A).clone()
        return Chunk(model, model.numpy().copy(), np.zeros((len(observations), H)))
    return infer


def run(cfg, envs=None, ids=(0,)):
    envs = envs or [FakeEnv() for _ in ids]
    calls = []
    res = run_batch(envs, fake_policy(calls), cfg, list(ids), (H, 32))
    return envs, calls, res


def test_synchronous_chunking_executes_first_s_actions():
    cfg = LoopConfig(H, stride=10, delay=0, guidance=False, max_steps=35)
    envs, calls, res = run(cfg)
    got = np.stack(envs[0].received)
    expected = np.array([[10 * (t // 10), t % 10] for t in range(35)], dtype=float)
    assert np.array_equal(got, expected)
    assert [c["t"] for c in calls] == [0, 10, 20, 30] and res.n_inference == 4
    assert all(c["leftover"] is None for c in calls)
    assert res.switch_steps == [0, 10, 20, 30]


def test_async_delay_keeps_old_chunk_for_d_steps():
    cfg = LoopConfig(H, stride=10, delay=3, guidance=False, max_steps=30)
    envs, calls, res = run(cfg)
    got = [tuple(a) for a in envs[0].received]
    for t, (start, j) in enumerate(got):
        if t < 13:
            assert (start, j) == (0, t)  # old chunk until the new one is ready at 13
        elif t < 23:
            assert (start, j) == (10, t - 10)
        else:
            assert (start, j) == (20, t - 20)
    assert res.switch_steps == [0, 13, 23]
    # inference at 10 saw the observation of step 10, not of step 13
    assert [c["t"] for c in calls] == [0, 10, 20]


def test_guidance_receives_leftover_aligned_with_inference_start():
    cfg = LoopConfig(H, stride=10, delay=3, guidance=True, max_steps=30)
    _, calls, _ = run(cfg)
    assert calls[0]["leftover"] is None
    for prev, call in zip(calls, calls[1:]):
        left = call["leftover"][0]
        assert left.shape == (H - 10, A)
        assert torch.equal(left[:, 0], torch.full((H - 10,), float(prev["t"])))  # from the previous chunk
        assert torch.equal(left[:, 1], torch.arange(10, H, dtype=torch.float32))  # indices s onward
        assert call["delay"] == 3 and call["mask_end"] == H - 10


def test_finished_episodes_stop_and_are_counted():
    cfg = LoopConfig(H, stride=10, delay=0, guidance=False, max_steps=40)
    envs, _, res = run(cfg, envs=[FakeEnv(25), FakeEnv(None)], ids=(0, 1))
    assert res.success.tolist() == [True, False]
    assert res.length.tolist() == [25, 40]
    assert len(envs[0].received) == 25 and len(envs[1].received) == 40
    assert np.isnan(res.actions[0, 25:]).all() and not np.isnan(res.actions[1]).any()


def test_noise_depends_on_episode_and_step_only():
    a = chunk_noise(0, [3, 4], 10, (H, 32))
    assert torch.equal(a[0], chunk_noise(0, [3], 10, (H, 32))[0])
    assert not torch.equal(a[0], a[1])
    assert not torch.equal(a, chunk_noise(0, [3, 4], 20, (H, 32)))


def test_config_validation():
    with pytest.raises(ValueError):
        LoopConfig(H, stride=5, delay=6, guidance=True, max_steps=10)  # d > s
    with pytest.raises(ValueError):
        LoopConfig(H, stride=45, delay=10, guidance=True, max_steps=10)  # s + d > H
    assert LoopConfig(H, stride=10, delay=2, guidance=True, max_steps=10).mask_end == 40


def test_stack_and_jumps():
    obs = stack_observations([{"a": np.zeros(2), "n": {"b": np.ones((3, 3))}}] * 4)
    assert obs["a"].shape == (4, 2) and obs["n"]["b"].shape == (4, 3, 3)
    actions = np.zeros((1, 6, 1))
    actions[0, 3:] = 1.0  # a jump at t = 3
    at_switch, other = switch_jumps(actions, [0, 3])
    assert at_switch == 1.0 and other == 0.0
