"""Chunk schedule of the closed loop with a fake policy and fake environments."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from closed_loop import (  # noqa: E402
    Chunk,
    LoopConfig,
    chunk_noise,
    run_batch,
    sparc,
    stack_observations,
    switch_jumps,
)
from planner import FixedPlanner, Plan  # noqa: E402

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
                      "mask_end": None if mask_end is None else mask_end.clone(), "noise": noise.clone(),
                      "n": len(observations), "steps": [int(o["step"][0]) for o in observations]})
        j = torch.arange(H, dtype=torch.float32)
        model = torch.stack([torch.full((H,), float(t)), j], dim=-1).expand(len(observations), H, A).clone()
        return Chunk(model, model.numpy().copy(), np.zeros((len(observations), H)), np.full(len(observations), 100))
    return infer


def run(cfg, planner, envs=None, ids=(0,)):
    envs = envs or [FakeEnv() for _ in ids]
    calls = []
    res = run_batch(envs, fake_policy(calls), cfg, planner, list(ids), (H, 32))
    return envs, calls, res


def test_synchronous_chunking_executes_first_s_actions():
    cfg = LoopConfig(H, delay=0, guidance=False, max_steps=35)
    envs, calls, res = run(cfg, FixedPlanner(H, 10))
    got = np.stack(envs[0].received)
    expected = np.array([[10 * (t // 10), t % 10] for t in range(35)], dtype=float)
    assert np.array_equal(got, expected)
    assert [c["t"] for c in calls] == [0, 10, 20, 30] and res.n_inference.tolist() == [4]
    assert all(c["leftover"] is None for c in calls)
    assert res.switch_steps == [[0, 10, 20, 30]]


def test_async_delay_keeps_old_chunk_for_d_steps():
    cfg = LoopConfig(H, delay=3, guidance=False, max_steps=30)
    envs, calls, res = run(cfg, FixedPlanner(H, 10))
    for t, (start, j) in enumerate(tuple(a) for a in envs[0].received):
        if t < 13:
            assert (start, j) == (0, t)
        elif t < 23:
            assert (start, j) == (10, t - 10)
        else:
            assert (start, j) == (20, t - 20)
    assert res.switch_steps == [[0, 13, 23]]
    assert [c["t"] for c in calls] == [0, 10, 20]


def test_guidance_receives_padded_leftover_and_mask():
    cfg = LoopConfig(H, delay=3, guidance=True, max_steps=30)
    _, calls, _ = run(cfg, FixedPlanner(H, 10))
    assert calls[0]["leftover"] is None and calls[0]["delay"] == 0
    for prev, call in zip(calls, calls[1:]):
        left = call["leftover"][0]
        assert left.shape == (H, A)
        assert torch.equal(left[: H - 10, 0], torch.full((H - 10,), float(prev["t"])))
        assert torch.equal(left[: H - 10, 1], torch.arange(10, H, dtype=torch.float32))
        assert torch.equal(left[H - 10 :], torch.zeros(10, A))  # padding
        assert call["delay"] == 3 and call["mask_end"].tolist() == [H - 10]


class AlternatingPlanner:
    """stride 6 then 12 then 6 ... with mask ends 8, 14, ...; checks per-chunk plans are honoured."""

    needs_entropy = False

    def __init__(self):
        self.k = 0

    def __call__(self, entropy, n_valid, delay):
        self.k += 1
        return Plan(6, 8) if self.k % 2 else Plan(12, 14)


def test_variable_stride_and_mask_follow_the_planner():
    cfg = LoopConfig(H, delay=4, guidance=True, max_steps=40)
    envs, calls, res = run(cfg, AlternatingPlanner())
    assert [c["t"] for c in calls] == [0, 6, 18, 24, 36]
    assert [c["mask_end"].tolist() if c["mask_end"] is not None else None for c in calls] == [None, [8], [14], [8], [14]]
    assert res.switch_steps == [[0, 10, 22, 28]]  # each switch d = 4 steps after the inference start
    # leftover of the inference at 18 comes from the chunk generated at 6, from index 12
    left = calls[2]["leftover"][0]
    assert left[0].tolist() == [6.0, 12.0]


def test_envs_with_different_schedules_are_batched_by_step():
    class PerEnv:
        needs_entropy = False

        def __init__(self):
            self.calls = 0

        def __call__(self, entropy, n_valid, delay):
            self.calls += 1
            # first two plans (env 0 and env 1 at t=0): 5 and 10, afterwards 10
            return Plan({1: 5, 2: 10}.get(self.calls, 10), 10)

    cfg = LoopConfig(H, delay=0, guidance=False, max_steps=21)
    envs, calls, res = run(cfg, PerEnv(), ids=(0, 1))
    assert [(c["t"], c["n"]) for c in calls] == [(0, 2), (5, 1), (10, 1), (15, 1), (20, 1)]
    assert res.n_inference.tolist() == [3, 3]  # env 0: 0, 5, 15; env 1: 0, 10, 20


def test_finished_episodes_stop_and_are_counted():
    cfg = LoopConfig(H, delay=0, guidance=False, max_steps=40)
    envs, _, res = run(cfg, FixedPlanner(H, 10), envs=[FakeEnv(25), FakeEnv(None)], ids=(0, 1))
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
        LoopConfig(H, delay=H, guidance=True, max_steps=10)


def test_stack_jumps_and_sparc():
    obs = stack_observations([{"a": np.zeros(2), "n": {"b": np.ones((3, 3))}}] * 4)
    assert obs["a"].shape == (4, 2) and obs["n"]["b"].shape == (4, 3, 3)
    actions = np.zeros((1, 6, 1))
    actions[0, 3:] = 1.0
    assert switch_jumps(actions, [[0, 3]]) == (1.0, 0.0)
    t = np.linspace(0, 2, 41)
    smooth = np.sin(np.pi * t / 2)  # one bell-shaped speed bump
    jerky = smooth + 0.3 * (np.arange(41) % 2)
    assert sparc(smooth, 20.0) > sparc(jerky, 20.0)
