"""Simulator protocol and the whole evaluation loop against the fake simulator (no RoboTwin, no model)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "libero_pro"))
import wire  # noqa: E402
from closed_loop import LoopConfig, Chunk, run_batch  # noqa: E402
from planner import FixedPlanner  # noqa: E402
from remote_env import RemoteEnv  # noqa: E402


class Pipe:
    def __init__(self):
        self.q = []

    def send_bytes(self, b):
        self.q.append(bytes(b))

    def recv_bytes(self):
        return self.q.pop(0)


def test_wire_roundtrip():
    msg = {"obs": {"pixels": {"a": np.arange(24, dtype=np.uint8).reshape(2, 4, 3)}, "agent_pos": np.ones(14, np.float32)},
           "x": [1, 2.5, "s", None, np.float64(3.0)], "flag": True}
    p = Pipe()
    wire.send(p, msg)
    out = wire.recv(p)
    assert out["obs"]["pixels"]["a"].dtype == np.uint8 and np.array_equal(out["obs"]["pixels"]["a"], msg["obs"]["pixels"]["a"])
    assert out["obs"]["agent_pos"].dtype == np.float32 and out["x"] == [1, 2.5, "s", None, 3.0] and out["flag"] is True


@pytest.fixture
def env(tmp_path):
    e = RemoteEnv(sys.executable, None, log_path=tmp_path / "sim.log", fake=True, fake_step_lim=40)
    yield e
    e.close()


def test_remote_env_episode(env):
    env.set_episode("lift_pot", 6)  # "unstable" in the fake: skipped to 7
    obs, info = env.reset()
    assert info["seed"] == 7 and env.step_lim == 40 and env.instruction == "fake lift_pot"
    assert obs["pixels"]["head_camera"].shape == (24, 32, 3) and obs["agent_pos"].shape == (14,)
    for t in range(40):
        obs, r, term, trunc, info = env.step(np.full(14, 0.1))
        if term or trunc:
            break
    assert trunc and not info["is_success"] and t == 39  # odd seed never succeeds
    assert np.allclose(obs["agent_pos"], 0.1)
    env.set_episode("lift_pot", 2)
    env.reset()
    for t in range(40):
        _, _, term, trunc, info = env.step(np.zeros(14))
        if term or trunc:
            break
    assert term and info["is_success"] and t + 1 == 22


def test_error_does_not_break_protocol(env):
    env.set_episode("lift_pot", 0)
    env.reset()
    _, _, term, _, info = env.step(np.zeros(3))  # wrong shape is fine for the fake; send garbage command instead
    env._send({"cmd": "nope"})
    with pytest.raises(Exception, match="unknown command"):
        env._reply("nope")
    obs, *_ = env.step(np.zeros(14))  # still in sync
    assert obs["agent_pos"].shape == (14,)


def test_run_batch_drives_remote_envs_in_parallel(tmp_path):
    envs = [RemoteEnv(sys.executable, None, fake=True, fake_step_lim=60) for _ in range(3)]
    try:
        for e, s in zip(envs, (0, 1, 4)):
            e.set_episode("t", s)

        def infer(observations, noise, leftover, delay, mask_end):
            state = np.stack([o["agent_pos"] for o in observations])
            acts = np.repeat(state[:, None], 50, axis=1)
            import torch
            return Chunk(torch.as_tensor(acts), acts)

        res = run_batch(envs, infer, LoopConfig(50, 4, False, 60), FixedPlanner(50, 10), [0, 1, 2], (50, 32))
        assert res.success.tolist() == [True, False, True]
        assert res.length.tolist() == [20, 60, 24]
    finally:
        for e in envs:
            e.close()


def test_eval_robotwin_end_to_end_fake(tmp_path):
    out = tmp_path / "run"
    cmd = [sys.executable, str(HERE / "scripts" / "eval_robotwin.py"), "--fake-sim", "--fake-policy",
           "--suites", "lift_pot,stamp_seal", "--episodes-per-task", "5", "--batch-size", "3",
           "--delay", "4", "--stride", "10", "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads((out / "lift_pot.json").read_text())
    assert res["n_episodes"] == 5 and res["n_errors"] == 0
    # fake: even seeds succeed; seed = 100000 + 1000 * i is even for every i
    assert res["success_rate"] == 1.0
    assert [e["init_state"] for e in res["episodes"]] == list(range(5))
    assert (out / "summary.md").exists() and (out / "stamp_seal.json").exists()
    r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # finished tasks are skipped
    assert "skip lift_pot" in r2.stderr


def test_policy_batch_keys_match_checkpoint():
    sys.path.insert(0, str(HERE / "scripts"))
    from eval_robotwin import to_policy_batch

    obs = [{"pixels": {c: np.zeros((240, 320, 3), np.uint8) for c in ("head_camera", "left_camera", "right_camera")},
            "agent_pos": np.zeros(14, np.float32)} for _ in range(2)]
    batch = to_policy_batch(obs, "lift pot")
    assert set(batch) == {"observation.images.cam_high", "observation.images.cam_left_wrist",
                          "observation.images.cam_right_wrist", "observation.state", "task"}
    assert batch["observation.images.cam_high"].shape == (2, 3, 240, 320)
    assert batch["observation.state"].shape == (2, 14) and batch["task"] == ["lift pot"] * 2


def _eval_cmd(out):
    return [sys.executable, str(HERE / "scripts" / "eval_robotwin.py"), "--fake-sim", "--fake-policy",
            "--suites", "lift_pot", "--episodes-per-task", "6", "--batch-size", "3", "--delay", "4",
            "--stride", "10", "--out", str(out)]


def test_batch_is_retried_after_a_simulator_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTWIN_FAKE_FAIL", str(tmp_path / "failed_once"))  # first reset of one server fails
    r = subprocess.run(_eval_cmd(tmp_path / "run"), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "simulator failure" in r.stderr
    res = json.loads((tmp_path / "run" / "lift_pot.json").read_text())
    assert res["n_errors"] == 0 and res["n_episodes"] == 6 and res["success_rate"] == 1.0


def test_persistent_failure_leaves_no_result(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOTWIN_FAKE_FAIL", "always")
    r = subprocess.run(_eval_cmd(tmp_path / "run"), capture_output=True, text=True, timeout=300)
    assert r.returncode != 0
    assert "CUDA out of memory" in r.stderr  # lib.sh retries on this wording
    assert not (tmp_path / "run" / "lift_pot.json").exists()


def test_lazy_observations_match_eager_ones(tmp_path):
    from remote_env import LazyObs, SimulatorError, resolve_observations

    lazy = [RemoteEnv(sys.executable, None, fake=True, fake_step_lim=60) for _ in range(2)]
    eager = RemoteEnv(sys.executable, None, fake=True, fake_step_lim=60, lazy_obs=False)
    try:
        for e in lazy + [eager]:
            e.set_episode("t", 4)
            e.reset()
        for t in range(5):
            a = np.full(14, 0.01 * t)
            for e in lazy:
                e.step_async(a)
            obs_l = [e.step_wait()[0] for e in lazy]
            obs_e = eager.step(a)[0]
            assert all(isinstance(o, LazyObs) and o.value is None for o in obs_l)
        got = resolve_observations(obs_l)
        for o in got:
            assert np.array_equal(o["pixels"]["head_camera"], obs_e["pixels"]["head_camera"])
            assert np.allclose(o["agent_pos"], obs_e["agent_pos"])
        lazy[0].step(np.zeros(14))
        stale = lazy[0].step(np.zeros(14))[0]
        lazy[0].step(np.zeros(14))
        with pytest.raises(SimulatorError, match="stale"):
            resolve_observations([stale])
    finally:
        for e in lazy + [eager]:
            e.close()
