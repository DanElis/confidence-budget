"""Step budget on openpi: runtime in the closed loop (openpi_budget.py) and its CPU fit (openpi_budget_fit.py)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from closed_loop import LoopConfig, run_batch  # noqa: E402
from eval_openpi import parse_args, progress_key  # noqa: E402
from openpi_budget import (  # noqa: E402
    FEATURE_DIM,
    BudgetRuntime,
    GivenStride,
    features,
    gripper_flip,
    make_budget_infer,
)
from openpi_budget_fit import calibrate, headroom, load_decisions, main, run_rate  # noqa: E402

H, GRID = 10, (2, 4, 6, 8)


def config(mode="explore", **extra):
    return {"mode": mode, "grid": list(GRID), "option_ticks": 12, "max_steps": 40, **extra}


def element(value=0.0):
    image = np.full((16, 16, 3), value, dtype=np.uint8)
    return {"observation/image": image, "observation/wrist_image": image, "observation/state": np.zeros(8), "prompt": "p"}


def test_features():
    plan = np.zeros((H, 32), np.float32)
    plan[:, 0] = np.arange(H)
    actions = -np.ones((H, 7), np.float32)
    actions[6:, 6] = 1.0
    assert gripper_flip(actions) == pytest.approx(0.6) and gripper_flip(-np.ones((H, 7))) == 1.0
    x = features(np.arange(8), plan, actions, None, 10, 40, None, None, 4, H)
    assert x.shape == (FEATURE_DIM,) and x[8] == 0.25 and x[9 + 7] == 9  # time, change of dimension 0
    old = plan[:7] + 1.0  # the rest of the old plan from this step: 7 actions, 1 away everywhere
    frames = (np.ones((2, 2)), np.zeros((2, 2)))
    x = features(np.zeros(8), plan, actions, old, 10, 40, frames, (np.zeros((2, 2)), np.zeros((2, 2))), 4, H)
    np.testing.assert_allclose(x[-6:], [1, 1, 1, 1, 0, 0.4], rtol=1e-6)  # has old, D, D gripper, motion, s / H


def test_explore_holds_s_for_an_option_and_does_not_depend_on_request_order():
    plan, actions = np.zeros((H, 32)), np.zeros((H, 7))
    one, two = BudgetRuntime(config(), 1, H, 2), BudgetRuntime(config(), 1, H, 2)
    a = [one.decide(5, t, element(), plan, actions) for t in (0, 4, 8, 12, 16)]
    two.decide(7, 0, element(), plan, actions)  # another episode in between
    b = [two.decide(5, t, element(), plan, actions) for t in (0, 4, 8, 12, 16)]
    assert [r["decision"] for r in a] == [1, 0, 0, 1, 0]  # option_ticks = 12
    assert [r["s"] for r in a] == [r["s"] for r in b] and a[0]["s"] == a[2]["s"] and a[0]["prob"] == 0.25
    assert a[1]["x"][-6] == 1.0 and a[0]["x"][-6] == 0.0  # the old plan from the second request on
    assert len({BudgetRuntime(config(), seed, H, 2).decide(5, 0, element(), plan, actions)["s"] for seed in range(40)}) > 1


def test_modes_and_feasibility():
    plan, actions = np.zeros((H, 32)), np.zeros((H, 7))
    fixed = BudgetRuntime(config("fixed", fixed=5), 0, H, 2).decide(0, 0, element(), plan, actions)
    assert fixed["s"] == 5 and fixed["action"] == -1  # off the grid
    only = BudgetRuntime(config("marginal", marginal=[0, 0, 1, 0]), 0, H, 2)
    assert {only.decide(e, 0, element(), plan, actions)["s"] for e in range(10)} == {6}
    with pytest.raises(ValueError, match="feasible"):
        BudgetRuntime(config(), 0, H, 4)  # s = 2 < d
    with pytest.raises(ValueError, match="marginal"):
        BudgetRuntime(config("marginal", marginal=[0.5, 0.5]), 0, H, 2)


class Env:
    def reset(self):
        return observation(), {}

    def step(self, action):
        return observation(), 0.0, False, False, {}


def observation():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    return {"pixels": {"image": image, "image2": image},
            "robot_state": {"eef": {"pos": np.zeros(3), "quat": np.array([0.0, 0.0, 0.0, 1.0])},
                            "gripper": {"qpos": np.zeros(2)}}}


class Client:
    def __init__(self, normalized=True):
        self.calls, self.normalized = 0, normalized

    def infer(self, element):
        self.calls += 1
        reply = {"actions": np.full((1, H, 7), self.calls, np.float32)}
        if self.normalized:
            reply["normalized_action"] = np.full((1, H, 32), self.calls, np.float32)
        return reply


def test_closed_loop_runs_the_budget_step_and_logs_the_decisions():
    runtime = BudgetRuntime(config(), 3, H, 2)
    infer = make_budget_infer(Client(), "task", H, runtime)
    res = run_batch([Env(), Env()], infer, LoopConfig(H, 2, False, 40, 3), GivenStride(H), [7, 8], (H, 32),
                    keep_entropy=True)
    for b, episode in enumerate((7, 8)):
        chunks = [c for c in res.chunks if c["env"] == b]
        steps = [c["step"] for c in chunks]
        assert steps == list(np.cumsum([0] + [c["stride"] for c in chunks[:-1]]))  # each request s after the last
        assert chunks[0]["budget_decision"] == 1 and all(len(c["budget_x"]) == FEATURE_DIM for c in chunks)
        assert all(c["stride"] == GRID[int(c["budget_action"])] for c in chunks)
        assert np.isfinite(res.actions[b]).all()  # d <= s: the plan never runs out
        assert set(runtime.episodes) == {7, 8}


def test_the_budget_needs_the_plan_in_model_units():
    infer = make_budget_infer(Client(normalized=False), "task", H, BudgetRuntime(config(), 0, H, 2))
    with pytest.raises(RuntimeError, match="sample_server"):
        run_batch([Env()], infer, LoopConfig(H, 2, False, 12), GivenStride(H), [0], (H, 32), keep_entropy=True)


def test_args_and_progress_key(tmp_path):
    path = tmp_path / "b.json"
    path.write_text(json.dumps(config()))
    base = ["--suites", "pos:y0.1", "--out", "x", "--delay", "2"]
    a = parse_args(base + ["--budget", str(path)])
    assert a.stride is None and progress_key(a)["budget_config"] == config()
    assert "budget_config" not in progress_key(parse_args(base + ["--stride", "4"]))
    for bad in (["--stride", "4"], ["--guidance", "paper"]):
        with pytest.raises(SystemExit):
            parse_args(base + ["--budget", str(path)] + bad)


# ---------------------------------------------------------------- fit

SUITE = "libero_object_pos_y0.1"


def write_run(path: Path, rows, stride=None):
    """rows: [(task, init, success, [(step, stride, action, prob, decision, x)])] -> a run directory as eval_openpi writes."""
    path.mkdir(parents=True)
    episodes, chunks = [], []
    for task, init, success, decisions in rows:
        length = 40
        episodes.append({"task_id": task, "init_state": init, "success": bool(success), "length": length,
                         "n_inference": len(decisions), "mean_stride": None, "sparc": 0.0})
        for step, s, action, prob, decision, x in decisions:
            chunks.append({"task_id": task, "init_state": init, "step": step, "stride": s, "budget_x": list(map(float, x)),
                           "budget_action": float(action), "budget_prob": prob, "budget_decision": float(decision)})
    (path / f"{SUITE}.json").write_text(json.dumps({"success_rate": 0.0, "episodes": episodes}))
    (path / f"{SUITE}_chunks.json").write_text(json.dumps(chunks))
    (path / "config.json").write_text(json.dumps({"stride": str(stride)}))
    return path


def explore_rows(seed, n_tasks=6, n_init=10):
    """The first decision of an episode decides it: s = 2 when feature 0 is positive, s = 8 otherwise. The later
    decisions do not matter."""
    rng = np.random.default_rng(seed)
    rows = []
    for task in range(n_tasks):
        for init in range(n_init):
            decisions, good = [], False
            for k, step in enumerate((0, 12, 24)):
                x = rng.standard_normal(FEATURE_DIM).astype(np.float32)
                a = int(rng.integers(4))
                if k == 0:
                    good = (a == 0) if x[0] > 0 else (a == 3)
                decisions.append((step, GRID[a], a, 0.25, 1, x))
                decisions.append((step + GRID[a], GRID[a], a, 0.25, 0, x))  # inside the option: not a decision
            rows.append((task, init, rng.random() < (0.95 if good else 0.05), decisions))
    return rows


def one_decision_rows(seed, n_tasks=6, n_init=10):
    """One option per episode, and it decides the episode: s = 2 when feature 0 is positive, s = 8 otherwise.
    (With several options and random features the network cannot tell which decision mattered.)"""
    rng = np.random.default_rng(seed)
    rows = []
    for task in range(n_tasks):
        for init in range(n_init):
            x = rng.standard_normal(FEATURE_DIM).astype(np.float32)
            a = int(rng.integers(4))
            good = (a == 0) if x[0] > 0 else (a == 3)
            rows.append((task, init, rng.random() < (0.95 if good else 0.05), [(0, GRID[a], a, 0.25, 1, x)]))
    return rows


def test_load_decisions(tmp_path):
    run = write_run(tmp_path / "r", explore_rows(0, 2, 3))
    data = load_decisions([run])
    assert len(data["x"]) == 2 * 3 * 3 and data["priv"].shape == (18, 0)  # decisions only
    assert list(data["step"][:3]) == [0, 12, 24] and len(np.unique(data["group"])) == 6
    bad = write_run(tmp_path / "f", [(0, 0, 1, [(0, 5, -1, 1.0, 1, np.zeros(FEATURE_DIM))])])
    with pytest.raises(RuntimeError, match="grid"):
        load_decisions([bad])
    assert run_rate(run) == pytest.approx(6 / 40)


def test_fit_finds_a_state_dependent_step_and_the_policy_deploys(tmp_path):
    runs = [write_run(tmp_path / f"e{k}", one_decision_rows(k)) for k in range(10)]
    out = tmp_path / "fit"
    rule_rate = (1 / 2 + 1 / 8) / 2  # the data's rule, half s = 2, half s = 8
    main(["fit", *map(str, runs), "--out", str(out), "--option-ticks", "12", "--target-rate", str(rule_rate)])
    summary = json.loads((out / "fit.json").read_text())
    assert summary["main"]["uplift"][1] > 0 and (out / "fit.md").exists() and summary["decisions"] == 10 * 60
    runtime = BudgetRuntime(config("policy", policy=str(out / "policy.pt")), 0, H, 2)
    x = np.zeros(FEATURE_DIM, np.float32)
    x[0] = 3.0
    hi = runtime.select(x, 0, 0)[2]
    x[0] = -3.0
    assert (hi, runtime.select(x, 0, 0)[2]) != (8, 2)  # never the opposite of the rule in the data


def test_calibrate_moves_the_rate(tmp_path):
    runs = [write_run(tmp_path / f"e{k}", explore_rows(k)) for k in range(3)]
    main(["fit", *map(str, runs), "--out", str(tmp_path / "fit"), "--option-ticks", "12", "--target-rate", "0.25"])
    policy = tmp_path / "fit" / "policy.pt"
    assert calibrate(runs[0], policy, run_rate(runs[0]), tmp_path / "same.pt") == policy  # matched: kept
    out = calibrate(runs[0], policy, 2 * run_rate(runs[0]), tmp_path / "recal.pt")
    assert out == tmp_path / "recal.pt" and out.exists()


def test_headroom_cross_fits_the_per_scene_step(tmp_path):
    """Scene 0 succeeds only with s = 2, scene 1 only with s = 8: a per-scene s gains 50 pp over any constant."""
    runs = {}
    for s in (2, 8):
        rows = [(task, init, (task == 0) == (s == 2), []) for task in (0, 1) for init in range(10)]
        runs[s] = write_run(tmp_path / f"s{s}", rows, stride=s)
    h = headroom(runs, n_boot=200)
    assert h["gain"] == pytest.approx(0.5) and h["pass"] and h["s_star"] == 8  # tie: the larger s
    same = {s: write_run(tmp_path / f"t{s}", [(0, i, i % 3 == 0, []) for i in range(10)], stride=s) for s in (2, 8)}
    assert headroom(same, n_boot=200)["gain"] == 0.0


def test_cli_headroom_and_config(tmp_path, capsys):
    runs = [write_run(tmp_path / f"s{s}", [(0, i, s == 4, []) for i in range(4)], stride=s) for s in (2, 4)]
    main(["headroom", *map(str, runs), "--out", str(tmp_path / "b")])
    passed, s_star, rate = capsys.readouterr().out.split()
    assert (passed, s_star) == ("0", "4") and float(rate) == 0.0 and (tmp_path / "b" / "headroom.md").exists()
    main(["config", "--out", str(tmp_path / "c.json"), "--mode", "marginal", "--option-ticks", "40", "--max-steps", "280",
          "--marginal", "[0.1, 0.2, 0.3, 0.4]"])
    assert json.loads((tmp_path / "c.json").read_text())["marginal"] == [0.1, 0.2, 0.3, 0.4]
