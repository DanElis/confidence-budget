"""Option budget invariants: explicit splits, decision timing, no privileged input, credit targets, offline gates."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from budget_features import ACTOR_DIM, BLOCKS, actor_features, gray, motion_features, plan_features, residual_features
from budget_items import item_id, make_splits, scenes_from_config, write_splits
from budget_options import OptionRuntime, check_pair, request_rng
from budget_privileged import N_OBJECTS, PRIV_DIM, ObjectTracker
from budget_q import QNet, load_policy
from budget_v2 import best_mask, nearest_stride, parse_args, select_candidate
from fit_budget import (cost_for_rate, doubly_robust, evaluate, fold_ids, group_bootstrap, lambda_targets, load_segments,
                        option_calls, request_rate, state_dependence_uplift)
import policy_server as ps

PAIRS = [(7, 0), (7, 12), (20, 0), (20, 12)]


def test_splits_share_scenes_regardless_of_their_size():
    config = {"defaults": {"enabled": True}, "tasks": {"libero_10": {
        str(t): {"activation_variants": [{}] * 5} for t in range(4)} | {"9": {"enabled": False, "activation_variants": [{}] * 5},
                                                                     "8": {"activation_variants": [{}] * 2}}}}
    scenes = scenes_from_config(config, "l1")
    assert ("libero_10", 9, "l1") not in scenes
    splits = make_splits(scenes, {"train": [0, 1], "val": [2], "test": [3, 4]}, seed=1)
    composition = {name: {i.rsplit("__", 1)[0] for i in ids} for name, ids in splits.items()}
    assert composition["train"] == composition["val"] == composition["test"]
    assert len(composition["val"]) == 4  # the scene without trajectories 2..4 is excluded everywhere
    assert len(splits["train"]) == 8 and len(splits["val"]) == 4
    assert item_id("libero_10", 3, "l2", 4) == "libero_10_task03__l2__traj04"
    limited = make_splits(scenes, {"train": [0], "test": [1]}, seed=1, scene_limit=2)
    assert len(limited["train"]) == len(limited["test"]) == 2
    with pytest.raises(ValueError, match="disjoint"):
        make_splits(scenes, {"train": [0], "test": [0]})


def test_split_files_are_json_lists_for_the_runner(tmp_path):
    paths = write_splits({"val": ["a", "b"]}, tmp_path)
    assert json.loads(paths["val"].read_text()) == ["a", "b"]


def test_tracker_uses_named_objects_and_ignores_robot_keys():
    tracker = ObjectTracker()
    obs = {"robot0_eef_pos": np.zeros(3), "robot0_gripper_qpos": np.array([.03, -.03]),
           "soup_1_pos": np.array([.1, 0, 0]), "basket_1_pos": np.array([0, .2, 0]), "milk_1_pos": np.array([9., 9, 9]),
           "soup_1_to_robot0_eef_pos": np.ones(3)}
    first = tracker(obs, "put the soup in the basket", 0)
    assert first.shape == (PRIV_DIM,) and first[N_OBJECTS] == 3 and first[10] == 0
    assert first[0] == pytest.approx(.1)
    assert first[7] == pytest.approx(.06)
    obs["soup_1_pos"] = np.array([.1, 0, .1])
    second = tracker(obs, "put the soup in the basket", 10)
    assert second[2] == pytest.approx(.1)  # lift
    assert second[3] == pytest.approx(.5)
    assert second[5] == pytest.approx(.01)  # per tick
    assert tracker({"robot0_gripper_qpos": np.zeros(2)}, "x", 0)[N_OBJECTS] == 0
    tracker.reset()
    assert tracker(obs, "unrelated words", 0)[10] == 1


def test_feature_blocks_cover_the_vector_and_exclude_object_state():
    assert sorted(BLOCKS.values())[0][0] == 0 and sorted(BLOCKS.values())[-1][1] == ACTOR_DIM
    edges = sorted(BLOCKS.values())
    assert all(a[1] == b[0] for a, b in zip(edges, edges[1:]))
    frames = np.zeros((8, 8)), np.zeros((8, 8))
    x = actor_features(np.ones(8), np.zeros(8), np.zeros((38, 7)), 7, 40, 600, frames, frames, np.ones(50), (12, 4))
    assert x.shape == (ACTOR_DIM,)
    assert x[8] == pytest.approx(1 / 12)
    start = BLOCKS["residual"][0]
    np.testing.assert_allclose(x[start:start + 4], 1)
    np.testing.assert_allclose(x[BLOCKS["context"][0] + 2:BLOCKS["context"][1]], [12 / 50, 4 / 50], rtol=1e-5)


def test_motion_and_plan_features_detect_their_causes():
    still = np.zeros((16, 16), np.float32)
    moved = still.copy()
    moved[:4, :4] = 1
    out = motion_features((moved, still), (still, still), 2)
    assert out[0] == pytest.approx(.5) and out[1:16].sum() == 0 and out[19] == 1
    assert motion_features((moved, still), None, 2).sum() == 0
    left = np.zeros((38, 7), np.float32)
    left[:, 6] = -1
    left[20:, 6] = 1
    plan = plan_features(left, 7, 12)
    assert plan[15] == 1 and plan[16] == pytest.approx(13 / 50)
    assert residual_features(None, 7, 38).sum() == 0
    assert gray(np.full((2, 2, 3), 255, np.uint8)).max() == 1


def request(step, left=38, uid=None, state=None):
    return {"budget_id": uid or f"id{step}", "budget_step": step, "budget_episode": "e", "prompt": "p",
            "budget_priv": np.eye(PRIV_DIM, dtype=np.float32)[N_OBJECTS] * 3,
            "observation/state": np.full(8, .01, np.float32) if state is None else state,
            "observation/image": np.zeros((8, 8, 3), np.uint8), "observation/wrist_image": np.zeros((8, 8, 3), np.uint8)}


def runtime(tmp_path, **config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"mode": "explore", "pairs": PAIRS, "delay": 7, "seed": 0, "option_ticks": 40,
                                "max_steps": 600, **config}))
    return OptionRuntime(path, tmp_path / "traces")


def test_option_holds_pair_and_clamps_only_first_request(tmp_path):
    rt = runtime(tmp_path, mode="fixed", fixed=[12, 12])  # off-grid constant, needed for rate matching
    state = {}
    first = rt.choose(request(0), state, np.zeros((0, 7)), 7, 50, None, np.zeros(2048))
    assert first["decision"] and (first["s"], first["r"]) == (12, 0)
    second = rt.choose(request(12), state, np.zeros((38, 7)), 7, 50, np.zeros(50), np.zeros(2048))
    assert not second["decision"] and (second["s"], second["r"]) == (12, 12)
    assert not (tmp_path / "traces/id12.npz").exists()
    third = rt.choose(request(48), state, np.zeros((38, 7)), 7, 50, np.zeros(50), np.zeros(2048))
    assert third["decision"]
    with np.load(tmp_path / "traces/id0.npz") as trace:
        assert int(trace["action"]) == -1 and trace["x"].shape == (ACTOR_DIM,) and trace["priv"].shape == (PRIV_DIM,)
    with pytest.raises(ValueError, match="does not match"):
        rt.choose(request(60), state, np.zeros((38, 7)), 4, 50, None, np.zeros(2048))
    with pytest.raises(ValueError, match="repeatable"):
        check_pair((40, 12), 7)


def test_missing_object_poses_fail_closed(tmp_path):
    rt = runtime(tmp_path)
    bad = request(0)
    bad["budget_priv"] = np.zeros(PRIV_DIM, np.float32)
    with pytest.raises(ValueError, match="no object poses"):
        rt.choose(bad, {}, np.zeros((0, 7)), 7, 50, None, np.zeros(2048))


def test_exploration_is_reproducible_and_independent_of_call_order(tmp_path):
    rt = runtime(tmp_path)
    a, b = request(0), request(40)
    x = np.zeros(ACTOR_DIM, np.float32)
    first = [rt.select(x, None, r, r["budget_step"]) for r in (a, b)]
    second = [rt.select(x, None, r, r["budget_step"]) for r in (b, a)][::-1]
    assert first == second
    assert first[0][1] == pytest.approx(.25)
    assert request_rng(1, a, 0).random() != request_rng(2, a, 0).random()
    draws = {rt.select(x, None, {**a, "budget_episode": str(i)}, 0)[0] for i in range(64)}
    assert draws == set(range(4))


def test_rule_uses_only_the_deployable_gripper_width(tmp_path):
    rt = runtime(tmp_path, mode="rule", option_ticks=1, rule={"open": [7, 0], "closed": [20, 12]})
    x = np.zeros(ACTOR_DIM, np.float32)
    x[6], x[7] = .04, -.04
    assert rt.select(x, None, request(0), 0)[2] == (7, 0)
    x[6], x[7] = .01, -.01
    assert rt.select(x, None, request(0), 0)[2] == (20, 12)


def test_decision_runs_after_prefix_and_before_denoising_once(tmp_path, monkeypatch):
    import shared_prefix

    events = []

    def fake(policy, batch, repeats, noise=None, before_integrate=None, latent_out=None, **kw):
        latent_out["latent"] = torch.ones(1, 2048)
        events.append("prefix")
        before_integrate(lambda x, t: torch.zeros_like(x))
        events.append(("denoise", int(kw["execution_horizon"][0]) if "execution_horizon" in kw else None))
        return torch.zeros(1, 50, 7)

    monkeypatch.setattr(shared_prefix, "predict_chunk_shared", fake)
    server = ps.Server.__new__(ps.Server)
    server.args = SimpleNamespace(device="cpu", seed=0, fresh_t=.5)
    server.horizon, server.a_max, server.policy = 50, 32, None
    server.batch = lambda obs: {}
    server.post = lambda action: action
    rt = runtime(tmp_path, mode="fixed", fixed=[20, 12])
    server.option_budget = rt
    choose = rt.choose

    def spy(obs, state, left, delay, horizon, residual, latent):
        events.append(("choose", residual is not None, latent.shape))
        return choose(obs, state, left, delay, horizon, residual, latent)

    rt.choose = spy
    state = {}
    obs = {**request(0), "prev_chunk_left_over": np.zeros((0, 7), np.float32), "inference_delay_steps": 7}
    assert server.infer(obs, state)["budget"]["r"] == 0
    assert events == ["prefix", ("choose", False, (2048,)), ("denoise", None)]
    events.clear()
    obs = {**request(20), "prev_chunk_left_over": np.zeros((30, 7), np.float32), "inference_delay_steps": 7}
    result = server.infer(obs, state)
    assert events[-1] == ("denoise", 19)  # the mask of THIS generation is d + r
    assert result["budget"]["s"] == 20 and result["actions"].shape == (50, 7)


def test_lambda_targets_end_in_the_outcome_and_do_not_cross_episodes():
    episode = np.array([0, 0, 0, 1, 1])
    value = np.array([.5, .6, .7, .2, .1])
    success = np.array([1., 1., 1., 0., 0.])
    np.testing.assert_allclose(lambda_targets(episode, value, success, 1.), success)
    np.testing.assert_allclose(lambda_targets(episode, value, success, 0.), [.6, .7, 1, .1, 0])
    # A telescoping sum: with lambda=0 the advantages of an episode add up to outcome - V(first).
    advantage = lambda_targets(episode, value, success, 0.) - value
    assert advantage[:3].sum() == pytest.approx(1 - .5)


def test_folds_keep_repeated_items_together():
    groups = np.array(["a", "a", "b", "c", "b", "d", "e", "e"])
    fold = fold_ids(groups, 2, 0)
    for name in np.unique(groups):
        assert len(set(fold[groups == name])) == 1


def test_group_bootstrap_counts_items_not_decisions():
    values = np.concatenate((np.ones(100), np.zeros(1), np.ones(1)))
    groups = np.array(["a"] * 100 + ["b", "c"])
    mean, low, high = group_bootstrap(values, groups)
    assert mean == pytest.approx(101 / 102) and low < .9 and high == 1


def test_rate_multiplier_reaches_the_target():
    q = np.random.default_rng(0).normal(size=(2000, 4))
    calls = option_calls(PAIRS, 40)
    cost = cost_for_rate(q, PAIRS, calls, 1 / 12)
    assert request_rate((q - cost * calls).argmax(1), PAIRS) == pytest.approx(1 / 12, abs=.005)


def test_doubly_robust_is_unbiased_under_logged_propensity():
    rng = np.random.default_rng(0)
    n = 200000
    action = rng.integers(0, 2, n)
    advantage = np.where(action == 1, 1., 0.) + rng.normal(size=n)
    wrong = np.zeros((n, 2), np.float32)  # even with a useless model
    dr = doubly_robust(wrong, action, np.full(n, .5), advantage)
    assert dr[:, 1].mean() == pytest.approx(1, abs=.02) and dr[:, 0].mean() == pytest.approx(0, abs=.02)


def synthetic(state_dependent, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, ACTOR_DIM)).astype(np.float32)
    action = rng.integers(0, 4, n)
    best = np.where(x[:, 0] > 0, 0, 3)  # the same request rate either way is not required by the test
    advantage = (.5 * (action == best) if state_dependent else .5 * (action == 0)) + rng.normal(0, .2, n)
    groups = np.array([f"g{i // 8}" for i in range(n)])
    return {"x": x, "latent": np.zeros((n, 4), np.float32), "action": action, "prob": np.full(n, .25),
            "group": groups}, advantage


def test_gate_detects_state_dependence_and_stays_near_zero_without_it():
    data, advantage = synthetic(True)
    calls = option_calls(PAIRS, 40)
    fold = fold_ids(data["group"], 4, 0)
    result = evaluate(data, advantage, PAIRS, fold, calls, 1 / 12, 0)
    assert result["uplift"][1] > .05
    data, advantage = synthetic(False)
    fold = fold_ids(data["group"], 4, 0)
    assert abs(evaluate(data, advantage, PAIRS, fold, calls, 1 / 12, 0)["uplift"][0]) < .05


def test_saved_policy_round_trip_and_ablated_block_is_ignored(tmp_path):
    data, advantage = synthetic(True, n=800)
    calls = option_calls(PAIRS, 40)
    fold = fold_ids(data["group"], 2, 0)
    result = evaluate(data, advantage, PAIRS, fold, calls, 1 / 12, 0, without="robot")
    torch.save(result["ckpt"], tmp_path / "policy.pt")
    policy = load_policy(tmp_path / "policy.pt")
    changed = data["x"][:4].copy()
    changed[:, :12] += 100
    np.testing.assert_array_equal(policy.q(data["x"][:4]), policy.q(changed))
    policy.save(tmp_path / "again.pt", 3.)
    assert load_policy(tmp_path / "again.pt").cost == 3
    assert QNet(3, PAIRS)(torch.zeros(2, 3)).shape == (2, 4)


def test_segments_keep_decisions_only_and_check_traces(tmp_path):
    x, priv = np.zeros(ACTOR_DIM, np.float32), np.zeros(PRIV_DIM, np.float32)
    for uid, step in (("a", 0), ("c", 40)):
        np.savez(tmp_path / f"{uid}.npz", x=x, priv=priv, latent=np.zeros(4, np.float16), action=1, prob=.25, step=step, pair=np.array([7, 12]))
    trace = [{"id": "a", "step": 0, "s": 7, "r": 0, "decision": True}, {"id": "b", "step": 7, "s": 7, "r": 12, "decision": False},
             {"id": "c", "step": 40, "s": 7, "r": 12, "decision": True}]
    rows = {"item": {"success": True, "suite": "s", "task_id": 1, "level": "l1", "controller_summary": {"budget_trace": trace}}}
    data = load_segments([(rows, tmp_path)])
    assert data["step"].tolist() == [0, 40] and data["success"].tolist() == [1, 1]
    trace[2]["s"] = 12
    with pytest.raises(RuntimeError, match="mismatch"):
        load_segments([(rows, tmp_path)])


def test_equal_rate_selection_helpers():
    assert best_mask({(12, 0): .4, (12, 4): .5, (12, 12): .5}, 12) == 4
    assert select_candidate({"a": (.9, .2), "b": (.4, .084), "c": (.5, .081)}, 1 / 12) == ("c", True)
    assert select_candidate({"a": (.9, .2), "b": (.1, .1)}, 1 / 12) == ("b", False)
    assert nearest_stride(1 / 11.4, 7) == 11 and nearest_stride(1., 7) == 7


def test_v2_arguments_fail_closed(tmp_path):
    out = str(tmp_path)
    assert parse_args(["--out", out]).delay == 7
    with pytest.raises(SystemExit):
        parse_args(["--out", out, "--strides", "4", "12"])  # below d
    with pytest.raises(SystemExit):
        parse_args(["--out", out, "--strides", "7", "40"])  # infeasible after another pair
    with pytest.raises(SystemExit):
        ps.parse_args(["--option-budget", "c.json", "--budget-head", "h.pt", "--budget-traces", "t", "--mode", "rtc", "--actions-per-chunk", "50"])
    assert ps.parse_args(["--option-budget", "c.json", "--budget-traces", "t", "--mode", "rtc", "--actions-per-chunk", "50"]).option_budget == "c.json"


# ---------------------------------------------------------------- openpi pi05_libero, H = 10

def test_features_scale_with_the_horizon():
    frames = np.zeros((8, 8)), np.zeros((8, 8))
    x = actor_features(np.ones(8), np.zeros(8), np.zeros((6, 7)), 2, 40, 600, frames, frames, None, (4, -1), horizon=10)
    assert x.shape == (ACTOR_DIM,)
    assert x[8] == pytest.approx(1 / 4)  # age = H - len(left) = 4
    np.testing.assert_allclose(x[BLOCKS["context"][0] + 1:BLOCKS["context"][1]], [.2, .4, -.1], rtol=1e-5)
    assert x[BLOCKS["residual"][0]:BLOCKS["residual"][1]].sum() == 0  # openpi: no residual
    assert plan_features(np.zeros((6, 7)), 2, 4, horizon=10)[17] == pytest.approx(.6)


def test_naive_pairs_and_the_short_horizon(tmp_path):
    check_pair((2, -1), 2, 10)
    check_pair((3, 4), 3, 10)
    for pair, d in (((9, -1), 2), ((5, 4), 3), ((2, -2), 2), ((2, 0), 3)):
        with pytest.raises(ValueError, match="repeatable"):
            check_pair(pair, d, 10)
    pairs = [(3, -1), (3, 2), (5, -1), (7, -1)]
    rt = runtime(tmp_path, pairs=pairs, delay=3, horizon=10, mode="fixed", fixed=[5, -1])
    state = {}
    first = rt.choose(request(0), state, np.zeros((0, 7)), 3, 10, None, np.zeros(0))
    assert (first["s"], first["r"], first["horizon"]) == (5, -1, 10)  # naive stays naive, not clamped to 0
    assert rt.choose(request(5), state, np.zeros((5, 7)), 3, 10, None, np.zeros(0))["r"] == -1
    with pytest.raises(ValueError, match="does not match"):
        rt.choose(request(10), state, np.zeros((5, 7)), 3, 50, None, np.zeros(0))
    explore = runtime(tmp_path, pairs=pairs, delay=3, horizon=10)
    x = np.zeros(ACTOR_DIM, np.float32)
    assert {explore.select(x, None, {**request(0), "budget_episode": str(i)}, 0)[2] for i in range(64)} == set(pairs)


def test_openpi_arguments_fail_closed(tmp_path):
    from budget_v2 import parse_pairs, reference_masks, runner_ticks

    out = str(tmp_path)
    base = ["--out", out, "--server", "openpi", "--reference-stride", "5"]
    a = parse_args(base + ["--delay", "3", "--latency", "0.134", "--pairs", "3:-1 3:2 3:4 5:-1 5:2 7:-1"])
    assert a.guidance == 5 and not a.rule and parse_pairs(a.pairs)[1] == (3, 2)
    assert parse_args(["--out", out]).guidance == 2 and parse_args(["--out", out]).rule
    assert reference_masks(parse_pairs(a.pairs), 5, 3, 10) == [-1, 2]
    assert reference_masks([(2, -1), (4, -1)], 5, 2, 10) == [-1]
    assert runner_ticks(0.091) == 2 and runner_ticks(0.134) == 3 and runner_ticks(0.15) == 3 and runner_ticks(0.1) == 2
    for bad in (["--delay", "3"],  # no pairs
                ["--delay", "3", "--latency", "0.091", "--pairs", "3:-1"],  # latency gives d = 2
                ["--delay", "3", "--pairs", "3:-1 5:4"],  # not repeatable
                ["--delay", "2", "--pairs", "2:-1 2:-1"],  # duplicate
                ["--delay", "2", "--pairs", "2:x"]):
        with pytest.raises(SystemExit):
            parse_args(base + bad)
    with pytest.raises(SystemExit):
        parse_args(["--out", out, "--pairs", "7:0"])  # explicit pairs are for openpi


def test_infrastructure_errors_are_collected_for_a_rerun_or_raise(tmp_path):
    from learn_budget import read_episodes

    (tmp_path / "v0.06").mkdir()
    (tmp_path / "v0.06/manifest.jsonl").write_text(json.dumps(
        {"item_id": "a__l1__traj00", "status": "error", "error": "TimeoutError('timed out during handshake')"}) + "\n")
    failed = []
    assert read_episodes(tmp_path, None, None, failed=failed) == {} and failed == ["a__l1__traj00"]
    with pytest.raises(RuntimeError, match="infrastructure error"):
        read_episodes(tmp_path)
