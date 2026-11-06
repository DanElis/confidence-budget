"""Budget learning invariants; GPU-free, with real upstream queue if installed."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from budget_head import BudgetHead, INPUT_DIM, action_pairs, features, valid_actions
from learn_budget import parse_args, ppo_update, read_episodes
import policy_server as ps


def test_current_r_depends_on_old_age_not_selected_s():
    pairs = action_pairs([4, 16], [0, 12, 32], 4)
    valid = valid_actions(pairs, 34, 4)  # previous stride=16
    assert valid.tolist() == [True, True, False, True, True, False]
    assert valid_actions(pairs, 0, 4).tolist() == [True, False, False, True, False, False]
    # Current s=16,r=32 is feasible after previous s=4, though their sum+d > H.
    assert valid_actions(pairs, 46, 4)[pairs.index((16, 32))]


def test_feature_layout_keeps_full_tail_and_history():
    vis = np.ones(1024, np.float32)
    x = features(vis, None, np.zeros(8), None, np.ones((34, 7)), 16, 4, 32)
    assert x.shape == (INPUT_DIM,)
    np.testing.assert_array_equal(x[1024:2048], 0)
    tail = x[2048:2048+350].reshape(50, 7)
    np.testing.assert_array_equal(tail[:34], 1)
    np.testing.assert_array_equal(tail[34:], 0)
    assert x[-1] == 0
    changed = features(2*vis, vis, np.zeros(8), np.ones(8), np.ones((34, 7)), 16, 4, 32)
    np.testing.assert_array_equal(changed[1024:2048], 1)
    assert changed[-1] == 1


def test_ppo_reuses_behavior_masks_and_updates_actor():
    torch.manual_seed(0)
    model = BudgetHead(4)
    rows = {}
    for i in range(8):
        x = torch.randn(1, INPUT_DIM)
        valid = torch.tensor([[True, False, True, False]])
        action = i % 2 * 2
        with torch.no_grad():
            dist, value = model(x, valid)
        rows[str(i)] = {"success": bool(i % 2), "samples": [
            {"x": x[0].numpy(), "valid": valid[0].numpy(), "action": np.array(action),
             "logp": np.array(float(dist.log_prob(torch.tensor([action])))), "value": np.array(float(value))}]}
    before = model.actor.weight.detach().clone()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    log = ppo_update(model, opt, rows, {"call_cost": .1, "reference_stride": 8, "max_steps": 600})
    assert log["logp_error"] < 2e-5
    assert not torch.equal(before, model.actor.weight)
    assert np.isfinite(log["loss"])
    rows["0"]["samples"][0]["logp"] += 1
    with pytest.raises(RuntimeError, match="off-policy"):
        ppo_update(model, opt, rows, {"call_cost": .1, "reference_stride": 8, "max_steps": 600})


def test_splits_and_server_options_fail_closed(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(["--out", str(tmp_path), "--train-traj", "0 2", "--val-traj", "2"])
    with pytest.raises(SystemExit):
        ps.parse_args(["--budget-head", "test.pt", "--mode", "rtc"])
    args = ps.parse_args(["--budget-head", "test.pt", "--budget-traces", "traces",
                          "--mode", "rtc", "--actions-per-chunk", "50"])
    assert args.budget_mode == "sample"


def test_reader_rejects_infrastructure_and_starvation(tmp_path):
    path = tmp_path / "manifest.jsonl"
    row = {"item_id": "a", "status": "error", "error": "model died"}
    path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="infrastructure"):
        read_episodes(tmp_path)
    row = {"item_id": "a", "status": "ok", "error": None, "steps": 20, "success": False,
           "controller_summary": {"budget_trace": [{"id": "one", "step": 0, "s": 8, "r": 0}],
                                  "policy_requests": 1, "latency_budget_steps": 4},
           "action_runtime": {"queue_underflow_steps": 5}}
    path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="starvation"):
        read_episodes(tmp_path)
    row["action_runtime"]["queue_underflow_steps"] = 4
    path.write_text(json.dumps(row))
    assert len(read_episodes(tmp_path, expected=1)) == 1
    with pytest.raises(RuntimeError, match="expected 2"):
        read_episodes(tmp_path, expected=2)


DYN = ROOT / "third_party/LIBERO-Dynamic/scripts"


@pytest.mark.skipif(not (DYN / "realtime_eval/control.py").exists(), reason="setup.sh required")
def test_real_queue_variable_stride_full_leftover_and_inflight_charge():
    sys.path.insert(0, str(DYN))
    from realtime_eval import control

    name = "realtime_eval.strategies.learned_budget_queue"
    spec = importlib.util.spec_from_file_location(name, ROOT / "learned_budget_queue.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    calls = []

    class Client:
        def infer(self, req):
            k = len(calls)
            calls.append(req)
            full = np.repeat(((k * 100 + np.arange(50)) * 1e-4)[:, None], 7, 1).astype(np.float32)
            return {"actions": full, "original_actions": full,
                    "budget": {"id": req["budget_id"], "step": req["budget_step"],
                               "s": [4, 16, 8][k % 3], "r": 0}}

    class Policy:
        supports_rtc_conditioning = True
        action_dim = 7
        client = Client()

        def _observation(self, obs, language):
            return {}

    ctrl = module.LearnedBudgetQueueController(
        policy=Policy(), language="task", control_freq=20, mode="learned_budget_queue",
        prefetch_threshold=10, action_chunk_size=8, chunk_size_threshold=None,
        latency_sampler=control.LatencySampler(source="fixed", fixed_seconds=.2))
    ctrl.reset()
    actions = control.ActionRuntime(7, empty_action="hold")
    executed = []
    for t in range(30):
        packet = control.ObservationPacket(episode_id="e", step_idx=t, sim_time_s=t/20,
            language="task", obs={}, queue_len=actions.queue_len, last_action=actions.last_action)
        ctrl.tick(packet, actions)
        executed.append(actions.next_action()[0][0])
    ctrl.close()
    assert [r["budget_step"] for r in calls] == [0, 4, 20, 28]
    assert actions.underflow_steps == 4
    assert [len(r["prev_chunk_left_over"]) for r in calls] == [0, 46, 34, 42]
    # Selected s of request 1 controls its own scheduling (next request at 20).
    assert executed[7] == pytest.approx(7e-4)
    assert executed[8] == pytest.approx(104e-4)
    assert executed[23] == pytest.approx(119e-4)
    assert executed[24] == pytest.approx(204e-4)
    assert len(ctrl.summary()["budget_trace"]) == ctrl.summary()["policy_requests"] == 4


def test_budget_selects_r_before_single_current_generation():
    # Exercise the real server dispatcher without pi0.5 or image weights.
    from types import SimpleNamespace
    server = ps.Server.__new__(ps.Server)
    server.args = SimpleNamespace(device="cpu", seed=0, mask_mode="rtc", n_samples=1)
    server.horizon, server.a_max, server.apc, server.exec_h = 50, 7, 50, 6
    server.rtc = object()
    server.batch = lambda obs: {}
    server.post = lambda action: action
    events = []

    class Budget:
        def choose(self, obs, state, left, delay, horizon):
            events.append("choose")
            return {"id": "a", "s": 16, "r": 12, "step": 4}

    server.budget = Budget()

    def infer_single(batch, noise, left, delay, mask_end, scale):
        events.append(("generate", int(mask_end[0])))
        return torch.zeros(1, 50, 7), None

    server.infer_single = infer_single
    result = server.infer({"observation/state": np.zeros(8), "prev_chunk_left_over": np.zeros((46, 7)),
                           "inference_delay_steps": 4}, {})
    assert events == ["choose", ("generate", 16)]
    assert result["actions"].shape == (50, 7)
    assert result["budget"]["s"] == 16
