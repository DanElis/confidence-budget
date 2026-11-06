"""Saved-result audits must not hide missing episodes or infer splits from LIMIT."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnose_budget import counts, frequency_distance, read_run, split_report


def episode(uid, task_id, traj):
    return {"item_id": uid, "suite": "suite", "task_id": task_id, "level": "l1", "traj_index": traj,
            "status": "ok", "success": True, "controller_summary": {"policy_requests": 2, "budget_trace": [
                {"step": 0, "s": 8, "r": 0}, {"step": 8, "s": 16, "r": 12}]}}


def test_audit_identifies_different_tasks_despite_equal_episode_counts():
    runs = {"train_1": [episode("a", 0, 0), episode("b", 0, 1)],
            "val_0": [episode("c", 0, 2), episode("d", 1, 2)],
            "test_learned": [episode("e", 0, 3), episode("f", 0, 4)]}
    report = "\n".join(split_report(runs))
    assert "validation: scenes absent from train = ['suite/1/l1']" in report
    assert "Shared task/level scenes across all splits: 1" in report
    assert counts(runs["train_1"], True) == {(16, 12): 2}
    # Different numbers of episodes alone must not change a normalized histogram.
    assert frequency_distance(counts(runs["train_1"]), counts(runs["train_1"] * 2)) == 0


def test_audit_rejects_duplicate_or_failed_results(tmp_path):
    row = episode("a", 0, 0)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n" + json.dumps(row))
    with pytest.raises(ValueError, match="duplicate item"):
        read_run(tmp_path)
    row.update(status="error", error="simulator died")
    manifest.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="invalid episode"):
        read_run(tmp_path)
    row.update(status="ok", error=None)
    row["controller_summary"]["policy_requests"] = 3
    manifest.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="incomplete decisions"):
        read_run(tmp_path)
