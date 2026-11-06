"""gate_summary.py: numbers of the RTC gate from a run's chunk logs, the threshold and the selection rule."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gate_summary import load, main, select, tau_for  # noqa: E402


def write_run(path: Path, divergence, guided, success=(0.5, 0.3), tau=0.2):
    path.mkdir()
    chunks = [{"step": 0, "divergence": -1.0, "guided": 0.0, "calls": 1.0}]  # first chunk: left out
    chunks += [{"step": 5 * (i + 1), "divergence": d, "guided": g, "calls": 1.0 + g} for i, (d, g) in enumerate(zip(divergence, guided))]
    (path / "libero_object_pos_y0.1_chunks.json").write_text(json.dumps(chunks))
    for name, rate in zip(("libero_object_pos_y0.1", "libero_object_pos_y0.2"), success):
        (path / f"{name}.json").write_text(json.dumps({"success_rate": rate}))
    (path / "config.json").write_text(json.dumps({"rtc_tau": str(tau)}))
    (path / "libero_object_pos_y0.3.progress.json").write_text("{}")  # a stopped suite: not a result
    return path


def test_load_leaves_out_requests_without_an_old_plan(tmp_path):
    d, guided, calls, success = load(write_run(tmp_path / "r", [0.1, 0.3], [1.0, 0.0]))
    np.testing.assert_array_equal(d, [0.1, 0.3])
    np.testing.assert_array_equal(calls, [2.0, 1.0])
    assert sorted(success) == [0.3, 0.5]


def test_runs_without_the_gate_are_refused(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "a_chunks.json").write_text(json.dumps([{"step": 0}]))
    with pytest.raises(SystemExit, match="gate"):
        load(run)


def test_tau_for_gives_the_share():
    d = np.arange(100) / 100
    for share in (0.25, 0.5, 0.75):
        assert np.mean(d < tau_for(d, share)) == pytest.approx(share, abs=0.02)
    assert tau_for(d, 0) == 0.0 and np.mean(d < tau_for(d, 1)) == 1.0


def test_select_prefers_success_then_the_larger_share_within_a_point():
    assert select({"a": (0.50, 0.25), "b": (0.48, 0.75)}) == "a"  # 2 pp worse: not a tie
    assert select({"a": (0.50, 0.25), "b": (0.495, 0.75)}) == "b"  # within 1 pp: more smoothing wins
    assert select({"a": (0.50, 0.25), "b": (0.50, 0.25)}) == "a"  # full tie: by name, deterministic


def test_cli(tmp_path, capsys):
    one = write_run(tmp_path / "q25", [0.1, 0.3, 0.2, 0.4], [1.0, 0.0, 1.0, 0.0], success=(0.5, 0.5), tau=0.25)
    two = write_run(tmp_path / "q75", [0.1, 0.3, 0.2, 0.4], [1.0, 1.0, 1.0, 0.0], success=(0.6, 0.4), tau=0.35)
    main([str(one), "--print", "share"])
    assert float(capsys.readouterr().out) == pytest.approx(0.5)
    main([str(one), str(two), "--select"])  # equal success, the larger share
    assert capsys.readouterr().out.split() == [str(two), "0.350000"]
    main([str(one), "--tau-for", "0.5"])
    assert float(capsys.readouterr().out) == pytest.approx(0.25)
    main([str(one), str(two)])
    assert "q75" in capsys.readouterr().out
