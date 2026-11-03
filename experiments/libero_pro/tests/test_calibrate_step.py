"""Threshold of the dispersion-based adaptive step (scripts/calibrate_step.py)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrate_step import calibrate, harmonic, main, steps  # noqa: E402
from hstar import HStarRule, smooth  # noqa: E402
from planner import AdaptivePlanner  # noqa: E402


def profiles(n=400, horizon=50, seed=0):
    g = np.random.default_rng(seed)
    return np.cumsum(g.uniform(0, 0.02, (n, horizon)), axis=1) * g.uniform(0.3, 3, (n, 1))


def test_steps_match_the_planner():
    disp = profiles(50)
    bar = np.stack([smooth(d, 3) for d in disp])
    for eta in (0.05, 0.3, 1.0, 100.0):
        planner = AdaptivePlanner(50, HStarRule("above", eta=eta, k=3, min_h=1), s_min=8, s_max=25)
        want = [planner(d, 0, 8).stride for d in disp]
        assert steps(bar, eta, 8, 8, 25).tolist() == want


def test_calibrate_matches_the_harmonic_mean_not_the_mean():
    eta, s = calibrate(profiles(), 12, 8, 8, 25)
    assert harmonic(s) == pytest.approx(12, abs=0.2) and s.mean() > harmonic(s)
    assert s.min() == 8 and s.max() > 12 and eta > 0


def test_unreachable_target():
    with pytest.raises(SystemExit):
        calibrate(profiles(), 30, 8, 8, 25)


def test_main_reads_both_log_formats_and_achieved(tmp_path, capsys):
    disp = profiles(100).round(5).tolist()
    pro = tmp_path / "pro"
    pro.mkdir()
    (pro / "suite_chunks.json").write_text(json.dumps([{"dispersion": d, "stride": 10} for d in disp]))
    (pro / "suite.json").write_text("{}")  # the summary of a suite is not a chunk log
    dyn = tmp_path / "dyn"
    dyn.mkdir()
    (dyn / "signals_0.jsonl").write_text("".join(json.dumps({"disp": d, "s": 7}) + "\n" for d in disp))
    args = ["--target", "12", "--delay", "8", "--s-min", "8", "--s-max", "25"]
    etas = []
    for run in (pro, dyn):
        main([str(run), *args, "--out", str(run / "step.json")])
        out = json.loads((run / "step.json").read_text())
        assert HStarRule.from_dict(out["rule"]).kind == "above"
        etas.append(out["rule"]["eta"])
    assert etas[0] == pytest.approx(etas[1])
    capsys.readouterr()
    main([str(pro), "--achieved"])
    main([str(dyn), "--achieved"])
    assert capsys.readouterr().out.split() == ["10", "7"]
