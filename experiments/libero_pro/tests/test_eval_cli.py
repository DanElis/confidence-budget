import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_libero import make_planner, parse_args, parse_rule  # noqa: E402


def test_parse_rule():
    r = parse_rule("kind=relative,q=0.6,k=3")
    assert r.kind == "relative" and r.q == 0.6 and r.k == 3


def test_planners_from_cli(tmp_path):
    fixed = make_planner(parse_args(["--out", "x", "--stride", "10"]), 50)
    assert fixed(None, None, 4).stride == 10
    cfg = tmp_path / "hstar.json"
    cfg.write_text(json.dumps({"rule": {"kind": "const", "c": 30},
                               "lambda_entropy": {"mu": 0.6, "sigma": 0.02, "zone": 10, "z_max": 2.0}}))
    budget = make_planner(parse_args(["--out", "x", "--planner", "budget", "--hstar-config", str(cfg), "--lam", "1"]), 50)
    assert budget.needs_entropy and budget.rule.c == 30 and budget.lam == 1.0
    ent = make_planner(parse_args(["--out", "x", "--planner", "budget", "--hstar-config", str(cfg),
                                   "--lam", "entropy", "--lam-config", str(cfg)]), 50)
    assert ent.lam is None and ent.lam_fn.mu == 0.6
