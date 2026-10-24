import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import analyze_signals as S  # noqa: E402

H, A, STRIDE = 50, 14, 25


def make_run(d: Path, linked: bool, seed: int):
    """Plans: the next chunk revises the old one at index i by noise of scale disp[i] (linked) or 0.1."""
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True)
    for task in ("lift_pot", "stamp_seal"):
        eps, chunks = [], []
        for i in range(30):
            success = bool(rng.random() < 0.5)
            length = 150 if success else 400
            plan = rng.normal(size=(H, A))
            for step in range(0, length, STRIDE):
                disp = 0.05 + 0.002 * np.arange(H) + rng.gamma(2.0, 0.05, H)
                chunks.append({"init_state": i, "step": step, "mask_end": 0, "dispersion": disp.tolist(),
                               "plan": plan.tolist(), "entropy": (np.full(H, 0.65) * math.log(800)).tolist(),
                               "n_valid": 800})
                scale = disp[STRIDE:] if linked else np.full(H - STRIDE, 0.1)
                new = np.empty((H, A))
                new[: H - STRIDE] = plan[STRIDE:] + rng.normal(size=(H - STRIDE, A)) * scale[:, None]
                new[H - STRIDE:] = rng.normal(size=(STRIDE, A))
                plan = new
            eps.append({"init_state": i, "success": success, "length": length})
        (d / f"{task}.json").write_text(json.dumps({"episodes": eps}))
        (d / f"{task}_chunks.json").write_text(json.dumps(chunks))


def test_plan_revision_test_detects_link(tmp_path):
    make_run(tmp_path / "sync_linked", True, 0)
    make_run(tmp_path / "sync_null", False, 1)
    out = tmp_path / "rep.md"
    S.main([str(tmp_path / "sync_linked"), str(tmp_path / "sync_null"), "--n-boot", "100", "--n-perm", "100",
            "--out", str(out)])
    text = out.read_text()
    assert "sync_linked, B dispersion: signal" in text
    assert "sync_null, B dispersion: no signal" in text, text[-800:]
    rows = S.load(tmp_path / "sync_linked")
    names = S.features(rows)
    eps = S.episodes(rows, names, 100)
    rho, lo, hi, n = S.test_b(eps, "dispersion", type("a", (), {"n_boot": 100})(), np.random.default_rng(0))
    assert rho > 0.2 and lo > 0.1
