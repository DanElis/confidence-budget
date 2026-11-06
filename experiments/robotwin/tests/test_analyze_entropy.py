import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import analyze_entropy as A  # noqa: E402


def make_run(d: Path, effect: float, seed=0):
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True)
    n_valid = 800
    for t, task in enumerate(("lift_pot", "stamp_seal")):
        eps, chunks = [], []
        for i in range(40):
            success = bool(rng.random() < 0.5)
            length = int(rng.integers(100, 200)) if success else 400
            base = 0.6 + 0.05 * t + (0 if success else effect)
            for step in range(0, length, 25):
                e = (base + 0.002 * np.arange(50) + rng.normal(0, 0.01, 50)) * math.log(n_valid)
                chunks.append({"task_id": 0, "init_state": i, "step": step, "entropy": e.tolist(), "n_valid": n_valid})
            eps.append({"task_id": 0, "init_state": i, "success": success, "length": length, "n_inference": 1})
        (d / f"{task}.json").write_text(json.dumps({"episodes": eps}))
        (d / f"{task}_chunks.json").write_text(json.dumps(chunks))


def test_detects_signal_and_its_absence(tmp_path):
    make_run(tmp_path / "signal", effect=0.02)
    make_run(tmp_path / "null", effect=0.0, seed=1)
    out = tmp_path / "rep.md"
    A.main([str(tmp_path / "signal"), str(tmp_path / "null"), "--n-boot", "200", "--n-perm", "500", "--out", str(out)])
    text = out.read_text()
    assert "- signal: signal." in text and "higher in episodes that fail" in text
    assert "- null: no episode-level signal" in text, text[-600:]
    rng = np.random.default_rng(0)
    rows = A.load_run(tmp_path / "signal", 10)
    A.zscore_within_task(rows)
    eps = A.episode_table(rows, 100)
    point, lo, hi = A.boot_auroc(eps, "zone", 200, rng)
    assert point > 0.9 and lo > 0.8


def test_auroc():
    s = np.array([0.1, 0.2, 0.3, 0.4])
    assert A.auroc(s, np.array([False, False, True, True])) == 1.0
    assert A.auroc(s, np.array([True, True, False, False])) == 0.0
