"""calibrate_hstar.py recovers an entropy rule that tracks a planted error onset."""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrate_hstar import main  # noqa: E402

H, A, N = 50, 7, 500


def fake_run(path, seed, onset_shift=0):
    rng = np.random.default_rng(seed)
    c = 240
    onset = rng.integers(10, 40, size=c) - onset_shift
    j = np.arange(H)[None, :]
    # entropy rises at the onset (sigmoid), error jumps at the same place
    e_norm = 0.60 + 0.06 / (1 + np.exp(-(j - onset[:, None]) / 1.5)) + rng.normal(0, 0.002, (c, H))
    entropy = (e_norm * math.log(N))[:, None, :].repeat(3, axis=1)
    err_scale = np.where(j >= onset[:, None], 1.0, 0.1)
    gt = rng.normal(size=(c, H, A))
    pred = gt + np.sqrt(err_scale)[..., None] * rng.normal(size=(c, H, A))
    np.savez(path, episode=np.repeat(np.arange(c // 4), 4), is_rtc=np.tile([False, True], c // 2),
             pred_raw=pred.astype(np.float32), gt_raw=gt.astype(np.float32), gt_pad=np.zeros((c, H), bool),
             entropy=entropy.astype(np.float32), n_valid=np.full(c, N),
             meta=np.array(json.dumps({"action_std": [1.0] * A})))


def test_calibration_finds_signal_and_conflict(tmp_path, capsys):
    base, conflict = tmp_path / "base.npz", tmp_path / "conflict.npz"
    fake_run(base, 0)
    fake_run(conflict, 1, onset_shift=8)
    out = tmp_path / "hstar.json"
    main([str(base), "--conflict", str(conflict), "--out", str(out)])
    res = json.loads(out.read_text())
    assert res["beats_constant_on_test"]
    chosen = res["families"][res["chosen_family"]]["test"]
    assert chosen["spearman"] > 0.7
    assert res["conflict_check"][0]["h_mean"] < res["conflict_check"][0]["reference_h_mean"]
    lam = res["lambda_entropy"]
    assert 0.59 < lam["mu"] < 0.62 and lam["zone"] == 10


def test_closed_loop_check_uses_recorded_chunks(tmp_path):
    base = tmp_path / "base.npz"
    fake_run(base, 0)
    run = tmp_path / "cl"
    run.mkdir()
    episodes, chunks = [], []
    for i in range(6):
        success = i % 2 == 0
        episodes.append({"task_id": 0, "init_state": i, "success": success})
        onset = 35 if success else 12  # failures: entropy rises early
        e = (0.60 + 0.06 / (1 + np.exp(-(np.arange(H) - onset) / 1.5))) * math.log(N)
        chunks.append({"task_id": 0, "init_state": i, "step": 0, "entropy": e.tolist(), "n_valid": N})
    (run / "suite.json").write_text(json.dumps({"episodes": episodes}))
    (run / "suite_chunks.json").write_text(json.dumps(chunks))
    out = tmp_path / "h.json"
    main([str(base), "--closed-loop", str(run), "--out", str(out)])
    res = json.loads(out.read_text())
    cl = res["closed_loop_check"]
    assert cl["n_success"] == 3 and cl["n_failure"] == 3
    assert cl["h_failure"] < cl["h_success"]
    assert res["lambda_entropy"]["source"] == "closed_loop"
