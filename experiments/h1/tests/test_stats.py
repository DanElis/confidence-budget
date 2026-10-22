import numpy as np

from h1.signal import ZONE_FROZEN
from h1.stats import (
    action_table,
    cluster_bootstrap,
    correlation_report,
    frozen_deviation,
    h1_verdict,
    partial_spearman,
    per_action_error,
    spearman,
)


def test_partial_spearman_removes_index_confound():
    rng = np.random.default_rng(0)
    j = np.repeat(np.arange(50), 200)
    x = j + rng.normal(0, 5, j.size)
    y = j + rng.normal(0, 5, j.size)  # related to x only through j
    assert spearman(x, y) > 0.7
    assert abs(partial_spearman(x, y, j)) < 0.03


def test_partial_spearman_keeps_within_index_link():
    rng = np.random.default_rng(1)
    j = np.repeat(np.arange(50), 200)
    z = rng.normal(0, 1, j.size)
    x = j + z
    y = 0.1 * j + z + rng.normal(0, 1, j.size)
    assert partial_spearman(x, y, j) > 0.5


def test_cluster_bootstrap_brackets_estimate():
    rng = np.random.default_rng(2)
    clusters = np.repeat(np.arange(40), 50)
    x = rng.normal(size=clusters.size)
    y = x + rng.normal(size=clusters.size)
    est, lo, hi = cluster_bootstrap(lambda idx: spearman(x[idx], y[idx]), clusters, n_boot=200)
    assert lo < est < hi and lo > 0


def test_per_action_error_scales_by_std():
    pred = np.zeros((1, 2, 2))
    gt = np.array([[[2.0, 0.0], [0.0, 4.0]]])
    err = per_action_error(pred, gt, np.array([2.0, 4.0]))
    assert np.allclose(err, [[0.5, 0.5]])


def _synthetic(n_episodes=30, chunks=4, horizon=50, steps=3, signal=1.0, seed=0):
    """Chunks whose per-action error rises with entropy at a fixed index when ``signal`` > 0."""
    rng = np.random.default_rng(seed)
    n = n_episodes * chunks * 2
    j = np.arange(horizon)[None, :]
    z = rng.normal(size=(n, horizon))
    n_valid = np.full(n, 500)
    entropy = (0.6 + 0.006 * j + 0.03 * z) * np.log(500)
    err_scale = np.sqrt(np.exp(0.03 * j + signal * z + 0.3 * rng.normal(size=(n, horizon))))
    gt = rng.normal(size=(n, horizon, 7))
    pred = gt + err_scale[..., None] * rng.choice([-1.0, 1.0], size=(n, horizon, 7))
    gt_pad = np.zeros((n, horizon), dtype=bool)
    gt_pad[:5, -10:] = True
    prev = np.full((n, horizon, 7), np.nan)
    prev[:, :25] = pred[:, :25] + 0.1
    return {
        "episode": np.repeat(np.arange(n_episodes), chunks * 2),
        "is_rtc": np.tile([False, True], n_episodes * chunks),
        "pred": pred.astype(np.float32),
        "pred_raw": pred.astype(np.float32),
        "gt_raw": gt.astype(np.float32),
        "gt_pad": gt_pad,
        "prev": prev.astype(np.float32),
        "entropy": np.repeat(entropy[:, None, :], steps, axis=1).astype(np.float32),
        "nonlang_share": np.full((n, steps, horizon), 0.9, dtype=np.float32),
        "n_valid": n_valid,
    }


def test_action_table_drops_padding_and_marks_zones():
    data = _synthetic()
    table = action_table(data, np.ones(7), delay=5, mask_end=25)
    assert table["error"].size == data["gt_pad"].size - data["gt_pad"].sum()
    assert np.all(table["zone"][table["j"] < 5] == ZONE_FROZEN)
    assert table["chunk_h_star"].shape == (data["episode"].size,)


def test_verdict_passes_with_signal_and_fails_without():
    for signal, expected in ((1.0, True), (0.0, False)):
        data = _synthetic(signal=signal)
        table = action_table(data, np.ones(7), delay=5, mask_end=25)
        rows = np.flatnonzero(table["is_rtc"] & (table["zone"] != ZONE_FROZEN))
        report = correlation_report(table, rows, n_boot=100, seed=0)
        assert h1_verdict(report)["passed"] is expected
        assert report["spearman"] > 0.3  # the raw coefficient is inflated by the index either way


def test_frozen_deviation():
    data = _synthetic()
    dev = frozen_deviation(data, delay=5)
    assert np.allclose(dev, 0.1, atol=1e-5)
