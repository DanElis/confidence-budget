"""Statistics for H1: does action entropy track action error, beyond the action index?"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy.stats import rankdata

from .signal import ZONE_FROZEN, mask_zones, plateau_horizon


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    y = np.asarray(y, dtype=np.float64) - np.mean(y)
    denom = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
    return float(np.dot(x, y) / denom) if denom > 0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def partial_spearman(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Spearman correlation after removing per-group mean ranks.

    With ``groups`` = action index this answers: at the same position in the
    chunk, do higher-entropy actions have larger errors? Both entropy and error
    grow with the index, so the plain coefficient is inflated by position alone.
    """
    rx, ry = rankdata(x), rankdata(y)
    _, gid = np.unique(groups, return_inverse=True)
    counts = np.bincount(gid)
    rx = rx - (np.bincount(gid, rx) / counts)[gid]
    ry = ry - (np.bincount(gid, ry) / counts)[gid]
    return pearson(rx, ry)


def cluster_bootstrap(
    stat: Callable[[np.ndarray], float],
    clusters: np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI, resampling whole clusters (episodes).

    ``stat`` receives an array of row indices and returns a scalar.
    """
    clusters = np.asarray(clusters)
    ids = np.unique(clusters)
    rows_by_cluster = {c: np.flatnonzero(clusters == c) for c in ids}
    estimate = stat(np.arange(clusters.shape[0]))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        sample = rng.choice(ids, size=ids.shape[0], replace=True)
        rows = np.concatenate([rows_by_cluster[c] for c in sample])
        values.append(stat(rows))
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return estimate, float("nan"), float("nan")
    alpha = (1.0 - level) / 2.0
    return estimate, float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def per_action_error(pred_raw: np.ndarray, gt_raw: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    """Mean squared error per action, each dimension divided by its dataset std. Shape (C, H)."""
    std = np.maximum(np.asarray(action_std, dtype=np.float64), 1e-6)
    diff = (pred_raw.astype(np.float64) - gt_raw.astype(np.float64)) / std
    return np.mean(diff**2, axis=-1)


def select_step(per_step: np.ndarray, denoise_step: int | None) -> np.ndarray:
    """(C, S, H) -> (C, H): one denoising step, or the mean over steps if ``None``."""
    if denoise_step is None:
        return per_step.mean(axis=1)
    return per_step[:, denoise_step, :]


def action_table(
    data: dict[str, np.ndarray],
    action_std: np.ndarray,
    delay: int,
    mask_end: int,
    denoise_step: int | None = -1,
    plateau: dict | None = None,
) -> dict[str, np.ndarray]:
    """One row per (chunk, action index) with a real demonstration target.

    ``denoise_step`` selects the denoising step whose attention is used
    (-1 = last, closest to the clean chunk); ``None`` averages all steps.

    Columns: chunk, episode, is_rtc, j, zone, entropy (nats), entropy_norm
    (entropy / log n_valid), error, nonlang_share, h_star, plateau_found.
    """
    plateau = plateau or {}
    n_chunks, horizon = data["gt_pad"].shape
    entropy = select_step(data["entropy"], denoise_step)  # (C, H)
    share = select_step(data["nonlang_share"], denoise_step)
    log_n = np.log(data["n_valid"].astype(np.float64))[:, None]
    error = per_action_error(data["pred_raw"], data["gt_raw"], action_std)
    zones = np.broadcast_to(mask_zones(horizon, delay, mask_end), (n_chunks, horizon))

    h_star = np.empty(n_chunks, dtype=np.int64)
    found = np.empty(n_chunks, dtype=bool)
    for c in range(n_chunks):
        h_star[c], found[c] = plateau_horizon(entropy[c], int(data["n_valid"][c]), **plateau)

    keep = ~data["gt_pad"]
    chunk_idx = np.broadcast_to(np.arange(n_chunks)[:, None], (n_chunks, horizon))
    j_idx = np.broadcast_to(np.arange(horizon)[None, :], (n_chunks, horizon))
    return {
        "chunk": chunk_idx[keep],
        "episode": np.broadcast_to(data["episode"][:, None], (n_chunks, horizon))[keep],
        "is_rtc": np.broadcast_to(data["is_rtc"][:, None], (n_chunks, horizon))[keep],
        "j": j_idx[keep],
        "zone": zones[keep],
        "entropy": entropy[keep],
        "entropy_norm": (entropy / log_n)[keep],
        "error": error[keep],
        "nonlang_share": share[keep],
        "h_star": np.broadcast_to(h_star[:, None], (n_chunks, horizon))[keep],
        "plateau_found": np.broadcast_to(found[:, None], (n_chunks, horizon))[keep],
        "chunk_h_star": h_star,
        "chunk_plateau_found": found,
    }


def frozen_deviation(data: dict[str, np.ndarray], delay: int) -> np.ndarray:
    """RMSE between each chunk and the previous leftover on the frozen prefix, model space. Shape (C,)."""
    if delay == 0:
        return np.full(data["pred"].shape[0], np.nan)
    diff = data["pred"][:, :delay] - data["prev"][:, :delay]
    return np.sqrt(np.nanmean(diff.astype(np.float64) ** 2, axis=(1, 2)))


def correlation_report(
    table: dict[str, np.ndarray], rows: np.ndarray, n_boot: int, seed: int
) -> dict[str, float]:
    """Raw and index-controlled Spearman on ``rows`` with episode-bootstrap CIs."""
    entropy, error, j, episode = table["entropy"], table["error"], table["j"], table["episode"]

    def raw(idx):
        r = rows[idx]
        return spearman(entropy[r], error[r])

    def partial(idx):
        r = rows[idx]
        return partial_spearman(entropy[r], error[r], j[r])

    raw_est, raw_lo, raw_hi = cluster_bootstrap(raw, episode[rows], n_boot, seed)
    par_est, par_lo, par_hi = cluster_bootstrap(partial, episode[rows], n_boot, seed)
    return {
        "n_pairs": int(rows.shape[0]),
        "n_episodes": int(np.unique(episode[rows]).shape[0]),
        "spearman": raw_est,
        "spearman_ci": [raw_lo, raw_hi],
        "partial_spearman": par_est,
        "partial_spearman_ci": [par_lo, par_hi],
    }


def difference_report(
    table: dict[str, np.ndarray], rows: np.ndarray, n_boot: int, seed: int
) -> dict[str, float]:
    """Index-controlled Spearman under RTC minus without guidance, paired by episode."""
    entropy, error, j = table["entropy"], table["error"], table["j"]
    is_rtc, episode = table["is_rtc"], table["episode"]

    def diff(idx):
        r = rows[idx]
        rtc, free = r[is_rtc[r]], r[~is_rtc[r]]
        return partial_spearman(entropy[rtc], error[rtc], j[rtc]) - partial_spearman(
            entropy[free], error[free], j[free]
        )

    est, lo, hi = cluster_bootstrap(diff, episode[rows], n_boot, seed)
    return {"partial_spearman_rtc_minus_free": est, "ci": [lo, hi]}


def horizon_report(table: dict[str, np.ndarray], is_rtc: bool) -> dict[str, float]:
    """Does h* separate low-error from high-error actions? Excludes the frozen prefix."""
    rows = (table["is_rtc"] == is_rtc) & (table["zone"] != ZONE_FROZEN)
    before = rows & (table["j"] < table["h_star"])
    after = rows & (table["j"] >= table["h_star"])
    chunk_rows = np.unique(table["chunk"][rows])
    found = table["chunk_plateau_found"][chunk_rows]
    h_star = table["chunk_h_star"][chunk_rows]
    mean_before = float(table["error"][before].mean()) if before.any() else float("nan")
    mean_after = float(table["error"][after].mean()) if after.any() else float("nan")
    return {
        "plateau_found_share": float(found.mean()) if found.size else float("nan"),
        "h_star_mean": float(h_star.mean()) if h_star.size else float("nan"),
        "h_star_median": float(np.median(h_star)) if h_star.size else float("nan"),
        "error_before_h_star": mean_before,
        "error_after_h_star": mean_after,
        "error_ratio_after_before": mean_after / mean_before if mean_before > 0 else float("nan"),
    }


def partial_by_step(
    data: dict[str, np.ndarray], action_std: np.ndarray, delay: int, mask_end: int, is_rtc: bool
) -> list[float]:
    """Index-controlled Spearman for every denoising step (no CI), outside the frozen prefix."""
    out = []
    for step in range(data["entropy"].shape[1]):
        t = action_table(data, action_std, delay, mask_end, denoise_step=step)
        rows = (t["is_rtc"] == is_rtc) & (t["zone"] != ZONE_FROZEN)
        out.append(partial_spearman(t["entropy"][rows], t["error"][rows], t["j"][rows]))
    return out


def h1_verdict(report: dict, criterion: str = "partial", threshold: float = 0.15) -> dict:
    """H1 holds if the RTC coefficient reaches ``threshold`` and its CI lies above zero.

    ``report`` is a ``correlation_report`` for RTC chunks outside the frozen prefix.
    """
    key = {"partial": "partial_spearman", "raw": "spearman"}[criterion]
    value, (lo, _hi) = report[key], report[key + "_ci"]
    passed = bool(np.isfinite(value) and value >= threshold and np.isfinite(lo) and lo > 0)
    return {"criterion": criterion, "threshold": threshold, "value": value, "ci_low": lo, "passed": passed}
