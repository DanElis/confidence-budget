#!/usr/bin/env python
"""Stage 2: choose the h* rule on offline H1 data and calibrate entropy-based lambda.

Reference horizon g of a chunk: the number of actions before the error against
the demonstration rises for good, i.e. the first index j where the mean error
over [j, j + k_err) exceeds tau_err = err_mult x the median error of the first
k_err actions on training chunks; g = H if that never happens.

Every candidate rule (absolute / relative / rise / const, see hstar.py) is
scored on training episodes by MAE(h*, g); the best of each family and the
overall best are then reported on held-out episodes together with
Spearman(h*, g), the error ratio after/before h*, the spread of h* and the share
of chunks with h* = H. A rule is only useful if it beats the best constant.

Optionally, conflict runs (e.g. h1_stale25_lerobot.npz) are compared with the
reference run: a useful rule gives a shorter h* under conflict.

Also writes ``lambda_entropy`` (mu, sigma of the normalised entropy of the first
``--zone`` actions of no-conflict RTC chunks) for ``--lam entropy``.

Example:
    python scripts/calibrate_hstar.py ../h1/runs/h1_lerobot.npz \\
        --conflict ../h1/runs/h1_stale25_lerobot.npz ../h1/runs/h1_stale100_lerobot.npz \\
        --out runs/hstar.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hstar import HStarRule, hstar  # noqa: E402

GRID = {
    "absolute": [dict(eta=e, k=k, tau=t) for e, k, t in itertools.product(
        [round(float(x), 3) for x in np.arange(0.55, 0.80, 0.01)], (3, 5, 7), (0.005, 0.01, 0.02, 0.05))],
    "relative": [dict(q=q, k=k) for q, k in itertools.product(
        [round(float(x), 3) for x in np.arange(0.2, 0.96, 0.05)], (1, 3, 5, 7))],
    "rise": [dict(k=k, min_h=m) for k, m in itertools.product((1, 3, 5, 7), (3, 5, 10, 15))],
    "const": [dict(c=c) for c in range(5, 51, 1)],
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path, nargs="+", help="H1 .npz files used for calibration (no conflict)")
    p.add_argument("--conflict", type=Path, nargs="*", default=[], help="conflict runs for the sanity check")
    p.add_argument("--closed-loop", type=Path, nargs="*", default=[],
                   help="eval_libero.py output dirs run with --record-entropy: h* in failed vs successful episodes, "
                        "and lambda_entropy calibrated on closed-loop chunks instead of H1 chunks")
    p.add_argument("--condition", choices=["rtc", "free", "both"], default="rtc")
    p.add_argument("--err-mult", type=float, default=2.0, help="tau_err = err_mult x median early error")
    p.add_argument("--k-err", type=int, default=5)
    p.add_argument("--zone", type=int, default=10, help="early actions used by entropy-based lambda")
    p.add_argument("--z-max", type=float, default=2.0)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def load(paths, condition: str) -> dict:
    parts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            std = np.maximum(np.asarray(meta["action_std"], dtype=np.float64), 1e-6)
            sel = np.ones(len(z["is_rtc"]), bool) if condition == "both" else (z["is_rtc"] == (condition == "rtc"))
            sel &= ~z["gt_pad"].any(axis=1)  # whole chunk inside the episode
            err = (((z["pred_raw"][sel] - z["gt_raw"][sel]) / std) ** 2).mean(-1)
            parts.append({
                "entropy": z["entropy"][sel][:, -1, :].astype(np.float64),
                "n_valid": z["n_valid"][sel],
                "error": err,
                "episode": z["episode"][sel] + 100_000 * len(parts),
            })
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def reference_horizon(error: np.ndarray, tau: float, k: int) -> np.ndarray:
    c, h = error.shape
    out = np.full(c, h, dtype=np.int64)
    for i in range(c):
        for j in range(h):
            if error[i, j : j + k].mean() > tau:
                out[i] = j
                break
    return out


def apply(rule: HStarRule, data: dict) -> np.ndarray:
    return np.array([hstar(e, int(n), rule) for e, n in zip(data["entropy"], data["n_valid"], strict=True)])


def score(h: np.ndarray, g: np.ndarray, error: np.ndarray) -> dict:
    horizon = error.shape[1]
    j = np.arange(horizon)[None, :]
    before, after = j < h[:, None], j >= h[:, None]
    eb = error[before].mean() if before.any() else float("nan")
    ea = error[after].mean() if after.any() else float("nan")
    rho = spearmanr(h, g).statistic if np.std(h) > 0 and np.std(g) > 0 else float("nan")
    return {"mae": float(np.abs(h - g).mean()), "spearman": float(rho), "ratio_after_before": float(ea / eb),
            "h_mean": float(h.mean()), "h_std": float(h.std()), "share_H": float((h == horizon).mean())}


def subset(data: dict, mask: np.ndarray) -> dict:
    return {k: v[mask] for k, v in data.items()}


def closed_loop_check(dirs, rule: HStarRule, zone: int) -> dict:
    """Mean h* per episode, split by success; early-zone entropy of every closed-loop chunk."""
    h_by_outcome = {True: [], False: []}
    zone_means = []
    for d in dirs:
        for chunks_file in sorted(Path(d).glob("*_chunks.json")):
            suite = json.loads(chunks_file.with_name(chunks_file.name.replace("_chunks", "")).read_text())
            outcome = {(e["task_id"], e["init_state"]): e["success"] for e in suite["episodes"]}
            per_episode = {}
            for c in json.loads(chunks_file.read_text()):
                if "entropy" not in c:
                    continue
                e, n = np.asarray(c["entropy"]), int(c["n_valid"])
                per_episode.setdefault((c["task_id"], c["init_state"]), []).append(hstar(e, n, rule))
                zone_means.append(float((e[:zone] / math.log(max(n, 2))).mean()))
            for key, hs in per_episode.items():
                if key in outcome:
                    h_by_outcome[outcome[key]].append(float(np.mean(hs)))
    mean = lambda x: float(np.mean(x)) if x else float("nan")  # noqa: E731
    return {"h_success": mean(h_by_outcome[True]), "h_failure": mean(h_by_outcome[False]),
            "n_success": len(h_by_outcome[True]), "n_failure": len(h_by_outcome[False]), "zone_means": zone_means}


def main(argv=None) -> None:
    args = parse_args(argv)
    data = load(args.data, args.condition)
    train = (data["episode"] % 2) == 0
    tr, te = subset(data, train), subset(data, ~train)
    tau = float(args.err_mult * np.median(tr["error"][:, : args.k_err].mean(axis=1)))
    g_tr = reference_horizon(tr["error"], tau, args.k_err)
    g_te = reference_horizon(te["error"], tau, args.k_err)
    print(f"chunks: train {len(g_tr)}, test {len(g_te)}; tau_err = {tau:.4f}; "
          f"reference horizon g: mean {g_te.mean():.1f}, median {np.median(g_te):.0f}, share H {np.mean(g_te == 50):.2f}")

    best = {}
    for kind, grid in GRID.items():
        scored = []
        for params in grid:
            rule = HStarRule(kind=kind, **params)
            scored.append((score(apply(rule, tr), g_tr, tr["error"])["mae"], rule))
        mae, rule = min(scored, key=lambda x: x[0])
        best[kind] = {"rule": rule.to_dict(), "train": score(apply(rule, tr), g_tr, tr["error"]),
                      "test": score(apply(rule, te), g_te, te["error"])}

    chosen_kind = min((k for k in best if k != "const"), key=lambda k: best[k]["train"]["mae"])
    chosen = HStarRule.from_dict(best[chosen_kind]["rule"])
    beats_const = best[chosen_kind]["test"]["mae"] < best["const"]["test"]["mae"]

    conflict = []
    if args.conflict:
        ref_h = apply(chosen, data)
        for path in args.conflict:
            other = load([path], args.condition)
            conflict.append({"run": path.stem, "h_mean": float(apply(chosen, other).mean()),
                             "reference_h_mean": float(ref_h.mean())})

    early = data["entropy"][:, : args.zone] / np.log(np.maximum(data["n_valid"], 2))[:, None]
    zone_mean = early.mean(axis=1)
    lam_source = "h1"
    closed = closed_loop_check(args.closed_loop, chosen, args.zone) if args.closed_loop else None
    if closed and closed["zone_means"]:
        zone_mean, lam_source = np.asarray(closed.pop("zone_means")), "closed_loop"
    elif closed:
        closed.pop("zone_means")
    lam_cfg = {"mu": float(zone_mean.mean()), "sigma": float(zone_mean.std()), "zone": args.zone,
               "z_max": args.z_max, "source": lam_source}

    result = {"rule": chosen.to_dict(), "chosen_family": chosen_kind, "beats_constant_on_test": bool(beats_const),
              "tau_err": tau, "families": best, "conflict_check": conflict, "lambda_entropy": lam_cfg,
              "closed_loop_check": closed,
              "inputs": [str(p) for p in args.data], "condition": args.condition}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print("\n| family | params | test MAE | Spearman(h*, g) | err after/before | mean h* | std h* | share h*=H |")
    print("|---|---|---|---|---|---|---|---|")
    for kind, b in best.items():
        params = {k: v for k, v in b["rule"].items() if k in GRID[kind][0]}
        t = b["test"]
        print(f"| {kind}{' *' if kind == chosen_kind else ''} | {params} | {t['mae']:.2f} | {t['spearman']:.3f} "
              f"| {t['ratio_after_before']:.2f} | {t['h_mean']:.1f} | {t['h_std']:.1f} | {t['share_H']:.2f} |")
    print(f"\nchosen: {chosen_kind} {chosen.to_dict()}; beats the best constant on test: {beats_const}")
    for c in conflict:
        print(f"conflict {c['run']}: mean h* {c['h_mean']:.1f} vs reference {c['reference_h_mean']:.1f}")
    if closed:
        print(f"closed loop: mean h* in successful episodes {closed['h_success']:.1f}, "
              f"in failed {closed['h_failure']:.1f} ({closed['n_success']} / {closed['n_failure']} episodes)")
    print(f"lambda_entropy: {lam_cfg}")
    print(f"written: {args.out}")
    if not math.isfinite(best[chosen_kind]["test"]["spearman"]):
        print("warning: h* is constant on the test chunks")


if __name__ == "__main__":
    main()
