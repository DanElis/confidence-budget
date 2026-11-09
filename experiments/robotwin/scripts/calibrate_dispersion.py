#!/usr/bin/env python
"""Threshold for the dispersion-based h* ("above" rule) giving a target mean h*.

h* = first action index where the smoothed dispersion across noise samples reaches eta (at least
min_h), as in AAC. eta is set so that the mean h* over the recorded chunks equals --target: an
adaptive configuration then has the same average horizon (and inference cost) as a constant one
that has been run already, and the comparison isolates adaptivity. Episode outcomes are not used.

Example:
    uv run --project ../libero_pro python scripts/calibrate_dispersion.py runs_samples/sync_s25_k8 \\
        --target 25 --out runs_samples/hstar_disp25.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "libero_pro"))
from hstar import HStarRule, hstar  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--target", type=float, required=True, help="mean h* to match")
    p.add_argument("--k", type=int, default=3, help="smoothing window")
    p.add_argument("--min-h", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def load_dispersion(runs) -> np.ndarray:
    rows = [c["dispersion"] for run in runs for f in sorted(Path(run).glob("*_chunks.json"))
            for c in json.loads(f.read_text()) if "dispersion" in c]
    if not rows:
        raise SystemExit(f"no dispersion in {runs} (run with --n-samples > 1)")
    return np.asarray(rows, dtype=np.float64)


def mean_h(disp: np.ndarray, eta: float, k: int, min_h: int) -> float:
    rule = HStarRule("above", eta=eta, k=k, min_h=min_h)
    return float(np.mean([hstar(d, 0, rule) for d in disp]))


def calibrate(disp: np.ndarray, target: float, k: int, min_h: int) -> tuple[float, float]:
    """Bisection on eta: mean h* grows with eta (monotone)."""
    lo, hi = float(disp.min()), float(disp.max()) + 1e-9
    for _ in range(40):
        mid = (lo + hi) / 2
        if mean_h(disp, mid, k, min_h) < target:
            lo = mid
        else:
            hi = mid
    eta = (lo + hi) / 2
    return eta, mean_h(disp, eta, k, min_h)


def main(argv=None) -> None:
    args = parse_args(argv)
    disp = load_dispersion(args.runs)
    eta, achieved = calibrate(disp, args.target, args.k, args.min_h)
    rule = HStarRule("above", eta=eta, k=args.k, min_h=args.min_h)
    hs = np.array([hstar(d, 0, rule) for d in disp])
    out = {"rule": rule.to_dict(), "target_mean_h": args.target, "mean_h": achieved,
           "h_quantiles": {q: float(np.quantile(hs, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
           "share_H": float((hs == disp.shape[1]).mean()), "n_chunks": int(len(disp)),
           "inputs": [str(r) for r in args.runs]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"eta = {eta:.4f}: mean h* {achieved:.1f} (target {args.target}), quantiles {out['h_quantiles']}, "
          f"share h* = H {out['share_H']:.2f}, {len(disp)} chunks -> {args.out}")


if __name__ == "__main__":
    main()
