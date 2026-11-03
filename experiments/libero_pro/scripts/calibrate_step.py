#!/usr/bin/env python
"""Threshold of the dispersion-based adaptive step giving the request rate of a constant step.

s = clip(h* - d, s_min, s_max), h* = first index where the smoothed dispersion across noise samples reaches eta. The
input is a run with a constant step, whose requests sample the episode uniformly in time; an adaptive schedule then
requests at a rate of about E[1 / s], so eta matches the harmonic mean of s, not the mean of h* (matching the mean on
RoboTwin gave 20.3 calls per episode against 11.4). The rate is only approximate in closed loop: compare the achieved
one (--achieved) with the control's. Episode outcomes are not used.

    python scripts/calibrate_step.py RUN --target 12 --delay 8 --s-min 8 --s-max 25 --out RUN/step.json
    python scripts/calibrate_step.py RUN --achieved    # rounded mean s of a finished adaptive run

RUN is an eval_libero.py run (*_chunks.json) or a LIBERO-Dynamic run of policy_server.py (signals_*.jsonl).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hstar import HStarRule, smooth  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("--achieved", action="store_true", help="print the rounded mean s of RUN and exit")
    p.add_argument("--target", type=float, help="constant step whose request rate to match")
    p.add_argument("--delay", type=int)
    p.add_argument("--s-min", type=int)
    p.add_argument("--s-max", type=int)
    p.add_argument("--k", type=int, default=3, help="smoothing window")
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    if not args.achieved and None in (args.target, args.delay, args.s_min, args.s_max, args.out):
        p.error("--target, --delay, --s-min, --s-max and --out are required")
    return args


def load_records(run: Path) -> list[dict]:
    rows = [c for f in sorted(run.glob("*_chunks.json")) for c in json.loads(f.read_text())]
    rows += [json.loads(line) for f in sorted(run.glob("signals_*.jsonl")) for line in f.read_text().splitlines()
             if line.strip()]
    return rows


def load_dispersion(run: Path) -> np.ndarray:
    rows = [r.get("dispersion", r.get("disp")) for r in load_records(run)]
    rows = [r for r in rows if r is not None]
    if not rows:
        raise SystemExit(f"no dispersion in {run} (run with --n-samples > 1)")
    return np.asarray(rows, dtype=np.float64)


def steps(bar: np.ndarray, eta: float, delay: int, s_min: int, s_max: int) -> np.ndarray:
    """s per chunk from the smoothed dispersion (N, H); the same as AdaptivePlanner and Server.plan_step."""
    horizon = bar.shape[1]
    hits = bar >= eta
    h = np.where(hits.any(1), hits.argmax(1), horizon)
    return np.clip(np.maximum(h, 1) - delay, s_min, max(min(s_max, horizon - delay), s_min))


def harmonic(s: np.ndarray) -> float:
    return float(1 / np.mean(1 / s))


def calibrate(disp: np.ndarray, target: float, delay: int, s_min: int, s_max: int, k: int = 3) -> tuple[float, np.ndarray]:
    """Bisection on eta: s grows with eta (monotone)."""
    bar = np.stack([smooth(d, k) for d in disp])
    lo, hi = float(bar.min()), float(bar.max()) + 1e-9
    if not harmonic(steps(bar, lo, delay, s_min, s_max)) <= target <= harmonic(steps(bar, hi, delay, s_min, s_max)):
        raise SystemExit(f"target {target} is out of reach for s in [{s_min}, {s_max}]")
    for _ in range(50):
        mid = (lo + hi) / 2
        if harmonic(steps(bar, mid, delay, s_min, s_max)) < target:
            lo = mid
        else:
            hi = mid
    eta = (lo + hi) / 2
    return eta, steps(bar, eta, delay, s_min, s_max)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.achieved:
        s = [r.get("stride", r.get("s")) for r in load_records(args.run)]
        s = [x for x in s if x is not None]
        if not s:
            raise SystemExit(f"no planned steps in {args.run}")
        print(int(round(np.mean(s))))
        return
    if args.s_min < args.delay:
        raise SystemExit("--s-min below the delay: the next request would start before the chunk arrives")
    disp = load_dispersion(args.run)
    eta, s = calibrate(disp, args.target, args.delay, args.s_min, args.s_max, args.k)
    rule = HStarRule("above", eta=eta, k=args.k, min_h=1)
    out = {"rule": rule.to_dict(), "target_s": args.target, "harmonic_s": harmonic(s), "mean_s": float(s.mean()),
           "s_quantiles": {q: float(np.quantile(s, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
           "share_s_min": float((s == s.min()).mean()), "share_s_max": float((s == s.max()).mean()),
           "n_chunks": int(len(s)), "input": str(args.run)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"eta = {eta:.5g}: harmonic s {out['harmonic_s']:.2f} (target {args.target}), mean s {out['mean_s']:.2f}, "
          f"quantiles {out['s_quantiles']}, {len(s)} chunks -> {args.out}")


if __name__ == "__main__":
    main()
