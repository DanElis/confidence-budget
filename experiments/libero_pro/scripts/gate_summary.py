#!/usr/bin/env python
"""RTC gate of eval_openpi.py (--rtc-gate) from a run's chunk logs: divergence D, share of guided requests, calls.

Requests without an old plan (the first chunk of an episode, D = -1) are left out of every number.

    python scripts/gate_summary.py runs/openpi_gate_cal_s5_d4                      # table
    python scripts/gate_summary.py runs/openpi_gate_cal_s5_d4 --tau-for 0.25 0.5    # tau with these guided shares
    python scripts/gate_summary.py runs/openpi_gate_q50_s5_d4 --print share         # one number, for the shell
    python scripts/gate_summary.py runs/openpi_gate_q25_s5_d4 runs/... --select     # "run tau" of the chosen gate

--select, fixed before the tuning runs: the highest mean success; candidates within SELECT_TIE of it count as equal,
and of those the one with the larger guided share wins (more smoothing at the same success).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SELECT_TIE = 0.01  # success rate: one percentage point


def load(run: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """-> D, guided, calls of requests with an old plan; success rate of each suite."""
    chunks = []
    for path in sorted(run.glob("*_chunks.json")):
        chunks += json.loads(path.read_text())
    if not chunks:
        raise SystemExit(f"{run}: no chunk logs")
    if any("divergence" not in c for c in chunks):
        raise SystemExit(f"{run}: chunk logs without the gate's fields (run with --rtc-gate)")
    rows = [c for c in chunks if c["divergence"] >= 0]
    d = np.array([c["divergence"] for c in rows], dtype=np.float64)
    guided = np.array([c["guided"] for c in rows], dtype=np.float64)
    calls = np.array([c["calls"] for c in rows], dtype=np.float64)
    success = [json.loads(p.read_text())["success_rate"] for p in sorted(run.glob("*.json"))
               if not p.name.endswith("_chunks.json") and p.name != "config.json" and not p.name.endswith(".progress.json")]
    return d, guided, calls, success


def tau_for(d: np.ndarray, share: float) -> float:
    """tau such that D < tau holds for the given share of the logged requests."""
    if share <= 0:
        return 0.0
    if share >= 1:
        return float(d.max()) + 1.0
    return float(np.quantile(d, share))


def select(candidates: dict[str, tuple[float, float]]) -> str:
    """{name: (mean success, guided share)} -> the chosen name."""
    best = max(success for success, _ in candidates.values())
    tied = [name for name, (success, _) in candidates.items() if success >= best - SELECT_TIE - 1e-9]
    return max(sorted(tied), key=lambda name: candidates[name][1])


def run_tau(run: Path) -> float:
    return float(json.loads((run / "config.json").read_text())["rtc_tau"])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--tau-for", type=float, nargs="+", default=None, help="print tau for these guided shares (first run)")
    p.add_argument("--print", choices=["share", "success", "calls"], default=None, help="print one number (first run)")
    p.add_argument("--select", action="store_true", help='print "run tau" of the chosen divergence gate')
    a = p.parse_args(argv)
    if a.select:
        candidates = {}
        for run in a.runs:
            _, guided, _, success = load(run)
            candidates[str(run)] = (float(np.mean(success)), float(guided.mean()))
        chosen = select(candidates)
        print(f"{chosen} {run_tau(Path(chosen)):.6f}")
        return
    if a.tau_for or a.print:
        d, guided, calls, success = load(a.runs[0])
        if a.tau_for:
            print(" ".join(f"{tau_for(d, q):.6f}" for q in a.tau_for))
        else:
            value = {"share": guided.mean(), "success": float(np.mean(success)), "calls": calls.mean()}[a.print]
            print(f"{value:.6f}")
        return
    print("| run | requests | guided, % | calls / request | D p10 | D p25 | D p50 | D p75 | D p90 | success, % |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for run in a.runs:
        d, guided, calls, success = load(run)
        q = np.quantile(d, [0.1, 0.25, 0.5, 0.75, 0.9])
        print(f"| {run.name} | {len(d)} | {100 * guided.mean():.1f} | {calls.mean():.3f} | "
              + " | ".join(f"{v:.4f}" for v in q) + f" | {100 * np.mean(success):.1f} |")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
