#!/usr/bin/env python
"""Compare closed-loop runs: success per suite, smoothness, policy calls, paired difference to a reference.

The difference to the reference is paired by (suite, task, init state): both
runs saw the same scenes and the same noise, so the paired estimate is much
tighter than comparing two success rates. 95% CI by bootstrap over episode pairs.

Example:
    python scripts/compare.py --ref runs/rtc_s10_d4 runs/budget_lam0.5_d4 runs/budget_entropy_d4 \\
        --suites libero_object_pos_x0.1,libero_object_pos_x0.2 --out runs/compare.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--ref", type=Path, default=None, help="reference run (default: the first one)")
    p.add_argument("--suites", default=None, help="comma-separated suite names (default: all suites of the reference)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def load_run(run: Path, suites: list[str]) -> dict:
    out = {}
    for name in suites:
        f = run / f"{name}.json"
        if f.exists():
            out[name] = json.loads(f.read_text())
    return out


def episode_table(run: dict) -> dict:
    return {(s, e["task_id"], e["init_state"]): e for s, r in run.items() for e in r["episodes"]}


def paired_diff(ref: dict, other: dict, n_boot: int, seed: int) -> tuple[float, float, float, int]:
    keys = sorted(set(ref) & set(other))
    if not keys:
        return float("nan"), float("nan"), float("nan"), 0
    d = np.array([float(other[k]["success"]) - float(ref[k]["success"]) for k in keys])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), len(keys)


def mean_key(run: dict, key: str) -> float:
    vals = [e.get(key) for r in run.values() for e in r["episodes"]]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def all_suites(run: Path) -> list[str]:
    return sorted(p.stem for p in run.glob("*.json") if p.stem != "config" and not p.stem.endswith("_chunks"))


def build_table(runs: list[Path], ref_dir: Path, suites: list[str] | None = None, n_boot: int = 2000,
                seed: int = 0, labels: dict | None = None) -> str:
    """Markdown table; ``runs`` may include ``ref_dir`` or not, the reference row comes first."""
    runs = [ref_dir] + [r for r in runs if r != ref_dir]
    suites = suites or all_suites(ref_dir)
    labels = labels or {}
    data = {r: load_run(r, suites) for r in runs}
    ref_eps = episode_table(data[ref_dir])

    short = {s: s.replace("libero_object_pos_", "") for s in suites}
    head = ["run"] + [short[s] for s in suites] + ["mean", f"Δ vs {ref_dir.name} [95% CI]", "pairs", "calls/ep.",
                                                    "SPARC", "|Δa| switch", "mean s"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in runs:
        run = data[r]
        rates = [100 * run[s]["success_rate"] if s in run else float("nan") for s in suites]
        valid = [x for x in rates if np.isfinite(x)]
        if r == ref_dir:
            delta = "—"
            pairs = len(ref_eps)
        else:
            m, lo, hi, pairs = paired_diff(ref_eps, episode_table(run), n_boot, seed)
            delta = f"{100 * m:+.1f} [{100 * lo:+.1f}; {100 * hi:+.1f}]"
        jumps = [run[s].get("jump_at_switch", float("nan")) for s in suites if s in run]
        lines.append("| " + " | ".join([labels.get(r, r.name)] + [f"{x:.1f}" for x in rates] +
                                       [f"{np.mean(valid):.1f}" if valid else "—", delta, str(pairs),
                                        f"{mean_key(run, 'n_inference'):.1f}", f"{mean_key(run, 'sparc'):.3f}",
                                        f"{np.nanmean(jumps):.4f}" if jumps else "—",
                                        f"{mean_key(run, 'mean_stride'):.1f}"]) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    args = parse_args(argv)
    ref_dir = args.ref or args.runs[0]
    suites = [s.strip() for s in args.suites.split(",") if s.strip()] if args.suites else None
    table = build_table(args.runs, ref_dir, suites, args.n_boot, args.seed)
    print(table)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table)


if __name__ == "__main__":
    main()
