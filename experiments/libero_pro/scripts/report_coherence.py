#!/usr/bin/env python
"""Report of the night_coherence.sh runs: per delay, all configurations against RTC with the same K samples.

Paired Δ with 95% bootstrap CI (scripts/compare.py), plus mean h*, share of chunks executed from a sample other than
0, and the matched-pair note. Works on partial results (missing runs are skipped).

    uv run python scripts/report_coherence.py --runs runs --out runs/coherence_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare import all_suites, build_table, load_run, mean_key

ORDER = ["naive", "rtc", "rtck", "coh", "ada_z1.5", "ada_z3", "budget_z1.5_lam0.5", "cohconf1", "naivecoh", "cohmatch"]


def runs_for_delay(root: Path, d: int) -> list[Path]:
    found = {p.name: p for p in root.glob(f"cs_*_d{d}") if p.is_dir()}
    out = []
    for key in ORDER:
        out += sorted(p for name, p in found.items() if name.startswith(f"cs_{key}_") and p not in out)
    return out


def extra_table(runs: list[Path], suites: list[str]) -> str:
    lines = ["| run | mean h* | sel ≠ 0 | episodes |", "|---|---|---|---|"]
    for r in runs:
        run = load_run(r, suites)
        n = sum(len(x["episodes"]) for x in run.values())
        lines.append(f"| {r.name} | {mean_key(run, 'mean_h_star'):.1f} | {mean_key(run, 'frac_sel'):.2f} | {n} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--delays", type=int, nargs="+", default=[8, 16])
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    parts = ["# Coherent sample, confidence mask, agreement-based adaptive s (LIBERO-Pro y levels)\n",
             "Δ is paired by (suite, task, init state) against `rtck` (RTC with the same K = 4 samples, sample 0); "
             "calls/ep. counts policy calls (each call = K unguided samples + 1 guided pass for K > 1).\n"]
    sync = args.runs / "cs_sync_s10"
    for d in args.delays:
        runs = runs_for_delay(args.runs, d)
        ref = next((r for r in runs if r.name.startswith("cs_rtck_")), None)
        if ref is None or not all_suites(ref):
            parts.append(f"## d = {d}\n\nno reference run yet\n")
            continue
        suites = all_suites(ref)
        table_runs = ([sync] if sync.exists() else []) + runs
        parts += [f"## d = {d}\n", build_table(table_runs, ref, suites), "", extra_table(table_runs, suites)]
        match = args.runs / f"cs_cohmatch_d{d}" / "config.json"
        if match.exists():
            parts.append(f"`cohmatch`: constant s = {json.loads(match.read_text())['stride']} "
                         f"(mean s of `ada_z1.5`, rounded).\n")
    text = "\n".join(parts)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
