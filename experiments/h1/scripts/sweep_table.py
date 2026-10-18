#!/usr/bin/env python
"""One-line-per-run summary of several analyze_h1.py outputs.

Example:
    python scripts/sweep_table.py runs/h1_exact runs/h1_shift0.5_exact --out runs/sweep.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COLUMNS = [
    ("run", "запуск"),
    ("rho_rtc", "ρ RTC"),
    ("ci", "95% CI"),
    ("rho_free", "ρ без guidance"),
    ("ramp", "маска free/RTC"),
    ("tail", "хвост free/RTC"),
    ("frozen", "RMSE замор. free/RTC"),
    ("ent_corr", "корр. энтропий"),
    ("d_ent_mask", "|ΔE| маска"),
    ("verdict", "H1"),
]


def row(run_dir: Path) -> dict[str, str]:
    summary = json.loads((run_dir / "summary.json").read_text())
    c = summary["correlations"]
    rtc, free = c["rtc"]["ramp+tail"], c["free"]["ramp+tail"]

    def zone(name: str) -> str:
        values = [c[k][name]["partial_spearman"] if name in c[k] else float("nan") for k in ("free", "rtc")]
        return f"{values[0]:.3f}/{values[1]:.3f}"

    frozen = summary["frozen_rmse_median"]
    with np.load(run_dir.with_suffix(".npz")) as data:
        entropy = data["entropy"][:, -1, :]
        is_rtc = data["is_rtc"]
    free_e, rtc_e = entropy[~is_rtc], entropy[is_rtc]
    delay, mask_end = summary["settings"]["delay"], summary["settings"]["mask_end"]
    lo, hi = rtc["partial_spearman_ci"]
    return {
        "run": run_dir.name,
        "rho_rtc": f"{rtc['partial_spearman']:.3f}",
        "ci": f"[{lo:.3f}; {hi:.3f}]",
        "rho_free": f"{free['partial_spearman']:.3f}",
        "ramp": zone("ramp"),
        "tail": zone("tail"),
        "frozen": f"{frozen['free']:.3f}/{frozen['rtc']:.3f}",
        "ent_corr": f"{np.corrcoef(free_e.ravel(), rtc_e.ravel())[0, 1]:.4f}",
        "d_ent_mask": f"{np.abs(rtc_e - free_e)[:, delay:mask_end].mean():.4f}",
        "verdict": "да" if summary["verdict"]["passed"] else "нет",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+", help="analysis directories (runs/<name>), next to runs/<name>.npz")
    p.add_argument("--out", type=Path, default=None, help="also write the table as Markdown")
    args = p.parse_args()

    rows = []
    for run in args.runs:
        run = run.with_suffix("") if run.suffix == ".npz" else run
        if (run / "summary.json").exists() and run.with_suffix(".npz").exists():
            rows.append(row(run))
        else:
            print(f"skip {run}: no summary.json or .npz")
    lines = ["| " + " | ".join(t for _, t in COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(r[k] for k, _ in COLUMNS) + " |" for r in rows]
    table = "\n".join(lines) + "\n"
    print(table)
    if args.out:
        args.out.write_text(table)


if __name__ == "__main__":
    main()
