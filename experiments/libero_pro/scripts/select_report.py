#!/usr/bin/env python
"""Pick the tuned values on the tuning levels (y) and report the picked configurations on x.

Selection rule, fixed before any x result is seen, applied per delay d:
  s*_sync  = argmax over s of the mean y success of y_sync_s{s}
  s*       = argmax over s of the mean y success of y_rtc_s{s}_d{d}       (best tuned RTC = reference)
  lambda*  = argmax over lambda of y_budget_lam{lambda}_d{d}
  z_max*   = argmax over z of y_budget_entropy_z{z}_d{d}
Ties go to the first value in grid order. Only y runs enter the selection; the x numbers of
the picked configurations are the reported result. The full x grid is printed as well, marked
as not to be used for choosing.

Example:
    python scripts/select_report.py --runs runs --delays 4 --out runs/report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare import all_suites, build_table  # noqa: E402

STRIDES = ["5", "10", "15", "20"]
SYNC_STRIDES = ["5", "10"]
LAMBDAS = ["0", "0.25", "0.5", "0.75", "1"]
ZMAXES = ["1", "2", "3"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--delays", nargs="+", default=["4"])
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def mean_success(run: Path) -> float:
    """Mean over suites of the success rate; nan if the run is missing or empty."""
    rates = [json.loads((run / f"{s}.json").read_text())["success_rate"] for s in all_suites(run)] if run.is_dir() else []
    return float(np.mean(rates)) if rates else float("nan")


def pick(runs: Path, pattern: str, values: list[str]) -> tuple[str | None, list[tuple[str, float]]]:
    scored = [(v, mean_success(runs / pattern.format(v))) for v in values]
    ok = [(v, m) for v, m in scored if np.isfinite(m)]
    if not ok:
        return None, scored
    best = max(m for _, m in ok)
    return next(v for v, m in ok if m == best), scored


def fmt_scores(scored) -> str:
    return ", ".join(f"{v}: {100 * m:.1f}" if np.isfinite(m) else f"{v}: —" for v, m in scored)


def existing(paths: list[Path]) -> list[Path]:
    return [p for p in paths if p.is_dir() and all_suites(p)]


def report(runs: Path, delays: list[str], n_boot: int = 2000) -> tuple[str, dict]:
    out, chosen = [], {}
    s_sync, sc = pick(runs, "y_sync_s{}", SYNC_STRIDES)
    out.append("# Report\n\nSelection on y (mean success, %), report on x.\n")
    out.append(f"- sync stride: **{s_sync}** ({fmt_scores(sc)})")
    for d in delays:
        s, s_sc = pick(runs, "y_rtc_s{}_d" + d, STRIDES)
        lam, l_sc = pick(runs, "y_budget_lam{}_d" + d, LAMBDAS)
        z, z_sc = pick(runs, "y_budget_entropy_z{}_d" + d, ZMAXES)
        chosen[d] = {"s_sync": s_sync, "s": s, "lam": lam, "z_max": z}
        out.append(f"- d = {d}: RTC stride **{s}** ({fmt_scores(s_sc)}); lambda **{lam}** ({fmt_scores(l_sc)}); "
                   f"z_max **{z}** ({fmt_scores(z_sc)})")
    out.append("")

    for d in delays:
        c = chosen[d]
        out.append(f"## d = {d}\n")
        if c["s"] is None:
            out.append("no y_rtc runs, nothing to report\n")
            continue
        ref = runs / f"x_rtc_s{c['s']}_d{d}"
        if not existing([ref]):
            out.append(f"missing reference {ref.name}\n")
            continue
        picked = {
            runs / f"x_sync_s{c['s_sync']}": f"sync s={c['s_sync']}",
            runs / f"x_naive_s{c['s']}_d{d}": f"naive async s={c['s']}",
            ref: f"RTC s={c['s']} (reference)",
            runs / f"x_budget_lam{c['lam']}_d{d}": f"budget lambda={c['lam']}",
            runs / f"x_budget_entropy_z{c['z_max']}_d{d}": f"budget lambda(entropy), z_max={c['z_max']}",
            runs / f"x_budget_lam0_d{d}": "budget lambda=0 (all slack to r)",
            runs / f"x_budget_lam1_d{d}": "budget lambda=1 (all slack to s)",
        }
        rows = [r for r in dict.fromkeys(picked) if r in existing(list(picked))]
        out.append("### Main table (configurations picked on y)\n")
        out.append(build_table(rows, ref, n_boot=n_boot, labels=picked))

        grid = existing([runs / f"x_sync_s{s}" for s in SYNC_STRIDES]
                        + [runs / f"x_{k}_s{s}_d{d}" for k in ("naive", "rtc") for s in STRIDES]
                        + [runs / f"x_budget_lam{v}_d{d}" for v in LAMBDAS]
                        + [runs / f"x_budget_entropy_z{v}_d{d}" for v in ZMAXES])
        out.append("### Full x grid (for reference only, not for choosing)\n")
        out.append(build_table(grid, ref, n_boot=n_boot))

        abl_ref = runs / f"x_budget_entropy_z2_d{d}"
        abl = existing([runs / f"abl_{n}_d{d}" for n in ("const_hstar", "budget_exact", "rtc_exact_s10", "rmin3", "rmin10")])
        if abl and existing([abl_ref]):
            out.append("### Ablations (vs budget lambda(entropy), z_max=2; pairs only where both ran)\n")
            out.append(build_table(abl, abl_ref, n_boot=n_boot))
    return "\n".join(out), chosen


def main(argv=None) -> None:
    args = parse_args(argv)
    text, chosen = report(args.runs, args.delays, args.n_boot)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        args.out.with_suffix(".json").write_text(json.dumps(chosen, indent=2))


if __name__ == "__main__":
    main()
