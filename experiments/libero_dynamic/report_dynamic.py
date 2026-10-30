#!/usr/bin/env python3
"""Compare LIBERO-Dynamic runs (run_dynamic.sh outputs) on the same items, paired by item id and speed.

    python3 report_dynamic.py runs/pt_async_k6 runs/ada --ref runs/pt_async_k6 [--speeds 0.06] [--out report.md]

Per run: success per speed and overall, paired Δ to the reference with a 95% bootstrap CI over common episodes,
policy requests per episode, action change between steps (from the runner's action metrics), the mean correction
norm (L2 per tick) and RTC scale of corrected_rtc_queue runs, and for runs of our
server the plan statistics from signals_*.jsonl (mean s, h*, share of chunks from a sample other than 0; with
--mask-mode transition the mean transition length L and the mean and median deviation of the old plan).
Standard library only (runs with any python3).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def find_number(d, words):
    """First numeric 'mean' (or plain number) under a key containing all words, searching nested dicts."""
    if isinstance(d, dict):
        for k, v in d.items():
            if all(w in str(k).lower() for w in words):
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, dict) and isinstance(v.get("mean"), (int, float)):
                    return float(v["mean"])
            r = find_number(v, words)
            if r is not None:
                return r
    return None


def load(run: Path, speeds) -> dict:
    """(speed, item_id) -> row with success, requests, action delta."""
    out = {}
    for v in speeds:
        for f in sorted((run / f"v{v}").rglob("manifest.jsonl")):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("status", "ok") != "ok" and r.get("error"):
                    continue  # infrastructure failure, not a policy failure
                cs = r.get("controller_summary") or {}
                out[(v, r["item_id"])] = {
                    "success": bool(r.get("success")),
                    "requests": cs.get("policy_requests"),
                    "delta": find_number(r.get("action_metrics"), ("delta",)),
                    "corr": cs.get("correction_abs_mean") if cs.get("correction_calls") else None,
                    "scale": cs.get("rtc_scale_mean"),
                }
    return out


def plan_stats(run: Path) -> str:
    rows = [json.loads(l) for f in run.glob("signals_*.jsonl") for l in f.read_text().splitlines() if l.strip()]
    trans = [r for r in rows if "L" in r]
    if trans:  # --mask-mode transition: mean transition length, mean (= --trans-ref of a constant run) and median deviation
        deltas = [r["delta"] for r in trans if r.get("delta") is not None]
        dev = f", Δ mean {statistics.mean(deltas):.4g}, med {statistics.median(deltas):.4g}" if deltas else ""
        return f"L {statistics.mean(r['L'] for r in trans):.2f}{dev}"
    rows = [r for r in rows if r.get("z") is not None]
    if not rows:
        return "—"
    s = [r["s"] for r in rows if "s" in r]
    h = [r["h_star"] for r in rows if "h_star" in r]
    sel = [r["sel"] != 0 for r in rows]
    parts = [f"sel≠0 {statistics.mean(sel):.2f}"]
    if s:
        parts = [f"s {statistics.mean(s):.1f}", f"h* {statistics.mean(h):.1f}"] + parts
    return ", ".join(parts)


def paired(ref: dict, other: dict, n_boot=2000, seed=0):
    keys = sorted(set(ref) & set(other))
    if not keys:
        return None
    d = [float(other[k]["success"]) - float(ref[k]["success"]) for k in keys]
    rng = random.Random(seed)
    boots = sorted(statistics.mean(rng.choices(d, k=len(d))) for _ in range(n_boot))
    return statistics.mean(d), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)], len(keys)


def mean_or_dash(xs, fmt):
    xs = [x for x in xs if x is not None]
    return format(statistics.mean(xs), fmt) if xs else "—"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--ref", type=Path, default=None)
    p.add_argument("--speeds", nargs="+", default=["0.03", "0.06", "0.09"])
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    ref_dir = a.ref or a.runs[0]
    runs = [ref_dir] + [r for r in a.runs if r != ref_dir]
    data = {r: load(r, a.speeds) for r in runs}
    head = ["run"] + [f"v={v}" for v in a.speeds] + ["overall", f"Δ vs {ref_dir.name} [95% CI]", "pairs",
                                                      "requests/ep", "|Δa|", "corr |Δa|", "RTC scale", "plan"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in runs:
        rows = data[r]
        per = []
        for v in a.speeds:
            xs = [x["success"] for (vv, _), x in rows.items() if vv == v]
            per.append(f"{100 * statistics.mean(xs):.1f} ({len(xs)})" if xs else "—")
        allx = [x["success"] for x in rows.values()]
        overall = f"{100 * statistics.mean(allx):.1f}" if allx else "—"
        if r == ref_dir:
            delta, pairs = "—", str(len(rows))
        else:
            res = paired(data[ref_dir], rows)
            delta = "—" if res is None else f"{100 * res[0]:+.1f} [{100 * res[1]:+.1f}; {100 * res[2]:+.1f}]"
            pairs = "0" if res is None else str(res[3])
        lines.append("| " + " | ".join([r.name, *per, overall, delta, pairs,
                                          mean_or_dash([x["requests"] for x in rows.values()], ".1f"),
                                          mean_or_dash([x["delta"] for x in rows.values()], ".4f"),
                                          mean_or_dash([x["corr"] for x in rows.values()], ".4f"),
                                          mean_or_dash([x["scale"] for x in rows.values()], ".3f"),
                                          plan_stats(r)]) + " |")
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        a.out.write_text(text)


if __name__ == "__main__":
    main()
