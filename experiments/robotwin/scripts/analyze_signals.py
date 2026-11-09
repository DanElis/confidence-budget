#!/usr/bin/env python
"""Dispersion across noise samples vs entropy as reliability signals, closed loop on RoboTwin.

Input: runs made with --n-samples K --record-entropy --record-plans (scripts/run_samples.sh).

Test A, episode outcome (as analyze_entropy.py): does the signal of the first chunks (steps < --early)
separate episodes that fail? z-scored within task; AUROC (failure = positive), bootstrap CI,
stratified permutation p-value.

Test B, plan revision (closed-loop analogue of H1; only meaningful WITHOUT guidance, because RTC pulls
the new chunk onto the old one): for consecutive chunks of an episode starting at t and t' = t + o,
drift_j = mean_joints |new[j] - old[o + j]| (same control step, planned from an older vs a fresher
observation). Is the old chunk's signal at index o + j higher where the plan gets revised more?
Both quantities are z-scored within each old index (the index effect - both grow along the chunk -
is removed), then Spearman over all (pair, index) points, CI by bootstrap of episodes.

Pre-registered primary features (decide the verdict; others are exploratory):
    dispersion  "tail": mean dispersion at action indices 25-49 (beyond the RTC mask for s = 25)
    entropy     "zone": mean normalised entropy of the first 10 actions (analyze_entropy.py)

Example:
    uv run --project ../libero_pro python scripts/analyze_signals.py runs_samples/sync_s25_k8 runs_samples/rtc_s25_d6_k8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_entropy import auroc  # noqa: E402

TAIL = slice(25, 50)
ZONE = slice(0, 10)
PRIMARY = {"dispersion": "tail", "entropy": "zone"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--early", type=int, default=100)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="default: <first run>/../signals_report.md")
    return p.parse_args(argv)


def load(run: Path) -> list[dict]:
    rows = []
    for f in sorted(run.glob("*_chunks.json")):
        task = f.name[: -len("_chunks.json")]
        res = run / f"{task}.json"
        if not res.exists():
            continue
        eps = {e["init_state"]: e for e in json.loads(res.read_text())["episodes"] if not e.get("error")}
        for c in json.loads(f.read_text()):
            if c["init_state"] not in eps or "dispersion" not in c:
                continue
            ep = eps[c["init_state"]]
            row = {"task": task, "episode": c["init_state"], "step": int(c["step"]), "success": bool(ep["success"]),
                   "length": int(ep["length"]), "mask_end": c.get("mask_end"),
                   "dispersion": np.asarray(c["dispersion"], np.float64)}
            if "entropy" in c:
                row["entropy"] = np.asarray(c["entropy"], np.float64) / math.log(max(int(c["n_valid"]), 2))
            if "plan" in c:
                row["plan"] = np.asarray(c["plan"], np.float64)
            rows.append(row)
    return rows


def features(rows: list[dict]) -> list[str]:
    names = []
    for sig in ("dispersion", "entropy"):
        if rows and sig in rows[0]:
            for r in rows:
                x = r[sig]
                r[f"{sig}:zone"], r[f"{sig}:tail"], r[f"{sig}:mean"] = x[ZONE].mean(), x[TAIL].mean(), x.mean()
            names += [f"{sig}:zone", f"{sig}:tail", f"{sig}:mean"]
    for task in {r["task"] for r in rows}:  # z-score within task
        sub = [r for r in rows if r["task"] == task]
        for n in names:
            v = np.array([r[n] for r in sub])
            mu, sd = v.mean(), v.std() or 1.0
            for r, x in zip(sub, v, strict=True):
                r["z:" + n] = (x - mu) / sd
    return names


def episodes(rows: list[dict], names: list[str], early: int) -> list[dict]:
    eps = {}
    for r in rows:
        eps.setdefault((r["task"], r["episode"]), []).append(r)
    out = []
    for (task, _), chunks in eps.items():
        chunks.sort(key=lambda r: r["step"])
        use = [r for r in chunks if r["step"] < early] or chunks[:1]
        out.append({"task": task, "success": chunks[0]["success"], "chunks": chunks,
                    **{n: float(np.mean([r["z:" + n] for r in use])) for n in names}})
    return out


def strata(eps):
    groups = {}
    for i, e in enumerate(eps):
        groups.setdefault(e["task"], []).append(i)
    return [np.array(g) for g in groups.values()]


def test_a(eps, feat, args, rng, level):
    score = np.array([e[feat] for e in eps])
    fail = np.array([not e["success"] for e in eps])
    groups = strata(eps)
    point = auroc(score, fail)
    if not np.isfinite(point):  # only successes or only failures: nothing to test
        return float("nan"), float("nan"), float("nan"), float("nan")
    boots = [auroc(score[idx], fail[idx]) for idx in
             (np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups]) for _ in range(args.n_boot))]
    a = (1 - level) / 2
    lo, hi = np.nanquantile(boots, [a, 1 - a])
    obs, hits = abs(point - 0.5), 0
    for _ in range(args.n_perm):
        perm = fail.copy()
        for g in groups:
            perm[g] = fail[rng.permutation(g)]
        hits += abs(auroc(score, perm) - 0.5) >= obs - 1e-12
    return point, float(lo), float(hi), (hits + 1) / (args.n_perm + 1)


def drift_points(eps, sig: str):
    """(episode id, old index, drift, signal) for every overlapping (pair, index)."""
    pts = []
    for ei, e in enumerate(eps):
        ch = [r for r in e["chunks"] if "plan" in r and sig in r]
        for old, new in zip(ch, ch[1:]):
            o = new["step"] - old["step"]
            h = len(old["plan"])
            if not 0 < o < h:
                continue
            drift = np.abs(new["plan"][: h - o] - old["plan"][o:]).mean(axis=1)
            for j, dval in enumerate(drift):
                pts.append((ei, o + j, dval, old[sig][o + j]))
    return np.array(pts) if pts else np.zeros((0, 4))


def test_b(eps, sig, args, rng):
    pts = drift_points(eps, sig)
    if len(pts) < 50:
        return None
    z = np.zeros((len(pts), 2))
    for i in np.unique(pts[:, 1]):  # remove the index effect
        m = pts[:, 1] == i
        for c, col in enumerate((2, 3)):
            v = pts[m, col]
            z[m, c] = (v - v.mean()) / (v.std() or 1.0)
    rho = spearmanr(z[:, 0], z[:, 1]).statistic
    ep_ids = np.unique(pts[:, 0])
    by_ep = {e: np.flatnonzero(pts[:, 0] == e) for e in ep_ids}
    boots = []
    for _ in range(min(args.n_boot, 500)):
        idx = np.concatenate([by_ep[e] for e in rng.choice(ep_ids, len(ep_ids))])
        boots.append(spearmanr(z[idx, 0], z[idx, 1]).statistic)
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return float(rho), float(lo), float(hi), len(pts)


def analyse(run: Path, args, rng):
    rows = load(run)
    if not rows:
        return f"## {run.name}\n\nno chunks with dispersion (run with --n-samples > 1)\n", {}
    names = features(rows)
    eps = episodes(rows, names, args.early)
    guided = any(r["mask_end"] not in (None, 0) and r["step"] > 0 for r in rows) and "rtc" in run.name
    md = [f"## {run.name}", "", f"{len(eps)} episodes ({sum(not e['success'] for e in eps)} failed), {len(rows)} chunks.", "",
          "### A. Episode outcome: AUROC for failure (> 0.5 = signal higher in failures)", "",
          "| feature | AUROC [CI] | permutation p |", "|---|---|---|"]
    verdict = {}
    n_expl = len(names) - len(PRIMARY)
    for n in names:
        sig, kind = n.split(":")
        primary = PRIMARY.get(sig) == kind
        level = 0.95 if primary else 1 - 0.05 / max(n_expl, 1)
        point, lo, hi, p = test_a(eps, n, args, rng, level)
        md.append(f"| {n}{' (primary)' if primary else ''} | {point:.3f} [{lo:.3f}; {hi:.3f}] | {p:.4f} |")
        if primary:
            verdict[f"A {sig}"] = (point, lo, hi, p)
    md += ["", "### B. Plan revision vs signal (z within action index), Spearman", ""]
    if guided:
        md.append("RTC run: the guidance pulls the new chunk onto the old one, so drift is suppressed by construction; "
                  "shown for completeness only.")
        md.append("")
    md += ["| signal | rho [95% CI] | points |", "|---|---|---|"]
    for sig in ("dispersion", "entropy"):
        if sig not in rows[0]:
            continue
        res = test_b(eps, sig, args, rng)
        if res is None:
            md.append(f"| {sig} | — (no plans recorded) | — |")
            continue
        rho, lo, hi, n_pts = res
        md.append(f"| {sig} | {rho:+.3f} [{lo:+.3f}; {hi:+.3f}] | {n_pts} |")
        if not guided:
            verdict[f"B {sig}"] = (rho, lo, hi, None)
    prof = np.mean([r["dispersion"] for r in rows], axis=0)
    md += ["", "### Dispersion profile along the chunk (mean over chunks; success / failure)", "",
           "| | " + " | ".join(f"a{i}" for i in range(0, 50, 5)) + " |", "|---|" + "---|" * 10]
    for outcome, label in ((True, "success"), (False, "failure")):
        sub = [r["dispersion"] for r in rows if r["success"] == outcome]
        if sub:
            pm = np.mean(sub, axis=0)
            md.append(f"| {label} | " + " | ".join(f"{pm[i]:.3f}" for i in range(0, 50, 5)) + " |")
    md.append(f"| all | " + " | ".join(f"{prof[i]:.3f}" for i in range(0, 50, 5)) + " |")
    return "\n".join(md) + "\n", verdict


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    parts, lines = ["# Dispersion vs entropy on RoboTwin", ""], ["## Verdict", ""]
    for run in args.runs:
        md, v = analyse(run, args, rng)
        parts.append(md)
        for key, (point, lo, hi, p) in v.items():
            if key.startswith("A"):
                sig = "not computable (one outcome only)" if not np.isfinite(p) else ("signal" if p < 0.05 else "no signal")
                lines.append(f"- {run.name}, {key} ({PRIMARY[key.split()[1]]}): {sig}, AUROC {point:.3f} "
                             f"[{lo:.3f}; {hi:.3f}], p = {p:.4f}")
            else:
                sig = "signal" if lo > 0 else ("opposite sign" if hi < 0 else "no signal")
                lines.append(f"- {run.name}, {key}: {sig}, rho {point:+.3f} [{lo:+.3f}; {hi:+.3f}]")
    lines += ["", "Dispersion is a usable reliability signal for the budget if B (sync) is positive and A agrees "
              "in sign (higher dispersion -> more revision / more failures)."]
    text = "\n".join(parts + lines) + "\n"
    print(text)
    out = args.out or args.runs[0].parent / "signals_report.md"
    out.write_text(text)


if __name__ == "__main__":
    main()
