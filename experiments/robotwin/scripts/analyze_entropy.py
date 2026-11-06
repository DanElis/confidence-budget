#!/usr/bin/env python
"""Does the cross-attention entropy of closed-loop chunks predict failure on RoboTwin?

Input: run directories made with --record-entropy (default runs/sync_s25 and runs/rtc_s25_d6):
<task>.json (episodes: success, length) and <task>_chunks.json (per chunk: init_state, step,
entropy over the H actions, n_valid). There is no demonstration in closed loop, so "before an
error" is approximated by (a) how early the entropy separates episodes that will fail and
(b) how it evolves towards the end of the episode.

Entropy is normalised by log(n_valid) and then z-scored within (run, task), so differences
between tasks (prompt length, scene) do not count as signal. Failure is the positive class;
AUROC > 0.5 means higher entropy in episodes that fail.

Per chunk features
    zone    mean normalised entropy of the first 10 actions (what the entropy-lambda rule reads)
    mean    mean over all H actions
    rise    mean of the last 10 minus the first 10 (how fast confidence decays within the chunk)
    h*      reliable horizon by the plateau rules of hstar.py (relative q=0.7 k=5; rise k=3)

Analyses (per run)
    1. episode level: mean feature over chunks that start in the first --early steps (same
       phase for every episode; failed episodes are longer), AUROC with 95% CI (bootstrap of
       episodes within tasks), and per task;
    2. early warning: AUROC of the k-th chunk alone, k = 0..5;
    3. time course: z(zone) by episode phase (quartiles of the episode's length), success vs failure;
    4. within-chunk profile: mean normalised entropy at action indices 0, 5, ..., 45, and h*.

Example:
    uv run --project ../libero_pro python scripts/analyze_entropy.py runs/sync_s25 runs/rtc_s25_d6
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "libero_pro"))
from hstar import HStarRule, hstar  # noqa: E402

RULES = {"h*_relative": HStarRule("relative", q=0.7, k=5), "h*_rise": HStarRule("rise", k=3, min_h=5)}
FEATURES = ("zone", "mean", "rise", *RULES)
PRIMARY = "zone"  # what the entropy-lambda rule reads; the verdict is decided on it alone


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="*", default=[Path("runs/sync_s25"), Path("runs/rtc_s25_d6")])
    p.add_argument("--early", type=int, default=100, help="episode-level features use chunks starting before this step")
    p.add_argument("--zone", type=int, default=10)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("runs/entropy_report.md"))
    return p.parse_args(argv)


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_run(run: Path, zone: int) -> list[dict]:
    """One record per chunk: task, episode, step, features, outcome, episode length."""
    rows = []
    for f in sorted(run.glob("*_chunks.json")):
        task = f.name[: -len("_chunks.json")]
        result = run / f"{task}.json"
        if not result.exists():
            continue
        episodes = {e["init_state"]: e for e in json.loads(result.read_text())["episodes"] if not e.get("error")}
        for c in json.loads(f.read_text()):
            if "entropy" not in c or c["init_state"] not in episodes:
                continue
            ep = episodes[c["init_state"]]
            e = np.asarray(c["entropy"], dtype=np.float64) / math.log(max(int(c["n_valid"]), 2))
            row = {"task": task, "episode": c["init_state"], "step": int(c["step"]), "success": bool(ep["success"]),
                   "length": int(ep["length"]), "profile": e,
                   "zone": float(e[:zone].mean()), "mean": float(e.mean()),
                   "rise": float(e[-zone:].mean() - e[:zone].mean())}
            for name, rule in RULES.items():
                row[name] = float(hstar(c["entropy"], int(c["n_valid"]), rule))
            rows.append(row)
    return rows


def zscore_within_task(rows: list[dict]) -> None:
    for task in {r["task"] for r in rows}:
        sub = [r for r in rows if r["task"] == task]
        for feat in FEATURES:
            v = np.array([r[feat] for r in sub])
            mu, sd = v.mean(), v.std() or 1.0
            for r, x in zip(sub, v, strict=True):
                r["z_" + feat] = (x - mu) / sd


def episode_table(rows: list[dict], early: int) -> list[dict]:
    eps = {}
    for r in rows:
        key = (r["task"], r["episode"])
        eps.setdefault(key, {"task": r["task"], "success": r["success"], "length": r["length"], "chunks": []})
        eps[key]["chunks"].append(r)
    out = []
    for e in eps.values():
        e["chunks"].sort(key=lambda r: r["step"])
        early_chunks = [r for r in e["chunks"] if r["step"] < early] or e["chunks"][:1]
        for feat in FEATURES:
            e[feat] = float(np.mean([r["z_" + feat] for r in early_chunks]))
        out.append(e)
    return out


def boot_auroc(eps: list[dict], feat: str, n_boot: int, rng, level: float = 0.95) -> tuple[float, float, float]:
    score = np.array([e[feat] for e in eps])
    fail = np.array([not e["success"] for e in eps])
    point = auroc(score, fail)
    by_task = {}
    for i, e in enumerate(eps):
        by_task.setdefault(e["task"], []).append(i)
    groups = [np.array(v) for v in by_task.values()]
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        boots.append(auroc(score[idx], fail[idx]))
    boots = np.array([b for b in boots if np.isfinite(b)])
    a = (1 - level) / 2
    lo, hi = (np.quantile(boots, [a, 1 - a]) if len(boots) else (float("nan"), float("nan")))
    return point, float(lo), float(hi)


def perm_pvalue(eps: list[dict], feat: str, n_perm: int, rng) -> float:
    """Two-sided permutation p-value of |AUROC - 0.5|, outcomes shuffled within each task."""
    score = np.array([e[feat] for e in eps])
    fail = np.array([not e["success"] for e in eps])
    groups = {}
    for i, e in enumerate(eps):
        groups.setdefault(e["task"], []).append(i)
    groups = [np.array(g) for g in groups.values()]
    obs = abs(auroc(score, fail) - 0.5)
    if not np.isfinite(obs):
        return float("nan")
    hits = 0
    for _ in range(n_perm):
        perm = fail.copy()
        for g in groups:
            perm[g] = fail[rng.permutation(g)]
        hits += abs(auroc(score, perm) - 0.5) >= obs - 1e-12
    return (hits + 1) / (n_perm + 1)


def analyse(run: Path, args, rng) -> tuple[str, dict]:
    rows = load_run(run, args.zone)
    if not rows:
        return f"## {run.name}\n\nno chunks with entropy (run without --record-entropy?)\n", {}
    zscore_within_task(rows)
    eps = episode_table(rows, args.early)
    n_fail = sum(not e["success"] for e in eps)
    md = [f"## {run.name}", "",
          f"{len(eps)} episodes ({n_fail} failed), {len(rows)} chunks, {len({r['task'] for r in rows})} tasks; "
          f"episode features from chunks starting before step {args.early}.", ""]

    md += ["### 1. Episode level: AUROC for failure (0.5 = no signal; > 0.5 = higher in failures)", "",
           f"Primary feature {PRIMARY}: 95% CI. Others exploratory: {100 * (1 - 0.05 / (len(FEATURES) - 1)):.1f}% CI "
           "(Bonferroni).", "",
           "| feature | AUROC [CI] | " + " | ".join(sorted({e['task'] for e in eps})) + " |",
           "|---|---|" + "---|" * len({e['task'] for e in eps})]
    summary = {}
    for feat in FEATURES:
        # primary feature (fixed in advance): 95% CI; the others are exploratory: Bonferroni over them
        level = 0.95 if feat == PRIMARY else 1 - 0.05 / (len(FEATURES) - 1)
        point, lo, hi = boot_auroc(eps, feat, args.n_boot, rng, level)
        summary[feat] = (point, lo, hi)
        if feat == PRIMARY:
            summary["_p"] = perm_pvalue(eps, feat, args.n_perm, rng)
        per_task = []
        for task in sorted({e["task"] for e in eps}):
            sub = [e for e in eps if e["task"] == task]
            per_task.append(auroc(np.array([e[feat] for e in sub]), np.array([not e["success"] for e in sub])))
        md.append(f"| {feat} | {point:.3f} [{lo:.3f}; {hi:.3f}] | " + " | ".join(f"{a:.2f}" for a in per_task) + " |")

    md += ["", "### 2. Early warning: AUROC of the k-th chunk of the episode alone (z(zone) / z(h*_relative))", "",
           "| k | episodes | failed | AUROC zone | AUROC h*_relative |", "|---|---|---|---|---|"]
    for k in range(6):
        sel = [e["chunks"][k] for e in eps if len(e["chunks"]) > k]
        if len(sel) < 10:
            break
        fail = np.array([not r["success"] for r in sel])
        md.append(f"| {k} | {len(sel)} | {int(fail.sum())} | {auroc(np.array([r['z_zone'] for r in sel]), fail):.3f} "
                  f"| {auroc(np.array([r['z_h*_relative'] for r in sel]), fail):.3f} |")

    md += ["", "### 3. Time course: mean z(zone) by phase of the episode (quartile of its length)", "",
           "| outcome | Q1 | Q2 | Q3 | Q4 (end) | chunks |", "|---|---|---|---|---|---|"]
    for outcome, name in ((True, "success"), (False, "failure")):
        sub = [r for r in rows if r["success"] == outcome]
        if not sub:
            continue
        q = np.minimum((4 * np.array([r["step"] for r in sub]) / np.maximum([r["length"] for r in sub], 1)).astype(int), 3)
        z = np.array([r["z_zone"] for r in sub])
        md.append(f"| {name} | " + " | ".join(f"{z[q == i].mean():+.2f}" if (q == i).any() else "—" for i in range(4))
                  + f" | {len(sub)} |")

    idx = list(range(0, len(rows[0]["profile"]), 5))
    md += ["", "### 4. Within-chunk profile: mean normalised entropy by action index, and h*", "",
           "| outcome | " + " | ".join(f"a{i}" for i in idx) + " | h*_relative | h*_rise |",
           "|---|" + "---|" * (len(idx) + 2)]
    for outcome, name in ((True, "success"), (False, "failure")):
        sub = [r for r in rows if r["success"] == outcome]
        if not sub:
            continue
        prof = np.mean([r["profile"] for r in sub], axis=0)
        md.append(f"| {name} | " + " | ".join(f"{prof[i]:.3f}" for i in idx)
                  + f" | {np.mean([r['h*_relative'] for r in sub]):.1f} | {np.mean([r['h*_rise'] for r in sub]):.1f} |")
    return "\n".join(md) + "\n", summary


def verdict(summaries: dict) -> str:
    lines = ["## Verdict", ""]
    for run, s in summaries.items():
        if not s:
            continue
        point, lo, hi = s[PRIMARY]
        pval = s["_p"]
        if not np.isfinite(pval):
            lines.append(f"- {run}: not computable (only one outcome)")
            continue
        if pval < 0.05:
            direction = "higher" if point > 0.5 else "lower"
            lines.append(f"- {run}: signal. {PRIMARY}: AUROC {point:.3f} [{lo:.3f}; {hi:.3f}], permutation p = {pval:.4f} "
                         f"({direction} in episodes that fail).")
        else:
            lines.append(f"- {run}: no episode-level signal in {PRIMARY}: AUROC {point:.3f} [{lo:.3f}; {hi:.3f}], "
                         f"permutation p = {pval:.3f}.")
        extra = [f"{f} {p:.3f} [{a:.3f}; {b:.3f}]" for f, v in s.items() if not f.startswith("_") and f != PRIMARY
                 for p, a, b in [v] if a > 0.5 or b < 0.5]
        if extra:
            lines.append(f"  exploratory features with a corrected CI away from 0.5: {', '.join(extra)}")
    lines += ["", "Usable for an entropy-based budget if the early AUROC is clearly above 0.5 in both runs "
              "(the signal survives RTC guidance, as in H1) and agrees in sign across tasks."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    parts, summaries = ["# Entropy vs failure on RoboTwin", ""], {}
    for run in args.runs:
        md, summary = analyse(run, args, rng)
        parts.append(md)
        summaries[run.name] = summary
    parts.append(verdict(summaries))
    text = "\n".join(parts)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)


if __name__ == "__main__":
    main()
