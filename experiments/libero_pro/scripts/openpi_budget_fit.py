#!/usr/bin/env python
"""CPU side of the step budget on openpi (openpi_budget.py, stage scripts/stages/openpi_budget.sh).

  headroom RUN...      constant-s runs on the same scenes and noise: cross-fitted gain of a per-scene best s over the
                       best constant s. The gate before any budget episode is spent (HEADROOM_MIN).
  fit RUN...           explore runs -> V, lambda-returns, advantage Q (ensemble), offline gates, policy.pt, fit.md.
                       The method is fit_budget.py of LIBERO-Dynamic (v2), with deployable features in V: LIBERO-Pro
                       exposes no object state to this client.
  calibrate RUN        re-fit the rate multiplier once on the policy's own validation states, if its rate is off.
  marginal RUN         action frequencies of a policy run: the state-independent control at equal frequencies.
  rate RUN             requests per control step, from the episodes.
  config               write a runtime config.
  report               the test table and the comparisons fixed before the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "libero_dynamic"))
from compare import all_suites, build_table, episode_table, load_run  # noqa: E402
from openpi_budget import FEATURE_DIM, FEATURES, GRID, load_qpolicy, write_config  # noqa: E402

HEADROOM_MIN = 0.02  # cross-fitted per-scene best s over the best constant: below 2 pp no budget is trained
AUROC_MIN = 0.65
AUROC_WINDOW = 50, 200  # libero_object episodes last at most 280 steps; successes mostly end before 200
AUROC_BINS = (0, 50, 100, 150, 200, 281)
EQUAL_RATE = 0.05


# ---------------------------------------------------------------- runs

def episodes(run: Path) -> dict:
    return episode_table(load_run(run, all_suites(run)))


def run_rate(run: Path) -> float:
    eps = episodes(run).values()
    return float(sum(e["n_inference"] for e in eps) / sum(e["length"] for e in eps))


def rate_matched(rate: float, target: float) -> bool:
    return abs(rate / target - 1) <= EQUAL_RATE


def load_decisions(runs, grid=GRID, behaviour=True) -> dict:
    """Option decisions of budget runs, in episode / step order. behaviour: the runs must log grid actions
    with propensities (explore / marginal / policy), as the Q fit needs."""
    cols = defaultdict(list)
    n_episode = 0
    for run in runs:
        run = Path(run)
        table = episodes(run)
        for suite in all_suites(run):
            chunks = json.loads((run / f"{suite}_chunks.json").read_text())
            by_episode = defaultdict(list)
            for c in chunks:
                by_episode[c["task_id"], c["init_state"]].append(c)
            for (task, init), rows in sorted(by_episode.items()):
                rows.sort(key=lambda c: c["step"])
                if "budget_x" not in rows[0] or rows[0]["step"] != 0 or not rows[0]["budget_decision"]:
                    raise RuntimeError(f"{run}/{suite} {task}/{init}: no budget decision at the first request")
                success = float(table[suite, task, init]["success"])
                for c in rows:
                    if not c["budget_decision"]:
                        continue
                    if behaviour and (c["budget_action"] < 0 or grid[int(c["budget_action"])] != c["stride"]):
                        raise RuntimeError(f"{run}/{suite}: a decision off the grid {grid}")
                    cols["x"].append(np.asarray(c["budget_x"], np.float32))
                    cols["action"].append(int(c["budget_action"]))
                    cols["prob"].append(float(c["budget_prob"]))
                    cols["step"].append(int(c["step"]))
                    cols["episode"].append(n_episode)
                    cols["group"].append(f"{suite}/{task}/{init}")  # the same item in every run: one fold
                    cols["scene"].append(f"{suite}/{task}")
                    cols["level"].append(suite)
                    cols["success"].append(success)
                n_episode += 1
    data = {k: np.asarray(v) for k, v in cols.items()}
    if not len(data.get("x", [])) or data["x"].shape[1:] != (FEATURE_DIM,):
        raise RuntimeError("no decisions, or features of another size")
    data["priv"] = np.zeros((len(data["x"]), 0), np.float32)  # fit_budget.value_inputs: no simulator state here
    return data


# ---------------------------------------------------------------- headroom

def headroom(runs: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """runs {s: dir}, same scenes and noise. Init states split by parity: the best s per scene (task, level) and the
    best constant s are chosen on one half and scored on the other, both ways. Ties go to the larger s (fewer calls)."""
    from fit_budget import group_bootstrap

    strides = sorted(runs)
    tables = {s: episodes(runs[s]) for s in strides}
    keys = sorted(set.intersection(*(set(t) for t in tables.values())))
    if not keys:
        raise RuntimeError("the runs share no episodes")
    success = np.array([[float(tables[s][k]["success"]) for s in strides] for k in keys])
    scene = np.array([f"{k[0]}/{k[1]}" for k in keys])
    half = np.array([k[2] % 2 for k in keys])

    def best(rows) -> int:
        mean = success[rows].mean(0)
        return max(range(len(strides)), key=lambda j: (mean[j], strides[j]))

    gain = np.zeros(len(keys))
    chosen = []
    for h in (0, 1):
        fit, held = half == h, half != h
        const = best(fit)
        for sc in np.unique(scene):
            rows = held & (scene == sc)
            j = best(fit & (scene == sc)) if (fit & (scene == sc)).any() else const
            gain[rows] = success[rows, j] - success[rows, const]
            chosen.append(strides[j])
    mean, low, high = group_bootstrap(gain, scene, n=n_boot, seed=seed)
    star = best(np.ones(len(keys), bool))
    return {"strides": strides, "success": dict(zip(map(str, strides), success.mean(0).tolist())), "pairs": len(keys),
            "gain": mean, "ci": [low, high], "pass": bool(mean >= HEADROOM_MIN), "s_star": strides[star],
            "rate_star": run_rate(runs[strides[star]]),
            "episode_oracle": float(success.max(1).mean() - success[:, star].mean()),
            "chosen_per_scene": dict(zip(map(str, strides), [chosen.count(s) for s in strides]))}


def headroom_markdown(h: dict) -> str:
    lines = ["# Headroom of a step budget (constant s, same scenes and noise)", "",
             "| s | " + " | ".join(str(s) for s in h["strides"]) + " |", "|---|" + "---|" * len(h["strides"]),
             "| success, % | " + " | ".join(f"{100 * h['success'][str(s)]:.1f}" for s in h["strides"]) + " |", "",
             f"Cross-fitted per-scene best s over the best constant s: {100 * h['gain']:+.2f} pp "
             f"[{100 * h['ci'][0]:+.2f}; {100 * h['ci'][1]:+.2f}] over {h['pairs']} episodes; required >= "
             f"{100 * HEADROOM_MIN:.0f} pp (fixed before the run): {'continue' if h['pass'] else 'STOP'}.",
             f"Best constant s* = {h['s_star']}, {h['rate_star']:.4f} requests per step: reference and target rate.",
             f"Per-scene choices (both halves): {h['chosen_per_scene']}.",
             f"Per-episode oracle over s* (biased upward by outcome noise, descriptive): {100 * h['episode_oracle']:+.2f} pp.",
             "", "The per-scene choice ignores the request rate and sees no within-episode adaptation; the offline gates "
             "of the fit test that."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- fit

def value_quality(data, value):
    from fit_budget import auroc

    bins = []
    for start, end in zip(AUROC_BINS, AUROC_BINS[1:]):
        mask = (data["step"] >= start) & (data["step"] < end)
        bins.append((start, end, int(mask.sum()), auroc(data["success"][mask], value[mask])))
    middle = (data["step"] >= AUROC_WINDOW[0]) & (data["step"] < AUROC_WINDOW[1])
    return auroc(data["success"][middle], value[middle]), bins


def evaluate(data, advantage, pairs, fold, calls, target, seed):
    from budget_q import prepare_inputs
    from fit_budget import _standardize, cost_for_rate, fit_q, group_bootstrap, request_rate, state_dependence_uplift

    mean, std = _standardize(data["x"])
    keep = np.ones(FEATURE_DIM, np.float32)
    z = prepare_inputs(data["x"], None, mean, std, keep, None)
    q, const, members = fit_q(z, data, advantage, pairs, fold, seed)
    cost = cost_for_rate(q, pairs, calls, target)
    uplift, greedy, marginal = state_dependence_uplift(q, q - cost * calls, data, advantage)
    rows = np.arange(len(q))
    gain = (advantage - const[rows, data["action"]]) ** 2 - (advantage - q[rows, data["action"]]) ** 2
    return {"uplift": group_bootstrap(uplift, data["group"], seed=seed),
            "mse_gain": group_bootstrap(gain, data["group"], seed=seed),
            "rate": request_rate(greedy, pairs), "marginal": marginal.tolist(), "cost": float(cost),
            "ckpt": {"pairs": pairs, "members": members, "mean": mean, "std": std, "keep": keep, "pca": None,
                     "calls": calls, "cost": float(cost), "features": FEATURES}}


def fit(data, grid, option_ticks, target, out: Path, lam=0.5, folds=5, seed=0) -> dict:
    import torch
    from fit_budget import fit_value, fold_ids, group_bootstrap, lambda_targets, option_calls

    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    pairs = [(int(s), 0) for s in grid]
    calls = option_calls(pairs, option_ticks)
    fold = fold_ids(data["group"], folds, seed)
    value = fit_value(data, fold, seed)
    middle, bins = value_quality(data, value)
    advantage = lambda_targets(data["episode"], value, data["success"], lam) - value
    main = evaluate(data, advantage, pairs, fold, calls, target, seed)
    torch.save(main.pop("ckpt"), out / "policy.pt")
    monte_carlo = evaluate(data, data["success"] - value, pairs, fold, calls, target, seed)["uplift"]
    by_s = [(s, int((data["action"] == a).sum()), *group_bootstrap(advantage[data["action"] == a],
                                                                    data["group"][data["action"] == a], seed=seed))
            for a, s in enumerate(grid) if (data["action"] == a).any()]
    gates = {"value_auroc": bool(middle >= AUROC_MIN), "state_dependence": bool(main["uplift"][1] > 0)}
    summary = {"pass": all(gates.values()), "gates": gates, "value_auroc_middle": middle, "value_bins": bins,
               "main": main, "monte_carlo_uplift": monte_carlo, "by_s": by_s, "grid": list(grid),
               "decisions": len(data["step"]), "episodes": int(data["episode"].max() + 1),
               "items": len(np.unique(data["group"])), "target_rate": target, "lambda": lam}
    (out / "fit.json").write_text(json.dumps(summary, indent=1))
    (out / "fit.md").write_text(fit_markdown(summary))
    return summary


def fit_markdown(s: dict) -> str:
    m = s["main"]
    per_episode = s["decisions"] / s["episodes"]
    lines = ["# Step budget on openpi: offline fit", "",
             f"{s['episodes']} episodes, {s['items']} items, {s['decisions']} option decisions ({per_episode:.1f} per "
             f"episode), lambda = {s['lambda']}, grid s = {s['grid']}.", "", "## Gates (fixed before the run)", "",
             f"- V out-of-fold AUROC for final success, steps {AUROC_WINDOW[0]}..{AUROC_WINDOW[1]}: "
             f"{s['value_auroc_middle']:.3f}; required >= {AUROC_MIN}: {s['gates']['value_auroc']}.",
             f"- Value of state dependence at equal action frequencies, per option: {m['uplift'][0]:+.5f} "
             f"[{m['uplift'][1]:+.5f}; {m['uplift'][2]:+.5f}]; CI low > 0: {s['gates']['state_dependence']}.",
             f"- Decision: {'continue' if s['pass'] else 'STOP: no evidence for a learnable state-dependent step'}.", "",
             f"Rough scale: x {per_episode:.1f} options = {100 * m['uplift'][0] * per_episode:+.2f} pp per episode "
             "(a sum of one-step improvements, not a prediction).",
             f"Held-out squared-error gain over the state-independent model: {m['mse_gain'][0]:+.6f} "
             f"[{m['mse_gain'][1]:+.6f}; {m['mse_gain'][2]:+.6f}].",
             f"Same uplift with Monte Carlo targets: {s['monte_carlo_uplift']}.",
             f"Greedy rate on logged states {m['rate']:.4f} (target {s['target_rate']:.4f}); frequencies: "
             + ", ".join(f"s={g}: {100 * f:.1f}%" for g, f in zip(s["grid"], m["marginal"])), "",
             "## V by episode time", "", "| steps | decisions | AUROC |", "|---|---:|---:|",
             *[f"| {a}..{b} | {n} | {v:.3f} |" for a, b, n, v in s["value_bins"]], "",
             "## Advantage by s (descriptive)", "", "| s | decisions | mean [95% CI] |", "|---:|---:|---:|",
             *[f"| {g} | {n} | {v:+.5f} [{lo:+.5f}; {hi:+.5f}] |" for g, n, v, lo, hi in s["by_s"]]]
    return "\n".join(lines) + "\n"


def calibrate(run: Path, policy_path: Path, target: float, out: Path) -> Path:
    """The policy's rate on its own states; if off by more than EQUAL_RATE, one new multiplier for those states."""
    from fit_budget import cost_for_rate, request_rate

    realized = run_rate(run)
    if rate_matched(realized, target):
        return policy_path
    policy = load_qpolicy(policy_path)
    data = load_decisions([run], [s for s, _ in policy.pairs])
    q = policy.q(data["x"])
    proxy = request_rate((q - policy.cost * policy.calls).argmax(1), policy.pairs)  # what the multiplier controls
    policy.save(out, cost_for_rate(q, policy.pairs, policy.calls, proxy * target / realized))
    return out


def retarget(runs, policy_path: Path, target: float, out: Path) -> Path:
    """The same Q with the multiplier for another target rate (in options, 1/s), on the states of the given runs."""
    from fit_budget import cost_for_rate

    policy = load_qpolicy(policy_path)
    data = load_decisions(runs, [s for s, _ in policy.pairs])
    policy.save(out, cost_for_rate(policy.q(data["x"]), policy.pairs, policy.calls, target))
    return out


def marginal(run: Path, grid) -> list:
    data = load_decisions([run], grid)
    freq = np.bincount(data["action"], minlength=len(grid)) / len(data["action"])
    return [float(f) for f in freq]


# ---------------------------------------------------------------- report

def paired(ref_dir: Path, other_dir: Path, level: float, n_boot: int = 2000, seed: int = 0):
    """Paired Δ of success (other - ref) by (suite, task, init state) -> (mean, low, high, pairs) at the CI level."""
    ref, other = episodes(ref_dir), episodes(other_dir)
    keys = sorted(set(ref) & set(other))
    d = np.array([float(other[k]["success"]) - float(ref[k]["success"]) for k in keys])
    boots = d[np.random.default_rng(seed).integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    tail = (1 - level) / 2
    return float(d.mean()), float(np.quantile(boots, tail)), float(np.quantile(boots, 1 - tail)), len(keys)


def report(fixed: Path, learned: Path, control: Path, headroom_json: Path | None = None, best: Path | None = None,
           level: float = 0.95) -> str:
    runs = [fixed, learned, control] + ([best] if best is not None and best != fixed else [])
    pct = f"{100 * level:g}%"
    lines = ["# Step budget on openpi: test", "", build_table(runs, fixed), "",
             f"| comparison | Δ success, pp [{pct} CI] | pairs | rate ratio | equal rate (within 5%) |",
             "|---|---:|---:|---:|---|"]
    rows = [("PRIMARY learned - fixed at its rate", fixed, learned), ("learned - marginal", control, learned)]
    if best is not None and best != fixed:
        rows.append(("learned - best constant (fewer calls)", best, learned))
    for label, ref_dir, other_dir in rows:
        d, lo, hi, n = paired(ref_dir, other_dir, level)
        ratio = run_rate(other_dir) / run_rate(ref_dir)
        lines.append(f"| {label} | {100 * d:+.1f} [{100 * lo:+.1f}; {100 * hi:+.1f}] | {n} | {ratio:.3f} | "
                     f"{abs(ratio - 1) <= EQUAL_RATE} |")
    lines += ["", f"Fixed before the run: a gain counts only if the primary {pct} CI is above zero at equal rate; the "
              "other rows are secondary. A budget that spends more calls is useful only if it also beats the best "
              "constant. The CI bootstraps episodes; episodes of one scene are not independent."]
    if headroom_json is not None:
        h = json.loads(Path(headroom_json).read_text())
        lines += ["", f"Headroom on the tuning levels: {100 * h['gain']:+.2f} pp, s* = {h['s_star']}."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- CLI

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("headroom")
    h.add_argument("runs", type=Path, nargs="+", help="constant-s runs (their config.json gives s)")
    h.add_argument("--out", type=Path, required=True, help="directory for headroom.md and headroom.json")
    f = sub.add_parser("fit")
    f.add_argument("runs", type=Path, nargs="+")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--grid", type=int, nargs="+", default=list(GRID))
    f.add_argument("--option-ticks", type=int, required=True)
    f.add_argument("--target-rate", type=float, required=True)
    f.add_argument("--folds", type=int, default=5)
    f.add_argument("--lam", type=float, default=0.5)
    f.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("calibrate")
    c.add_argument("run", type=Path)
    c.add_argument("--policy", type=Path, required=True)
    c.add_argument("--target-rate", type=float, required=True)
    c.add_argument("--out", type=Path, required=True)
    rt = sub.add_parser("retarget")
    rt.add_argument("runs", type=Path, nargs="+")
    rt.add_argument("--policy", type=Path, required=True)
    rt.add_argument("--target-rate", type=float, required=True)
    rt.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("marginal")
    m.add_argument("run", type=Path)
    m.add_argument("--grid", type=int, nargs="+", default=list(GRID))
    r = sub.add_parser("rate")
    r.add_argument("run", type=Path)
    g = sub.add_parser("config")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--mode", required=True)
    g.add_argument("--grid", type=int, nargs="+", default=list(GRID))
    g.add_argument("--option-ticks", type=int, required=True)
    g.add_argument("--max-steps", type=int, required=True)
    g.add_argument("--fixed", type=int, default=None)
    g.add_argument("--marginal", default=None, help="JSON list")
    g.add_argument("--policy", default=None)
    t = sub.add_parser("report")
    t.add_argument("--fixed", type=Path, required=True)
    t.add_argument("--learned", type=Path, required=True)
    t.add_argument("--marginal", type=Path, required=True)
    t.add_argument("--headroom", type=Path, default=None)
    t.add_argument("--best", type=Path, default=None, help="the best constant step: a secondary row")
    t.add_argument("--level", type=float, default=0.95, help="CI level (Bonferroni over reference steps)")
    t.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)

    if a.cmd == "headroom":
        runs = {int(json.loads((run / "config.json").read_text())["stride"]): run for run in a.runs}
        result = headroom(runs)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "headroom.json").write_text(json.dumps(result, indent=1))
        (a.out / "headroom.md").write_text(headroom_markdown(result))
        print(int(result["pass"]), result["s_star"], f"{result['rate_star']:.6f}")
    elif a.cmd == "fit":
        if (a.out / "fit.json").exists() and (a.out / "policy.pt").exists():  # rollouts may already use this policy
            summary = json.loads((a.out / "fit.json").read_text())
        else:
            summary = fit(load_decisions(a.runs, tuple(a.grid)), a.grid, a.option_ticks, a.target_rate, a.out, a.lam,
                          a.folds, a.seed)
        print(int(summary["pass"]))
    elif a.cmd == "calibrate":
        print(calibrate(a.run, a.policy, a.target_rate, a.out))
    elif a.cmd == "retarget":
        print(retarget(a.runs, a.policy, a.target_rate, a.out))
    elif a.cmd == "marginal":
        print(json.dumps(marginal(a.run, tuple(a.grid))))
    elif a.cmd == "rate":
        print(f"{run_rate(a.run):.6f}")
    elif a.cmd == "config":
        config = {"mode": a.mode, "grid": a.grid, "option_ticks": a.option_ticks, "max_steps": a.max_steps}
        if a.fixed is not None:
            config["fixed"] = a.fixed
        if a.marginal is not None:
            config["marginal"] = json.loads(a.marginal)
        if a.policy is not None:
            config["policy"] = str(Path(a.policy).resolve())
        write_config(a.out, **config)
    elif a.cmd == "report":
        a.out.write_text(report(a.fixed, a.learned, a.marginal, a.headroom, a.best, a.level))


if __name__ == "__main__":
    main()
