#!/usr/bin/env python3
"""Option budget experiment on LIBERO-Dynamic: explore -> offline gates -> validation -> one-shot test.

Pi0.5 stays frozen. Run through night_budget_v2.sh on the server only; the design is in BUDGET.md. Each finished
rollout is recorded and skipped on restart with the same arguments, so a failure does not repeat earlier stages.
Infrastructure errors abort the run and never count as policy failures.

--server openpi: openpi's pi05_libero (H = 10, openpi_dynamic_server.py) instead of LeRobot pi0.5 (H = 50); the grid
is an explicit --pairs list (r = -1: no guidance, naive), the runner latency is --latency (the measured request
time, ceil(latency * 20) must equal d), no gripper rule, and Q has no latent variant (the server logs none).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from budget_items import make_splits, read_scenes, write_splits
from budget_q import load_policy
from fit_budget import cost_for_rate, fit_round, load_segments
from learn_budget import metrics, paired_delta, read_episodes

HERE = Path(__file__).resolve().parent
EQUAL_RATE = .05


def nearest_stride(rate, delay, horizon=50):
    return int(np.clip(round(1 / rate), max(delay, 1), horizon - delay))


def rate_of(rows):
    return metrics(rows, 0, 1, 1)["rate"]


def rate_matched(rate, target, tolerance=EQUAL_RATE):
    return abs(rate / target - 1) <= tolerance


def best_mask(results, s):
    """Validation success of constant pairs with stride s; ties go to the weaker attachment (r = -1, naive, first)."""
    return min((r for stride, r in results if stride == s), key=lambda r: (-results[s, r], r))


def parse_pairs(text):
    """'s:r s:r ...' -> [(s, r)]"""
    try:
        pairs = [tuple(int(v) for v in item.split(":")) for item in text.split()]
    except ValueError:
        pairs = []
    if not pairs or any(len(p) != 2 for p in pairs) or len(set(pairs)) != len(pairs):
        raise ValueError(f"invalid --pairs {text!r}: expected distinct s:r items")
    return pairs


def runner_ticks(latency, hz=20):
    """Delay in ticks from a fixed latency in seconds, ceil(latency * hz) (at least 1), rounded against float noise."""
    return max(1, int(np.ceil(round(latency * hz, 9))))


def select_candidate(candidates, target):
    """{name: (validation success, rate)}: best success among equal-rate candidates, else the closest rate."""
    matched = [k for k, (_, rate) in candidates.items() if rate_matched(rate, target)]
    if matched:
        return max(sorted(matched), key=lambda k: candidates[k][0]), True
    return min(sorted(candidates), key=lambda k: abs(candidates[k][1] / target - 1)), False


class Experiment:
    def __init__(self, args):
        self.args = args
        self.out = args.out.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ("smoke_only", "force")}
        path = self.out / "config.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise RuntimeError(f"{path} differs from the current arguments; choose a fresh OUT")
        path.write_text(json.dumps(config, indent=1))
        self.runs, self.traces, self.configs, self.marks = (self.out / n for n in ("rollouts", "traces", "configs", "finished"))
        for directory in (self.runs, self.traces, self.configs, self.marks):
            directory.mkdir(exist_ok=True)
        self.openpi = args.server == "openpi"
        self.pairs = parse_pairs(args.pairs) if args.pairs else [(s, r) for s in args.strides for r in args.masks]
        self.horizon = 10 if self.openpi else 50
        self.latency = args.latency if args.latency is not None else args.delay / 20
        scenes = read_scenes(args.config_dir or HERE / "third_party/LIBERO-Dynamic/libero/libero/dynamic_configs", args.levels.split())
        splits = {name: [int(t) for t in getattr(args, f"{name}_traj").split()] for name in ("train", "val", "test")}
        self.items = make_splits(scenes, splits, args.seed, args.scene_limit)
        self.items["smoke"] = self.items["train"][:2]
        self.files = write_splits(self.items, self.out / "splits")
        print({k: len(v) for k, v in self.items.items()}, flush=True)

    def rollout(self, name, split, mode, seed=0, steps=None, option_ticks=None, **extra):
        a = self.args
        run, traces, mark = self.runs / name, self.traces / name, self.marks / name
        if not mark.exists():
            for partial in (run, traces):  # only directories of this experiment; never reused unless finished
                if partial.exists():
                    shutil.rmtree(partial)
            traces.mkdir()
            config = self.configs / f"{name}.json"
            config.write_text(json.dumps({"mode": mode, "pairs": self.pairs, "delay": a.delay, "seed": seed, "horizon": self.horizon,
                                          "option_ticks": a.option_ticks if option_ticks is None else option_ticks,
                                          "max_steps": a.max_steps, "rule_width": a.rule_width, **extra}, indent=1))
            env = os.environ.copy()
            env.pop("CORR_HEAD", None)
            env.update(RUNS=str(self.runs), SPEEDS=str(a.speed), LEVELS=a.levels, TRAJ="0 1 2 3 4", LIMIT="",
                       ITEM_IDS_FILE=str(self.files[split]), SERVERS=str(a.servers), WORKERS=str(a.workers),
                       PORT=str(a.port), KEEP_CHUNK="1", MAX_STEPS=str(steps or a.max_steps),
                       RUNNER_ARGS=f"--control-mode learned_budget_queue --seed {a.seed}")
            if self.openpi:
                env["SERVER"] = "openpi"
            cmd = ["bash", str(HERE / "run_dynamic.sh"), name, "rtc", "8", str(self.latency),
                   "--actions-per-chunk", str(self.horizon), "--max-guidance-weight", str(a.guidance),
                   "--option-budget", str(config), "--budget-traces", str(traces)]
            print(f"{time.strftime('%F %T')} rollout {name}: {split} {mode} {extra}", flush=True)
            with (self.out / f"{name}.log").open("w") as log:
                process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    rc = process.wait(timeout=a.rollout_timeout_hours * 3600)
                    if rc:
                        raise RuntimeError(f"rollout {name} exited {rc}; see {self.out / (name + '.log')}")
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        rows = read_episodes(run, None, len(self.items[split]))
        if set(rows) != set(self.items[split]):
            raise RuntimeError(f"rollout {name} ran other items than its explicit list")
        if not mark.exists():
            mark.write_text(json.dumps({"time": time.strftime("%F %T")}))
            print(json.dumps({"run": name, **metrics(rows, 0, 1, 1)}), flush=True)
        return rows

    def segments(self, names_rows):
        return load_segments([(rows, self.traces / name) for name, rows in names_rows.items()])

    def smoke(self):
        a = self.args
        fixed = list(self.pairs[-1]) if self.openpi else [a.strides[0], a.masks[-1]]
        runs = {"smoke_explore": self.rollout("smoke_explore", "smoke", "explore", steps=64, seed=9000),
                "smoke_fixed": self.rollout("smoke_fixed", "smoke", "fixed", steps=64, fixed=fixed)}
        if a.rule:
            runs["smoke_rule"] = self.rollout("smoke_rule", "smoke", "rule", steps=64, option_ticks=1, rule=self.rule())
        data = self.segments({"smoke_explore": runs["smoke_explore"]})
        walls = [r["controller_summary"]["policy_latency_mean_s"] for rows in runs.values() for r in rows.values()]
        report = {"decisions": len(data["step"]), "object_poses": float(data["priv"][:, 9].mean()),
                  "fallback_share": float(data["priv"][:, 10].mean()),
                  "residual_present": float(data["x"][:, 54].mean()), "request_wall_s": float(np.mean(walls)),
                  "ticks_required": float(np.ceil(np.mean(walls) * 20))}
        (self.out / "smoke_done.json").write_text(json.dumps(report, indent=1))
        print(json.dumps({"smoke": report}), flush=True)

    def rule(self):
        return {"open": self.args.rule_open, "closed": self.args.rule_closed}

    def fit(self, name, runs):
        """A finished fit is kept: refitting after a restart could change a policy that rollouts already used."""
        a, out = self.args, self.out / name
        if (out / "fit.json").exists() and (out / "policy.pt").exists():
            summary = json.loads((out / "fit.json").read_text())
        else:
            torch.set_num_threads(4)
            summary = fit_round(self.segments(runs), self.pairs, a.option_ticks, 1 / a.reference_stride, out, lam=a.lam,
                                folds=a.folds, seed=a.seed, width=a.rule_width, pca_dim=0 if self.openpi else 32)
        print(json.dumps({name: {k: summary[k] for k in ("pass", "gates", "value_auroc_middle")},
                          "uplift": summary["main"]["uplift"]}), flush=True)
        return summary

    def recalibrated(self, name, policy_path, target, val_rows):
        """Validation at the target rate: the multiplier is re-fitted once on on-policy states, without outcomes."""
        rows = val_rows(name, policy_path)
        if not rate_matched(rate_of(rows), target):
            policy = load_policy(policy_path)
            data = self.segments({name: rows})
            cost = cost_for_rate(policy.q(data["x"], data["latent"]), policy.pairs, policy.calls, target)
            policy_path = policy_path.with_name(policy_path.stem + "_recal.pt")
            policy.save(policy_path, cost)
            name += "_recal"
            rows = val_rows(name, policy_path)
        return name, policy_path, rows

    def run(self):
        a = self.args
        self.smoke()
        if a.smoke_only:
            return
        target = 1 / a.reference_stride
        # Exploration: uniform pair per option; the rollouts are the only input of the first fit.
        explored = {}
        for k in range(a.explore_passes):
            name = f"explore_{k}"
            explored[name] = self.rollout(name, "train", "explore", seed=a.seed + 100 + k)
        first = self.fit("fit_1", explored)
        if not first["pass"] and not a.force:
            message = "STOP at the offline gates: see fit_1/fit.md. No validation or test episodes were spent."
            (self.out / "stopped.txt").write_text(message + "\n")
            print(message, flush=True)
            return

        def val_policy(name, policy):
            return self.rollout(name, "val", "policy", policy=str(policy))

        masks = reference_masks(self.pairs, a.reference_stride, a.delay, self.horizon) if self.openpi else a.masks
        val_fixed = {(a.reference_stride, r): self.rollout(f"val_fixed_{a.reference_stride}_{r}", "val", "fixed",
                                                           fixed=[a.reference_stride, r]) for r in masks}
        results = {pair: metrics(rows, 0, 1, 1)["sr"] for pair, rows in val_fixed.items()}
        reference = (a.reference_stride, best_mask(results, a.reference_stride))
        rule_stride = None
        if a.rule:
            val_rule = self.rollout("val_rule", "val", "rule", option_ticks=1, rule=self.rule())
            rule_stride = nearest_stride(rate_of(val_rule), a.delay, self.horizon)

        candidates, policies = {}, {}
        name, path, rows = self.recalibrated("val_pi1", self.out / "fit_1/policy.pt", target, val_policy)
        policies[name], candidates[name] = (path, rows), (metrics(rows, 0, 1, 1)["sr"], rate_of(rows))
        # Second round: data near the first policy's state distribution, pooled with the uniform data.
        pooled = dict(explored)
        for k in range(a.refine_passes):
            run = f"refine_{k}"
            pooled[run] = self.rollout(run, "train", "epsilon", seed=a.seed + 200 + k, policy=str(path), epsilon=a.epsilon)
        if a.refine_passes:
            if self.fit("fit_2", pooled)["pass"]:
                name, path, rows = self.recalibrated("val_pi2", self.out / "fit_2/policy.pt", target, val_policy)
                policies[name], candidates[name] = (path, rows), (metrics(rows, 0, 1, 1)["sr"], rate_of(rows))
        chosen, matched = select_candidate(candidates, target)
        path, chosen_rows = policies[chosen]
        data = self.segments({chosen: chosen_rows})
        marginal = (np.bincount(data["action"], minlength=len(self.pairs)) / len(data["action"])).tolist()
        selection = {"policy": chosen, "path": str(path), "validation": candidates, "validation_rate_matched": matched,
                     "reference": reference, "validation_fixed": {str(k): v for k, v in results.items()},
                     "rule_stride": rule_stride, "marginal": marginal}
        (self.out / "selection.json").write_text(json.dumps(selection, indent=1))

        test = {"fixed": self.rollout("test_fixed", "test", "fixed", seed=2000, fixed=list(reference)),
                "learned": self.rollout("test_learned", "test", "policy", seed=2000, policy=str(path)),
                "marginal": self.rollout("test_marginal", "test", "marginal", seed=2000, marginal=marginal)}
        if a.rule:
            test["rule"] = self.rollout("test_rule", "test", "rule", seed=2000, option_ticks=1, rule=self.rule())
            test["rule fixed"] = self.rollout("test_rule_fixed", "test", "fixed", seed=2000, fixed=[rule_stride, reference[1]])
        write_report(self.out / "report.md", test, selection, a, first)
        print(f"DONE: {self.out / 'report.md'}", flush=True)


def write_report(path, runs, selection, args, fit):
    lines = ["# Option budget on LIBERO-Dynamic", "",
             ("Frozen openpi pi05_libero (H = 10); r = -1 is naive (no guidance); one fixed latency for every pair. "
              if args.server == "openpi" else "Frozen pi0.5. ") + "Same scenes in all splits, listed explicitly; one exploration seed: pilot-level evidence about "
             "held-out trajectories, not transfer to new tasks or training-seed variance.", "",
             f"d={args.delay} (latency {args.latency if args.latency is not None else args.delay / 20} s), v={args.speed}, levels {args.levels}, option {args.option_ticks} ticks, reference stride {args.reference_stride}.",
             f"Offline uplift per option [95% CI]: {fit['main']['uplift']}.", f"Selection: `{json.dumps(selection)}`", "",
             "| policy | episodes | SR % | calls/ep | calls/tick | request wall, s |", "|---|---:|---:|---:|---:|---:|"]
    for name, rows in runs.items():
        m = metrics(rows, 0, 1, 1)
        lines.append(f"| {name} | {m['episodes']} | {100*m['sr']:.2f} | {m['calls']:.2f} | {m['rate']:.4f} | {m['request_wall_mean_s']:.3f} |")
    lines += ["", "| comparison | delta, pp [95% CI] | rate ratio | equal rate (within 5%) |", "|---|---:|---:|---|"]
    for label, reference, candidate in (("PRIMARY learned - fixed", "fixed", "learned"), ("learned - marginal", "marginal", "learned"),
                                        ("rule - its fixed", "rule fixed", "rule")):
        if reference not in runs:
            continue
        delta, low, high = paired_delta(runs[reference], runs[candidate])
        ratio = rate_of(runs[candidate]) / rate_of(runs[reference])
        lines.append(f"| {label} | {delta:+.2f} [{low:+.2f}; {high:+.2f}] | {ratio:.3f} | {abs(ratio - 1) <= EQUAL_RATE} |")
    lines += ["", "A gain counts only if the primary CI is above zero at equal rate; the other rows are secondary. "
              "CI bootstraps episodes, and the episodes of one scene are not independent. Latency is the simulated "
              "fixed d: compare the request wall time with d/20 before any real-time claim."]
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config-dir", type=Path, default=None)
    p.add_argument("--delay", type=int, default=7)
    p.add_argument("--strides", type=int, nargs="+", default=[7, 12, 20])
    p.add_argument("--masks", type=int, nargs="+", default=[0, 4, 12])
    p.add_argument("--reference-stride", type=int, default=12)
    p.add_argument("--option-ticks", type=int, default=40)
    p.add_argument("--speed", type=float, default=.06)
    p.add_argument("--levels", default="l1 l2 l3")
    p.add_argument("--train-traj", default="0 1")
    p.add_argument("--val-traj", default="2")
    p.add_argument("--test-traj", default="3 4")
    p.add_argument("--scene-limit", type=int, default=None)
    p.add_argument("--explore-passes", type=int, default=3)
    p.add_argument("--refine-passes", type=int, default=1)
    p.add_argument("--epsilon", type=float, default=.3)
    p.add_argument("--lam", type=float, default=.5)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--rule-width", type=float, default=.07)
    p.add_argument("--rule-open", type=int, nargs=2, default=[7, 0], help="pair while the gripper is open")
    p.add_argument("--rule-closed", type=int, nargs=2, default=[20, 12], help="pair while the gripper is closed")
    p.add_argument("--guidance", type=float, default=None, help="beta of RTC; default 2 (LeRobot), 5 (openpi, the paper)")
    p.add_argument("--server", choices=["lerobot", "openpi"], default="lerobot")
    p.add_argument("--pairs", default=None, help="openpi: explicit grid 's:r ...'; r = -1 is naive")
    p.add_argument("--latency", type=float, default=None, help="runner latency in seconds; default d / 20")
    p.add_argument("--no-rule", dest="rule", action="store_false", help="skip the gripper-phase rule")
    p.add_argument("--servers", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--port", type=int, default=8124)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--rollout-timeout-hours", type=float, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--force", action="store_true", help="continue after failed offline gates")
    a = p.parse_args(argv)
    if a.guidance is None:
        a.guidance = 5. if a.server == "openpi" else 2.
    if a.latency is not None and runner_ticks(a.latency) != a.delay:
        p.error(f"--latency {a.latency} gives {runner_ticks(a.latency)} ticks at 20 Hz, not d = {a.delay}")
    if a.server == "openpi":
        return check_openpi(p, a)
    if a.pairs:
        p.error("--pairs is for --server openpi")
    if a.delay < 1 or min(a.explore_passes, a.option_ticks, a.servers, a.workers, a.folds - 1) <= 0 or not 0 < a.epsilon < 1 or not 0 <= a.lam <= 1:
        p.error("invalid limits")
    if a.reference_stride < a.delay or min(a.strides) < a.delay or 0 not in a.masks:
        p.error("strides must be >= d, and r=0 must be on the grid")
    if max(a.strides) + max(a.masks) + a.delay > 50:
        p.error("every grid pair must stay feasible after every other: max s + max r + d <= 50")
    return a


def reference_masks(pairs, stride, delay, horizon):
    """Masks of the grid that are repeatable at the reference stride: candidates of the constant control."""
    return sorted({r for _, r in pairs if stride + max(r, 0) + delay <= horizon})


def check_openpi(p, a):
    from budget_options import NAIVE, check_pair

    a.rule = False  # the gripper rule's pairs are for H = 50
    if not a.pairs:
        p.error("--server openpi needs an explicit --pairs grid")
    try:
        pairs = parse_pairs(a.pairs)
        for pair in pairs + [(a.reference_stride, NAIVE)]:
            check_pair(pair, a.delay, 10)
    except ValueError as error:
        p.error(str(error))
    if not reference_masks(pairs, a.reference_stride, a.delay, 10):
        p.error("no mask of the grid is repeatable at --reference-stride")
    if a.delay < 1 or min(a.explore_passes, a.option_ticks, a.servers, a.workers, a.folds - 1) <= 0 or not 0 < a.epsilon < 1 or not 0 <= a.lam <= 1:
        p.error("invalid limits")
    return a


if __name__ == "__main__":
    Experiment(parse_args()).run()
