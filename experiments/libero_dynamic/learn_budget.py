#!/usr/bin/env python3
"""On-policy PPO of joint before-generation (s,r), with frozen pi0.5.

Orchestrates existing benchmark subprocesses in their simulator environment.
Run from libero_pro's environment, via night_budget.sh, on the server only.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from budget_head import BudgetHead, action_pairs, load_head, save_head

HERE = Path(__file__).resolve().parent


def read_episodes(run, traces=None, expected=None):
    rows = {}
    for manifest in sorted(Path(run).rglob("manifest.jsonl")):
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok" or row.get("error"):
                raise RuntimeError(f"infrastructure error: {row.get('item_id')}: {row.get('error')}")
            key = row["item_id"]
            if key in rows:
                raise RuntimeError(f"duplicate episode {key}")
            cs = row["controller_summary"]
            decisions = cs.get("budget_trace", [])
            if not decisions or len(decisions) != cs["policy_requests"]:
                raise RuntimeError(f"incomplete budget trajectory: {key}")
            steps = [int(d["step"]) for d in decisions]
            if steps[0] != 0 or any(steps[i+1] != steps[i] + decisions[i]["s"] for i in range(len(steps)-1)):
                raise RuntimeError(f"incorrect request schedule: {key}")
            delay = cs["latency_budget_steps"]
            underflow = row["action_runtime"]["queue_underflow_steps"]
            if underflow > min(delay, row["steps"]):
                raise RuntimeError(f"queue starvation after startup: {key}, {underflow} ticks")
            if not row["steps"] or not isinstance(row.get("success"), bool):
                raise RuntimeError(f"invalid episode result: {key}")
            if traces is not None:
                samples = []
                for decision in decisions:
                    with np.load(Path(traces) / f"{decision['id']}.npz", allow_pickle=False) as f:
                        sample = {k: np.array(f[k]) for k in f.files}
                    if int(sample["step"]) != decision["step"] or not sample["valid"][int(sample["action"])]:
                        raise RuntimeError("trace/manifest mismatch")
                    samples.append(sample)
                row["samples"] = samples
            rows[key] = row
    if not rows or (expected is not None and len(rows) != expected):
        raise RuntimeError(f"expected {expected} episodes, got {len(rows)} in {run}")
    return rows


def metrics(rows, cost, reference_stride, max_steps):
    rows = list(rows.values())
    calls = np.array([r["controller_summary"]["policy_requests"] for r in rows])
    steps = np.array([r["steps"] for r in rows])
    sr = np.mean([r["success"] for r in rows])
    budget_times = [d["budget_elapsed_s"] for r in rows for d in r["controller_summary"]["budget_trace"]
                    if "budget_elapsed_s" in d]
    return {"sr": float(sr), "calls": float(calls.mean()), "rate": float(calls.sum() / steps.sum()),
            "score": float(sr - cost * calls.mean() * reference_stride / max_steps), "episodes": len(rows),
            "request_wall_mean_s": float(np.mean([r["controller_summary"].get("policy_latency_mean_s", 0) for r in rows])),
            "budget_wall_mean_s": float(np.mean(budget_times)) if budget_times else 0.0,
            "gpu_peak_allocated_mb": max((d.get("gpu_peak_allocated_mb", 0) for r in rows
                                            for d in r["controller_summary"]["budget_trace"]), default=0)}


def ppo_update(model, optimizer, rows, config, epochs=4, batch_size=256, entropy=.01):
    samples, targets = [], []
    penalty = config["call_cost"] * config["reference_stride"] / config["max_steps"]
    for row in rows.values():
        trajectory = row["samples"]
        n = len(trajectory)
        samples.extend(trajectory)
        # Undiscounted episodic Monte Carlo returns: no incentive to delay reward
        # by choosing long s. A request in flight at termination still incurs cost.
        targets.extend(float(row["success"]) - penalty * np.arange(n, 0, -1))
    x = torch.from_numpy(np.stack([s["x"] for s in samples]))
    valid = torch.from_numpy(np.stack([s["valid"] for s in samples]))
    action = torch.tensor([int(s["action"]) for s in samples])
    old_logp = torch.tensor([float(s["logp"]) for s in samples])
    old_value = torch.tensor([float(s["value"]) for s in samples])
    returns = torch.tensor(np.asarray(targets), dtype=torch.float32)
    # Require exactly the behavior policy/mask before any update.
    with torch.no_grad():
        dist, _ = model(x, valid)
        error = float((dist.log_prob(action) - old_logp).abs().max())
        if error > 2e-5:
            raise RuntimeError(f"off-policy or mismatched feasibility masks: logp error {error}")
    adv = returns - old_value
    adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)
    logs = []
    for _ in range(epochs):
        stop = False
        for idx in torch.randperm(len(samples)).split(batch_size):
            dist, value = model(x[idx], valid[idx])
            log_ratio = dist.log_prob(action[idx]) - old_logp[idx]
            ratio = log_ratio.exp()
            actor = -torch.minimum(ratio * adv[idx], ratio.clamp(.8, 1.2) * adv[idx]).mean()
            critic = .5 * (value - returns[idx]).square().mean()
            ent = dist.entropy().mean()
            loss = actor + critic - entropy * ent
            optimizer.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            if not torch.isfinite(loss) or not torch.isfinite(grad):
                raise RuntimeError("nonfinite PPO update")
            optimizer.step()
            kl = float(((ratio - 1) - log_ratio).mean().detach())
            logs.append({"loss": float(loss.detach()), "entropy": float(ent.detach()), "kl": kl})
            if kl > .03:
                stop = True
                break
        if stop:
            break
    return {"samples": len(samples), "logp_error": error,
            **{k: float(np.mean([r[k] for r in logs])) for k in logs[0]}}


def paired_delta(reference, candidate, seed=0):
    if set(reference) != set(candidate):
        raise RuntimeError("unpaired test items")
    delta = np.array([float(candidate[k]["success"]) - float(reference[k]["success"]) for k in sorted(reference)])
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(2000)])
    lo, hi = np.quantile(boot, [.025, .975])
    return 100 * delta.mean(), 100 * lo, 100 * hi


def report(path, runs, config, selection):
    lines = ["# LIBERO-Dynamic: joint before-generation budget", "",
             "Frozen pi0.5 + frozen ImageNet ResNet-18; only actor/critic trained. H=50.", "",
             "Selection uses validation only. Split is by trajectory index within the same tasks/levels; "
             "this does not establish transfer to new tasks. One training seed: pilot evidence only.", "",
             f"Selection: `{json.dumps(selection, sort_keys=True)}`", "",
             "| policy | SR % | calls/ep | calls/tick | delta vs matched fixed, pp [95% CI] |",
             "|---|---:|---:|---:|---:|"]
    ref = runs["matched fixed"]
    for name, rows in runs.items():
        m = metrics(rows, config["call_cost"], config["reference_stride"], config["max_steps"])
        delta, lo, hi = paired_delta(ref, rows)
        lines.append(f"| {name} | {100*m['sr']:.2f} | {m['calls']:.2f} | {m['rate']:.4f} | {delta:+.2f} [{lo:+.2f}; {hi:+.2f}] |")
    ratio = metrics(runs["learned"], 0, 1, 1)["rate"] / metrics(ref, 0, 1, 1)["rate"]
    lines.extend(["", f"Learned/matched-fixed call-rate ratio: {ratio:.3f}.",
                  "If test rates differ by >5%, do not claim a gain at equal frequency.",
                  "CI bootstraps episodes within the selected tasks, not training seeds or new tasks.",
                  "Fixed simulated latency includes the declared encoder+actor+policy budget; "
                  "inspect wallclock_policy_elapsed_s before making real-time claims.",
                  "This pipeline and its baselines use the same full-plan queue; earlier K=10 runs are not its primary control."])
    path.write_text("\n".join(lines) + "\n")


class Experiment:
    def __init__(self, args):
        self.args = args
        self.out = args.out.resolve()
        if self.out.exists() and any(self.out.iterdir()):
            raise RuntimeError(f"OUT must be empty; refusing to overwrite {self.out}")
        self.out.mkdir(parents=True, exist_ok=True)
        self.runs = self.out / "rollouts"
        self.config = {"pairs": action_pairs(args.strides, args.masks, args.delay), "delay": args.delay,
                       "call_cost": args.call_cost, "reference_stride": 8, "max_steps": args.max_steps,
                       "encoder": "torchvision.ResNet18_Weights.IMAGENET1K_V1", "seed": args.seed}
        torch.manual_seed(args.seed)
        torch.set_num_threads(4)
        self.model = BudgetHead(len(self.config["pairs"]))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.head = self.out / "current.pt"
        self.best = self.out / "best.pt"
        (self.out / "config.json").write_text(json.dumps(vars(args) | {"out": str(self.out)}, indent=2))

    def save(self, step):
        save_head(self.head, self.model, self.config, self.optimizer, step)

    def rollout(self, name, split, mode="sample", fixed=(8, 4), seed=0, steps=None, limit=None):
        a = self.args
        run = self.runs / name
        if run.exists():
            raise RuntimeError(f"refusing to reuse on-policy rollout directory {run}; choose a fresh OUT")
        traces = self.out / "traces" / name
        traces.mkdir(parents=True)
        env = os.environ.copy()
        env.pop("CORR_HEAD", None)
        # Upstream seeds env with seed+worker_slot; slots come from a static round-robin
        # over the item list, so equal lists and worker counts would also stay paired.
        servers = a.servers if name.startswith("train_") else 1
        env.update(RUNS=str(self.runs), SPEEDS=str(a.speed), LEVELS=a.levels, TRAJ=split,
                   LIMIT=str(limit or (a.train_episodes if split == a.train_traj else a.eval_episodes)),
                   SERVERS=str(servers), WORKERS=str(servers), KEEP_CHUNK="1",
                   MAX_STEPS=str(steps or a.max_steps),
                   RUNNER_ARGS=f"--control-mode learned_budget_queue --seed {a.seed}")
        cmd = ["bash", str(HERE / "run_dynamic.sh"), name, "rtc", "8", str(a.delay / 20),
               "--actions-per-chunk", "50", "--max-guidance-weight", "2", "--budget-head", str(self.head),
               "--budget-traces", str(traces), "--budget-mode", mode,
               "--budget-fixed", f"{fixed[0]},{fixed[1]}", "--budget-seed", str(seed)]
        print(f"{time.strftime('%F %T')} rollout {name} split={split} mode={mode}", flush=True)
        with (self.out / f"{name}.log").open("w") as log:
            process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                rc = process.wait(timeout=a.rollout_timeout_hours * 3600)
                if rc:
                    raise RuntimeError(f"rollout {name} exited {rc}; see {self.out / (name + '.log')}")
            finally:
                # Stop the complete bash/server/worker group on timeout or interruption.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        rows = read_episodes(run, traces, int(env["LIMIT"]))
        print(json.dumps({"run": name, **metrics(rows, a.call_cost, 8, a.max_steps)}), flush=True)
        return rows

    def score(self, rows):
        return metrics(rows, self.args.call_cost, 8, self.args.max_steps)["score"]

    def run(self):
        a = self.args
        self.save(0)
        smoke = self.rollout("smoke", a.train_traj, "fixed", fixed=(a.strides[0], 0), steps=32, limit=1)
        sampled = self.rollout("smoke_sample", a.train_traj, steps=32, limit=1, seed=9000)
        smoke_update = ppo_update(self.model, self.optimizer, sampled, self.config, epochs=1)
        print(json.dumps({"smoke_ppo": smoke_update}), flush=True)
        self.model, _ = load_head(self.head)  # discard smoke updates, including optimizer state
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=a.lr)
        if a.smoke_only:
            (self.out / "smoke_done.json").write_text(json.dumps({"fixed": metrics(smoke, 0, 1, 1),
                "sampled": metrics(sampled, 0, 1, 1), "ppo": smoke_update}, indent=2))
            return
        fixed_val = {}
        for s, r in self.config["pairs"]:
            if s + r + a.delay <= 50:
                fixed_val[(s, r)] = self.rollout(f"val_fixed_{s}_{r}", a.val_traj, "fixed", (s, r), 1000)
        best_fixed = max(fixed_val, key=lambda pair: self.score(fixed_val[pair]))
        # Mild initial preference, retaining substantial exploration over the grid.
        with torch.no_grad():
            self.model.actor.bias[self.config["pairs"].index(best_fixed)] = 1.5
        self.save(0)
        selected_rows = self.rollout("val_0", a.val_traj, seed=1000)
        best_score, selected_step = self.score(selected_rows), 0
        save_head(self.best, self.model, self.config, step=0)
        started = time.monotonic()
        last_step = last_validated = 0
        for step in range(1, a.iterations + 1):
            if step > 1 and time.monotonic() - started >= a.train_hours * 3600:
                break
            rows = self.rollout(f"train_{step}", a.train_traj, seed=a.seed + 10000 + step)
            update = ppo_update(self.model, self.optimizer, rows, self.config, epochs=a.ppo_epochs)
            self.save(step)
            last_step = step
            record = {"iteration": step, **update}
            if step % a.val_every == 0 or step == a.iterations:
                val = self.rollout(f"val_{step}", a.val_traj, seed=1000)
                last_validated = step
                score = self.score(val)
                record["val_score"] = score
                if score > best_score:
                    best_score, selected_step, selected_rows = score, step, val
                    save_head(self.best, self.model, self.config, step=step)
            with (self.out / "train.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if last_step != last_validated:
            val = self.rollout(f"val_{last_step}", a.val_traj, seed=1000)
            score = self.score(val)
            if score > best_score:
                best_score, selected_step, selected_rows = score, last_step, val
                save_head(self.best, self.model, self.config, step=last_step)
            print(json.dumps({"final_validation_iteration": last_step, "val_score": score}), flush=True)
        learned_rate = metrics(selected_rows, a.call_cost, 8, a.max_steps)["rate"]
        close = [p for p, rows in fixed_val.items()
                 if abs(metrics(rows, a.call_cost, 8, a.max_steps)["rate"] / learned_rate - 1) <= .05]
        matched = max(close, key=lambda p: self.score(fixed_val[p])) if close else min(
            fixed_val, key=lambda p: abs(metrics(fixed_val[p], a.call_cost, 8, a.max_steps)["rate"] / learned_rate - 1))
        counts = np.bincount([int(s["action"]) for row in selected_rows.values() for s in row["samples"]],
                             minlength=len(self.config["pairs"])).astype(float)
        # Initial r=0 guarantees marginal support for every runtime feasibility mask.
        self.config["marginal"] = (counts / counts.sum()).tolist()
        self.model, _ = load_head(self.best)
        save_head(self.head, self.model, self.config, step=selected_step)
        save_head(self.best, self.model, self.config, step=selected_step)
        selection = {"iteration": selected_step, "fixed": best_fixed, "matched_fixed": matched,
                     "validation_score": best_score, "validation_rate": learned_rate}
        (self.out / "selection.json").write_text(json.dumps(selection, indent=2))
        runs = {"matched fixed": self.rollout("test_matched", a.test_traj, "fixed", matched, 2000),
                "learned": self.rollout("test_learned", a.test_traj, seed=2000),
                "marginal random": self.rollout("test_random", a.test_traj, "random", seed=2000)}
        runs["objective fixed"] = runs["matched fixed"] if matched == best_fixed else self.rollout(
            "test_fixed", a.test_traj, "fixed", best_fixed, 2000)
        report(self.out / "report.md", runs, self.config, selection)
        (self.out / "done.json").write_text(json.dumps(selection, indent=2))
        print(f"DONE: {self.out / 'report.md'}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--strides", type=int, nargs="+", default=[4, 8, 16])
    p.add_argument("--masks", type=int, nargs="+", default=[0, 4, 12])
    p.add_argument("--delay", type=int, default=4)
    p.add_argument("--speed", type=float, default=.06)
    p.add_argument("--levels", default="l1")
    p.add_argument("--train-traj", default="0 1")
    p.add_argument("--val-traj", default="2")
    p.add_argument("--test-traj", default="3 4")
    p.add_argument("--train-episodes", type=int, default=16)
    p.add_argument("--eval-episodes", type=int, default=16)
    p.add_argument("--servers", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--iterations", type=int, default=12)
    p.add_argument("--train-hours", type=float, default=6)
    p.add_argument("--rollout-timeout-hours", type=float, default=4)
    p.add_argument("--val-every", type=int, default=2)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--call-cost", type=float, default=.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke-only", action="store_true")
    a = p.parse_args(argv)
    splits = [set(getattr(a, f"{s}_traj").split()) for s in ("train", "val", "test")]
    if any(not s for s in splits) or any(splits[i] & splits[j] for i in range(3) for j in range(i)):
        p.error("trajectory splits must be nonempty and disjoint")
    if any(int(t) not in range(5) for split in splits for t in split):
        p.error("trajectory indices must be 0..4")
    if a.delay < 1 or min(a.train_episodes, a.eval_episodes, a.servers, a.max_steps, a.val_every,
                          a.iterations, a.ppo_epochs, a.lr, a.train_hours, a.rollout_timeout_hours) <= 0 or a.call_cost < 0:
        p.error("invalid positive limits or delay/cost")
    action_pairs(a.strides, a.masks, a.delay)
    return a


if __name__ == "__main__":
    Experiment(parse_args()).run()
