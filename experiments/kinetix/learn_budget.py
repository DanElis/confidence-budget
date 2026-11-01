"""Restartable overnight experiment: frozen Kinetix policy + learned (s, r) budget.

Run via bash night_learned.sh. No demonstrations, pretrained-policy gradients,
or external experiment tracker. Train/validation/test RNG streams are disjoint.
PPO acts at request boundaries, with Monte Carlo returns over control ticks.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import subprocess
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

import budget
import budget_head as head
import budget_loop
import preflight
import run_points


def stamp(message):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_pickle(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(jax.device_get(value), f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def tree_hash(tree):
    digest = hashlib.sha256()
    for x in jax.tree.leaves(tree):
        a = np.ascontiguousarray(jax.device_get(x))
        digest.update(str((a.shape, a.dtype)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def objective(result, args, total_steps):
    return (np.asarray(result["solved"]) - args.call_cost * np.asarray(result["n_infer"]) * args.reference_stride / total_steps
            - args.smooth_cost * np.asarray(result["action_change"]) / total_steps)


def check_result(result, logs):
    for k, value in result.items():
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"non-finite rollout field {k}; not a policy failure")
    if np.any((result["solved"] < 0) | (result["solved"] > 1)) or np.any(result["elapsed"] < 1):
        raise RuntimeError("invalid outcome or elapsed time")
    if not np.all(np.isfinite(logs["action"])):
        raise RuntimeError("non-finite executed actions; not a policy failure")


class Experiment:
    def __init__(self, args):
        self.a = args
        self.out = args.out.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache = self.out / "evaluations"
        self.cache.mkdir(exist_ok=True)
        paths = run_points.DEFAULT_LEVELS
        if args.levels:
            paths = tuple(p for p in paths if run_points.level_name(p).removeprefix("worlds_l_") in args.levels)
            if len(paths) != len(args.levels):
                raise ValueError(f"unknown or duplicate levels: {args.levels}")
        self.names = [run_points.level_name(p).removeprefix("worlds_l_") for p in paths]
        os.chdir(run_points.RTC_DIR)
        env, ep, _, self.levels, self.flow, ckpt, obs_dim, action_dim = run_points.load(args.run_path, paths, args.step)
        self.total_steps = int(ep.max_timesteps)
        self.n_levels = len(paths)
        self.eval_fn, self.horizon = budget_loop.make_rollout(
            env, ep, obs_dim, action_dim, args.eval_envs, args.eval_envs,
            planner=head.make_planner(args.delay, self.total_steps, record_features=False))
        self.train_fn, _ = budget_loop.make_rollout(
            env, ep, obs_dim, action_dim, args.train_envs, args.train_envs,
            planner=head.make_planner(args.delay, self.total_steps))
        self.pairs = head.action_pairs(self.horizon, args.delay, tuple(args.strides))
        self.inputs = obs_dim + 2 * self.horizon * action_dim + 4 + self.n_levels
        self.base = budget.LoopConfig(d=args.delay, s=args.reference_stride, guidance=True,
                                     mask_end=self.horizon - args.reference_stride)
        self.base.validate(self.horizon)
        dummy = head.init_network(jax.random.key(0), self.inputs, len(self.pairs), args.width, self.n_levels)
        self.dummy = head.control(dummy, self.inputs, self.pairs, self.n_levels, delay=args.delay)
        source = hashlib.sha256()
        for name in ("learn_budget.py", "budget_head.py", "budget.py", "budget_loop.py", "run_points.py", "preflight.py"):
            source.update((run_points.HERE / name).read_bytes())
        policy_sha = {p: hashlib.sha256((ckpt / "policies" / f"{run_points.level_name(p)}.pkl").read_bytes()).hexdigest()
                      for p in paths}
        settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        revision = subprocess.check_output(["git", "-C", str(run_points.HERE), "rev-parse", "HEAD"], text=True).strip()
        upstream_revision = subprocess.check_output(["git", "-C", str(run_points.RTC_DIR), "rev-parse", "HEAD"], text=True).strip()
        if not upstream_revision.startswith("9296f31"):
            raise RuntimeError("expected RTC checkout @9296f31; run experiments/kinetix/setup.sh first")
        upstream_diff = subprocess.check_output(["git", "-C", str(run_points.RTC_DIR), "diff", "HEAD", "--"])
        identity = dict(settings=settings, sources_sha256=source.hexdigest(), policy_sha256=policy_sha,
                        levels=self.names, pairs=self.pairs.tolist(), total_steps=self.total_steps,
                        horizon=self.horizon, inputs=self.inputs, checkpoint=str(ckpt),
                        upstream_commit=upstream_revision, upstream_diff_sha256=hashlib.sha256(upstream_diff).hexdigest(),
                        versions={name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "flax", "optax", "numpy")},
                        devices=[d.device_kind for d in jax.devices()])
        manifest = self.out / "experiment.json"
        if manifest.exists():
            old = json.loads(manifest.read_text())
            if old["identity"] != identity:
                raise RuntimeError("experiment configuration/source/checkpoint changed; choose a NEW --out directory")
        else:
            atomic_json(manifest, dict(identity=identity, git_commit=revision,
                                      primary="mean paired SR difference vs per-model per-level frequency-matched fixed budget selected on validation",
                                      seed_protocol="training >=10000, validation 1000.., test 2000..; no test-based selection",
                                      notes="state-based policy; fixed set of levels; simulated d independent of wall time"))
        self.identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        stamp(f"devices {jax.devices()}, levels={self.names}, H={self.horizon}, d={args.delay}, T={self.total_steps}, "
              f"actions={self.pairs.tolist()}, checkpoint={ckpt}")

    def rollout(self, controller, seed, *, train=False, cfg=None):
        fn = self.train_fn if train else self.eval_fn
        cfg = self.base if cfg is None else cfg
        key = jax.random.key(seed)
        rngs = jax.random.split(key, self.n_levels)
        noise = jax.random.fold_in(key, 1_000_003)
        decision_key = jax.random.fold_in(key, 2_000_003)
        start = time.monotonic()
        result, logs = jax.device_get(fn(cfg, rngs, self.levels, self.flow, noise, controller, decision_key))
        check_result(result, logs)
        stamp(f"{'train' if train else 'eval'} seed={seed} SR={100 * result['solved'].mean():.2f}% "
              f"calls={result['n_infer'].mean():.1f} s={result['mean_s'].mean():.2f} "
              f"elapsed={time.monotonic() - start:.1f}s")
        return result, logs

    def evaluate(self, controller, tag, seeds):
        rows, histograms = [], []
        fingerprint = tree_hash(controller)
        for seed in seeds:
            path = self.cache / f"{tag}_rng{seed}.npz"
            if path.exists():
                with np.load(path) as z:
                    if str(z["fingerprint"]) != fingerprint or str(z["identity"]) != self.identity:
                        raise RuntimeError(f"cached result does not match controller: {path}; use a new --out")
                    result = {k[4:]: z[k].copy() for k in z.files if k.startswith("res_")}
                    hist = z["hist"].copy()
                stamp(f"skip {path.name}")
            else:
                result, logs = self.rollout(controller, seed)
                hist = head.action_histogram(logs, len(self.pairs))
                tmp = path.with_suffix(".tmp")
                with tmp.open("wb") as f:
                    np.savez_compressed(f, fingerprint=fingerprint, identity=self.identity, hist=hist,
                                        **{f"res_{k}": v for k, v in result.items()})
                os.replace(tmp, path)
            rows.append(result)
            histograms.append(hist)
        return {k: np.concatenate([r[k] for r in rows], axis=1) for k in rows[0]}, np.sum(histograms, axis=0)

    def fixed_controller(self, ids):
        return dict(self.dummy, mode=jnp.asarray(2, jnp.int32), fixed=jnp.asarray(ids, jnp.int32),
                    initial_mask=jnp.asarray(self.a.delay + self.pairs[ids, 1]))

    def select_fixed(self):
        """The reference is selected on validation, separately per task. Never on test."""
        scores, rates, successes = [], [], []
        for i, (s, r) in enumerate(self.pairs):
            controller = self.fixed_controller(np.full(self.n_levels, i, np.int32))
            result, _ = self.evaluate(controller, f"val_fixed_s{s}_r{r}", self.a.val_seeds)
            scores.append(objective(result, self.a, self.total_steps).mean(1))
            rates.append((result["n_infer"] / np.maximum(result["elapsed"], 1)).mean(1))
            successes.append(result["solved"].mean(1))
        self.grid_rates, self.grid_successes = np.asarray(rates), np.asarray(successes)
        ids = np.argmax(np.asarray(scores), axis=0).astype(np.int32)
        atomic_json(self.out / "fixed_selection.json", dict(levels=self.names, action_ids=ids.tolist(),
                    pairs=self.pairs[ids].tolist(), validation_scores=np.asarray(scores).tolist()))
        stamp(f"validation-selected fixed budgets: {dict(zip(self.names, self.pairs[ids].tolist()))}")
        return ids

    def match_frequency(self, result):
        """Best validation SR among constants within 5% of the learned validation rate;
        otherwise nearest rate. The test report flags any remaining cost mismatch.
        """
        target = (result["n_infer"] / np.maximum(result["elapsed"], 1)).mean(1)
        relative_error = np.abs(self.grid_rates / target[None] - 1)
        eligible = relative_error <= 0.05
        best = np.argmax(np.where(eligible, self.grid_successes, -np.inf), axis=0)
        return np.where(eligible.any(0), best, np.argmin(relative_error, axis=0)).astype(np.int32)

    def validate_controller(self, controller, tag):
        result, _ = self.evaluate(controller, tag, self.a.val_seeds)
        return float(objective(result, self.a, self.total_steps).mean()), float(result["solved"].mean())

    def train(self, model_seed, fixed):
        folder = self.out / f"model_{model_seed}"
        folder.mkdir(exist_ok=True)
        done, latest, best_path = folder / "done.json", folder / "latest.pkl", folder / "best.pkl"
        if done.exists():
            with best_path.open("rb") as f:
                return pickle.load(f)["controller"]
        opt, update = head.make_update(self.a.lr)
        rng = np.random.default_rng(30_000 + model_seed)
        if latest.exists():
            with latest.open("rb") as f:
                saved = pickle.load(f)
            ctrl = jax.tree.map(jnp.asarray, saved["controller"])
            opt_state = jax.tree.map(jnp.asarray, saved["optimizer"])
            start_step, best = saved["step"], saved["best"]
            rng.bit_generator.state = saved["numpy_rng"]
        else:
            net = head.init_network(jax.random.key(model_seed), self.inputs, len(self.pairs), self.a.width, self.n_levels)
            prior = np.full((self.n_levels, len(self.pairs)), 0.3 / (len(self.pairs) - 1), np.float32)
            prior[np.arange(self.n_levels), fixed] = 0.7
            net["task_bias"] = jnp.log(prior)
            net["actor"]["w"] = jnp.zeros_like(net["actor"]["w"])
            ctrl = head.control(net, self.inputs, self.pairs, self.n_levels, delay=self.a.delay, fixed=fixed)
            result, logs = self.rollout(ctrl, 10_000 + model_seed * 100_000, train=True)
            x = np.asarray(logs["head_x"])[np.asarray(logs["due"], bool)]
            mean, std = x.mean(0), np.maximum(x.std(0), 0.05)
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
                raise RuntimeError("non-finite normalisation statistics")
            ctrl = dict(ctrl, mean=jnp.asarray(mean), std=jnp.asarray(std))
            del x, logs, result
            opt_state = opt.init(ctrl["net"])
            best, _ = self.validate_controller(ctrl, f"val_model{model_seed}_step0")
            start_step = 0
            atomic_pickle(best_path, dict(controller=ctrl, step=0, score=best))
            atomic_pickle(latest, dict(controller=ctrl, optimizer=opt_state, step=0, best=best,
                                       numpy_rng=rng.bit_generator.state))
        deadline = time.monotonic() + self.a.train_hours * 3600 / len(self.a.model_seeds)
        step = start_step
        for step in range(start_step + 1, self.a.iterations + 1):
            result, logs = self.rollout(ctrl, 10_000 + model_seed * 100_000 + step, train=True)
            batch = head.training_batch(result, logs, call_cost=self.a.call_cost, smooth_cost=self.a.smooth_cost,
                                        reference_stride=self.a.reference_stride, delay=self.a.delay)
            del logs
            metrics, stop_early = {}, False
            for epoch in range(self.a.epochs):
                order = rng.permutation(len(batch["target"]))
                for begin in range(0, len(order), self.a.minibatch):
                    idx = order[begin:begin + self.a.minibatch]
                    valid_n = len(idx)
                    idx = np.pad(idx, (0, self.a.minibatch - valid_n), mode="edge")
                    mini = {k: jnp.asarray(v[idx]) for k, v in batch.items()}
                    mini["valid"] = jnp.asarray(np.arange(self.a.minibatch) < valid_n, jnp.float32)
                    net, candidate_state, m = update(ctrl["net"], opt_state, mini, ctrl["mean"], ctrl["std"])
                    metrics = {k: float(v) for k, v in jax.device_get(m).items()}
                    if not all(np.isfinite(v) for v in metrics.values()):
                        raise RuntimeError("non-finite PPO update; last good checkpoint remains intact")
                    if metrics["kl"] > 0.03:
                        stop_early = True
                        break
                    ctrl, opt_state = dict(ctrl, net=net), candidate_state
                if stop_early:
                    break
            row = dict(step=step, sr=float(result["solved"].mean()), calls=float(result["n_infer"].mean()),
                       mean_s=float(result["mean_s"].mean()), mean_soft_r=float(result["mean_mask"].mean() - self.a.delay),
                       samples=len(batch["target"]), **metrics)
            del batch, result
            reached_limit = time.monotonic() >= deadline
            if step % self.a.val_every == 0 or step == self.a.iterations or reached_limit:
                score, sr = self.validate_controller(ctrl, f"val_model{model_seed}_step{step}")
                row.update(val_score=score, val_sr=sr)
                if score > best:
                    best = score
                    atomic_pickle(best_path, dict(controller=ctrl, step=step, score=score))
            with (folder / "train.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            atomic_pickle(latest, dict(controller=ctrl, optimizer=opt_state, step=step, best=best,
                                       numpy_rng=rng.bit_generator.state))
            stamp(f"model={model_seed} iteration={step} {row}")
            if reached_limit:
                stamp(f"model={model_seed}: training time allocation reached; best validation checkpoint retained")
                break
        with best_path.open("rb") as f:
            selected = pickle.load(f)
        atomic_json(done, dict(iterations=step, best_score=best, selected_step=selected["step"]))
        return selected["controller"]


def paired_ci(differences, n_boot=2000):
    """M x L x E differences. Resample training seeds and episodes within each fixed level.

    Episode indices are shared across model seeds because all see the same test
    episodes. This avoids treating repeated evaluations of one episode as independent.
    """
    d = np.asarray(differences, float)
    rng = np.random.default_rng(7654)
    m, levels, episodes = d.shape
    boots = np.empty(n_boot)
    for i in range(n_boot):
        chosen = d[rng.integers(m, size=m)]
        boots[i] = np.mean([chosen[:, l, rng.integers(episodes, size=episodes)].mean() for l in range(levels)])
    return float(d.mean()), *np.percentile(boots, [2.5, 97.5]).tolist()


def report(exp, fixed, learned, random_controls, matched_refs):
    a = exp.a
    selected_steps = {seed: json.loads((exp.out / f"model_{seed}" / "done.json").read_text())["selected_step"]
                      for seed in a.model_seeds[:len(learned)]}
    lines = ["# Learned budget on Kinetix", "",
             f"Frozen public flow policies, H={exp.horizon}, d={a.delay}; levels={len(exp.names)}. "
             f"Evaluated model seeds={a.model_seeds[:len(learned)]}; held-out test RNGs={a.test_seeds}. "
             f"{'COMPLETE' if len(learned) == len(a.model_seeds) else 'PARTIAL: some model seeds still pending'}.", "",
             f"Selected checkpoint iterations (validation only): {selected_steps}.", "",
             "The primary reference is a per-model, per-level constant (s,r) selected ONLY on validation to match the learned call rate. "
             "The global fixed reference in the table is selected by the training objective, separately per level. "
             "Marginal-random controls use each learned model's per-level action frequencies on validation. "
             "They remove state dependence but do not enforce exactly equal test cost. "
             "All policies use identical reset/action-noise streams and flow batches on the same test items.", "",
             "| policy | SR % | paired delta vs fixed, pp [95% CI] | calls/ep | calls/tick | mean step | mean action change |",
             "|---|---|---|---|---|---|---|"]
    def add(name, results, reference):
        stack = {k: np.stack([r[k] for r in results]) for k in results[0]}
        d = stack["solved"] - reference
        mean, lo, hi = paired_ci(100 * d)
        delta = "—" if name == "fixed per level" else f"{mean:+.2f} [{lo:+.2f}; {hi:+.2f}]"
        rate = stack["n_infer"] / np.maximum(stack["elapsed"], 1)
        jump = stack["action_change"] / np.maximum(stack["elapsed"] - 1, 1)
        lines.append(f"| {name} | {100 * stack['solved'].mean():.2f} | {delta} | "
                     f"{stack['n_infer'].mean():.2f} | {rate.mean():.4f} | {stack['mean_s'].mean():.2f} | {jump.mean():.4f} |")
    ref = fixed["solved"][None]
    add("fixed per level", [fixed], ref)
    for seed, r in zip(a.model_seeds, learned):
        add(f"learned seed {seed}", [r], ref)
    add("learned mean over training seeds", learned, ref)
    add("frequency-matched fixed mean", matched_refs, ref)
    add("marginal-random mean", random_controls, ref)
    m, lo, hi = paired_ci(100 * (np.stack([r["solved"] for r in learned])
                                - np.stack([r["solved"] for r in random_controls])))
    primary = paired_ci(100 * (np.stack([r["solved"] for r in learned])
                              - np.stack([r["solved"] for r in matched_refs])))
    lines += ["", f"PRIMARY learned vs frequency-matched fixed: {primary[0]:+.2f} pp [{primary[1]:+.2f}; {primary[2]:+.2f}].",
              f"State-dependent vs marginal-random (exploratory): {m:+.2f} pp [{lo:+.2f}; {hi:+.2f}].",
              "CI resamples training seeds and episodes within the FIXED levels; it does not establish transfer to new levels. "
              "Only the aggregate learned-vs-frequency-matched-fixed SR difference is primary; other comparisons are exploratory.", "",
              "## Cost and smoothness checks", ""]
    fixed_rate = float((fixed["n_infer"] / np.maximum(fixed["elapsed"], 1)).mean())
    learned_rate = float(np.mean([np.mean(r["n_infer"] / np.maximum(r["elapsed"], 1)) for r in learned]))
    random_rate = float(np.mean([np.mean(r["n_infer"] / np.maximum(r["elapsed"], 1)) for r in random_controls]))
    fixed_jump = float((fixed["action_change"] / np.maximum(fixed["elapsed"] - 1, 1)).mean())
    learned_jump = float(np.mean([np.mean(r["action_change"] / np.maximum(r["elapsed"] - 1, 1)) for r in learned]))
    matched_rate = float(np.mean([np.mean(r["n_infer"] / np.maximum(r["elapsed"], 1)) for r in matched_refs]))
    matched_jump = float(np.mean([np.mean(r["action_change"] / np.maximum(r["elapsed"] - 1, 1)) for r in matched_refs]))
    lines += [f"- Learned/fixed call-rate ratio: {learned_rate / fixed_rate:.3f}.",
              f"- Learned/frequency-matched-fixed call-rate ratio: {learned_rate / matched_rate:.3f}.",
              f"- Learned/marginal-random call-rate ratio: {learned_rate / random_rate:.3f}.",
              f"- Learned/fixed mean-action-change ratio: {learned_jump / max(fixed_jump, 1e-12):.3f}.",
              f"- Learned/frequency-matched-fixed action-change ratio: {learned_jump / max(matched_jump, 1e-12):.3f}.",
              "- If call rates differ by more than 5%, do not claim improvement at equal frequency; a calibrated control is required.",
              "- Action changes are in motor-command units, not measured physical trajectory smoothness.",
              "- Selection reward: success - call_cost*N*reference_stride/T - smooth_cost*sum(|delta action|)/T.",
              "- Step-0 warm-start checkpoints may win validation; inspect model_*/best.pkl step and train.jsonl.", "",
              "## Success by level", "", "| level | fixed | learned mean | random mean |", "|---|---|---|---|"]
    for l, name in enumerate(exp.names):
        lines.append(f"| {name} | {100 * fixed['solved'][l].mean():.2f} | "
                     f"{100 * np.mean([r['solved'][l].mean() for r in learned]):.2f} | "
                     f"{100 * np.mean([r['solved'][l].mean() for r in random_controls]):.2f} |")
    (exp.out / "report.md").write_text("\n".join(lines) + "\n")
    stamp(f"wrote {exp.out / 'report.md'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=run_points.HERE / "runs" / "learned_d2")
    p.add_argument("--run-path", type=Path, default=run_points.HERE / "assets" / "bc")
    p.add_argument("--step", type=int, default=-1)
    p.add_argument("--levels", nargs="+")
    p.add_argument("--delay", type=int, default=2)
    p.add_argument("--strides", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--reference-stride", type=int, default=2)
    p.add_argument("--train-envs", type=int, default=32)
    p.add_argument("--eval-envs", type=int, default=256)
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--train-hours", type=float, default=4.0, help="total training allocation, divided among model seeds; evaluation is extra")
    p.add_argument("--model-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--val-seeds", type=int, nargs="+", default=[1000, 1001])
    p.add_argument("--test-seeds", type=int, nargs="+", default=[2000, 2001, 2002, 2003])
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--minibatch", type=int, default=512)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--call-cost", type=float, default=0.1)
    p.add_argument("--smooth-cost", type=float, default=0.05)
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end run, separate output directory")
    p.add_argument("--preflight", action="store_true", help="real-model equivalence check against the existing fixed loop, then exit")
    a = p.parse_args()
    a.out, a.run_path = a.out.resolve(), a.run_path.resolve()
    if a.smoke or a.preflight:
        a.levels = a.levels or ["grasp_easy"]
        a.train_envs = a.eval_envs = 4
        a.iterations, a.epochs, a.val_every = 1, 1, 1
        a.model_seeds, a.val_seeds, a.test_seeds = [0], [1000], [2000]
    counts = (a.train_envs, a.eval_envs, a.iterations, a.epochs, a.minibatch, a.val_every, a.width)
    if min(counts) < 1 or a.train_hours <= 0 or min(a.call_cost, a.smooth_cost) < 0:
        p.error("counts and train-hours must be positive; costs nonnegative")
    if not all(np.isfinite(v) for v in (a.lr, a.train_hours, a.call_cost, a.smooth_cost)) or a.lr <= 0:
        p.error("lr/train-hours/costs must be finite; lr must be positive")
    if not a.model_seeds or len(set(a.model_seeds)) != len(a.model_seeds) or any(s < 0 for s in a.model_seeds):
        p.error("model seeds must be nonnegative and unique")
    if not (a.val_seeds and a.test_seeds) or len(set(a.val_seeds)) != len(a.val_seeds) or len(set(a.test_seeds)) != len(a.test_seeds):
        p.error("validation/test seeds must be nonempty and unique")
    if any(not 1000 <= s < 2000 for s in a.val_seeds) or any(not 2000 <= s < 10000 for s in a.test_seeds):
        p.error("validation seeds must be in [1000,2000), test seeds in [2000,10000)")
    if a.iterations >= 100000:
        p.error("iterations must be <100000 (disjoint model training RNG ranges)")
    if a.reference_stride not in a.strides:
        p.error("reference-stride must be one of --strides")
    if not any(d.platform == "gpu" for d in jax.devices()):
        raise SystemExit("GPU required for real-model runs; use the CPU pytest preflight for synthetic tests")
    exp = Experiment(a)
    if len(exp.pairs) < 2:
        raise SystemExit("at least two feasible budgets are needed")
    if a.preflight:
        if (exp.out / "preflight_pass.json").exists():
            stamp("preflight already passed for these sources/settings/checkpoints")
            return
        # Scientific comparisons all use exp.eval_fn, including fixed controls.
        # Verify sampled-actor mode against fixed mode in that SAME executable.
        ids = np.full(exp.n_levels, np.flatnonzero((exp.pairs == (a.reference_stride, 0)).all(1))[0], np.int32)
        cfg = dataclasses.replace(exp.base, mask_end=a.delay)
        constant = head.constant_actor(exp.fixed_controller(ids))
        controlled, logs = exp.rollout(constant, 3000, cfg=cfg)
        sampled, sampled_logs = exp.rollout(dict(constant, mode=jnp.asarray(0, jnp.int32)), 3000, cfg=cfg)
        diagnostics = preflight.compare_rollouts(controlled, logs, sampled, sampled_logs, delay=a.delay)
        if diagnostics.get("max_abs_action_diff", float("inf")) != 0:
            diagnostics["failures"].append("common executable: actions must be bit-identical")
        diagnostics["passed"] = not diagnostics["failures"]
        env, ep, _, levels, flow, _, od, ad = run_points.load(a.run_path, tuple(
            p for p in run_points.DEFAULT_LEVELS if run_points.level_name(p).removeprefix("worlds_l_") in exp.names), a.step)
        reference, _ = budget_loop.make_rollout(env, ep, od, ad, a.eval_envs, a.eval_envs)
        key = jax.random.key(3000)
        expected, expected_logs = jax.device_get(reference(cfg, jax.random.split(key, exp.n_levels), levels, flow,
                                                         jax.random.fold_in(key, 1_000_003)))
        legacy = preflight.compare_rollouts(controlled, logs, expected, expected_logs, delay=a.delay)
        # A separately compiled legacy graph is not used in train/test. Its
        # action drift is diagnostic, while timing/outcome/metric failures remain fatal.
        legacy_fatal = [f for f in legacy["failures"] if not f.startswith("first-episode action max error")]
        diagnostics["failures"].extend(f"legacy: {f}" for f in legacy_fatal)
        diagnostics["passed"] = not diagnostics["failures"]
        diagnostics["legacy_separate_executable"] = legacy
        atomic_json(exp.out / "preflight_diagnostics.json", diagnostics)
        stamp(f"preflight diagnostics: {json.dumps(diagnostics, allow_nan=False)}")
        if not diagnostics["passed"]:
            trace_path = exp.out / "preflight_failure.npz"
            tmp = trace_path.with_suffix(".tmp")
            with tmp.open("wb") as f:
                np.savez_compressed(f, **{f"controlled_{k}": v for k, v in controlled.items()},
                                    **{f"reference_{k}": v for k, v in expected.items()},
                                    **{f"sampled_{k}": v for k, v in sampled.items()},
                                    **{f"controlled_log_{k}": logs[k] for k in ("action", "due", "end")},
                                    **{f"sampled_log_{k}": sampled_logs[k] for k in ("action", "due", "end")},
                                    **{f"reference_log_{k}": expected_logs[k] for k in ("action", "due", "end")})
            os.replace(tmp, trace_path)
            raise AssertionError("Preflight failed:\n" + "\n".join(diagnostics["failures"]))
        atomic_json(exp.out / "preflight_pass.json", dict(passed=True, seed=3000, pairs=exp.pairs.tolist(),
                                                          diagnostics=diagnostics))
        stamp("preflight PASS: fixed and constant sampled actor are bit-identical in the common eval executable; legacy timing/outcomes agree")
        return
    fixed_ids = exp.select_fixed()
    learned_ctrls = [exp.train(seed, fixed_ids) for seed in a.model_seeds]
    # Freeze models and controls BEFORE reading any test outcomes.
    random_ctrls, matched_ctrls, matched_ids = [], [], []
    for seed, ctrl in zip(a.model_seeds, learned_ctrls):
        val_result, hist = exp.evaluate(ctrl, f"val_selected_model{seed}", a.val_seeds)
        probs = hist / np.maximum(hist.sum(1, keepdims=True), 1)
        if np.any(hist.sum(1) == 0):
            raise RuntimeError("no validation requests for marginal control")
        random_ctrls.append(dict(ctrl, mode=jnp.asarray(3, jnp.int32), marginal=jnp.asarray(probs, jnp.float32)))
        ids = exp.match_frequency(val_result)
        matched_ids.append(ids.tolist())
        matched_ctrls.append(exp.fixed_controller(ids))
    atomic_json(exp.out / "test_plan.json", dict(fixed_ids=fixed_ids.tolist(), model_seeds=a.model_seeds,
                controller_hashes=[tree_hash(c) for c in learned_ctrls], test_seeds=a.test_seeds,
                matched_fixed_ids=matched_ids,
                marginal_probabilities=[np.asarray(c["marginal"]).tolist() for c in random_ctrls]))
    fixed, _ = exp.evaluate(exp.fixed_controller(fixed_ids), "test_fixed", a.test_seeds)
    learned, random_results, matched_refs = [], [], []
    for seed, ctrl, random_ctrl, matched_ctrl in zip(a.model_seeds, learned_ctrls, random_ctrls, matched_ctrls):
        r, _ = exp.evaluate(ctrl, f"test_learned_model{seed}", a.test_seeds)
        learned.append(r)
        r, _ = exp.evaluate(random_ctrl, f"test_marginal_model{seed}", a.test_seeds)
        random_results.append(r)
        r, _ = exp.evaluate(matched_ctrl, f"test_matched_fixed_model{seed}", a.test_seeds)
        matched_refs.append(r)
        report(exp, fixed, learned, random_results, matched_refs)
    stamp("DONE; no test outcomes were used for training or model selection")


if __name__ == "__main__":
    main()
