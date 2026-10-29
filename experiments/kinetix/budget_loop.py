"""Confidence budget and lambda on Kinetix: per-environment schedules (budget.py) on the public BC checkpoints.

Each --config is ``label:key=value,...`` with keys of budget.LoopConfig (d defaults to --delay). Examples:

    naive:s=2                                    naive async, constant s (= eval_flow naive)
    rtc:guidance=1,s=2                           RTC, constant s, mask to H - s (= eval_flow realtime)
    rtc_k8:guidance=1,s=2,n_samples=8            same, plus dispersion across 8 noise samples (logged)
    b_const:guidance=1,planner=budget,rule=const,c=8,lam=0.5     budget with constant h* (a fixed (s, r) pair)
    b05:guidance=1,planner=budget,rule=above,eta=0.03,k=1,min_h=1,lam=0.5,n_samples=8
    bz:guidance=1,planner=budget,rule=above,eta=0.03,lam_mode=z,mu=0.01,sigma=0.005,n_samples=8
    ada:guidance=1,planner=adaptive,rule=above,eta=0.03,n_samples=8
    b_aac:guidance=1,planner=budget,rule=aac,min_h=1,lam=0.5,n_samples=8      h* as in AAC, no threshold to calibrate
    ada_aac:guidance=1,planner=adaptive,rule=aac,min_h=1,n_samples=8
    conf1:guidance=1,s=2,n_samples=8,mask_mode=rtc_x_conf,conf_sigma=1,conf_ref=0.05   RTC mask * exp(-D_old / sigma)
    coh:guidance=1,s=2,n_samples=8,select=coherent                  noise of the sample closest to the old leftover
    med:guidance=1,s=2,n_samples=8,select=medoid                    control: consensus sample, no reference to the old plan

conf_ref: median dispersion of a reference run (``--calibrate LABEL`` prints it); conf_sigma is in its units.

Protocol (CLAUDE.md): environments, resets and noise are shared by all configs of a run (paired); Δ is reported
against the first config with the same d, with a 95% bootstrap CI over environments (stratified by level).
Calibrate eta / mu / sigma on one seed (``--calibrate``, from a config with n_samples > 1) and report on another.

Outputs in --out: <label>.npz (per-environment results, logs of the first --n-log environments per level, config),
results.csv (per config x level), report.md.

    ./run.sh budget_loop.py --delay 2 --config naive:s=2 --config rtc:guidance=1,s=2 --out runs/budget_d2
    ./run.sh budget_loop.py --out runs/budget_d2 --calibrate rtc_k8 [--target-h 6]
"""

import argparse
import dataclasses
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import run_points  # noqa: E402  (puts the RTC sources on sys.path)

import flax.nnx as nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import kinetix.environment.wrappers as wrappers  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import model as _model  # noqa: E402
import train_expert  # noqa: E402

sys.path.insert(0, str(run_points.HERE))
import budget  # noqa: E402

PER_ENV = ("solved", "ret", "length", "unfinished", "n_infer", "mean_s", "mean_h", "mean_mask", "mean_lam",
           "frac_sel", "jump_switch", "jump_other")


def parse_config(spec: str, delay: int) -> tuple[str, budget.LoopConfig]:
    label, _, body = spec.partition(":")
    fields = {f.name: f for f in dataclasses.fields(budget.LoopConfig)}
    kw = {"d": delay}
    for item in filter(None, body.split(",")):
        key, _, value = item.partition("=")
        if key not in fields:
            raise SystemExit(f"unknown key {key!r} in {spec!r}; keys: {sorted(fields)}")
        default = fields[key].default
        if key == "mask_end":
            kw[key] = None if value in ("", "none", "None") else int(value)
        elif isinstance(default, bool):
            kw[key] = value.lower() in ("1", "true", "yes")
        elif isinstance(default, int):
            kw[key] = int(value)
        elif isinstance(default, float):
            kw[key] = float(value)
        else:
            kw[key] = value
    return label, budget.LoopConfig(**kw)


def make_rollout(env, env_params, obs_dim, action_dim, num_evals, n_log):
    env = train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(wrappers.AutoReplayWrapper(train_expert.NoisyActionWrapper(env))), num_evals
    )
    model_cfg = _model.ModelConfig()
    horizon = model_cfg.action_chunk_size
    num_steps = int(env_params.max_timesteps)

    def policy_of(state_dict):
        policy = _model.FlowPolicy(obs_dim=obs_dim, action_dim=action_dim, config=model_cfg, rngs=nnx.Rngs(0))
        graphdef, state = nnx.split(policy)
        state.replace_by_pure_dict(state_dict)
        return nnx.merge(graphdef, state)

    def flow(policy, obs, noise, n):  # FlowPolicy.action with given noise
        dt = 1 / n

        def step(carry, _):
            x_t, t = carry
            return (x_t + dt * policy(obs, x_t, t), t + dt), None

        (x_1, _), _ = jax.lax.scan(step, (noise, 0.0), length=n)
        return x_1

    def guided(policy, obs, noise, prev, weights, n, max_w):  # FlowPolicy.realtime_action, per-env weights
        dt = 1 / n

        def step(carry, _):
            x_t, t = carry

            def one(o, x, y, w):
                def denoiser(x):
                    v = policy(o[None], x[None], t)[0]
                    return x + v * (1 - t), v

                x_1, vjp_fun, v = jax.vjp(denoiser, x, has_aux=True)
                correction = vjp_fun((y - x_1) * w[:, None])[0]
                inv_r2 = (t**2 + (1 - t) ** 2) / ((1 - t) ** 2)
                c = jnp.nan_to_num((1 - t) / t, posinf=max_w)
                return v + jnp.minimum(c * inv_r2, max_w) * correction

            return (x_t + dt * jax.vmap(one)(obs, x_t, prev, weights), t + dt), None

        (x_1, _), _ = jax.lax.scan(step, (noise, 0.0), length=n)
        return x_1

    @jax.jit
    def reset(rngs, levels):
        # as eval_flow.eval: rng, key = split(rng); reset with key
        keys = jax.vmap(jax.random.split)(rngs)
        obs, state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(keys[:, 1], levels, env_params)
        return obs, state, keys[:, 0]

    def rollout(cfg: budget.LoopConfig, rngs, levels, state_dicts, noise_key):
        obs0, state0, step_keys = reset(rngs, levels)

        def step_fn(t, state, action):
            def one(key, s, a):
                obs, s, _reward, done, info = env.step(jax.random.fold_in(key, t), s, a, env_params)
                return obs, s, done, info

            return jax.vmap(one)(step_keys, state, action)

        def sample_fn(obs, noise):
            return jax.vmap(lambda sd, o, z: flow(policy_of(sd), o, z, cfg.num_flow_steps))(state_dicts, obs, noise)

        def guided_fn(obs, noise, prev, weights):
            return jax.vmap(
                lambda sd, o, z, y, w: guided(policy_of(sd), o, z, y, w, cfg.num_flow_steps, cfg.max_guidance_weight)
            )(state_dicts, obs, noise, prev, weights)

        def weights_fn(mask_end):
            w = lambda e: _model.get_prefix_weights(cfg.d, e, horizon, cfg.schedule)  # noqa: E731
            return jax.vmap(jax.vmap(w))(mask_end)

        return budget.run_loop(
            cfg, horizon, action_dim, num_steps, obs0, state0, step_fn, sample_fn, guided_fn, weights_fn,
            noise_key, n_log,
        )

    return jax.jit(rollout, static_argnums=0), horizon


# ---------------------------------------------------------------- report


def bootstrap_delta(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """a, b (L, B) paired per-env outcomes -> mean over levels of (a - b), 95% CI; resampling envs within levels."""
    rng = np.random.default_rng(seed)
    diff = a - b
    n_levels, n = diff.shape
    boots = np.zeros(n_boot)
    for lvl in range(n_levels):
        idx = rng.integers(0, n, size=(n_boot, n))
        boots += diff[lvl][idx].mean(1)
    boots /= n_levels
    return diff.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def write_report(out: pathlib.Path, labels: list[str], level_names: list[str]):
    runs = {lab: np.load(out / f"{lab}.npz") for lab in labels}
    cfgs = {lab: json.loads(str(runs[lab]["config"])) for lab in labels}
    ref_of_d = {}
    for lab in labels:
        ref_of_d.setdefault(cfgs[lab]["d"], lab)
    rows, lines = [], ["# Kinetix: confidence budget", ""]
    lines += [
        "Solve rate of the first episode, mean over levels; Δ to the first config with the same d, "
        "95% bootstrap CI over environments. infer/ep — policy calls per episode (incl. the first chunk); "
        "s, h*, mask end (= d + r), λ — means over plans; sel — share of chunks where a sample other than 0 was "
        "executed; jump — mean L2 action change at chunk switches / elsewhere.",
        "",
        "| config | d | SR, % | Δ, p.p. [95% CI] | infer/ep | s | h* | mask end | λ | sel | jump switch / other "
        "| unfinished | compile, s | run, s | peak GPU, MB |",
        "|---" * 15 + "|",
    ]
    for lab in labels:
        r, c = runs[lab], cfgs[lab]
        ref = ref_of_d[c["d"]]
        solved = 100 * r["solved"]
        if ref == lab:
            delta = "—"
        else:
            m, lo, hi = bootstrap_delta(solved, 100 * runs[ref]["solved"])
            delta = f"{m:+.1f} [{lo:+.1f}; {hi:+.1f}] vs {ref}"
        lines.append(
            f"| {lab} | {c['d']} | {solved.mean():.1f} | {delta} | {r['n_infer'].mean():.1f} | {r['mean_s'].mean():.2f} "
            f"| {r['mean_h'].mean():.2f} | {r['mean_mask'].mean():.2f} | {r['mean_lam'].mean():.2f} "
            f"| {r['frac_sel'].mean():.2f} "
            f"| {r['jump_switch'].mean():.3f} / {r['jump_other'].mean():.3f} | {int(r['unfinished'].sum())} "
            f"| {float(r['compile_s']):.0f} | {float(r['run_s']):.0f} | {float(r['peak_gpu_mb']):.0f} |"
        )
        for i, lvl in enumerate(level_names):
            rows.append({"label": lab, **{k: c[k] for k in ("d", "guidance", "planner")}, "level": lvl,
                         **{k: float(np.mean(r[k][i])) for k in PER_ENV},
                         "compile_s": float(r["compile_s"]), "run_s": float(r["run_s"])})
    lines += ["", "## Configs", ""]
    lines += [f"- `{lab}`: `{json.dumps(cfgs[lab])}`" for lab in labels]
    lines += ["", "## Solve rate per level, %", "", "| level | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for i, lvl in enumerate(level_names):
        lines.append(f"| {lvl} | " + " | ".join(f"{100 * runs[lab]['solved'][i].mean():.1f}" for lab in labels) + " |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)


def calibrate(out: pathlib.Path, label: str, target: float | None, k: int, min_h: int, zone: int):
    """From the logged plans of ``label`` (no outcomes): conf_ref = median dispersion; with ``target``, eta of the
    'above' rule giving mean h* = target, and mu / sigma for lam_mode z."""
    r = np.load(out / f"{label}.npz")
    disp = r["log_disp"][r["log_due"]]  # (n_plans, H)
    if not np.any(disp):
        raise SystemExit(f"{label}: no dispersion logged (needs n_samples > 1 and --n-log > 0)")
    print(f"{label}: {len(disp)} plans; median dispersion {np.median(disp):.5g} -> use conf_ref={np.median(disp):.5g}")
    print(f"profile (mean dispersion per action index): {np.round(disp.mean(0), 5).tolist()}")
    if target is None:
        return

    def mean_h(eta):
        cfg = budget.LoopConfig(rule="above", eta=float(eta), k=k, min_h=min_h, n_samples=2)
        return float(np.mean(np.asarray(budget.h_star(cfg, jnp.asarray(disp)))))

    lo, hi = float(disp.min()), float(disp.max()) + 1e-9
    for _ in range(40):  # mean h* grows with eta
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mean_h(mid) < target else (lo, mid)
    eta = (lo + hi) / 2
    early = disp[:, :zone].mean(1)
    print(f"eta={eta:.5g} -> mean h* {mean_h(eta):.2f} (target {target}); "
          f"early zone {zone}: mu={early.mean():.5g} sigma={early.std():.5g}")
    print(f"use: rule=above,eta={eta:.5g},k={k},min_h={min_h}  /  lam_mode=z,mu={early.mean():.5g},"
          f"sigma={early.std():.5g},zone={zone}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", action="append", default=[], help="label:key=value,... (repeatable)")
    ap.add_argument("--delay", type=int, default=2, help="default d for configs that do not set it")
    ap.add_argument("--out", default=str(run_points.HERE / "runs" / "budget"))
    ap.add_argument("--run-path", default=str(run_points.HERE / "assets" / "bc"))
    ap.add_argument("--step", type=int, default=-1)
    ap.add_argument("--num-evals", type=int, default=2048)
    ap.add_argument("--levels", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-log", type=int, default=64, help="environments per level whose per-step logs are stored")
    ap.add_argument("--calibrate", default=None, metavar="LABEL", help="only calibrate from <out>/<LABEL>.npz")
    ap.add_argument("--target-h", type=float, default=None)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--min-h", type=int, default=1)
    ap.add_argument("--zone", type=int, default=2)
    args = ap.parse_args()
    out = pathlib.Path(args.out).resolve()

    if args.calibrate:
        calibrate(out, args.calibrate, args.target_h, args.k, args.min_h, args.zone)
        return
    if not args.config:
        raise SystemExit("give at least one --config")
    configs = [parse_config(s, args.delay) for s in args.config]
    labels = [lab for lab, _ in configs]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"duplicate labels in {labels}")

    run_path = pathlib.Path(args.run_path).resolve()
    out.mkdir(parents=True, exist_ok=True)
    level_paths = run_points.DEFAULT_LEVELS
    if args.levels:
        level_paths = tuple(p for p in level_paths if run_points.level_name(p).removeprefix("worlds_l_") in args.levels)
        assert len(level_paths) == len(args.levels), f"unknown level in {args.levels}"
    level_names = [run_points.level_name(p).removeprefix("worlds_l_") for p in level_paths]

    os.chdir(run_points.RTC_DIR)
    print(f"jax {jax.__version__} {jax.devices()}; {len(level_paths)} levels, num_evals={args.num_evals}", flush=True)
    env, env_params, _static, levels, state_dicts, ckpt_dir, obs_dim, action_dim = run_points.load(
        run_path, level_paths, args.step
    )
    rollout, horizon = make_rollout(env, env_params, obs_dim, action_dim, args.num_evals, args.n_log)
    for _, cfg in configs:
        cfg.validate(horizon)
    print(f"checkpoints {ckpt_dir}, H={horizon}, max_timesteps={int(env_params.max_timesteps)}", flush=True)

    rngs = jax.random.split(jax.random.key(args.seed), len(level_paths))  # as eval_flow.main
    noise_key = jax.random.fold_in(jax.random.key(args.seed), 1_000_003)
    meta = dict(seed=args.seed, num_evals=args.num_evals, levels=level_names, step=args.step)
    for label, cfg in configs:
        path = out / f"{label}.npz"
        cfg_json = json.dumps({**dataclasses.asdict(cfg), **meta}, sort_keys=True)
        if path.exists() and str(np.load(path)["config"]) == cfg_json:
            print(f"skip {label} (done)", flush=True)
            continue
        t0 = time.perf_counter()
        compiled = rollout.lower(cfg, rngs, levels, state_dicts, noise_key).compile()
        compile_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        res, logs = jax.device_get(compiled(rngs, levels, state_dicts, noise_key))
        run_s = time.perf_counter() - t0
        peak = run_points.peak_gpu_mb()
        logs = {f"log_{k}": np.moveaxis(v, 0, 2) for k, v in logs.items()}  # (L, n_log, T, ...)
        np.savez_compressed(path, config=cfg_json, compile_s=compile_s, run_s=run_s, peak_gpu_mb=peak,
                            **{k: np.asarray(v) for k, v in res.items()}, **logs)
        print(f"{label:12s} d={cfg.d}: solved {100 * res['solved'].mean():.1f}%  infer/ep {res['n_infer'].mean():.1f}  "
              f"s {res['mean_s'].mean():.2f}  h* {res['mean_h'].mean():.2f}  compile {compile_s:.0f}s  run {run_s:.0f}s  "
              f"peak GPU {peak:.0f} MB", flush=True)
    write_report(out, labels, level_names)
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
