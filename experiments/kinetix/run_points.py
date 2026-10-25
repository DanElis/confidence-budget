"""Naive and RTC on the public Kinetix BC checkpoints at chosen (delay, execute_horizon) points.

Reuses eval_flow.eval from real-time-chunking-kinetix@9296f31 unchanged; only the sweep is replaced:
eval_flow.main runs every (d, s) pair for four methods, here only s = max(d, 1) for naive and RTC.
Setup (levels, env, policy loading, sharding, per-level rngs) is copied from eval_flow.main, so for a given
seed the numbers match what eval_flow.main would report at the same points.

Per point: compile and run time, peak GPU memory, solve rate / return / length per level.
Results are appended to <out>/results.csv after every point (re-runs skip finished points);
<out>/report.md is rewritten at the end.

Run through run.sh (sets XLA_PYTHON_CLIENT_PREALLOCATE=false and the RTC venv).
"""

import argparse
import collections
import dataclasses
import functools
import inspect
import os
import pathlib
import pickle
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

HERE = pathlib.Path(__file__).resolve().parent
RTC_DIR = HERE / "third_party" / "real-time-chunking-kinetix"
sys.path.insert(0, str(RTC_DIR / "src"))

import flax.nnx as nnx  # noqa: E402
import jax  # noqa: E402
from jax.experimental import shard_map  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import kinetix.environment.env as kenv  # noqa: E402
import kinetix.environment.env_state as kenv_state  # noqa: E402
import pandas as pd  # noqa: E402

import eval_flow  # noqa: E402
import model as _model  # noqa: E402
import train_expert  # noqa: E402

DEFAULT_LEVELS = tuple(inspect.signature(eval_flow.main).parameters["level_paths"].default)
METHODS = {
    "naive": eval_flow.NaiveMethodConfig(),
    "rtc": eval_flow.RealtimeMethodConfig(),  # exp schedule, max guidance weight 5 (eval_flow defaults)
}
# Published mean solve rates (%) over the 12 levels, docs/STATUS.md section 6.
PUBLISHED = {
    "naive (REMAC)": [82.8, 70.2, 63.9, 52.5, 45.1],
    "RTC (REMAC)": [81.7, 77.8, 73.3, 66.5, 58.8],
    "naive (A2C2, s = d)": [90.8, 78.1, 75.7, 59.3, 51.2],
    "RTC (A2C2, s = d)": [90.9, 88.2, 84.1, 74.8, 62.9],
}
KEY_COLS = ["method", "delay", "execute_horizon", "num_evals", "seed", "step"]


def level_name(path: str) -> str:
    return path.replace("/", "_").replace(".json", "")


def build(run_path: pathlib.Path, level_paths, step: int):
    """Copy of the setup part of eval_flow.main."""
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)
    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)

    log_dirs = sorted((p for p in run_path.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    ckpt_dir = log_dirs[step]
    state_dicts = []
    for path in level_paths:
        with (ckpt_dir / "policies" / f"{level_name(path)}.pkl").open("rb") as f:
            state_dicts.append(pickle.load(f))
    state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *state_dicts))

    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)[
        0
    ].shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    pspec = jax.sharding.PartitionSpec("x")
    sharding = jax.sharding.NamedSharding(mesh, pspec)

    @functools.partial(jax.jit, static_argnums=(0,), in_shardings=sharding, out_shardings=sharding)
    @functools.partial(shard_map.shard_map, mesh=mesh, in_specs=(None, pspec, pspec, pspec), out_specs=pspec)
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
    def _eval(config: eval_flow.EvalConfig, rng: jax.Array, level: kenv_state.EnvState, state_dict):
        policy = _model.FlowPolicy(obs_dim=obs_dim, action_dim=action_dim, config=config.model, rngs=nnx.Rngs(rng))
        graphdef, state = nnx.split(policy)
        state.replace_by_pure_dict(state_dict)
        policy = nnx.merge(graphdef, state)
        eval_info, _ = eval_flow.eval(config, env, rng, level, policy, env_params, static_env_params)
        return eval_info

    return _eval, levels, state_dicts, ckpt_dir


def peak_gpu_mb() -> float:
    stats = jax.devices()[0].memory_stats() or {}
    return stats.get("peak_bytes_in_use", float("nan")) / 2**20


def write_report(df: pd.DataFrame, out: pathlib.Path, ckpt_dir: pathlib.Path):
    df = df.copy()
    df["solved"] = 100 * df["returned_episode_solved"]
    mean = df.groupby(["method", "delay", "execute_horizon", "num_evals"], as_index=False).agg(
        solved=("solved", "mean"),
        n_levels=("level", "nunique"),
        compile_s=("compile_s", "first"),
        run_s=("run_s", "first"),
        peak_gpu_mb=("peak_gpu_mb", "first"),
    )
    lines = [
        "# Kinetix: naive and RTC on public BC checkpoints",
        "",
        f"Checkpoints: `{ckpt_dir.parent.name}/{ckpt_dir.name}`. "
        "s = max(d, 1); RTC = exact guidance, exp schedule, β = 5; 5 flow steps; solve rate of the first episode.",
        "",
        "## Mean solve rate over levels, %",
        "",
    ]
    for n, g in mean.groupby("num_evals"):
        delays = sorted(g["delay"].unique())
        lines += [f"num_evals = {n} per level, levels = {g['n_levels'].max()}", ""]
        lines += ["| | " + " | ".join(f"d = {d}" for d in delays) + " |", "|---" * (len(delays) + 1) + "|"]
        for m in METHODS:
            row = g[g["method"] == m].set_index("delay")["solved"]
            lines.append(f"| {m} (ours) | " + " | ".join(f"{row[d]:.1f}" if d in row else "—" for d in delays) + " |")
        for name, vals in PUBLISHED.items():
            lines.append(f"| {name} | " + " | ".join(f"{vals[d]:.1f}" for d in delays) + " |")
        lines.append("")

    lines += ["## Time and memory per point", "", "| method | d | s | num_evals | compile, s | run, s | peak GPU, MB |"]
    lines.append("|---|---|---|---|---|---|---|")
    for r in mean.sort_values(["num_evals", "method", "delay"]).itertuples():
        lines.append(
            f"| {r.method} | {r.delay} | {r.execute_horizon} | {r.num_evals} | {r.compile_s:.1f} | {r.run_s:.1f} "
            f"| {r.peak_gpu_mb:.0f} |"
        )

    lines += ["", "## Solve rate per level, %", ""]
    piv = df.pivot_table(index="level", columns=["num_evals", "method", "delay"], values="solved")
    cols = list(piv.columns)
    lines.append("| level | " + " | ".join(f"{m} d={d} (n={n})" for n, m, d in cols) + " |")
    lines.append("|---" * (len(cols) + 1) + "|")
    for lvl, row in piv.iterrows():
        lines.append(f"| {level_name(lvl).removeprefix('worlds_l_')} | " + " | ".join(f"{v:.1f}" for v in row) + " |")
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-path", default=str(HERE / "assets" / "bc"), help="dir with <epoch>/policies/*.pkl")
    ap.add_argument("--step", type=int, default=-1, help="epoch index as in eval_flow (-1 = last)")
    ap.add_argument("--out", default=str(HERE / "runs" / "reproduce"))
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--delays", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--num-evals", type=int, default=2048, help="environments per level")
    ap.add_argument("--levels", nargs="+", default=None, help="subset, e.g. grasp_easy catapult (default: all 12)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_path = pathlib.Path(args.run_path).resolve()
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    level_paths = DEFAULT_LEVELS
    if args.levels:
        level_paths = tuple(p for p in DEFAULT_LEVELS if level_name(p).removeprefix("worlds_l_") in args.levels)
        assert len(level_paths) == len(args.levels), f"unknown level in {args.levels}"
    assert len(level_paths) % jax.local_device_count() == 0, "device count must divide the number of levels"

    os.chdir(RTC_DIR)  # level paths are relative to the repo root
    print(f"jax {jax.__version__} {jax.devices()}; {len(level_paths)} levels, num_evals={args.num_evals}", flush=True)
    t0 = time.perf_counter()
    _eval, levels, state_dicts, ckpt_dir = build(run_path, level_paths, args.step)
    print(f"setup {time.perf_counter() - t0:.1f}s, checkpoints {ckpt_dir}", flush=True)

    csv = out / "results.csv"
    done = set()
    if csv.exists():
        prev = pd.read_csv(csv)
        done = set(map(tuple, prev[KEY_COLS].drop_duplicates().itertuples(index=False)))

    # same per-level rngs for every point, as in eval_flow.main
    rngs = jax.random.split(jax.random.key(args.seed), len(level_paths))
    for d in args.delays:
        s = max(d, 1)
        for m in args.methods:
            key = (m, d, s, args.num_evals, args.seed, args.step)
            if key in done:
                print(f"skip {key}", flush=True)
                continue
            c = eval_flow.EvalConfig(
                num_evals=args.num_evals, inference_delay=d, execute_horizon=s, method=METHODS[m], step=args.step
            )
            t0 = time.perf_counter()
            compiled = _eval.lower(c, rngs, levels, state_dicts).compile()
            compile_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            res = jax.device_get(compiled(rngs, levels, state_dicts))
            run_s = time.perf_counter() - t0
            rows = collections.defaultdict(list)
            for i, path in enumerate(level_paths):
                for k, v in zip(KEY_COLS, key):
                    rows[k].append(v)
                rows["level"].append(path)
                for k, v in res.items():
                    rows[k].append(float(v[i]))
                rows["compile_s"].append(compile_s)
                rows["run_s"].append(run_s)
                rows["peak_gpu_mb"].append(peak_gpu_mb())
            pd.DataFrame(rows).to_csv(csv, mode="a", header=not csv.exists(), index=False)
            sr = 100 * sum(rows["returned_episode_solved"]) / len(level_paths)
            print(
                f"{m:5s} d={d} s={s}: solved {sr:.1f}%  compile {compile_s:.1f}s  run {run_s:.1f}s  "
                f"peak GPU {peak_gpu_mb():.0f} MB",
                flush=True,
            )

    write_report(pd.read_csv(csv), out, ckpt_dir)
    print(f"wrote {csv} and {out / 'report.md'}")


if __name__ == "__main__":
    main()
