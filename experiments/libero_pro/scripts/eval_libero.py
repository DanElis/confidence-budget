#!/usr/bin/env python
"""Closed-loop evaluation of PI0.5 on LIBERO / LIBERO-PRO with simulated inference delay.

Execution modes (see closed_loop.py):
    --delay 0 --guidance none          synchronous chunking (robot waits for every chunk)
    --delay d --guidance none          naive asynchronous switching
    --delay d --guidance lerobot|exact RTC

Planners (see planner.py):
    --planner fixed --stride s                     constant replanning step (ordinary RTC)
    --planner budget --hstar-config F --lam 0.5    confidence budget with fixed lambda
    --planner budget --hstar-config F --lam entropy --lam-config F   lambda from entropy

K noise samples per inference (--n-samples K > 1, see selection.py): K unguided samples give the dispersion,
the agreement of the old plan with the new samples and the choice of the executed sample; with guidance the
chosen noise is then generated once more with RTC.
    --select coherent|medoid                       executed sample (default first = noise 0)
    --mask-mode rtc_x_conf|conf --conf-sigma S --conf-ref R   RTC mask weighted by exp(-D_old / (S * R))
    --signal agreement --planner adaptive|budget --hstar-rule kind=above,eta=Z,k=1,min_h=1
                                                   h* = first index where the old plan leaves the new samples' cloud

Suites: stock LIBERO names (``libero_object``) and LIBERO-PRO position levels
(``pos:x0.2``, ``pos:y0.1``, ``pos:all``). Results go to ``--out`` as one JSON
per suite (re-running skips finished suites) plus ``summary.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering on GPU servers

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "h1"))

from closed_loop import Chunk, LoopConfig, run_batch, sparc, stack_observations, switch_jumps  # noqa: E402
from hstar import HStarRule  # noqa: E402
from libero_pro_suites import (  # noqa: E402
    MAX_STEPS,
    POSITION_LEVELS,
    import_libero_quietly,
    register_position_suites,
    suite_name,
)
from planner import AdaptivePlanner, BudgetPlanner, EntropyLambda, FixedPlanner  # noqa: E402
from selection import (  # noqa: E402
    MASK_MODES,
    SELECTS,
    agreement,
    conf_weights,
    dispersion,
    expand_noise,
    joint_batch,
    select_index,
    split_joint,
)

log = logging.getLogger("eval_libero")
CONTROL_HZ = 20.0  # LIBERO / robosuite control frequency
ACTION_DIM = 7  # LIBERO: 6-DoF end-effector delta + gripper; the policy pads to max_action_dim


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="lerobot/pi05_libero_finetuned")
    p.add_argument("--suites", default="libero_object,pos:all",
                   help="comma-separated: LIBERO suite names and/or pos:<level> / pos:all for LIBERO-PRO")
    p.add_argument("--libero-pro-root", type=Path, default=ROOT / "third_party" / "LIBERO-PRO")
    p.add_argument("--task-ids", type=int, nargs="*", default=None)
    p.add_argument("--episodes-per-task", type=int, default=50, help="init states per task (LIBERO has 50)")
    p.add_argument("--batch-size", type=int, default=10, help="environments run in parallel")
    p.add_argument("--max-steps", type=int, default=None, help=f"default: {MAX_STEPS} for libero_object-based suites")
    p.add_argument("--delay", type=int, default=0, help="d, simulated latency in control steps")
    p.add_argument("--guidance", choices=["none", "lerobot", "exact"], default="none")
    p.add_argument("--schedule", default="exp", choices=["exp", "linear", "ones", "zeros"])
    p.add_argument("--max-guidance-weight", type=float, default=10.0)
    # planner
    p.add_argument("--planner", choices=["fixed", "budget", "adaptive"], default="fixed")
    p.add_argument("--stride", type=int, default=None, help="fixed: s; default: the checkpoint's n_action_steps")
    p.add_argument("--mask-end", type=int, default=None, help="fixed: RTC soft-mask end; default H - s")
    p.add_argument("--hstar-config", type=Path, default=None, help="budget: JSON with the h* rule (calibrate_hstar.py)")
    p.add_argument("--hstar-rule", default=None,
                   help='budget: inline rule, e.g. "kind=relative,q=0.7,k=5" (overrides --hstar-config)')
    p.add_argument("--lam", default="0.5", help="budget: lambda in [0, 1], or 'entropy'")
    p.add_argument("--lam-config", type=Path, default=None, help="budget, --lam entropy: JSON with mu, sigma, zone, z_max")
    p.add_argument("--r-min", type=int, default=5)
    p.add_argument("--s-min", type=int, default=5, help="adaptive: lower bound of s")
    p.add_argument("--s-max", type=int, default=None, help="adaptive: upper bound of s (default H - d)")
    p.add_argument("--mask-r", type=int, default=None, help="adaptive: mask end d + r (default H - s)")
    # K samples
    p.add_argument("--n-samples", type=int, default=1, help="K noise samples per inference")
    p.add_argument("--select", choices=SELECTS, default="first")
    p.add_argument("--mask-mode", choices=MASK_MODES, default="rtc")
    p.add_argument("--conf-sigma", type=float, default=1.0, help="confidence mask scale, in units of --conf-ref")
    p.add_argument("--conf-ref", type=float, default=1.0, help="dispersion scale (median dispersion of a reference run)")
    p.add_argument("--sample-call", choices=["separate", "joint"], default="separate",
                   help="with guidance: K unguided samples then one guided pass (separate, two calls), or K unguided "
                        "and K guided rows in one call (joint)")
    p.add_argument("--signal", choices=["entropy", "agreement"], default="entropy",
                   help="what the budget / adaptive planner's h* rule reads (agreement needs --n-samples > 1)")
    # outputs
    p.add_argument("--record-entropy", action="store_true", help="store cross-attention entropy of every chunk")
    p.add_argument("--save-actions", action="store_true", help="store executed actions (.npy)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def resolve_suites(spec: str, pro_root: Path) -> list[str]:
    names, levels = [], []
    for item in (x.strip() for x in spec.split(",") if x.strip()):
        if item == "pos:all":
            levels += list(POSITION_LEVELS)
        elif item.startswith("pos:"):
            levels.append(item[4:])
        else:
            names.append(item)
    if levels:
        if not pro_root.exists():
            raise SystemExit(f"LIBERO-PRO not found at {pro_root}; see README (git clone ... third_party/LIBERO-PRO)")
        levels = list(dict.fromkeys(levels))
        register_position_suites(pro_root, levels=tuple(levels))
        names += [suite_name(level) for level in levels]
    return names


def parse_rule(text: str) -> HStarRule:
    fields = {}
    for part in text.split(","):
        key, value = part.split("=")
        key = key.strip()
        fields[key] = value.strip() if key == "kind" else (int(value) if key in ("k", "min_h", "c") else float(value))
    return HStarRule.from_dict(fields)


def make_planner(args, horizon: int):
    if args.planner == "fixed":
        return FixedPlanner(horizon, args.stride, args.mask_end)
    if args.hstar_rule:
        rule = parse_rule(args.hstar_rule)
    elif args.hstar_config:
        rule = HStarRule.from_dict(json.loads(args.hstar_config.read_text())["rule"])
    else:
        raise SystemExit(f"--planner {args.planner} needs --hstar-config or --hstar-rule")
    if args.planner == "adaptive":
        return AdaptivePlanner(horizon, rule, s_min=getattr(args, "s_min", 5), s_max=getattr(args, "s_max", None),
                               mask_r=getattr(args, "mask_r", None))
    if args.lam == "entropy":
        if args.lam_config is None:
            raise SystemExit("--lam entropy needs --lam-config")
        cfg = json.loads(args.lam_config.read_text())
        cfg = cfg.get("lambda_entropy", cfg)
        lam_fn = EntropyLambda(**{k: cfg[k] for k in ("mu", "sigma", "zone", "z_max") if k in cfg})
        return BudgetPlanner(horizon, rule, r_min=args.r_min, lam=None, lam_fn=lam_fn)
    return BudgetPlanner(horizon, rule, r_min=args.r_min, lam=float(args.lam))


def check_sample_args(args) -> None:
    k = args.n_samples
    if k < 1:
        raise SystemExit("--n-samples must be >= 1")
    if k == 1 and (args.select != "first" or args.mask_mode != "rtc" or args.signal == "agreement"):
        raise SystemExit("--select, --mask-mode and --signal agreement need --n-samples > 1")
    if k > 1 and args.record_entropy:
        raise SystemExit("--record-entropy is not supported with --n-samples > 1")
    if k > 1 and args.planner != "fixed" and args.signal != "agreement":
        raise SystemExit("with --n-samples > 1 the budget / adaptive planner needs --signal agreement")
    if args.mask_mode != "rtc" and args.guidance == "none":
        raise SystemExit("--mask-mode needs guidance")


def load_policy(path: str, device: str):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import get_policy_class

    config = PreTrainedConfig.from_pretrained(path)
    config.compile_model = False
    policy = get_policy_class(config.type).from_pretrained(path, config=config)
    return policy.to(device).eval()


def episode_sparc(actions: np.ndarray) -> float:
    """SPARC of the commanded end-effector speed (norm of the translational action part)."""
    a = actions[np.isfinite(actions).all(axis=-1)]
    return sparc(np.linalg.norm(a[:, :3], axis=-1), CONTROL_HZ) if len(a) else float("nan")


def main() -> None:
    args = parse_args()
    check_sample_args(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    import_libero_quietly()
    from h1.attention import CrossAttentionRecorder
    from h1.guidance import BatchRTCProcessor, install_batch_rtc, make_rtc_config
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.utils import parse_camera_names, preprocess_observation
    from lerobot.policies import make_pre_post_processors
    from lerobot.utils.constants import ACTION
    from libero.libero import benchmark

    suites = resolve_suites(args.suites, args.libero_pro_root)
    policy = load_policy(args.policy, args.device)
    horizon = int(policy.config.chunk_size)
    if args.planner == "fixed" and args.stride is None:
        args.stride = int(policy.config.n_action_steps)
    planner = make_planner(args, horizon)
    max_steps = args.max_steps or MAX_STEPS
    cfg = LoopConfig(horizon, args.delay, args.guidance != "none", max_steps, args.seed)
    default_end = horizon - (args.stride or 10)
    rtc_config = make_rtc_config(args.schedule, args.max_guidance_weight, default_end)
    rtc_proc = install_batch_rtc(policy, args.guidance, rtc_config) if args.guidance != "none" else None
    weights_proc = BatchRTCProcessor(rtc_config, jacobian=False)  # RTC mask for scoring samples, also without guidance
    need_entropy = args.record_entropy or (planner.needs_entropy and args.signal == "entropy")
    recorder = CrossAttentionRecorder(policy.model).attach() if need_entropy else None

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.policy,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    env_cfg = LiberoEnvConfig(task=suites[0], episode_length=max_steps)
    env_pre, env_post = env_cfg.get_env_processors()
    camera_names = parse_camera_names(env_cfg.camera_name)
    noise_shape = (horizon, int(policy.config.max_action_dim))
    log.info("H=%d d=%d guidance=%s planner=%s max_steps=%d", horizon, args.delay, args.guidance, planner, max_steps)

    def to_batch(observations, task_description: str):
        obs = preprocess_observation(stack_observations(observations))
        obs["task"] = [task_description] * len(observations)
        return preprocessor(env_pre(obs))

    def infer_samples(observations, noise, leftover, delay, mask_end, leftover_disp, leftover_len,
                      task_description: str) -> Chunk:
        """K unguided samples -> dispersion, agreement, choice; with guidance the chosen noise is generated with RTC.

        --sample-call separate: K unguided samples, then one guided pass for the chosen noise (two calls).
        --sample-call joint: one call with K unguided and K guided rows (same noises); the chosen guided row is
        executed. Same choice as separate, one call of a 2x larger batch.
        """
        n, k, dev = len(observations), args.n_samples, args.device
        noise_k = expand_noise(noise, k)  # (n*K, H, A_max), row i*K is the episode's noise
        guided_rows, w = None, None
        if leftover is not None:
            rtc_w = weights_proc.prefix_weights(delay, mask_end, n, horizon, horizon).to(dev)
            w = conf_weights(args.mask_mode, rtc_w, None if leftover_disp is None else leftover_disp.to(dev),
                             delay, mask_end, args.conf_sigma * args.conf_ref)
        if leftover is not None and args.guidance != "none" and args.sample_call == "joint":
            noise2, left2, mask2, w2 = joint_batch(noise_k, leftover, mask_end, w, k)
            rtc_proc.weight_override = w2
            try:
                with torch.no_grad():
                    out = policy.predict_action_chunk(
                        to_batch([o for o in observations for _ in range(2 * k)], task_description),
                        noise=noise2.to(dev), inference_delay=delay, prev_chunk_left_over=left2.to(dev),
                        execution_horizon=mask2)
            finally:
                rtc_proc.weight_override = None
            samples, guided_rows = split_joint(out.float(), n, k)
        else:
            with torch.no_grad():  # no leftover: the RTC processor leaves these unguided
                samples = policy.predict_action_chunk(to_batch([o for o in observations for _ in range(k)],
                                                               task_description), noise=noise_k.to(dev))
            samples = samples.float().reshape(n, k, horizon, -1)
        real = samples[..., :ACTION_DIM]
        disp = dispersion(real)
        sel, z = torch.zeros(n, dtype=torch.long), torch.zeros(n, horizon)
        if leftover is None:  # first chunk of the episode
            model = samples[:, 0]
        else:
            left = leftover[..., :ACTION_DIM].to(dev)
            sel = select_index(args.select, real, left, w)
            z = agreement(real, left, leftover_len)
            if guided_rows is not None:
                model = guided_rows[torch.arange(n, device=guided_rows.device), sel.to(guided_rows.device)]
            elif args.guidance != "none":
                chosen = noise_k.view(n, k, *noise.shape[1:])[torch.arange(n), sel]
                rtc_proc.weight_override = w
                try:
                    with torch.no_grad():
                        model = policy.predict_action_chunk(
                            to_batch(observations, task_description), noise=chosen.to(dev), inference_delay=delay,
                            prev_chunk_left_over=leftover.to(dev), execution_horizon=mask_end)
                finally:
                    rtc_proc.weight_override = None
            else:
                model = samples[torch.arange(n, device=samples.device), sel.to(samples.device)]
        model = model.float()
        actions = env_post({ACTION: postprocessor(model.clone())})[ACTION]
        disp_np, z_np = disp.cpu().numpy(), z.numpy()
        return Chunk(model, torch.as_tensor(actions).float().cpu().numpy(),
                     signal=z_np if args.signal == "agreement" else None, dispersion=disp_np,
                     extras={"dispersion": disp_np, "agree": z_np, "sel": sel.numpy()})

    infer_time = {"s": 0.0, "n": 0}  # wall time per inference (policy calls + pre/post-processing), per suite

    def make_infer(task_description: str):
        def infer(observations, noise, leftover, delay, mask_end, leftover_disp=None, leftover_len=None) -> Chunk:
            started = time.perf_counter()
            chunk = infer_once(observations, noise, leftover, delay, mask_end, leftover_disp, leftover_len)
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            infer_time["s"] += time.perf_counter() - started
            infer_time["n"] += 1
            return chunk

        def infer_once(observations, noise, leftover, delay, mask_end, leftover_disp, leftover_len) -> Chunk:
            if args.n_samples > 1:
                return infer_samples(observations, noise, leftover, delay, mask_end, leftover_disp, leftover_len,
                                     task_description)
            batch = to_batch(observations, task_description)
            kwargs = {}
            if leftover is not None:
                kwargs = {"inference_delay": delay, "prev_chunk_left_over": leftover.to(args.device),
                          "execution_horizon": mask_end}
            ctx = recorder.record() if recorder else contextlib.nullcontext()
            with torch.no_grad(), ctx:
                pred = policy.predict_action_chunk(batch, noise=noise.to(args.device), **kwargs)
            actions = env_post({ACTION: postprocessor(pred.clone())})[ACTION]
            entropy = n_valid = None
            if recorder:
                rec = recorder.result()
                entropy, n_valid = rec.entropy[:, -1].numpy(), rec.n_valid.numpy()
            return Chunk(pred.float(), torch.as_tensor(actions).float().cpu().numpy(), entropy, n_valid)
        return infer

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {k: str(v) for k, v in vars(args).items()} | {"horizon": horizon, "max_steps": max_steps,
                                                          "planner_repr": repr(planner)}
    (args.out / "config.json").write_text(json.dumps(meta, indent=2))

    for name in suites:
        out_file = args.out / f"{name}.json"
        if out_file.exists():
            log.info("skip %s (done)", name)
            continue
        suite = benchmark.get_benchmark_dict()[name]()
        task_ids = args.task_ids if args.task_ids is not None else list(range(suite.n_tasks))
        episodes, batches, chunk_log, actions_all = [], [], [], []
        started = time.time()
        infer_time.update(s=0.0, n=0)
        for tid in task_ids:
            task = suite.get_task(tid)
            infer = make_infer(task.language)
            for start in range(0, args.episodes_per_task, args.batch_size):
                ids = list(range(start, min(start + args.batch_size, args.episodes_per_task)))
                envs = [LiberoEnv(task_suite=suite, task_id=tid, task_suite_name=name, camera_name=camera_names,
                                  init_states=True, episode_length=max_steps, episode_index=i, n_envs=1,
                                  **env_cfg.gym_kwargs) for i in ids]
                policy.reset()
                global_ids = [tid * 1000 + i for i in ids]  # noise depends on task and init state only
                try:
                    res = run_batch(envs, infer, cfg, planner, global_ids, noise_shape,
                                    keep_entropy=args.record_entropy or args.n_samples > 1)
                finally:
                    for env in envs:
                        env.close()
                jump_switch, jump_other = switch_jumps(res.actions, res.switch_steps)
                for b, i in enumerate(ids):
                    strides = [c["stride"] for c in res.chunks if c["env"] == b]
                    hs = [c["h_star"] for c in res.chunks if c["env"] == b and c["h_star"] is not None]
                    sels = [c["sel"] != 0 for c in res.chunks if c["env"] == b and "sel" in c and c["step"] > 0]
                    episodes.append({
                        "task_id": tid, "init_state": i, "success": bool(res.success[b]),
                        "length": int(res.length[b]), "n_inference": int(res.n_inference[b]),
                        "sparc": episode_sparc(res.actions[b]),
                        "mean_stride": float(np.mean(strides)) if strides else None,
                        "mean_h_star": float(np.mean(hs)) if hs else None,
                        "frac_sel": float(np.mean(sels)) if sels else None,
                    })
                batches.append({"task_id": tid, "init_states": ids, "jump_at_switch": jump_switch,
                                "jump_other": jump_other})
                for c in res.chunks:
                    chunk_log.append({"task_id": tid, "init_state": ids[c["env"]]} | {k: v for k, v in c.items() if k != "env"})
                if args.save_actions:
                    actions_all.append(res.actions)
                done = [e for e in episodes if e["task_id"] == tid]
                log.info("%s task %d: %d/%d success after %d episodes (%.0f s)", name, tid,
                         sum(e["success"] for e in done), len(done), len(done), time.time() - started)
        result = summarize(name, episodes, batches, task_ids)
        result["seconds"] = time.time() - started
        result["infer_seconds_per_call"] = infer_time["s"] / max(infer_time["n"], 1)
        out_file.write_text(json.dumps(result, indent=1))
        (args.out / f"{name}_chunks.json").write_text(json.dumps(chunk_log))
        if actions_all:
            np.save(args.out / f"{name}_actions.npy", np.concatenate(actions_all))
        log.info("%s: success %.1f%% over %d episodes", name, 100 * result["success_rate"], len(episodes))

    write_summary(args.out, suites)
    if recorder:
        recorder.detach()


def summarize(name: str, episodes: list[dict], batches: list[dict], task_ids) -> dict:
    mean = lambda xs: float(np.nanmean(xs)) if len(xs) and np.isfinite(xs).any() else float("nan")  # noqa: E731
    return {
        "suite": name,
        "success_rate": mean([e["success"] for e in episodes]),
        "by_task": {tid: mean([e["success"] for e in episodes if e["task_id"] == tid]) for tid in task_ids},
        "n_episodes": len(episodes),
        "mean_inference": mean([e["n_inference"] for e in episodes]),
        "mean_sparc": mean([e["sparc"] for e in episodes]),
        "jump_at_switch": mean([b["jump_at_switch"] for b in batches]),
        "jump_other": mean([b["jump_other"] for b in batches]),
        "mean_stride": mean([e["mean_stride"] for e in episodes if e["mean_stride"] is not None]),
        "mean_h_star": mean([e["mean_h_star"] for e in episodes if e["mean_h_star"] is not None]),
        "episodes": episodes,
        "batches": batches,
    }


def write_summary(out: Path, suites: list[str]) -> None:
    rows = ["| suite | success, % | episodes | inferences / ep. | SPARC | |Δa| switch / other | mean s | mean h* |",
            "|---|---|---|---|---|---|---|---|"]
    for name in suites:
        f = out / f"{name}.json"
        if f.exists():
            r = json.loads(f.read_text())
            rows.append(f"| {name} | {100 * r['success_rate']:.1f} | {r['n_episodes']} | {r['mean_inference']:.1f} "
                        f"| {r.get('mean_sparc', float('nan')):.3f} | {r['jump_at_switch']:.4f} / {r['jump_other']:.4f} "
                        f"| {r.get('mean_stride', float('nan')):.1f} | {r.get('mean_h_star', float('nan')):.1f} |")
    (out / "summary.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
    # Results are on disk. Skip interpreter teardown: EGL / MuJoCo contexts can hang there and keep
    # the GPU memory of a finished run allocated.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
