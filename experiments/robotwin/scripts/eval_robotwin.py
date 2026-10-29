#!/usr/bin/env python
"""Closed-loop evaluation of PI0.5 on RoboTwin 2.0 (Easy, demo_clean) with simulated inference delay.

Same execution modes and planners as experiments/libero_pro/scripts/eval_libero.py:
    --delay 0 --guidance none          synchronous chunking
    --delay d --guidance none          naive asynchronous switching
    --delay d --guidance lerobot|exact RTC;  --planner budget ...  confidence budget

The simulator runs in RoboTwin's own Python (see setup_robotwin.sh), one server process per
parallel environment (remote_env.py); the policy runs here. Episode i of a task uses seed
100000 * (1 + --seed) + 1000 * i (unstable scenes are skipped deterministically), so every
method sees the same scenes and the same flow-matching noise; results can be paired.

Output: one JSON per task in --out (same format as eval_libero, so compare.py works with
--suites <task,...>), <task>_chunks.json, summary.md. Finished tasks are skipped.

Example:
    uv run --project ../libero_pro python scripts/eval_robotwin.py --suites lift_pot,stamp_seal \\
        --episodes-per-task 25 --stride 25 --delay 6 --guidance lerobot --out runs/rtc_s25_d6
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

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
LIBERO_PRO = HERE.parent / "libero_pro"
for p in (HERE, LIBERO_PRO, LIBERO_PRO / "scripts", HERE.parent / "h1"):
    sys.path.insert(0, str(p))

import eval_libero as EL  # noqa: E402  (planner, policy loading, summaries)
from closed_loop import Chunk, LoopConfig, run_batch, sparc, stack_observations, switch_jumps  # noqa: E402
from remote_env import RemoteEnv, SimulatorError, resolve_observations  # noqa: E402
from selection import (  # noqa: E402
    MASK_MODES,
    SELECTS,
    agreement,
    conf_weights,
    dispersion,
    joint_batch,
    select_index,
    split_joint,
)

log = logging.getLogger("eval_robotwin")
BATCH_ATTEMPTS = 3
CONTROL_HZ = 25.0  # RoboTwin control frequency
ARM_JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
ACTION_DIM = 14  # 14-d action without the two grippers
CAMERAS = ("head_camera", "left_camera", "right_camera")
# RoboTwin camera names -> feature names of SidneyXie/pi05_robotwin (its README's --rename_map)
RENAME = {"observation.images.head_camera": "observation.images.cam_high",
          "observation.images.left_camera": "observation.images.cam_left_wrist",
          "observation.images.right_camera": "observation.images.cam_right_wrist"}

# 8 tasks with mid success for the checkpoint (41-60% on Easy; README of SidneyXie/pi05_robotwin),
# all with a 400-500 step limit: room to get better or worse, and cheap episodes.
TASKS_DEFAULT = ("lift_pot,move_can_pot,place_a2b_left,place_bread_skillet,place_fan,place_mouse_pad,"
                 "stamp_seal,turn_switch")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="SidneyXie/pi05_robotwin")
    p.add_argument("--suites", default=TASKS_DEFAULT, help="comma-separated RoboTwin task names")
    p.add_argument("--episodes-per-task", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=8, help="simulator processes run in parallel")
    p.add_argument("--robotwin-root", type=Path, default=HERE / "third_party" / "RoboTwin")
    p.add_argument("--robotwin-python", default=None, help="default: <robotwin-root>/.venv/bin/python")
    p.add_argument("--max-steps", type=int, default=None, help="default: RoboTwin's per-task step limit")
    p.add_argument("--delay", type=int, default=0)
    p.add_argument("--guidance", choices=["none", "lerobot", "exact"], default="none")
    p.add_argument("--schedule", default="exp", choices=["exp", "linear", "ones", "zeros"])
    p.add_argument("--max-guidance-weight", type=float, default=10.0)
    p.add_argument("--planner", choices=["fixed", "budget", "adaptive"], default="fixed",
                   help="adaptive: s = h* - d from the --signal (AAC-style adaptive chunking)")
    p.add_argument("--s-min", type=int, default=5, help="adaptive: smallest step")
    p.add_argument("--stride", type=int, default=None, help="fixed: s; default: the checkpoint's n_action_steps")
    p.add_argument("--mask-end", type=int, default=None)
    p.add_argument("--hstar-config", type=Path, default=None)
    p.add_argument("--hstar-rule", default=None)
    p.add_argument("--lam", default="0.5")
    p.add_argument("--lam-config", type=Path, default=None)
    p.add_argument("--r-min", type=int, default=5)
    p.add_argument("--record-entropy", action="store_true", help="store entropy and the extras below per chunk")
    p.add_argument("--n-samples", type=int, default=1,
                   help="K > 1: generate K chunks per observation (different noise) and store their dispersion per "
                        "action index; the robot executes sample 0, which uses the same noise as a K = 1 run")
    p.add_argument("--signal", choices=["entropy", "dispersion", "agreement"], default="entropy",
                   help="what the budget / adaptive planner's h* rule reads (dispersion, agreement need --n-samples > 1)")
    # current idea (libero_pro/selection.py): K unguided + K guided rows in one call with a shared prefix
    p.add_argument("--select", choices=SELECTS, default="first", help="executed sample: coherent = closest to the old plan")
    p.add_argument("--mask-mode", choices=MASK_MODES, default="rtc", help="rtc_x_conf: RTC mask * exp(-D_old / sigma)")
    p.add_argument("--conf-sigma", type=float, default=1.0)
    p.add_argument("--conf-ref", type=float, default=1.0)
    p.add_argument("--shared-prefix", action="store_true", help="encode each observation's prefix once for the K samples")
    p.add_argument("--s-max", type=int, default=None, help="adaptive: largest step (default H - d)")
    p.add_argument("--mask-r", type=int, default=None, help="adaptive: mask end d + r (default H - s)")
    p.add_argument("--record-plans", action="store_true", help="store each chunk's actions (policy space) for drift analysis")
    p.add_argument("--save-actions", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fake-sim", action="store_true", help="synthetic simulator (tests the plumbing)")
    p.add_argument("--fake-policy", action="store_true",
                   help="hold the current joint state instead of running the model (tests the simulator)")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def expand_noise(noise: torch.Tensor, k: int) -> torch.Tensor:
    """(B, H, A) -> (B*K, H, A): sample 0 of each row is the given noise, the others are drawn
    deterministically from it (same scenes, same extra samples in every run)."""
    out = []
    for row in noise:
        out.append(row)
        seed = int(abs(float(row.flatten()[:16].double().sum())) * 1e9) % (2**62)
        g = torch.Generator().manual_seed(seed)
        out += [torch.randn(row.shape, generator=g, dtype=row.dtype) for _ in range(k - 1)]
    return torch.stack(out)


def joint_mode(args) -> bool:
    """The current idea's K-sample path (one call, unguided + guided rows) instead of the old one (K guided rows,
    sample 0 executed, dispersion only logged; kept for run_samples.sh)."""
    return args.n_samples > 1 and (args.select != "first" or args.mask_mode != "rtc" or args.signal == "agreement"
                                   or args.shared_prefix)


def check_args(args) -> None:
    if args.signal in ("dispersion", "agreement") and args.n_samples < 2:
        raise SystemExit(f"--signal {args.signal} needs --n-samples >= 2")
    if joint_mode(args) and args.guidance == "none":
        raise SystemExit("--select / --mask-mode / --signal agreement / --shared-prefix need guidance (they use the old plan)")
    if args.select != "first" and args.n_samples < 2:
        raise SystemExit("--select needs --n-samples >= 2")


def episode_sparc(actions: np.ndarray) -> float:
    """SPARC of the commanded joint speed of both arms (grippers excluded)."""
    a = actions[np.isfinite(actions).all(axis=-1)]
    if len(a) < 5:
        return float("nan")
    return sparc(np.linalg.norm(np.diff(a[:, ARM_JOINTS], axis=0), axis=-1) * CONTROL_HZ, CONTROL_HZ)


def to_policy_batch(observations: list, instruction: str) -> dict:
    from lerobot.envs.utils import preprocess_observation

    obs = preprocess_observation(stack_observations(observations))
    obs = {RENAME.get(k, k): v for k, v in obs.items()}
    obs["task"] = [instruction] * len(observations)
    return obs


class Simulators:
    """Pool of RemoteEnv servers, restarted when one dies."""

    def __init__(self, n: int, args, log_dir: Path):
        self.n, self.args, self.log_dir = n, args, log_dir
        self.python = args.robotwin_python or str(args.robotwin_root / ".venv" / "bin" / "python")
        if args.fake_sim:
            self.python = sys.executable
        self.envs = [self._start(i) for i in range(n)]

    def _start(self, i: int) -> RemoteEnv:
        return RemoteEnv(self.python, None if self.args.fake_sim else self.args.robotwin_root.resolve(),
                         log_path=self.log_dir / f"sim{i}.log", fake=self.args.fake_sim, cameras=CAMERAS)

    def heal(self, restart_all: bool = False) -> None:
        """Re-sync live servers (read replies still owed) and restart dead ones (or all)."""
        for i, env in enumerate(self.envs):
            env.drain()
            if restart_all or not env.alive:
                log.warning("restarting simulator %d", i)
                env.close()
                self.envs[i] = self._start(i)

    def close(self) -> None:
        for env in self.envs:
            env.close()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    tasks = [t.strip() for t in args.suites.split(",") if t.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sim_logs").mkdir(exist_ok=True)

    if args.fake_policy:
        policy, horizon, n_action_steps, noise_shape = None, 50, 50, (50, 32)
    else:
        from h1.guidance import install_batch_rtc, make_rtc_config
        from lerobot.policies import make_pre_post_processors

        policy = EL.load_policy(args.policy, args.device)
        horizon, n_action_steps = int(policy.config.chunk_size), int(policy.config.n_action_steps)
        noise_shape = (horizon, int(policy.config.max_action_dim))
    if args.planner == "fixed" and args.stride is None:
        args.stride = n_action_steps
    check_args(args)
    planner = EL.make_planner(args, horizon)
    recorder = None
    rtc_proc = weights_proc = None
    if policy is not None:
        if args.guidance != "none":
            from h1.guidance import BatchRTCProcessor

            default_end = horizon - (args.stride or 10)
            rtc_config = make_rtc_config(args.schedule, args.max_guidance_weight, default_end)
            rtc_proc = install_batch_rtc(policy, args.guidance, rtc_config)
            weights_proc = BatchRTCProcessor(rtc_config, jacobian=False)
        if args.record_entropy or (planner.needs_entropy and args.signal == "entropy"):
            from h1.attention import CrossAttentionRecorder

            recorder = CrossAttentionRecorder(policy.model).attach()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config, pretrained_path=args.policy,
            preprocessor_overrides={"device_processor": {"device": args.device}})

    meta = {k: str(v) for k, v in vars(args).items()} | {"horizon": horizon, "planner_repr": repr(planner)}
    (args.out / "config.json").write_text(json.dumps(meta, indent=2))
    log.info("H=%d d=%d guidance=%s planner=%s tasks=%s", horizon, args.delay, args.guidance, planner, tasks)

    def infer_joint(observations, noise, leftover, delay, mask_end, leftover_disp, leftover_len, instruction) -> Chunk:
        """K unguided + K guided rows in one call (shared prefix optional): dispersion, plan agreement z (the
        planner's signal), choice of the executed sample, confidence-weighted RTC mask."""
        from shared_prefix import predict_chunk_shared

        n, k, dev = len(observations), args.n_samples, args.device

        def predict_rows(repeats, noise_rows, **kw):
            if args.shared_prefix:
                batch = preprocessor(to_policy_batch(observations, instruction))
                return predict_chunk_shared(policy, batch, repeats, noise=noise_rows.to(dev), **kw)
            batch = preprocessor(to_policy_batch([o for o in observations for _ in range(repeats)], instruction))
            return policy.predict_action_chunk(batch, noise=noise_rows.to(dev), **kw)

        noise_k = expand_noise(noise, k)
        sel, z = torch.zeros(n, dtype=torch.long), torch.zeros(n, horizon)
        with torch.no_grad():
            if leftover is None:  # first chunk of an episode
                samples = predict_rows(k, noise_k).float().reshape(n, k, horizon, -1)
                real = samples[..., :ACTION_DIM]
                model = samples[:, 0]
            else:
                rtc_w = weights_proc.prefix_weights(delay, mask_end, n, horizon, horizon).to(dev)
                w = conf_weights(args.mask_mode, rtc_w, None if leftover_disp is None else leftover_disp.to(dev),
                                 delay, mask_end, args.conf_sigma * args.conf_ref)
                noise2, left2, mask2, w2 = joint_batch(noise_k, leftover, mask_end, w, k)
                rtc_proc.weight_override = w2.to(dev)
                try:
                    out = predict_rows(2 * k, noise2, inference_delay=delay, prev_chunk_left_over=left2.to(dev),
                                       execution_horizon=mask2)
                finally:
                    rtc_proc.weight_override = None
                unguided, guided = split_joint(out.float(), n, k)
                real = unguided[..., :ACTION_DIM]
                left = leftover[..., :ACTION_DIM].to(dev)
                sel = select_index(args.select, real, left, w)
                z = agreement(real, left, leftover_len)
                model = guided[torch.arange(n, device=guided.device), sel.to(guided.device)]
        disp = dispersion(real).cpu().numpy()
        z_np = z.numpy()
        actions = postprocessor(model.clone())
        extras = {"dispersion": disp, "agree": z_np, "sel": sel.numpy()}
        if args.record_plans:
            extras["plan"] = model[:, :, :ACTION_DIM].cpu().numpy()
        return Chunk(model, torch.as_tensor(actions).float().cpu().numpy(),
                     signal=z_np if args.signal == "agreement" else (disp if args.signal == "dispersion" else None),
                     dispersion=disp, extras=extras)

    def make_infer(instruction: str):
        def infer(observations, noise, leftover, delay, mask_end, leftover_disp=None, leftover_len=None) -> Chunk:
            observations = resolve_observations(observations)  # render the due environments together
            if policy is not None and joint_mode(args):
                return infer_joint(observations, noise, leftover, delay, mask_end, leftover_disp, leftover_len,
                                   instruction)
            if policy is None:  # hold position: tests the simulator side only
                state = np.stack([np.asarray(o["agent_pos"], np.float32) for o in observations])
                acts = np.repeat(state[:, None], horizon, axis=1)
                n = len(observations)  # flat dummy entropy so entropy-based planners run too
                extras = {"dispersion": np.tile(np.linspace(0.05, 0.5, horizon), (n, 1))} if args.n_samples > 1 else {}
                if args.record_plans:
                    extras["plan"] = acts[..., :ACTION_DIM]
                sig = extras.get("dispersion") if args.signal in ("dispersion", "agreement") else None
                return Chunk(torch.as_tensor(acts), acts, np.full((n, horizon), 3.0), np.full(n, 100),
                             signal=sig, extras=extras, dispersion=extras.get("dispersion") if joint_mode(args) else None)
            k = args.n_samples
            n = len(observations)
            if k > 1:  # K chunks per observation with different noise; sample 0 keeps the episode's noise
                observations = [o for o in observations for _ in range(k)]
                noise = expand_noise(noise, k)
                if leftover is not None:
                    leftover, mask_end = leftover.repeat_interleave(k, 0), mask_end.repeat_interleave(k, 0)
            batch = preprocessor(to_policy_batch(observations, instruction))
            kwargs = {}
            if leftover is not None:
                kwargs = {"inference_delay": delay, "prev_chunk_left_over": leftover.to(args.device),
                          "execution_horizon": mask_end}
            ctx = recorder.record() if recorder else contextlib.nullcontext()
            with torch.no_grad(), ctx:
                pred = policy.predict_action_chunk(batch, noise=noise.to(args.device), **kwargs)
            entropy = n_valid = None
            if recorder:
                rec = recorder.result()
                entropy, n_valid = rec.entropy[:, -1].numpy()[::k], rec.n_valid.numpy()[::k]
            pred = pred.float()
            extras, signal = {}, None
            if k > 1:
                samples = pred[:, :, :ACTION_DIM].reshape(n, k, horizon, ACTION_DIM)
                # dispersion: std across samples in the normalised action space, averaged over the 14 joints
                dispersion = samples.std(dim=1, unbiased=True).mean(dim=-1).cpu().numpy()
                extras["dispersion"] = dispersion
                if args.signal == "dispersion":
                    signal = dispersion
                pred = pred[::k]
            if args.record_plans:
                extras["plan"] = pred[:, :, :ACTION_DIM].cpu().numpy()
            actions = postprocessor(pred.clone())
            return Chunk(pred, torch.as_tensor(actions).float().cpu().numpy(), entropy, n_valid,
                         signal=signal, extras=extras)
        return infer

    sims = Simulators(min(args.batch_size, args.episodes_per_task), args, args.out / "sim_logs")
    try:
        for task_index, task in enumerate(tasks):
            out_file = args.out / f"{task}.json"
            if out_file.exists():
                log.info("skip %s (done)", task)
                continue
            episodes, batches, chunk_log, actions_all = [], [], [], []
            started = time.time()
            for start in range(0, args.episodes_per_task, sims.n):
                ids = list(range(start, min(start + sims.n, args.episodes_per_task)))
                res = None
                batch_started = time.time()
                failure = None
                for attempt in range(BATCH_ATTEMPTS):  # retry the batch with re-synced / fresh simulators
                    envs = sims.envs[: len(ids)]
                    try:
                        for env, i in zip(envs, ids, strict=True):
                            env.set_episode(task, 100_000 * (1 + args.seed) + 1000 * i)
                        max_steps = args.max_steps or max(env.step_lim for env in envs)
                        cfg = LoopConfig(horizon, args.delay, args.guidance != "none", max_steps, args.seed)
                        if policy is not None:
                            policy.reset()
                        instruction = task.replace("_", " ")
                        res = run_batch(envs, make_infer(instruction), cfg, planner,
                                        [task_index * 1000 + i for i in ids], noise_shape,
                                        keep_entropy=args.record_entropy or args.record_plans or args.n_samples > 1)
                        errors = [e.episode_error for e in envs if e.episode_error]
                        if not errors:
                            break
                        raise SimulatorError(f"{len(errors)} episode(s) hit a simulator error: {errors[0]}")
                    except (SimulatorError, EOFError, OSError) as e:
                        res, failure = None, str(e)
                        log.warning("%s episodes %s: simulator failure (%s), attempt %d/%d", task, ids, e,
                                    attempt + 1, BATCH_ATTEMPTS)
                        sims.heal(restart_all=attempt > 0)
                        time.sleep(30 * attempt)
                if res is None:
                    # Never record simulator failures as policy failures: give up on this task without a
                    # result file, so that a re-run evaluates it again. Out of GPU memory in the renderer
                    # is reported in the wording lib.sh retries on (after waiting for free memory).
                    oom = any(s in failure.lower() for s in ("cannot create buffer", "out of memory", "vk_error"))
                    raise SystemExit(f"{task}: simulator keeps failing ({failure})"
                                     + ("; CUDA out of memory in the simulator renderer" if oom else ""))
                jump_switch, jump_other = switch_jumps(res.actions, res.switch_steps)
                for b, i in enumerate(ids):
                    strides = [c["stride"] for c in res.chunks if c["env"] == b]
                    hs = [c["h_star"] for c in res.chunks if c["env"] == b and c["h_star"] is not None]
                    episodes.append({
                        "task_id": 0, "init_state": i, "seed": envs[b].used_seed, "success": bool(res.success[b]),
                        "error": False, "length": int(res.length[b]), "n_inference": int(res.n_inference[b]),
                        "sparc": episode_sparc(res.actions[b]),
                        "mean_stride": float(np.mean(strides)) if strides else None,
                        "mean_h_star": float(np.mean(hs)) if hs else None,
                    })
                batches.append({"task_id": 0, "init_states": ids, "jump_at_switch": jump_switch,
                                "jump_other": jump_other})
                for c in res.chunks:
                    chunk_log.append({"task_id": 0, "init_state": ids[c["env"]]} | {k: v for k, v in c.items() if k != "env"})
                if args.save_actions:
                    actions_all.append(res.actions)
                el = time.time() - started
                log.info("%s: %d/%d success after %d episodes (%.0f s, %.0f episodes/h, sim %.0f%% of time)",
                         task, sum(e["success"] for e in episodes), len(episodes), len(episodes), el,
                         3600 * len(episodes) / max(el, 1e-6),
                         100 * sum(e.sim_seconds for e in envs) / max((time.time() - batch_started) * len(envs), 1e-6))
                for env in envs:
                    env.sim_seconds = 0.0
            result = EL.summarize(task, episodes, batches, [0])
            result["seconds"] = time.time() - started
            result["n_errors"] = sum(bool(e.get("error")) for e in episodes)
            out_file.write_text(json.dumps(result, indent=1))
            (args.out / f"{task}_chunks.json").write_text(json.dumps(chunk_log))
            if actions_all:
                width = max(a.shape[1] for a in actions_all)
                padded = [np.pad(a, ((0, 0), (0, width - a.shape[1]), (0, 0)), constant_values=np.nan) for a in actions_all]
                np.save(args.out / f"{task}_actions.npy", np.concatenate(padded))
            log.info("%s: success %.1f%% over %d episodes", task, 100 * result["success_rate"], len(episodes))
    finally:
        sims.close()
    EL.write_summary(args.out, tasks)
    if recorder:
        recorder.detach()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
