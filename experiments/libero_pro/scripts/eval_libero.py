#!/usr/bin/env python
"""Closed-loop evaluation of PI0.5 on LIBERO / LIBERO-PRO with simulated inference delay.

Modes (see closed_loop.py):
    --delay 0 --guidance none          synchronous chunking (robot waits for every chunk)
    --delay d --guidance none          naive asynchronous switching
    --delay d --guidance lerobot|exact RTC

Suites: stock LIBERO names (``libero_object``) and LIBERO-PRO position levels
(``pos:x0.2``, ``pos:y0.5``, ``pos:all``). Results go to ``--out`` as one JSON
per suite (re-running skips finished suites) plus ``summary.md``.

Example:
    python scripts/eval_libero.py --suites libero_object,pos:x0.2 --episodes-per-task 20 \\
        --libero-pro-root third_party/LIBERO-PRO --out runs/sync_s10
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

from closed_loop import Chunk, LoopConfig, run_batch, stack_observations, switch_jumps  # noqa: E402
from libero_pro_suites import (  # noqa: E402
    MAX_STEPS,
    POSITION_LEVELS,
    import_libero_quietly,
    register_position_suites,
    suite_name,
)

log = logging.getLogger("eval_libero")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="lerobot/pi05_libero_finetuned")
    p.add_argument("--suites", default="libero_object,pos:all",
                   help="comma-separated: LIBERO suite names and/or pos:<level> / pos:all for LIBERO-PRO")
    p.add_argument("--libero-pro-root", type=Path, default=ROOT / "third_party" / "LIBERO-PRO")
    p.add_argument("--task-ids", type=int, nargs="*", default=None)
    p.add_argument("--episodes-per-task", type=int, default=50, help="init states per task (LIBERO has 50)")
    p.add_argument("--batch-size", type=int, default=10, help="environments run in parallel")
    p.add_argument("--max-steps", type=int, default=None, help=f"default: {MAX_STEPS} for libero_object-based suites")
    p.add_argument("--stride", type=int, default=None, help="s; default: the checkpoint's n_action_steps")
    p.add_argument("--delay", type=int, default=0, help="d, simulated latency in control steps")
    p.add_argument("--guidance", choices=["none", "lerobot", "exact"], default="none")
    p.add_argument("--mask-end", type=int, default=None, help="RTC soft-mask end; default H - s")
    p.add_argument("--schedule", default="exp", choices=["exp", "linear", "ones", "zeros"])
    p.add_argument("--max-guidance-weight", type=float, default=10.0)
    p.add_argument("--record-entropy", action="store_true", help="store cross-attention entropy of every chunk")
    p.add_argument("--save-actions", action="store_true", help="store executed actions (smoothness analysis)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


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
        register_position_suites(pro_root, levels=tuple(dict.fromkeys(levels)))
        names += [suite_name(level) for level in dict.fromkeys(levels)]
    return names


def load_policy(path: str, device: str):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import get_policy_class

    config = PreTrainedConfig.from_pretrained(path)
    config.compile_model = False
    policy = get_policy_class(config.type).from_pretrained(path, config=config)
    return policy.to(device).eval()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    import_libero_quietly()
    from h1.attention import CrossAttentionRecorder
    from h1.guidance import install_rtc, make_rtc_config
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.utils import parse_camera_names, preprocess_observation
    from lerobot.policies import make_pre_post_processors
    from lerobot.utils.constants import ACTION
    from libero.libero import benchmark

    suites = resolve_suites(args.suites, args.libero_pro_root)
    policy = load_policy(args.policy, args.device)
    horizon = int(policy.config.chunk_size)
    stride = args.stride or int(policy.config.n_action_steps)
    max_steps = args.max_steps or MAX_STEPS
    cfg = LoopConfig(horizon, stride, args.delay, args.guidance != "none", max_steps, args.mask_end, args.seed)
    if args.guidance != "none":
        install_rtc(policy, args.guidance, make_rtc_config(args.schedule, args.max_guidance_weight, cfg.mask_end))
    recorder = CrossAttentionRecorder(policy.model).attach() if args.record_entropy else None

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.policy,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    env_cfg = LiberoEnvConfig(task=suites[0], episode_length=max_steps)
    env_pre, env_post = env_cfg.get_env_processors()
    camera_names = parse_camera_names(env_cfg.camera_name)
    noise_shape = (horizon, int(policy.config.max_action_dim))
    log.info("H=%d s=%d d=%d guidance=%s mask_end=%d max_steps=%d", horizon, stride, args.delay,
             args.guidance, cfg.mask_end, max_steps)

    def make_infer(task_description: str):
        def infer(observations, noise, leftover, delay, mask_end) -> Chunk:
            obs = preprocess_observation(stack_observations(observations))
            obs["task"] = [task_description] * len(observations)
            batch = preprocessor(env_pre(obs))
            kwargs = {}
            if leftover is not None:
                kwargs = {"inference_delay": delay, "prev_chunk_left_over": leftover.to(args.device),
                          "execution_horizon": mask_end}
            ctx = recorder.record() if recorder else contextlib.nullcontext()
            with torch.no_grad(), ctx:
                pred = policy.predict_action_chunk(batch, noise=noise.to(args.device), **kwargs)
            actions = env_post({ACTION: postprocessor(pred.clone())})[ACTION]
            entropy = recorder.result().entropy[:, -1].numpy() if recorder else None
            return Chunk(pred.float(), torch.as_tensor(actions).float().cpu().numpy(), entropy)
        return infer

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {k: str(v) for k, v in vars(args).items()} | {"horizon": horizon, "stride": stride, "mask_end": cfg.mask_end,
                                                          "max_steps": max_steps}
    (args.out / "config.json").write_text(json.dumps(meta, indent=2))

    for name in suites:
        out_file = args.out / f"{name}.json"
        if out_file.exists():
            log.info("skip %s (done)", name)
            continue
        suite = benchmark.get_benchmark_dict()[name]()
        task_ids = args.task_ids if args.task_ids is not None else list(range(suite.n_tasks))
        episodes, batches, entropies, actions_all = [], [], [], []
        started = time.time()
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
                    res = run_batch(envs, infer, cfg, global_ids, noise_shape)
                finally:
                    for env in envs:
                        env.close()
                jump_switch, jump_other = switch_jumps(res.actions, res.switch_steps)
                for b, i in enumerate(ids):
                    episodes.append({"task_id": tid, "init_state": i, "success": bool(res.success[b]),
                                     "length": int(res.length[b])})
                batches.append({"task_id": tid, "init_states": ids, "n_inference": res.n_inference,
                                "jump_at_switch": jump_switch, "jump_other": jump_other})
                if args.record_entropy and res.chunk_entropy:
                    entropies.append({"task_id": tid, "init_states": ids, "steps": res.chunk_steps,
                                      "entropy": np.stack(res.chunk_entropy).tolist()})
                if args.save_actions:
                    actions_all.append(res.actions)
                done = [e for e in episodes if e["task_id"] == tid]
                log.info("%s task %d: %d/%d success after %d episodes (%.0f s)", name, tid,
                         sum(e["success"] for e in done), len(done), len(done), time.time() - started)
        sr = float(np.mean([e["success"] for e in episodes]))
        by_task = {tid: float(np.mean([e["success"] for e in episodes if e["task_id"] == tid])) for tid in task_ids}
        result = {"suite": name, "success_rate": sr, "by_task": by_task, "n_episodes": len(episodes),
                  "mean_inference": float(np.mean([b["n_inference"] for b in batches])),
                  "jump_at_switch": float(np.nanmean([b["jump_at_switch"] for b in batches])),
                  "jump_other": float(np.nanmean([b["jump_other"] for b in batches])),
                  "episodes": episodes, "batches": batches, "seconds": time.time() - started}
        out_file.write_text(json.dumps(result, indent=1))
        if entropies:
            (args.out / f"{name}_entropy.json").write_text(json.dumps(entropies))
        if actions_all:
            np.save(args.out / f"{name}_actions.npy", np.concatenate(actions_all))
        log.info("%s: success %.1f%% over %d episodes", name, 100 * sr, len(episodes))

    write_summary(args.out, suites)
    if recorder:
        recorder.detach()


def write_summary(out: Path, suites: list[str]) -> None:
    rows = ["| suite | success, % | episodes | inferences / batch | |Δa| at switch | |Δa| other |",
            "|---|---|---|---|---|---|"]
    for name in suites:
        f = out / f"{name}.json"
        if f.exists():
            r = json.loads(f.read_text())
            rows.append(f"| {name} | {100 * r['success_rate']:.1f} | {r['n_episodes']} | {r['mean_inference']:.1f} "
                        f"| {r['jump_at_switch']:.4f} | {r['jump_other']:.4f} |")
    (out / "summary.md").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
