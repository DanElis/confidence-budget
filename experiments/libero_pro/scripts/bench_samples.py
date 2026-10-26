#!/usr/bin/env python
"""Time and check K noise samples per observation for PI0.5: repeated observations vs one shared prefix.

On real LIBERO observations of --n environments, for the same noise:
    k1        one sample per observation (plain RTC call)
    rep_K     K samples, observation repeated K times (what eval_libero did before --shared-prefix)
    shared_K  K samples, prefix encoded once (shared_prefix.py)
    rep_2K / shared_2K   the joint call of --sample-call joint: K unguided + K guided rows with RTC
Prints the median wall time per call and the max |shared - repeated| difference of the actions.

    uv run python scripts/bench_samples.py --n 5 --k 4 --repeats 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "h1"))
sys.path.insert(0, str(ROOT / "scripts"))

from closed_loop import stack_observations  # noqa: E402
from eval_libero import load_policy  # noqa: E402
from libero_pro_suites import MAX_STEPS, import_libero_quietly  # noqa: E402
from selection import expand_noise, joint_batch  # noqa: E402
from shared_prefix import predict_chunk_shared  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="lerobot/pi05_libero_finetuned")
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--n", type=int, default=5, help="environments (distinct observations)")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--delay", type=int, default=8)
    p.add_argument("--mask-end", type=int, default=13)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import_libero_quietly()
    from h1.guidance import install_batch_rtc, make_rtc_config
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.utils import parse_camera_names, preprocess_observation
    from lerobot.policies import make_pre_post_processors
    from libero.libero import benchmark

    policy = load_policy(args.policy, args.device)
    horizon, a_max = int(policy.config.chunk_size), int(policy.config.max_action_dim)
    proc = install_batch_rtc(policy, "lerobot", make_rtc_config("exp", 10.0, args.mask_end))
    preprocessor, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=args.policy,
                                               preprocessor_overrides={"device_processor": {"device": args.device}})
    env_cfg = LiberoEnvConfig(task=args.suite, episode_length=MAX_STEPS)
    env_pre, _ = env_cfg.get_env_processors()
    suite = benchmark.get_benchmark_dict()[args.suite]()
    envs = [LiberoEnv(task_suite=suite, task_id=0, task_suite_name=args.suite,
                      camera_name=parse_camera_names(env_cfg.camera_name), init_states=True, episode_length=MAX_STEPS,
                      episode_index=i, n_envs=1, **env_cfg.gym_kwargs) for i in range(args.n)]
    observations = [env.reset()[0] for env in envs]
    task = suite.get_task(0).language

    def to_batch(obs_list):
        obs = preprocess_observation(stack_observations(obs_list))
        obs["task"] = [task] * len(obs_list)
        return preprocessor(env_pre(obs))

    n, k = args.n, args.k
    g = torch.Generator().manual_seed(0)
    noise = torch.randn(n, horizon, a_max, generator=g)
    noise_k = expand_noise(noise, k)
    sync = (lambda: torch.cuda.synchronize()) if args.device.startswith("cuda") else (lambda: None)  # noqa: E731

    def timed(fn):
        with torch.no_grad():
            out = fn()  # warm-up
            sync()
            times = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                fn()
                sync()
                times.append(time.perf_counter() - t0)
        return out.float().cpu(), float(np.median(times))

    def repeated(m, rows, **kw):
        return lambda: policy.predict_action_chunk(to_batch([o for o in observations for _ in range(m)]),
                                                   noise=rows.to(args.device), **kw)

    def shared(m, rows, **kw):
        return lambda: predict_chunk_shared(policy, to_batch(observations), m, noise=rows.to(args.device), **kw)

    results = {}
    base, results["k1"] = timed(lambda: policy.predict_action_chunk(to_batch(observations), noise=noise.to(args.device)))
    out_rep, results[f"rep_{k}"] = timed(repeated(k, noise_k))
    out_sh, results[f"shared_{k}"] = timed(shared(k, noise_k))
    diff_k = float((out_rep - out_sh).abs().max())

    leftover = torch.zeros(n, horizon, base.shape[-1])
    leftover[:, : horizon - args.delay] = base[:, args.delay:]  # the first chunk shifted by s = d
    mask_end = torch.full((n,), args.mask_end)
    w = proc.prefix_weights(args.delay, mask_end, n, horizon, horizon)
    noise2, left2, mask2, w2 = joint_batch(noise_k, leftover, mask_end, w, k)
    kw = dict(inference_delay=args.delay, prev_chunk_left_over=left2.to(args.device), execution_horizon=mask2)
    proc.weight_override = w2.to(args.device)
    try:
        out_rep2, results[f"rep_{2 * k}_joint"] = timed(repeated(2 * k, noise2, **kw))
        out_sh2, results[f"shared_{2 * k}_joint"] = timed(shared(2 * k, noise2, **kw))
    finally:
        proc.weight_override = None
    diff_2k = float((out_rep2 - out_sh2).abs().max())
    for env in envs:
        env.close()

    print(f"n={n} K={k} H={horizon} (median of {args.repeats} calls, seconds per call)")
    for name, t in results.items():
        print(f"  {name:16s} {t:.3f}  ({t / results['k1']:.2f}x k1)")
    print(f"max |shared - repeated|: K samples {diff_k:.2e}, joint 2K {diff_2k:.2e} "
          f"(action scale: {float(base.abs().mean()):.3f})")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
