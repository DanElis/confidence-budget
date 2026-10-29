#!/usr/bin/env python
"""Wall time of one PI0.5 inference vs batch size, with K noise samples per observation.

The batch size n is the number of distinct observations in one call (n = 1: one robot; n > 1: environments
batched in simulation for throughput). For each n in --n, on real LIBERO observations and the same noise:
    k1                 one sample per observation (plain call)
    rep_K / shared_K   K samples per observation: observation repeated K times / prefix encoded once
    rep_2K / shared_2K the joint call of --sample-call joint: K unguided + K guided rows (RTC on the guided ones)
Prints the median seconds per call, the ratio to k1 at the same n, and max |shared - repeated|.

    uv run python scripts/bench_samples.py --n 1 2 5 10 --k 2 4 --repeats 10
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
    p.add_argument("--n", type=int, nargs="+", default=[1, 2, 5, 10], help="batch sizes (distinct observations)")
    p.add_argument("--k", type=int, nargs="+", default=[2, 4], help="samples per observation")
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
    n_max = max(args.n)
    envs = [LiberoEnv(task_suite=suite, task_id=0, task_suite_name=args.suite,
                      camera_name=parse_camera_names(env_cfg.camera_name), init_states=True, episode_length=MAX_STEPS,
                      episode_index=i % 50, n_envs=1, **env_cfg.gym_kwargs) for i in range(n_max)]
    all_obs = [env.reset()[0] for env in envs]
    for env in envs:
        env.close()
    task = suite.get_task(0).language
    sync = (lambda: torch.cuda.synchronize()) if args.device.startswith("cuda") else (lambda: None)  # noqa: E731

    def timed(fn):
        with torch.no_grad():
            out = fn()  # warm-up (also the output)
            sync()
            times = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                fn()
                sync()
                times.append(time.perf_counter() - t0)
        return out.float().cpu(), float(np.median(times))

    rows = []
    for n in args.n:
        observations = all_obs[:n]

        def to_batch(obs_list):
            obs = preprocess_observation(stack_observations(obs_list))
            obs["task"] = [task] * len(obs_list)
            return preprocessor(env_pre(obs))

        def repeated(m, noise_rows, **kw):
            return lambda: policy.predict_action_chunk(to_batch([o for o in observations for _ in range(m)]),
                                                       noise=noise_rows.to(args.device), **kw)

        def shared(m, noise_rows, **kw):
            return lambda: predict_chunk_shared(policy, to_batch(observations), m, noise=noise_rows.to(args.device),
                                                **kw)

        noise = torch.randn(n, horizon, a_max, generator=torch.Generator().manual_seed(n))
        base, t_k1 = timed(lambda: policy.predict_action_chunk(to_batch(observations), noise=noise.to(args.device)))
        rows.append((n, "k1", 1, t_k1, t_k1, float("nan")))
        leftover = torch.zeros(n, horizon, base.shape[-1])
        leftover[:, : horizon - args.delay] = base[:, args.delay:]  # the chunk shifted by s = d
        mask_end = torch.full((n,), args.mask_end)
        w = proc.prefix_weights(args.delay, mask_end, n, horizon, horizon)
        for k in args.k:
            noise_k = expand_noise(noise, k)
            out_rep, t_rep = timed(repeated(k, noise_k))
            out_sh, t_sh = timed(shared(k, noise_k))
            diff = float((out_rep - out_sh).abs().max())
            rows += [(n, f"rep_{k}", k, t_rep, t_k1, float("nan")), (n, f"shared_{k}", k, t_sh, t_k1, diff)]
            noise2, left2, mask2, w2 = joint_batch(noise_k, leftover, mask_end, w, k)
            kw = dict(inference_delay=args.delay, prev_chunk_left_over=left2.to(args.device), execution_horizon=mask2)
            proc.weight_override = w2.to(args.device)
            try:
                out_rep2, t_rep2 = timed(repeated(2 * k, noise2, **kw))
                out_sh2, t_sh2 = timed(shared(2 * k, noise2, **kw))
            finally:
                proc.weight_override = None
            diff2 = float((out_rep2 - out_sh2).abs().max())
            rows += [(n, f"rep_{2 * k}_joint", 2 * k, t_rep2, t_k1, float("nan")),
                     (n, f"shared_{2 * k}_joint", 2 * k, t_sh2, t_k1, diff2)]
        print(f"n={n} done", flush=True)

    print(f"\nPI0.5, H={horizon}, median of {args.repeats} calls")
    print("| batch n | variant | rows / obs. | rows in call | s / call | × k1 (same n) | max |shared - rep| |")
    print("|---|---|---|---|---|---|---|")
    for n, name, m, t, t1, diff in rows:
        d = "" if np.isnan(diff) else f"{diff:.1e}"
        print(f"| {n} | {name} | {m} | {n * m} | {t:.3f} | {t / t1:.2f} | {d} |")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
