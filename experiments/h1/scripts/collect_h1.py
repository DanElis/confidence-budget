#!/usr/bin/env python
"""Collect paired free / RTC chunks with cross-attention entropy on LIBERO demonstrations.

For each sampled episode, chunks are generated every ``--stride`` frames. At each
frame the PI0.5 policy produces one chunk without guidance and one with RTC
guidance toward the leftover of the previous chunk, from the same observation
and noise. Per action we store the prediction, the demonstration target and the
entropy of the action expert's attention to the vision-language tokens at every
denoising step. ``analyze_h1.py`` turns the file into the H1 verdict.

Example:
    python scripts/collect_h1.py --episodes 100 --delay 5 --stride 25 --out runs/h1_d5_s25.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from h1.attention import CrossAttentionRecorder  # noqa: E402
from h1.guidance import GUIDANCE_MODES, install_rtc, make_rtc_config  # noqa: E402
from h1.rollout import (  # noqa: E402
    Generation,
    Observation,
    RolloutConfig,
    chunk_frames,
    run_episode,
    stack_records,
)

log = logging.getLogger("collect_h1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="lerobot/pi05_libero_finetuned", help="PI0.5 checkpoint (Hub id or path)")
    p.add_argument("--dataset", default="HuggingFaceVLA/libero", help="LeRobot dataset repo id")
    p.add_argument("--dataset-root", default=None, help="local dataset root, if already downloaded")
    p.add_argument("--full-dataset", action="store_true",
                   help="load every data file instead of only the selected episodes")
    p.add_argument("--episodes", type=int, default=100, help="number of episodes to sample")
    p.add_argument("--episode-ids", type=int, nargs="*", default=None, help="explicit episode ids (overrides --episodes)")
    p.add_argument("--max-chunks", type=int, default=None, help="recorded chunks per episode (default: all)")
    p.add_argument("--delay", type=int, default=5, help="inference delay d, controller steps")
    p.add_argument("--stride", type=int, default=25, help="replanning step s, controller steps")
    p.add_argument("--mask-end", type=int, default=None, help="end of the soft mask (default: H - s, as in RTC)")
    p.add_argument("--guidance", choices=GUIDANCE_MODES, default="exact", help="exact VJP (paper) or LeRobot's")
    p.add_argument("--schedule", default="exp", choices=["exp", "linear", "ones", "zeros"])
    p.add_argument("--max-guidance-weight", type=float, default=10.0, help="beta, guidance weight clip")
    p.add_argument("--prev-source", choices=["rtc", "free", "stale"], default="rtc",
                   help="previous chunk: last RTC chunk, last free chunk, or a stale plan (see --stale-frames)")
    p.add_argument("--stale-frames", type=int, default=0,
                   help="with --prev-source stale: the previous chunk is planned from an observation this many "
                        "frames older than it should be")
    p.add_argument("--prev-shift", type=float, default=0.0,
                   help="offset added to the leftover, in std of the normalised actions (random sign per dimension)")
    p.add_argument("--shift-gripper", action="store_true", help="also shift the last action dimension (gripper)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, required=True, help="output .npz")
    p.add_argument("--save-every", type=int, default=10, help="checkpoint the output every N episodes")
    return p.parse_args()


def load_policy(path: str, device: str):
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import get_policy_class

    config = PreTrainedConfig.from_pretrained(path)
    if config.type != "pi05":
        raise SystemExit(f"this script hooks the PI0.5 action expert; got policy type {config.type!r}")
    config.compile_model = False  # hooks need the eager graph
    policy = get_policy_class(config.type).from_pretrained(path, config=config)
    policy.to(device).eval()
    return policy


def episode_rows(dataset) -> tuple[dict[int, dict[int, int]], np.ndarray]:
    """Map episode -> frame_index -> row of the loaded dataset; also return the global ``index`` column."""
    table = dataset.hf_dataset.data
    episodes = table.column("episode_index").to_numpy()
    frames = table.column("frame_index").to_numpy()
    rows: dict[int, dict[int, int]] = {}
    for row, (ep, fr) in enumerate(zip(episodes.tolist(), frames.tolist(), strict=True)):
        rows.setdefault(int(ep), {})[int(fr)] = row
    return rows, table.column("index").to_numpy()


def check_episodes(rows, index, meta, episode_ids, filtered: bool) -> list[str]:
    """Problems that would make LeRobot's action windows wrong for the selected episodes.

    LeRobot builds the future-action window from ``dataset_from_index`` /
    ``dataset_to_index`` in the metadata, so the loaded rows must agree with it.
    """
    problems = []
    for ep in episode_ids:
        info = meta.episodes[ep]
        length, start = int(info["length"]), int(info["dataset_from_index"])
        frames = rows.get(ep)
        if not frames:
            problems.append(f"episode {ep}: no rows in the loaded data")
            continue
        if sorted(frames) != list(range(length)):
            problems.append(f"episode {ep}: {len(frames)} frames loaded, metadata says {length}")
            continue
        row_ids = np.array([frames[f] for f in range(length)])
        if not np.array_equal(index[row_ids], start + np.arange(length)):
            problems.append(f"episode {ep}: global index disagrees with metadata")
        elif not filtered and not np.array_equal(row_ids, index[row_ids]):
            problems.append(f"episode {ep}: row order differs from the global index")
    return problems


def load_dataset(args, meta, episode_ids, delta_timestamps):
    """Load only the selected episodes; fall back to the full dataset if the metadata is inconsistent.

    Some converted datasets (e.g. HuggingFaceVLA/libero at the time of writing)
    map episodes to the wrong parquet files, so the per-episode download finds
    no rows. Loading every file avoids the mapping.
    """
    from lerobot.datasets import LeRobotDataset

    if not args.full_dataset:
        try:
            dataset = LeRobotDataset(args.dataset, root=args.dataset_root, episodes=episode_ids,
                                     delta_timestamps=delta_timestamps)
            rows, index = episode_rows(dataset)
            problems = check_episodes(rows, index, meta, episode_ids, filtered=True)
            if not problems:
                return dataset, rows
            log.warning("per-episode loading is inconsistent: %s", "; ".join(problems[:3]))
        except ValueError as err:  # e.g. 'Instruction "train" corresponds to no data!'
            log.warning("per-episode loading failed: %s", err)
        log.warning("falling back to the full dataset (downloads every data file once)")
    dataset = LeRobotDataset(args.dataset, root=args.dataset_root, delta_timestamps=delta_timestamps,
                             force_cache_sync=True)
    rows, index = episode_rows(dataset)
    problems = check_episodes(rows, index, meta, episode_ids, filtered=False)
    if problems:
        raise SystemExit("dataset is inconsistent with its metadata:\n" + "\n".join(problems[:10]))
    return dataset, rows


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    from lerobot.datasets import LeRobotDatasetMetadata, resolve_delta_timestamps
    from lerobot.policies import make_pre_post_processors

    policy = load_policy(args.policy, args.device)
    horizon = int(policy.config.chunk_size)
    mask_end = args.mask_end if args.mask_end is not None else horizon - args.stride
    cfg = RolloutConfig(horizon, args.delay, args.stride, mask_end, args.prev_source, args.seed,
                        stale_frames=args.stale_frames, prev_shift=args.prev_shift,
                        shift_last_dim=args.shift_gripper)
    install_rtc(policy, args.guidance, make_rtc_config(args.schedule, args.max_guidance_weight, mask_end))
    recorder = CrossAttentionRecorder(policy.model).attach()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.policy,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    meta = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
    rng = np.random.default_rng(args.seed)
    if args.episode_ids:
        episode_ids = sorted(args.episode_ids)
    else:
        n = min(args.episodes, meta.total_episodes)
        episode_ids = sorted(rng.choice(meta.total_episodes, size=n, replace=False).tolist())
    dataset, rows = load_dataset(args, meta, episode_ids, resolve_delta_timestamps(policy.config, meta))
    action_std = np.asarray(meta.stats["action"]["std"], dtype=np.float64)
    noise_shape = (1, horizon, int(policy.config.max_action_dim))

    def generate(batch, noise, leftover) -> Generation:
        kwargs = {}
        if leftover is not None:
            kwargs = {
                "inference_delay": cfg.delay,
                "prev_chunk_left_over": leftover.unsqueeze(0).to(args.device),
                "execution_horizon": cfg.mask_end,
            }
        with torch.no_grad(), recorder.record():  # ExactRTCProcessor re-enables grad for its VJP
            pred = policy.predict_action_chunk(batch, noise=noise.to(args.device), **kwargs)
        attn = recorder.result()
        pred_raw = torch.as_tensor(postprocessor(pred.clone())).float().cpu().numpy()[0]
        return Generation(
            pred=pred[0].float(),
            pred_raw=pred_raw,
            entropy=attn.entropy[0].numpy(),
            nonlang_share=attn.nonlang_share[0].numpy(),
            n_valid=int(attn.n_valid[0]),
        )

    run_meta = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "horizon": horizon,
        "delay": cfg.delay,
        "stride": cfg.stride,
        "mask_end": cfg.mask_end,
        "num_inference_steps": int(policy.config.num_inference_steps),
        "fps": float(meta.fps),
        "action_std": action_std.tolist(),
        "episode_ids": episode_ids,
    }

    records = []
    started = time.time()
    for n_done, episode in enumerate(episode_ids, start=1):
        frames_of_ep = rows[episode]

        def get_observation(frame: int, episode=episode, frames_of_ep=frames_of_ep) -> Observation:
            item = dataset[frames_of_ep[frame]]
            if int(item["episode_index"]) != episode or int(item["frame_index"]) != frame:
                raise RuntimeError(f"row lookup mismatch for episode {episode}, frame {frame}")
            gt_raw = item["action"].float().numpy()
            gt_pad = item["action_is_pad"].bool().numpy()
            batch = preprocessor(default_collate([item]))
            return Observation(batch=batch, gt_raw=gt_raw, gt_pad=gt_pad)

        offset = int(rng.integers(0, cfg.stride))
        max_frames = None if args.max_chunks is None else args.max_chunks + 1  # first frame seeds the chain
        frames = chunk_frames(len(frames_of_ep), cfg.stride, offset, max_frames)
        if len(frames) < 2:
            log.warning("episode %d is too short for stride %d, skipped", episode, cfg.stride)
            continue
        records += run_episode(episode, frames, get_observation, generate, noise_shape, cfg)
        elapsed = time.time() - started
        log.info("episode %d (%d/%d): %d chunks so far, %.0f s", episode, n_done, len(episode_ids), len(records), elapsed)
        if n_done % args.save_every == 0 or n_done == len(episode_ids):
            save(args.out, records, run_meta)
    recorder.detach()
    log.info("saved %d chunks to %s", len(records), args.out)


def save(path: Path, records, run_meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = stack_records(records)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(run_meta)), **arrays)
    tmp.replace(path)


if __name__ == "__main__":
    main()
