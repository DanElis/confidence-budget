#!/usr/bin/env python
"""Download the LIBERO-Dynamic strict non-benchmark training set (pi05_raw, LeRobot v2.1) and lay it out for the
correction head (libero_pro venv). Idempotent: finished steps are skipped.

    uv run python ../libero_dynamic/correction/prepare_data.py --data DATA [--workers 16] [--limit N]

DATA/
  raw/pi05_raw/<suite>/{meta,data,videos}   the release (1800 episodes, 284,502 frames, ~1.1 GB)
  episodes.json   per episode: suite, episode_index, task, start (global frame index), length, speed, level, traj,
                  video paths, split ("train" / "val": 2 episodes per task x speed, seed 0)
  demo.npz        actions (N, 7), states (N, 8) of all frames in global order
  frames_<S>.u8   uint8 memmap (N, 2, S, S, 3): head_image of the agentview and wrist frames
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CAMERAS, HEAD_SIZE, REPO, SUITES, head_image  # noqa: E402

log = logging.getLogger("prepare_data")


def scalar(x):
    a = np.asarray(x).reshape(-1)
    return a[0].item() if a.size else None


def decode_video(path: str) -> np.ndarray:
    """All frames of an MP4 as (T, H, W, 3) uint8 (sequential decode, PyAV)."""
    import av

    with av.open(path) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def read_episodes(raw: Path, limit: int | None) -> tuple[list[dict], np.ndarray, np.ndarray]:
    import pandas as pd

    episodes, actions, states, start = [], [], [], 0
    for suite in SUITES:
        root = raw / "pi05_raw" / suite
        tasks = {}
        for line in (root / "meta" / "tasks.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                tasks[int(r["task_index"])] = str(r["task"])
        for f in sorted((root / "data").glob("chunk-*/episode_*.parquet")):
            df = pd.read_parquet(f)
            a = np.stack(df["action"].to_numpy()).astype(np.float32)
            s = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
            if a.shape[1] != 7 or s.shape[1] != 8:
                raise RuntimeError(f"{f}: action {a.shape}, state {s.shape}")
            ep = int(scalar(df["episode_index"].iloc[0]))
            chunk = f.parent.name
            row = {"suite": suite, "episode_index": ep, "task": tasks[int(scalar(df["task_index"].iloc[0]))],
                   "start": start, "length": len(a),
                   "speed": round(float(scalar(df["dynamic.speed_mps"].iloc[0])), 4),
                   "level": int(scalar(df["dynamic.level"].iloc[0])),
                   "traj": int(scalar(df["dynamic.trajectory_index"].iloc[0])),
                   "videos": [str(root / "videos" / chunk / cam / f"episode_{ep:06d}.mp4") for cam in CAMERAS]}
            episodes.append(row)
            actions.append(a)
            states.append(s)
            start += len(a)
            if limit and len(episodes) >= limit:
                return episodes, np.concatenate(actions), np.concatenate(states)
    return episodes, np.concatenate(actions), np.concatenate(states)


def assign_split(episodes: list[dict], n_val: int = 2, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    groups: dict[tuple, list[int]] = {}
    for i, e in enumerate(episodes):
        groups.setdefault((e["task"], e["speed"]), []).append(i)
    for key in sorted(groups):
        idx = groups[key]
        val = set(rng.permutation(idx)[: n_val].tolist()) if len(idx) > n_val else set()
        for i in idx:
            episodes[i]["split"] = "val" if i in val else "train"


def decode_into(args) -> int:
    path, size, n_total, ep = args
    mm = np.memmap(path, dtype=np.uint8, mode="r+", shape=(n_total, 2, size, size, 3))
    for c, video in enumerate(ep["videos"]):
        frames = decode_video(video)
        if len(frames) != ep["length"]:
            raise RuntimeError(f"{video}: {len(frames)} frames, parquet has {ep['length']}")
        for t, fr in enumerate(frames):
            mm[ep["start"] + t, c] = head_image(fr, size)
    mm.flush()
    return ep["length"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--size", type=int, default=HEAD_SIZE)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="first N episodes only (smoke test)")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    a.data.mkdir(parents=True, exist_ok=True)
    raw = a.data / "raw"

    if not (raw / "download.done").exists():
        from huggingface_hub import snapshot_download

        t0 = time.time()
        snapshot_download(repo_id=REPO, repo_type="dataset", local_dir=str(raw),
                          allow_patterns=[f"pi05_raw/*/{d}/*" for d in ("meta", "data", "videos")])
        (raw / "download.done").write_text("ok\n")
        log.info("downloaded in %.0f s", time.time() - t0)

    meta = a.data / "episodes.json"
    if not meta.exists():
        episodes, actions, states = read_episodes(raw, a.limit)
        assign_split(episodes)
        np.savez(a.data / "demo.npz", actions=actions, states=states)
        meta.write_text(json.dumps(episodes, indent=0))
        log.info("%d episodes, %d frames, %d val episodes", len(episodes), len(actions),
                 sum(e["split"] == "val" for e in episodes))
    episodes = json.loads(meta.read_text())
    n_total = episodes[-1]["start"] + episodes[-1]["length"]

    frames = a.data / f"frames_{a.size}.u8"
    done = a.data / f"frames_{a.size}.done"
    if not done.exists():
        np.memmap(frames, dtype=np.uint8, mode="w+", shape=(n_total, 2, a.size, a.size, 3)).flush()
        t0, n = time.time(), 0
        with ProcessPoolExecutor(a.workers) as ex:
            for i, m in enumerate(ex.map(decode_into, [(str(frames), a.size, n_total, e) for e in episodes],
                                         chunksize=4)):
                n += m
                if i % 100 == 0:
                    log.info("decoded %d / %d episodes (%d frames, %.0f frames/s)", i + 1, len(episodes), n,
                             n / max(time.time() - t0, 1e-6))
        done.write_text(f"{n_total}\n")
        log.info("frames: %s (%.1f GB)", frames, frames.stat().st_size / 1e9)


if __name__ == "__main__":
    main()
