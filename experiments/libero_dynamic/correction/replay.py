#!/usr/bin/env python
"""Offline replay of the asynchronous RTC process along the training demonstrations (libero_pro venv).

For every episode the runner's rtc_queue schedule (common.request_schedule) is replayed on the recorded frames: a
request at t_r sees frame t_r, its chunk is RTC-guided towards the unexecuted rest of the previous chunk
(chunk_prev[s_prev:K], model space, padded to H as the client's leftover) with delay d, and it executes indices
[d, d + s) at ticks t_r + j. pi0.5 runs exactly as policy_server.py with --mode rtc --return-latent (the Server class
is reused: same preprocessing, deterministic request noise, LeRobot guidance via BatchRTCProcessor, the mask of
--rtc-execution-horizon, the first request of an episode unguided); episodes are batched in lockstep (one request
of each episode per call). --mode sync is the ablation: no RTC, chunks from frame t_r executed at [0, s).

Per request the output stores the episode, t_r, s, the RTC scale, the previous request (or -1), the whole chunk in
environment space (H, 7) and the latent (2048, float16). Shards per group of episodes make the run restartable.

    uv run --with websockets --with msgpack python ../libero_dynamic/correction/replay.py --data DATA --name rtc_s4 \
        --mode rtc --K 10 --delay 4 --step 4
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))
from common import client_image, request_schedule  # noqa: E402

log = logging.getLogger("replay")


def collate(rows: list[dict]) -> dict:
    """Concatenate preprocessed single-observation batches (fixed-length tokens: padding="max_length")."""
    out = {}
    for k, v in rows[0].items():
        if isinstance(v, torch.Tensor):
            out[k] = torch.cat([r[k] for r in rows], 0)
        elif isinstance(v, list):
            out[k] = [x for r in rows for x in r[k]]
        else:
            out[k] = v
    return out


def episode_plan(e: dict, args, rng: np.random.Generator) -> list[tuple[int, int, float]]:
    """(t_r, s, scale) of one episode."""
    lo, hi = args.step_range if args.step_range else (args.step, args.step)
    s_min = max(args.delay, 1)
    steps = (lambda k: int(rng.integers(max(lo, s_min), hi + 1))) if hi > lo else max(lo, s_min)
    offset = int(rng.integers(0, args.max_offset + 1)) if args.max_offset else 0
    sched = request_schedule(e["length"], args.delay, steps, offset)
    lo_f, hi_f = args.scale_range
    return [(t, s, float(rng.uniform(lo_f, hi_f)) if hi_f > lo_f else lo_f) for t, s in sched]


def load_frames(e: dict) -> list[np.ndarray]:
    from prepare_data import decode_video

    return [decode_video(v) for v in e["videos"]]


def run_group(server, eps: list[dict], plans: list[list], states: np.ndarray, args) -> dict:
    from policy_server import leftover_tensor, request_noise
    from shared_prefix import predict_chunk_shared

    dev, H, K = args.device, server.horizon, args.K
    mask_end = torch.tensor([server.exec_h])
    frames = [load_frames(e) for e in eps]
    prev: list[dict | None] = [None] * len(eps)  # model-space chunk and s of the previous request
    rec = {k: [] for k in ("ep", "t", "s", "scale", "prev", "chunk", "latent")}
    for r in range(max(len(p) for p in plans)):
        rows = [i for i, p in enumerate(plans) if r < len(p)]
        obs, scales = [], []
        for i in rows:
            t, s, f = plans[i][r]
            o = {"observation/image": client_image(frames[i][0][t]),
                 "observation/wrist_image": client_image(frames[i][1][t]),
                 "observation/state": states[eps[i]["start"] + t], "prompt": eps[i]["task"]}
            if args.mode == "rtc" and prev[i] is not None:
                o["prev_chunk_left_over"] = prev[i]["model"][prev[i]["s"]:K].numpy()
            obs.append(o)
            scales.append(f)
        batch = collate([server.batch(o) for o in obs])
        noise = torch.cat([request_noise(o, args.seed, H, server.a_max) for o in obs]).to(dev)
        info: dict = {}
        guided = args.mode == "rtc" and r > 0
        with torch.no_grad():
            if guided:
                lefts = [leftover_tensor(o, H)[0] for o in obs]
                if any(x is None for x in lefts):
                    raise RuntimeError("a guided round has an episode without a leftover")
                left = torch.cat(lefts).to(dev)
                w = server.weights_proc.prefix_weights(args.delay, mask_end, len(rows), H, H)
                server.rtc.weight_override = (torch.tensor(scales)[:, None] * w).to(dev)
                try:
                    chunk = predict_chunk_shared(server.policy, batch, 1, noise=noise, latent_out=info,
                                                 inference_delay=args.delay, prev_chunk_left_over=left,
                                                 execution_horizon=mask_end).float()
                finally:
                    server.rtc.weight_override = None
            else:
                chunk = predict_chunk_shared(server.policy, batch, 1, noise=noise, latent_out=info).float()
            env = torch.stack([server.post(chunk[:, j, :]) for j in range(H)], dim=1).float().cpu()
        for n, i in enumerate(rows):
            t, s, f = plans[i][r]
            rec["ep"].append(eps[i]["id"])
            rec["t"].append(t)
            rec["s"].append(s)
            rec["scale"].append(f if guided else 1.0)
            rec["prev"].append(prev[i]["local"] if prev[i] is not None else -1)
            rec["chunk"].append(env[n].numpy())
            rec["latent"].append(info["latent"][n].cpu().numpy().astype(np.float16))
            prev[i] = {"model": chunk[n].cpu(), "s": s, "local": len(rec["ep"]) - 1}
    return {"ep": np.asarray(rec["ep"], np.int32), "t": np.asarray(rec["t"], np.int32),
            "s": np.asarray(rec["s"], np.int16), "scale": np.asarray(rec["scale"], np.float32),
            "prev": np.asarray(rec["prev"], np.int32), "chunk": np.stack(rec["chunk"]).astype(np.float32),
            "latent": np.stack(rec["latent"])}


def merge(shards: list[Path], out: Path, attrs: dict) -> None:
    parts = [dict(np.load(f)) for f in shards]
    res, base = {k: [] for k in parts[0]}, 0
    for p in parts:
        p["prev"] = np.where(p["prev"] >= 0, p["prev"] + base, -1).astype(np.int32)
        for k in res:
            res[k].append(p[k])
        base += len(p["ep"])
    np.savez(out, **{k: np.concatenate(v) for k, v in res.items()}, attrs=json.dumps(attrs))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--mode", choices=["rtc", "sync"], default="rtc")
    p.add_argument("--K", type=int, default=10, help="returned chunk length (runner --replan-steps)")
    p.add_argument("--delay", type=int, default=4, help="d = ceil(latency * 20); 0 for --mode sync")
    p.add_argument("--step", type=int, default=4, help="replanning step s (rtc_k10: s = d = 4)")
    p.add_argument("--step-range", type=int, nargs=2, default=None, help="s ~ U{lo..hi} per request")
    p.add_argument("--scale-range", type=float, nargs=2, default=[1.0, 1.0], help="RTC mask scale ~ U[lo, hi]")
    p.add_argument("--max-offset", type=int, default=0, help="first request at a random tick in [0, max-offset]")
    p.add_argument("--split", choices=["all", "train", "val"], default="all")
    p.add_argument("--limit", type=int, default=None, help="first N episodes (smoke test)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy", default="lerobot/pi05_libero_finetuned")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-guidance-weight", type=float, default=2.0)
    p.add_argument("--rtc-execution-horizon", type=int, default=6)
    a = p.parse_args(argv)
    if a.mode == "rtc" and a.K < (a.step_range[1] if a.step_range else a.step) + a.delay:
        p.error("K must be >= s_max + d (no underflow in the runner)")
    return a


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    out = args.data / f"replay_{args.name}.npz"
    if out.exists():
        log.info("%s exists", out)
        return
    import policy_server as ps

    episodes = json.loads((args.data / "episodes.json").read_text())
    for i, e in enumerate(episodes):
        e["id"] = i
    if args.split != "all":
        episodes = [e for e in episodes if e["split"] == args.split]
    episodes = episodes[: args.limit] if args.limit else episodes
    states = np.load(args.data / "demo.npz")["states"]
    rng = np.random.default_rng(args.seed)
    plans = [episode_plan(e, args, rng) for e in episodes]
    attrs = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    log.info("%d episodes, %d requests; %s", len(episodes), sum(map(len, plans)), attrs)

    server = ps.Server(ps.parse_args(["--mode", "rtc", "--policy", args.policy, "--device", args.device,
                                      "--actions-per-chunk", str(args.K), "--return-latent",
                                      "--max-guidance-weight", str(args.max_guidance_weight),
                                      "--rtc-execution-horizon", str(args.rtc_execution_horizon)]))
    shard_dir = args.data / f"replay_{args.name}"
    shard_dir.mkdir(exist_ok=True)
    shards, t0, done = [], time.time(), 0
    for g in range(0, len(episodes), args.batch):
        f = shard_dir / f"group_{g // args.batch:05d}.npz"
        shards.append(f)
        n = sum(len(p) for p in plans[g: g + args.batch])
        if not f.exists():
            res = run_group(server, episodes[g: g + args.batch], plans[g: g + args.batch], states, args)
            np.savez(shard_dir / "tmp.npz", **res)
            (shard_dir / "tmp.npz").rename(f)
            done += n
            log.info("group %d: %d / %d episodes, %.1f requests/s", g // args.batch, g + args.batch,
                     len(episodes), done / max(time.time() - t0, 1e-6))
    merge(shards, out, attrs)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
