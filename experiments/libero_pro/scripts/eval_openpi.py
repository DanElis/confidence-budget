#!/usr/bin/env python
"""Fixed-step closed loop on LIBERO / LIBERO-PRO with PI0.5 served by openpi (scripts/serve_policy.py --env LIBERO).

The environment, init states, step limit and result files are those of eval_libero.py, so compare.py pairs the runs by
scene. The observation follows the AAC authors' client (Adaptive-Action-Chunking/libero-pro@1c83fa7): images rotated by
180 degrees at 256 x 256 (the server resizes them), state = position, axis-angle, gripper; the returned environment
actions are executed as they are. The noise comes from the server's own generator: only scenes are shared with the
LeRobot runs, and repeats are not identical.

    python scripts/eval_openpi.py --suites pos:x0.2 --stride 5 --out runs/openpi_s5 --port 8765
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from closed_loop import Chunk, LoopConfig, run_batch, switch_jumps  # noqa: E402
from eval_libero import ACTION_DIM, episode_sparc, resolve_suites, summarize, write_summary  # noqa: E402
from libero_pro_suites import MAX_STEPS, import_libero_quietly  # noqa: E402
from planner import FixedPlanner  # noqa: E402

log = logging.getLogger("eval_openpi")
HORIZON = 10  # action_horizon of the pi05_libero config at the pinned openpi commit
RESOLUTION = 256  # LIBERO_ENV_RESOLUTION of openpi's and the AAC authors' clients


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suites", required=True)
    p.add_argument("--libero-pro-root", type=Path, default=ROOT / "third_party" / "LIBERO-PRO")
    p.add_argument("--task-ids", type=int, nargs="*", default=None)
    p.add_argument("--episodes-per-task", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--stride", type=int, required=True)
    p.add_argument("--delay", type=int, default=0)
    p.add_argument("--horizon", type=int, default=HORIZON, help="H the server must return")
    p.add_argument("--resolution", type=int, default=RESOLUTION)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0, help="only recorded: the noise is drawn by the server")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if not 1 <= a.stride <= a.horizon:
        p.error(f"need 1 <= stride <= horizon, got s={a.stride}, H={a.horizon}")
    return a


def pack_array(obj):
    """openpi_client.msgpack_numpy encoding."""
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj: dict):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class OpenPiClient:
    """The websocket protocol of openpi's server: metadata on connect, then one reply per observation.

    A text frame is the server's traceback. A failure raises and the suite leaves no result: it must not count as a
    policy failure. No keepalive: the server blocks its event loop during the first, compiling call.
    """

    def __init__(self, host: str, port: int):
        import msgpack
        from websockets.sync.client import connect

        self._msgpack = msgpack
        self._ws = connect(f"ws://{host}:{port}", compression=None, max_size=None, ping_interval=None, proxy=None)
        self.metadata = self._unpack(self._ws.recv())

    def _unpack(self, frame):
        if isinstance(frame, str):
            raise RuntimeError(f"openpi server error:\n{frame}")
        return self._msgpack.unpackb(frame, object_hook=unpack_array)

    def infer(self, element: dict) -> dict:
        self._ws.send(self._msgpack.packb(element, default=pack_array))
        return self._unpack(self._ws.recv())

    def close(self) -> None:
        self._ws.close()


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """(x, y, z, w) -> axis * angle, as _quat2axisangle of openpi's LIBERO client."""
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = math.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return np.asarray(quat[:3]) * 2.0 * math.acos(w) / den


def make_element(obs: dict, prompt: str) -> dict:
    """An observation of lerobot's LiberoEnv (obs_type pixels_agent_pos) as openpi's LiberoInputs reads it."""
    state = obs["robot_state"]
    return {
        "observation/image": np.ascontiguousarray(obs["pixels"]["image"][::-1, ::-1]),
        "observation/wrist_image": np.ascontiguousarray(obs["pixels"]["image2"][::-1, ::-1]),
        "observation/state": np.concatenate(
            (state["eef"]["pos"], quat2axisangle(state["eef"]["quat"]), state["gripper"]["qpos"])
        ).astype(np.float64),
        "prompt": str(prompt),
    }


def make_infer(client, prompt: str, horizon: int):
    def infer(observations, noise, leftover, delay, mask_end) -> Chunk:
        rows = []
        for obs in observations:
            actions = np.asarray(client.infer(make_element(obs, prompt))["actions"], dtype=np.float32)
            if actions.shape != (horizon, ACTION_DIM):
                raise RuntimeError(f"server returned actions {actions.shape}, expected {(horizon, ACTION_DIM)}")
            rows.append(actions)
        actions = np.stack(rows)
        return Chunk(torch.from_numpy(actions.copy()), actions)

    return infer


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    import_libero_quietly()
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.utils import parse_camera_names
    from libero.libero import benchmark

    suites = resolve_suites(args.suites, args.libero_pro_root)
    client = OpenPiClient(args.host, args.port)
    log.info("server metadata: %s", client.metadata)
    planner = FixedPlanner(args.horizon, args.stride)
    cfg = LoopConfig(args.horizon, args.delay, False, args.max_steps, args.seed)
    env_cfg = LiberoEnvConfig(task=suites[0], episode_length=args.max_steps, observation_height=args.resolution,
                              observation_width=args.resolution)
    camera_names = parse_camera_names(env_cfg.camera_name)
    noise_shape = (args.horizon, ACTION_DIM)

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {k: str(v) for k, v in vars(args).items()} | {"horizon": args.horizon, "max_steps": args.max_steps,
                                                          "planner_repr": repr(planner), "policy": "openpi pi05_libero"}
    (args.out / "config.json").write_text(json.dumps(meta, indent=2))

    for name in suites:
        out_file = args.out / f"{name}.json"
        if out_file.exists():
            log.info("skip %s (done)", name)
            continue
        suite = benchmark.get_benchmark_dict()[name]()
        task_ids = args.task_ids if args.task_ids is not None else list(range(suite.n_tasks))
        episodes, batches, chunk_log = [], [], []
        started = time.time()
        for tid in task_ids:
            task = suite.get_task(tid)
            infer = make_infer(client, task.language, args.horizon)
            for start in range(0, args.episodes_per_task, args.batch_size):
                ids = list(range(start, min(start + args.batch_size, args.episodes_per_task)))
                envs = [LiberoEnv(task_suite=suite, task_id=tid, task_suite_name=name, camera_name=camera_names,
                                  init_states=True, episode_length=args.max_steps, episode_index=i, n_envs=1,
                                  **env_cfg.gym_kwargs) for i in ids]
                try:
                    res = run_batch(envs, infer, cfg, planner, [tid * 1000 + i for i in ids], noise_shape)
                finally:
                    for env in envs:
                        env.close()
                jump_switch, jump_other = switch_jumps(res.actions, res.switch_steps)
                for b, i in enumerate(ids):
                    strides = [c["stride"] for c in res.chunks if c["env"] == b]
                    episodes.append({
                        "task_id": tid, "init_state": i, "success": bool(res.success[b]),
                        "length": int(res.length[b]), "n_inference": int(res.n_inference[b]),
                        "sparc": episode_sparc(res.actions[b]),
                        "mean_stride": float(np.mean(strides)) if strides else None,
                        "mean_h_star": None, "frac_sel": None,
                    })
                batches.append({"task_id": tid, "init_states": ids, "jump_at_switch": jump_switch,
                                "jump_other": jump_other})
                for c in res.chunks:
                    chunk_log.append({"task_id": tid, "init_state": ids[c["env"]]} | {k: v for k, v in c.items() if k != "env"})
                done = [e for e in episodes if e["task_id"] == tid]
                log.info("%s task %d: %d/%d success after %d episodes (%.0f s)", name, tid,
                         sum(e["success"] for e in done), len(done), len(done), time.time() - started)
        result = summarize(name, episodes, batches, task_ids)
        result["seconds"] = time.time() - started
        out_file.write_text(json.dumps(result, indent=1))
        (args.out / f"{name}_chunks.json").write_text(json.dumps(chunk_log))
        log.info("%s: success %.1f%% over %d episodes", name, 100 * result["success_rate"], len(episodes))

    write_summary(args.out, suites)
    client.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # as eval_libero.py: EGL / MuJoCo teardown can hang
