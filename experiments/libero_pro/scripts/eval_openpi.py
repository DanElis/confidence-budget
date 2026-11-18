#!/usr/bin/env python
"""Closed loop on LIBERO / LIBERO-PRO with PI0.5 served by openpi: constant step, or AAC with N samples.

The environment, init states, step limit and result files are those of eval_libero.py, so compare.py pairs the runs by
scene. The observation follows the AAC authors' client (Adaptive-Action-Chunking/libero-pro@1c83fa7): images rotated by
180 degrees at 256 x 256 (the server resizes them), state = position, axis-angle, gripper; the returned environment
actions are executed as they are.

Noise is sent with every request, per (seed, episode, step) as in closed_loop.py, and the executed sample is number 0, so
conditions share it while their trajectories coincide. openpi's stock server (scripts/serve_policy.py) ignores it and
draws its own, and has no N samples: --n-samples, --signal aac and --guidance need scripts/openpi_sample_server.py.

RTC: the old plan in model units is the target, its first d actions frozen and the soft mask ending at H - s by default.
--guidance paper is the RTC paper (Black et al., 2025): Eq. 2 with the Jacobian of the network, its weights, beta = 5, the
environment's action dimensions only. --guidance lerobot is eval_libero.py's mode: no Jacobian, LeRobot's --schedule,
beta = 10, all dimensions.

--rtc-gate decides per request whether to guide. The chunk is first generated without guidance, as naive would (same
noise, same request); D = mean |new - old| over the overlap with the old plan (i < H - s, the 7 action dimensions, model
units). divergence: guide (a second call, same noise) if D < --rtc-tau, else execute the unguided chunk, which is then
exactly naive's. random: guide with probability --rtc-share, independent of the state, the control at equal share.
--rtc-tau 0 never guides: naive execution with D logged, for calibration. D, the decision and the number of calls are
in the chunk log (divergence, guided, calls); gate_summary.py reads them.

--budget CONFIG: the step s per option from openpi_budget.py (naive switching); its features and decisions go to the
chunk log (budget_x, budget_action, budget_prob, budget_decision), read by openpi_budget_fit.py.

    python scripts/eval_openpi.py --suites pos:x0.2 --stride 5 --out runs/openpi_s5 --port 8765
    python scripts/eval_openpi.py --suites pos:x0.2 --signal aac --n-samples 20 --out runs/openpi_aac --port 8765
    python scripts/eval_openpi.py --suites pos:x0.2 --stride 4 --delay 4 --guidance paper --out runs/openpi_rtc --port 8765
    python scripts/eval_openpi.py --suites pos:y0.2 --stride 5 --delay 4 --guidance paper --rtc-gate divergence --rtc-tau 0.1 ...
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aac import paper_h_star  # noqa: E402
from closed_loop import Chunk, LoopConfig, run_batch, switch_jumps  # noqa: E402
from eval_libero import ACTION_DIM, episode_sparc, resolve_suites, summarize, write_summary  # noqa: E402
from libero_pro_suites import MAX_STEPS, import_libero_quietly  # noqa: E402
from planner import AdaptivePlanner, FixedPlanner, GivenHStar  # noqa: E402
from selection import expand_noise  # noqa: E402

log = logging.getLogger("eval_openpi")
HORIZON = 10  # action_horizon of the pi05_libero config at the pinned openpi commit
MODEL_ACTION_DIM = 32  # action_dim of the model: the noise has this size, the environment reads the first 7
RESOLUTION = 256  # LIBERO_ENV_RESOLUTION of openpi's and the AAC authors' clients
SAMPLE_TOLERANCE = 0.05  # normalised units: sample 0 of a batch against the same noise alone
GUIDANCE_PROBE_SHIFT = 1.0  # normalised units: how far from the free plan check_guidance puts its target
GUIDANCE_PROBE_MAX = 0.9  # share of that distance the frozen actions may keep, on average


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suites", required=True)
    p.add_argument("--libero-pro-root", type=Path, default=ROOT / "third_party" / "LIBERO-PRO")
    p.add_argument("--task-ids", type=int, nargs="*", default=None)
    p.add_argument("--episodes-per-task", type=int, default=50)
    p.add_argument("--first-episode", type=int, default=0,
                   help="first init state: episodes first .. first + episodes-per-task - 1 (new scenes of a task)")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--stride", type=int, default=None, help="constant step; with --signal aac leave it out")
    p.add_argument("--signal", choices=["aac"], default=None, help="aac: the step from aac.py, s = max(h* - d, d)")
    p.add_argument("--n-samples", type=int, default=1, help="N samples per request; sample 0 is executed")
    p.add_argument("--delay", type=int, default=0)
    p.add_argument("--guidance", choices=["none", "paper", "lerobot"], default="none", help="RTC, needs --delay and --stride")
    p.add_argument("--schedule", default="exp", choices=["exp", "linear", "ones", "zeros"], help="--guidance lerobot only")
    p.add_argument("--max-guidance-weight", type=float, default=None, help="beta; default 5 for paper, 10 for lerobot")
    p.add_argument("--mask-end", type=int, default=None, help="RTC soft-mask end; default H - s")
    p.add_argument("--rtc-gate", choices=["none", "divergence", "random"], default="none", help="guide per request")
    p.add_argument("--rtc-tau", type=float, default=None, help="divergence: guide if D < tau (0: never)")
    p.add_argument("--rtc-share", type=float, default=None, help="random: probability to guide")
    p.add_argument("--budget", type=Path, default=None,
                   help="JSON config of openpi_budget.py: the step per option, naive switching; no --stride")
    p.add_argument("--horizon", type=int, default=HORIZON, help="H the server must return")
    p.add_argument("--resolution", type=int, default=RESOLUTION)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0, help="of the noise")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    # A chunk runs for s + d steps and the next inference must start after the previous switch: d <= s <= H - d.
    if a.budget is not None:
        if a.stride is not None or a.signal is not None or a.n_samples != 1 or a.guidance != "none":
            p.error("--budget chooses the step itself: no --stride, --signal, --n-samples or --guidance")
        a.budget_config = json.loads(a.budget.read_text())
    elif a.signal == "aac":
        if a.stride is not None or a.n_samples < 2:
            p.error("--signal aac takes no --stride and needs --n-samples > 1")
        if 2 * a.delay > a.horizon:
            p.error(f"need 2 d <= H = {a.horizon}, got d={a.delay}")
    elif a.stride is None or not max(a.delay, 1) <= a.stride <= a.horizon - a.delay:
        p.error(f"need max(d, 1) <= --stride <= H - d, got s={a.stride}, d={a.delay}, H={a.horizon}")
    if a.guidance == "none":
        if a.mask_end is not None:
            p.error("--mask-end needs --guidance")
    else:
        if a.signal is not None or a.n_samples != 1 or a.delay < 1:
            p.error("--guidance needs one sample per request, a constant --stride and --delay > 0")
        if a.mask_end is not None and not a.delay <= a.mask_end <= a.horizon - a.stride:
            p.error(f"need d <= --mask-end <= H - s, got {a.mask_end}, d={a.delay}, s={a.stride}, H={a.horizon}")
    if a.max_guidance_weight is None:
        a.max_guidance_weight = 5.0 if a.guidance == "paper" else 10.0
    if a.rtc_gate == "none":
        if a.rtc_tau is not None or a.rtc_share is not None:
            p.error("--rtc-tau and --rtc-share need --rtc-gate")
    elif a.guidance == "none":
        p.error("--rtc-gate needs --guidance")
    elif a.rtc_gate == "divergence" and (a.rtc_tau is None or a.rtc_tau < 0 or a.rtc_share is not None):
        p.error("--rtc-gate divergence needs --rtc-tau >= 0 and no --rtc-share")
    elif a.rtc_gate == "random" and (a.rtc_share is None or not 0 <= a.rtc_share <= 1 or a.rtc_tau is not None):
        p.error("--rtc-gate random needs --rtc-share in [0, 1] and no --rtc-tau")
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


def make_rtc_weights(schedule: str, max_weight: float, horizon: int):
    """(delay, mask_end) -> (H,) weights, computed by the code the LeRobot runs use."""
    from h1.guidance import BatchRTCProcessor, make_rtc_config

    processor = BatchRTCProcessor(make_rtc_config(schedule, max_weight, horizon), jacobian=False)

    def weights(delay: int, mask_end: int) -> np.ndarray:
        return processor.prefix_weights(delay, mask_end, 1, horizon, horizon)[0].numpy().astype(np.float32)

    return weights


def paper_weights(delay: int, mask_end: int, horizon: int, action_dim: int = ACTION_DIM,
                  model_dim: int = MODEL_ACTION_DIM) -> np.ndarray:
    """(H, A) weights of the RTC paper: 1 for i < d, c_i (e^c_i - 1) / (e - 1) for d <= i < end, then 0, where
    c_i = (end - i) / (end - d + 1) and end = H - s. The model's padding dimensions are not actions, so they stay at 0."""
    row = np.zeros(horizon, dtype=np.float32)
    row[:delay] = 1.0
    for i in range(delay, mask_end):
        c = (mask_end - i) / (mask_end - delay + 1)
        row[i] = c * (math.exp(c) - 1) / (math.e - 1)
    weights = np.zeros((horizon, model_dim), dtype=np.float32)
    weights[:, :action_dim] = row[:, None]
    return weights


@dataclass(frozen=True)
class RtcGate:
    """Whether to guide a request: divergence (D < tau) or random (probability share, seeded by the request's noise)."""

    mode: str
    overlap: int  # H - s: actions of the old plan the new chunk overlaps
    tau: float = 0.0
    share: float = 0.0
    seed: int = 0

    def divergence(self, plan: np.ndarray, old: np.ndarray) -> float:
        n = self.overlap
        return float(np.abs(plan[:n, :ACTION_DIM] - old[:n, :ACTION_DIM]).mean())

    def guide(self, divergence: float, noise: np.ndarray) -> bool:
        if self.mode == "divergence":
            return divergence < self.tau
        # the noise is fixed per (seed, episode, step): the same coin in every run, whatever the state
        u = np.random.default_rng([self.seed, zlib.crc32(np.ascontiguousarray(noise).tobytes())]).random()
        return bool(u < self.share)


def make_infer(client, prompt: str, horizon: int, n_samples: int = 1, aac: bool = False, rtc_weights=None,
               max_weight: float = 10.0, jacobian: bool = False, gate: RtcGate | None = None):
    """rtc_weights (paper_weights, make_rtc_weights): guide toward the old plan, which is then kept in model units.
    gate: first the unguided chunk, then guidance only where the gate says so."""

    def infer(observations, noise, leftover, delay, mask_end) -> Chunk:
        rows, plans, signal, divergence, guided, calls = [], [], [], [], [], []
        for i, (obs, row) in enumerate(zip(observations, noise)):
            element = make_element(obs, prompt) | {"noise": expand_noise(row[None], n_samples).numpy()}
            rtc = {}
            if leftover is not None:
                rtc = {"rtc_target": leftover[i].numpy(), "rtc_weights": rtc_weights(delay, int(mask_end[i])),
                       "rtc_max_weight": max_weight, "rtc_jacobian": jacobian}
            if rtc and gate is not None:
                free = client.infer(element)
                d_i = gate.divergence(np.asarray(free["normalized_action"], dtype=np.float32)[0], rtc["rtc_target"])
                use = gate.guide(d_i, element["noise"])
                reply = client.infer(element | rtc) if use else free
                divergence.append(d_i), guided.append(float(use)), calls.append(2.0 if use else 1.0)
            else:
                reply = client.infer(element | rtc)
                divergence.append(-1.0), guided.append(float(bool(rtc))), calls.append(1.0)
            actions = np.asarray(reply["actions"], dtype=np.float32)
            if actions.ndim == 2:  # openpi's stock server
                actions = actions[None]
            if actions.shape != (n_samples, horizon, ACTION_DIM):
                raise RuntimeError(f"server returned actions {actions.shape}, expected {(n_samples, horizon, ACTION_DIM)}")
            rows.append(actions[0])
            if rtc_weights is not None:
                plans.append(np.asarray(reply["normalized_action"], dtype=np.float32)[0])
            if aac:
                signal.append(paper_h_star(np.asarray(reply["normalized_action"])[..., :ACTION_DIM], actions))
        actions = np.stack(rows)
        model = np.stack(plans) if rtc_weights is not None else actions
        extras = {} if gate is None else {"divergence": np.array(divergence), "guided": np.array(guided),
                                          "calls": np.array(calls)}  # -1: first chunk of the episode, no old plan
        return Chunk(torch.from_numpy(model.copy()), actions, signal=np.array(signal) if aac else None, extras=extras)

    return infer


def probe_element() -> dict:
    image = np.zeros((RESOLUTION, RESOLUTION, 3), dtype=np.uint8)
    return {"observation/image": image, "observation/wrist_image": image, "observation/state": np.zeros(8),
            "prompt": "pick up the milk and place it in the basket"}


def check_guidance(client, rtc_weights, delay: int, horizon: int, max_weight: float, jacobian: bool = False,
                   action_dim: int = MODEL_ACTION_DIM) -> tuple[float, float]:
    """On the real model: zero weights must change nothing, and guidance must move the frozen actions toward the target.

    Only "toward": with the Jacobian the pull is much weaker than without it (RMSE 0.043 against 0.004 on LeRobot).
    """
    element = probe_element() | {"noise": np.random.default_rng(0).standard_normal((1, horizon, action_dim), dtype=np.float32)}
    free = np.asarray(client.infer(element)["normalized_action"])[0]
    target = free + GUIDANCE_PROBE_SHIFT

    def guided(weights):
        reply = client.infer(element | {"rtc_target": target, "rtc_weights": weights, "rtc_max_weight": max_weight,
                                        "rtc_jacobian": jacobian})
        return np.asarray(reply["normalized_action"])[0]

    unmasked = float(np.abs(guided(np.zeros(horizon, dtype=np.float32)) - free).max())
    if unmasked > SAMPLE_TOLERANCE:
        raise RuntimeError(f"RTC with zero weights differs from no RTC by {unmasked:.4f}")
    moved = guided(rtc_weights(delay, delay))[:delay, :ACTION_DIM]
    frozen = float(np.abs(moved - target[:delay, :ACTION_DIM]).mean())
    if frozen > GUIDANCE_PROBE_MAX * GUIDANCE_PROBE_SHIFT:
        raise RuntimeError(f"RTC leaves the frozen actions {frozen:.4f} from a target {GUIDANCE_PROBE_SHIFT} away")
    return unmasked, frozen


def check_samples(client, n_samples: int, horizon: int, action_dim: int = MODEL_ACTION_DIM) -> float:
    """The batch must behave as single requests: sample 0 as with its noise alone, and the samples must differ."""
    element = probe_element()
    noise = np.random.default_rng(0).standard_normal((n_samples, horizon, action_dim), dtype=np.float32)
    many = np.asarray(client.infer(element | {"noise": noise})["normalized_action"])
    one = np.asarray(client.infer(element | {"noise": noise[:1]})["normalized_action"])
    diff = float(np.abs(many[0] - one[0]).max())
    if diff > SAMPLE_TOLERANCE:
        raise RuntimeError(f"sample 0 of a batch differs from the same noise alone by {diff:.4f}")
    if float(many[..., :ACTION_DIM].std(0).max()) == 0.0:
        raise RuntimeError("the samples of a batch coincide: the noise is not used")
    return diff


def progress_key(args) -> dict:
    """What finished tasks of a suite must have in common with the run that resumes them."""
    names = ["stride", "signal", "n_samples", "delay", "horizon", "resolution", "seed", "episodes_per_task", "batch_size",
             "max_steps"]
    if args.guidance != "none":  # only with RTC: progress files of runs without it stay valid
        names += ["guidance", "schedule", "max_guidance_weight", "mask_end"]
    if args.rtc_gate != "none":
        names += ["rtc_gate", "rtc_tau", "rtc_share"]
    if getattr(args, "budget", None) is not None:
        names += ["budget_config"]  # its content, not the file name
    if getattr(args, "first_episode", 0):
        names += ["first_episode"]  # only when set: progress files of earlier runs stay valid
    return {k: getattr(args, k) for k in names}


def load_progress(path: Path, key: dict) -> tuple[list, list, list, float]:
    """Episodes, batches, chunks and seconds of the tasks a stopped run finished."""
    if not path.exists():
        return [], [], [], 0.0
    saved = json.loads(path.read_text())
    if saved["key"] != key:
        raise SystemExit(f"{path} was written with other settings: {saved['key']}")
    return saved["episodes"], saved["batches"], saved["chunks"], saved["seconds"]


def save_progress(path: Path, key: dict, episodes: list, batches: list, chunks: list, seconds: float) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"key": key, "episodes": episodes, "batches": batches, "chunks": chunks, "seconds": seconds}))
    os.replace(tmp, path)


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
    aac = args.signal == "aac"
    if args.n_samples > 1:
        log.info("sample 0 of %d against the same noise alone: max diff %.4f", args.n_samples,
                 check_samples(client, args.n_samples, args.horizon))
    rtc_weights, jacobian = None, args.guidance == "paper"
    if args.guidance != "none":
        rtc_weights = ((lambda delay, end: paper_weights(delay, end, args.horizon)) if jacobian
                       else make_rtc_weights(args.schedule, args.max_guidance_weight, args.horizon))
        log.info("RTC check: zero weights differ by %.4f, frozen actions %.4f from the target",
                 *check_guidance(client, rtc_weights, args.delay, args.horizon, args.max_guidance_weight, jacobian))
    gate = None
    if args.rtc_gate != "none":
        gate = RtcGate(args.rtc_gate, args.horizon - args.stride, args.rtc_tau or 0.0, args.rtc_share or 0.0, args.seed)
        log.info("RTC gate: %s", gate)
    runtime = None
    if args.budget is not None:
        from openpi_budget import BudgetRuntime, GivenStride, make_budget_infer

        runtime = BudgetRuntime(args.budget_config, args.seed, args.horizon, args.delay)
        log.info("budget: %s", args.budget_config)
        planner = GivenStride(args.horizon)
        if runtime.uses_rtc:  # RTC of the paper on the pairs with r >= 0
            log.info("RTC check: zero weights differ by %.4f, frozen actions %.4f from the target",
                     *check_guidance(client, lambda delay, end: paper_weights(delay, end, args.horizon), args.delay,
                                     args.horizon, 5.0, True))
    elif aac:
        planner = AdaptivePlanner(args.horizon, GivenHStar(), s_min=1)
    else:
        planner = FixedPlanner(args.horizon, args.stride, args.mask_end)
    guided = rtc_weights is not None or (runtime is not None and runtime.uses_rtc)  # the loop passes the old plan
    cfg = LoopConfig(args.horizon, args.delay, guided, args.max_steps, args.seed)
    env_cfg = LiberoEnvConfig(task=suites[0], episode_length=args.max_steps, observation_height=args.resolution,
                              observation_width=args.resolution)
    camera_names = parse_camera_names(env_cfg.camera_name)
    noise_shape = (args.horizon, MODEL_ACTION_DIM)

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
        progress = args.out / f"{name}.progress.json"  # a stopped suite resumes after its last finished task
        key = progress_key(args)
        episodes, batches, chunk_log, seconds = load_progress(progress, key)
        finished = {e["task_id"] for e in episodes}
        if finished:
            log.info("%s: resuming, tasks %s done in %.0f s", name, sorted(finished), seconds)
        started = time.time() - seconds
        for tid in task_ids:
            if tid in finished:
                continue
            task = suite.get_task(tid)
            if runtime is not None:
                infer = make_budget_infer(client, task.language, args.horizon, runtime, make_element,
                                          lambda delay, end: paper_weights(delay, end, args.horizon))
            else:
                infer = make_infer(client, task.language, args.horizon, args.n_samples, aac, rtc_weights,
                                   args.max_guidance_weight, jacobian, gate)
            last = args.first_episode + args.episodes_per_task
            for start in range(args.first_episode, last, args.batch_size):
                ids = list(range(start, min(start + args.batch_size, last)))
                envs = [LiberoEnv(task_suite=suite, task_id=tid, task_suite_name=name, camera_name=camera_names,
                                  init_states=True, episode_length=args.max_steps, episode_index=i, n_envs=1,
                                  **env_cfg.gym_kwargs) for i in ids]
                try:
                    res = run_batch(envs, infer, cfg, planner, [tid * 1000 + i for i in ids], noise_shape,
                                    keep_entropy=gate is not None or runtime is not None)  # extras -> chunk log
                finally:
                    for env in envs:
                        env.close()
                jump_switch, jump_other = switch_jumps(res.actions, res.switch_steps)
                for b, i in enumerate(ids):
                    strides = [c["stride"] for c in res.chunks if c["env"] == b]
                    hs = [c["h_star"] for c in res.chunks if c["env"] == b and c["h_star"] is not None]
                    episodes.append({
                        "task_id": tid, "init_state": i, "success": bool(res.success[b]),
                        "length": int(res.length[b]), "n_inference": int(res.n_inference[b]),
                        "sparc": episode_sparc(res.actions[b]),
                        "mean_stride": float(np.mean(strides)) if strides else None,
                        "mean_h_star": float(np.mean(hs)) if hs else None, "frac_sel": None,
                    })
                batches.append({"task_id": tid, "init_states": ids, "jump_at_switch": jump_switch,
                                "jump_other": jump_other})
                for c in res.chunks:
                    chunk_log.append({"task_id": tid, "init_state": ids[c["env"]]} | {k: v for k, v in c.items() if k != "env"})
                done = [e for e in episodes if e["task_id"] == tid]
                log.info("%s task %d: %d/%d success after %d episodes (%.0f s)", name, tid,
                         sum(e["success"] for e in done), len(done), len(done), time.time() - started)
            save_progress(progress, key, episodes, batches, chunk_log, time.time() - started)
        result = summarize(name, episodes, batches, task_ids)
        result["seconds"] = time.time() - started
        (args.out / f"{name}_chunks.json").write_text(json.dumps(chunk_log))
        out_file.write_text(json.dumps(result, indent=1))  # last: this file marks the suite as finished
        progress.unlink(missing_ok=True)
        log.info("%s: success %.1f%% over %d episodes", name, 100 * result["success_rate"], len(episodes))

    write_summary(args.out, suites)
    client.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # as eval_libero.py: EGL / MuJoCo teardown can hang
