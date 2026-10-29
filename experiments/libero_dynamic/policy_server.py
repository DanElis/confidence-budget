#!/usr/bin/env python
"""Our Pi0.5 policy behind the OpenPI websocket protocol of LIBERO-Dynamic (runs in the libero_pro venv).

Wire format and semantics follow LIBERO-Dynamic's scripts/serve_lerobot_pi05_openpi_ws.py (@b66cb6b):
    request:  observation/image, observation/wrist_image (HWC, already rotated / resized by the client),
              observation/state, prompt; RTC also prev_chunk_left_over (unconsumed original actions, model space)
              and inference_delay_steps
    response: actions (environment space), original_actions (model space), server_timing; plus "plan" (ours)
    first message after connect: metadata with rtc_conditioning_supported
The returned chunk is truncated to --actions-per-chunk (their default 10); the RTC mask ends at
--rtc-execution-horizon (their default: actions_per_chunk). Guidance is LeRobot's (no Jacobian), as theirs.

Differences from their server, all opt-in except the noise:
* noise is deterministic per request (hash of state, prompt, leftover length and --seed), so configurations
  share it while their trajectories coincide (paired comparisons);
* --n-samples K > 1 (RTC only): K unguided + K guided rows in one call with a shared prefix
  (libero_pro/selection.py, shared_prefix.py): --select coherent|medoid, --mask-mode rtc_x_conf|conf;
* --mask-mode transition (RTC, one sample): before denoising, the old plan is checked under the new observation
  (one extra expert pass on the cached prefix, selection.plan_residual) and the RTC mask becomes an exponential
  transition whose length grows with the old plan's deviation near the stitch (selection.transition_weights):
  a far-off new plan is blended in slowly, a near one quickly. --trans-power 0 is the constant-length control.
  The old plan is the full chunk kept on the server (the client's leftover is only the returned part).

    uv run --project ../libero_pro --with websockets --with msgpack python policy_server.py --port 8124 --mode rtc
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LIBERO_PRO = HERE.parent / "libero_pro"
for p in (LIBERO_PRO, LIBERO_PRO / "scripts", HERE.parent / "h1"):
    sys.path.insert(0, str(p))

from selection import (  # noqa: E402
    MASK_MODES,
    SELECTS,
    agreement,
    conf_weights,
    plan_residual,
    transition_weights,
    dispersion,
    expand_noise,
    joint_batch,
    select_index,
    split_joint,
)

log = logging.getLogger("policy_server")
ACTION_DIM = 7


# ---------------------------------------------------------------- wire format (same as LIBERO-Dynamic / openpi)

def pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj: dict) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def as_image(value: Any) -> torch.Tensor:
    array = np.array(value, copy=True)
    if array.ndim != 3:
        raise ValueError(f"image must be HWC or CHW, got shape {array.shape}")
    t = torch.as_tensor(array, dtype=torch.float32)
    if t.numel() and float(t.max()) > 1.5:
        t = t / 255.0
    if t.shape[-1] == 3:
        t = t.permute(2, 0, 1)
    return t.contiguous()


def request_noise(obs: dict, seed: int, horizon: int, a_max: int) -> torch.Tensor:
    """(1, H, A_max) noise determined by the request content, so runs share it while trajectories coincide."""
    h = hashlib.sha256()
    h.update(np.asarray(obs["observation/state"], dtype=np.float32).tobytes())
    h.update(str(obs.get("prompt", "")).encode())
    h.update(str(len(np.asarray(obs.get("prev_chunk_left_over", [])))).encode())
    h.update(str(seed).encode())
    g = torch.Generator().manual_seed(int.from_bytes(h.digest()[:8], "little") % (2**63))
    return torch.randn(1, horizon, a_max, generator=g)


def leftover_tensor(obs: dict, horizon: int) -> tuple[torch.Tensor | None, int]:
    left = np.asarray(obs.get("prev_chunk_left_over", []), dtype=np.float32)
    if left.size == 0:
        return None, 0
    left = left.reshape(-1, left.shape[-1])[:horizon]
    out = torch.zeros(1, horizon, left.shape[-1])
    out[0, : len(left)] = torch.as_tensor(left)
    return out, len(left)


def aac_h_star(disp: np.ndarray, min_h: int = 1) -> int:
    """AAC [13] Eq. 5 on a per-action dispersion profile: h* = max(argmax_h (D_{h+1} - D_h), min_h)."""
    horizon = len(disp)
    prefix_mean = np.cumsum(disp) / np.arange(1, horizon + 1)
    h = int(np.argmax(np.diff(prefix_mean))) + 1  # diff index j is D_{j+2} - D_{j+1}
    return int(min(max(h, min_h), horizon))


# ---------------------------------------------------------------- policy

class Server:
    def __init__(self, args):
        from eval_libero import load_policy
        from h1.guidance import BatchRTCProcessor, install_batch_rtc, make_rtc_config
        from lerobot.policies import make_pre_post_processors

        self.args = args
        self.policy = load_policy(args.policy, args.device)
        self.horizon = int(self.policy.config.chunk_size)
        self.a_max = int(self.policy.config.max_action_dim)
        self.apc = int(args.actions_per_chunk)
        self.exec_h = int(args.rtc_execution_horizon or self.apc)
        rtc_config = make_rtc_config(args.schedule, args.max_guidance_weight, self.exec_h)
        self.rtc = install_batch_rtc(self.policy, "lerobot", rtc_config) if args.mode == "rtc" else None
        self.weights_proc = BatchRTCProcessor(rtc_config, jacobian=False)
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config, pretrained_path=args.policy,
            preprocessor_overrides={"device_processor": {"device": args.device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}})
        from hstar import HStarRule

        self.rule = HStarRule("above", eta=args.eta, k=1, min_h=1)
        self.log = open(args.log_jsonl, "a") if args.log_jsonl else None

    @property
    def metadata(self) -> dict:
        a = self.args
        return {"policy": "confidence_budget_pi05", "checkpoint": a.policy, "actions_per_chunk": self.apc,
                "rtc_conditioning_supported": a.mode == "rtc", "rtc_execution_horizon": self.exec_h,
                "n_samples": a.n_samples, "select": a.select, "mask_mode": a.mask_mode,
                "conf_sigma": a.conf_sigma, "conf_ref": a.conf_ref, "seed": a.seed, "adaptive": a.adaptive,
                "rule": a.rule, "eta": a.eta, "aac_min_h": a.aac_min_h, "s_min": a.s_min, "s_max": a.s_max,
                "trans_len": a.trans_len, "trans_ref": a.trans_ref, "trans_power": a.trans_power,
                "trans_min": a.trans_min, "trans_max": a.trans_max, "trans_window": a.trans_window,
                "fresh_t": a.fresh_t}

    def batch(self, obs: dict) -> dict:
        return self.pre({"observation.images.image": as_image(obs["observation/image"]),
                         "observation.images.image2": as_image(obs["observation/wrist_image"]),
                         "observation.state": torch.as_tensor(np.array(obs["observation/state"], dtype=np.float32)),
                         "task": str(obs.get("prompt", ""))})

    def infer(self, obs: dict, state: dict) -> dict:
        a, dev = self.args, self.args.device
        batch = self.batch(obs)
        noise = request_noise(obs, a.seed, self.horizon, self.a_max)
        left, left_len = leftover_tensor(obs, self.horizon) if self.rtc is not None else (None, 0)
        delay = int(obs.get("inference_delay_steps", 0) or 0)
        mask_end = torch.tensor([self.exec_h])
        plan: dict[str, Any] = {"sel": 0}
        with torch.no_grad():
            if a.mask_mode == "transition":
                chunk, plan = self.infer_transition(batch, noise, left, left_len, delay, mask_end, state)
            elif a.n_samples == 1:
                kw = {} if left is None else dict(inference_delay=delay, prev_chunk_left_over=left.to(dev),
                                                   execution_horizon=mask_end)
                chunk = self.policy.predict_action_chunk(batch, noise=noise.to(dev), **kw).float()
            else:
                chunk, plan = self.infer_samples(batch, noise, left, left_len, delay, mask_end, state)
        chunk = chunk[:, : int(plan.get("len", self.apc))]
        original = chunk[0].detach().cpu().numpy().astype(np.float32)
        actions = torch.stack([self.post(chunk[:, i, :]) for i in range(chunk.shape[1])], dim=1)
        return {"actions": actions[0].detach().cpu().numpy().astype(np.float32), "original_actions": original,
                "plan": plan}

    def old_plan(self, left, left_len, state):
        """The client's leftover is the unconsumed tail of the chunk we returned (length state["len"]); the old plan
        beyond it is in the full chunk kept here. -> (leftover (1, H, A) padded, its length, start in the full chunk)"""
        start = max(self.apc - left_len, 0)
        if left is None or "full" not in state:
            return left, left_len, start
        start = max(int(state["len"]) - left_len, 0)
        rest = state["full"][start:]
        out = torch.zeros(1, self.horizon, rest.shape[-1])
        out[0, : len(rest)] = rest.cpu()
        return out, len(rest), start

    def infer_transition(self, batch, noise, left, left_len, delay, mask_end, state):
        """RTC with the transition mask: check of the old plan under this observation, then guided denoising."""
        from shared_prefix import predict_chunk_shared

        a, dev = self.args, self.args.device
        left, left_len, _ = self.old_plan(left, left_len, state)
        plan: dict[str, Any] = {"sel": 0, "len": self.apc}
        if left is None:
            chunk = predict_chunk_shared(self.policy, batch, 1, noise=noise.to(dev)).float()
        else:
            info: dict[str, Any] = {}

            def set_weights(velocity):
                r = plan_residual(velocity, left[..., :ACTION_DIM].to(dev), noise.to(dev), a.fresh_t)[0].cpu()
                w, length, delta = transition_weights(r, delay, left_len, self.horizon, a.trans_len, a.trans_ref,
                                                      a.trans_power, a.trans_min, a.trans_max, a.trans_window)
                self.rtc.weight_override = w[None].to(dev)
                info.update(r=r[:left_len].numpy(), L=length, delta=delta)

            try:
                chunk = predict_chunk_shared(self.policy, batch, 1, noise=noise.to(dev), inference_delay=delay,
                                             prev_chunk_left_over=left.to(dev), execution_horizon=mask_end,
                                             before_integrate=set_weights).float()
            finally:
                self.rtc.weight_override = None
            plan.update(L=round(info["L"], 3), delta=None if np.isnan(info["delta"]) else round(info["delta"], 5))
            if self.log is not None:
                self.log.write(json.dumps({"prompt": state.get("prompt"), "req": state.get("req"), "delay": delay,
                                           "left_len": left_len, **plan,
                                           "r": np.round(info["r"], 5).tolist()}) + "
")
                self.log.flush()
        state.update(full=chunk[0].detach().cpu(), len=self.apc)
        return chunk, plan

    def infer_samples(self, batch, noise, left, left_len, delay, mask_end, state):
        """K samples in one call. With --adaptive also the replanning step from the agreement of the old plan with
        the new samples; the step is realised through the length of the returned chunk (s + d, see plan_step)."""
        from shared_prefix import predict_chunk_shared

        a, dev, k = self.args, self.args.device, self.args.n_samples
        noise_k = expand_noise(noise, k)
        disp_start = max(self.apc - left_len, 0)
        if a.adaptive:  # the old plan beyond the returned s + d actions (soft mask, agreement)
            left, left_len, disp_start = self.old_plan(left, left_len, state)
        z = None
        if left is None:  # first chunk of an episode: K unguided samples, sample 0 executed
            samples = predict_chunk_shared(self.policy, batch, k, noise=noise_k.to(dev)).float().reshape(1, k, self.horizon, -1)
            real = samples[..., :ACTION_DIM]
            chosen, sel = samples[:, 0], 0
        else:
            # dispersion of the previous chunk aligned with the leftover
            left_disp = None
            prev = state.get("disp")
            if prev is not None and a.mask_mode != "rtc":
                seg = prev[disp_start: disp_start + left_len]
                left_disp = torch.zeros(1, self.horizon)
                left_disp[0, : len(seg)] = torch.as_tensor(seg)
                left_disp = left_disp.to(dev)
            rtc_w = self.weights_proc.prefix_weights(delay, mask_end, 1, self.horizon, self.horizon).to(dev)
            w = conf_weights(a.mask_mode, rtc_w, left_disp, delay, mask_end, a.conf_sigma * a.conf_ref)
            noise2, left2, mask2, w2 = joint_batch(noise_k, left, mask_end, w, k)
            self.rtc.weight_override = w2.to(dev)
            try:
                out = predict_chunk_shared(self.policy, batch, 2 * k, noise=noise2.to(dev), inference_delay=delay,
                                           prev_chunk_left_over=left2.to(dev), execution_horizon=mask2)
            finally:
                self.rtc.weight_override = None
            unguided, guided = split_joint(out.float(), 1, k)
            real = unguided[..., :ACTION_DIM]
            sel = int(select_index(a.select, real, left[..., :ACTION_DIM].to(dev), w)[0])
            z = agreement(real, left[..., :ACTION_DIM].to(dev), torch.tensor([left_len]))[0].numpy()
            chosen = guided[:, sel]
        disp = dispersion(real)[0].cpu().numpy()
        plan: dict[str, Any] = {"sel": sel}
        length = self.apc
        if a.adaptive:
            h, s = self.plan_step(z, delay, disp)
            length = s + delay
            plan.update(h_star=h, s=s)
        plan["len"] = length
        state.update(full=chosen[0].detach(), len=length, disp=disp)
        if self.log is not None:
            self.log.write(json.dumps({"prompt": state.get("prompt"), "req": state.get("req"), "delay": delay,
                                       "left_len": left_len, **plan,
                                       "z": None if z is None else np.round(z, 3).tolist(),
                                       "disp": np.round(disp, 5).tolist()}) + "\n")
            self.log.flush()
        return chosen, plan

    def plan_step(self, z: np.ndarray | None, delay: int, disp: np.ndarray | None = None) -> tuple[int, int]:
        """Reliable horizon h* and replanning step s = clip(h* - d, s_min, s_max).

        --rule agree: h* = first index where the old plan leaves the cloud of new samples (z > eta), H if never;
            the first chunk of an episode (no old plan) gets s_min.
        --rule aac:   h* = argmax_h (D_{h+1} - D_h), D_h = mean dispersion of the new samples over the first h
            actions (AAC [13], Eq. 5, with the dispersion in place of the action entropy), at least --aac-min-h.

        With the runner's --prefetch-threshold d, a chunk of length s + d is executed for s steps before the next
        request (its last d actions cover the next inference), so s is the replanning step."""
        a = self.args
        s_min = a.s_min if a.s_min is not None else max(delay, 1)
        s_max = max(min(a.s_max, self.horizon - delay), s_min)
        if a.rule == "aac":
            if disp is None:
                return self.horizon, s_min
            h = aac_h_star(np.asarray(disp, dtype=np.float64), a.aac_min_h)
        else:
            if z is None:
                return self.horizon, s_min
            h = int(self.rule(z, 0))
        return h, int(np.clip(h - delay, s_min, s_max))


# ---------------------------------------------------------------- websocket

async def serve(server: Server, host: str, port: int) -> None:
    import msgpack
    import websockets

    packer = msgpack.Packer(default=pack_array)

    async def handler(ws):
        state: dict = {}  # per connection: the previous chunk's dispersion
        await ws.send(packer.pack(server.metadata))
        while True:
            try:
                obs = msgpack.unpackb(await ws.recv(), object_hook=unpack_array)
                if len(np.asarray(obs.get("prev_chunk_left_over", []))) == 0:
                    state.clear()  # new episode (or a non-RTC request)
                    state.update(prompt=str(obs.get("prompt", "")), req=0)
                state["req"] = state.get("req", 0) + 1
                t0 = time.monotonic()
                result = server.infer(obs, state)
                result["server_timing"] = {"infer_ms": (time.monotonic() - t0) * 1000.0}
                await ws.send(packer.pack(result))
            except websockets.ConnectionClosed:
                break
            except Exception:
                await ws.send(traceback.format_exc())
                await ws.close(code=1011, reason="Internal server error")
                raise

    async with websockets.serve(handler, host, port, compression=None, max_size=None):
        await asyncio.Future()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="lerobot/pi05_libero_finetuned")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8124)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mode", choices=["plain", "rtc"], default="plain")
    p.add_argument("--actions-per-chunk", type=int, default=10)
    p.add_argument("--rtc-execution-horizon", type=int, default=None)
    p.add_argument("--max-guidance-weight", type=float, default=10.0)
    p.add_argument("--schedule", default="exp")
    p.add_argument("--n-samples", type=int, default=1)
    p.add_argument("--select", choices=SELECTS, default="first")
    p.add_argument("--mask-mode", choices=(*MASK_MODES, "transition"), default="rtc")
    p.add_argument("--conf-sigma", type=float, default=1.0)
    p.add_argument("--conf-ref", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--adaptive", action="store_true",
                   help="replanning step from plan agreement; run the runner with --prefetch-threshold d")
    p.add_argument("--rule", choices=["agree", "aac"], default="agree",
                   help="adaptive: h* from plan agreement (z > eta) or from the growth of the samples' dispersion (AAC)")
    p.add_argument("--eta", type=float, default=3.0, help="adaptive, agree: z threshold")
    p.add_argument("--aac-min-h", type=int, default=1, help="adaptive, aac: lower bound of h*")
    p.add_argument("--s-min", type=int, default=None, help="adaptive: lower bound of s (default d)")
    p.add_argument("--s-max", type=int, default=20, help="adaptive: upper bound of s")
    p.add_argument("--trans-len", type=float, default=6.0,
                   help="transition: L0, the transition length at delta = --trans-ref")
    p.add_argument("--trans-ref", type=float, default=1.0,
                   help="transition: reference deviation (median delta of a --trans-power 0 run, report_dynamic.py 'plan')")
    p.add_argument("--trans-power", type=float, default=1.0, help="transition: L ~ delta^p; 0 = constant L0")
    p.add_argument("--trans-min", type=float, default=1.0, help="transition: lower bound of L")
    p.add_argument("--trans-max", type=float, default=30.0, help="transition: upper bound of L")
    p.add_argument("--trans-window", type=int, default=5,
                   help="transition: delta = RMS residual of the old plan over [d, d + window)")
    p.add_argument("--fresh-t", type=float, default=0.5,
                   help="transition: flow time of the old-plan check (LeRobot: 1 = noise, 0 = clean)")
    p.add_argument("--log-jsonl", default=None, help="append per-request signals (z, dispersion, plan) here")
    args = p.parse_args(argv)
    if args.adaptive and args.n_samples < 2:
        p.error("--adaptive needs --n-samples > 1 and --mode rtc")
    if args.n_samples > 1 and args.mode != "rtc":
        p.error("--n-samples > 1 needs --mode rtc (selection and the mask use the old plan)")
    if args.mask_mode == "transition" and (args.mode != "rtc" or args.n_samples != 1 or args.adaptive):
        p.error("--mask-mode transition needs --mode rtc and one sample")
    if not 0 < args.fresh_t < 1:
        p.error("--fresh-t must be in (0, 1)")
    if args.n_samples == 1 and (args.select != "first" or args.mask_mode not in ("rtc", "transition")):
        p.error("--select / --mask-mode need --n-samples > 1")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args()
    server = Server(args)
    log.info("serving on %s:%d %s", args.host, args.port, server.metadata)
    asyncio.run(serve(server, args.host, args.port))


if __name__ == "__main__":
    main()
