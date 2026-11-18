#!/usr/bin/env python
"""openpi's pi05_libero (H = 10) behind the websocket protocol of LIBERO-Dynamic. Runs in the venv of
libero_pro/third_party/openpi; run_dynamic.sh starts it with SERVER=openpi.

The same wire format and semantics as policy_server.py (our LeRobot pi0.5), which follows LIBERO-Dynamic's
scripts/serve_lerobot_pi05_openpi_ws.py:
    request:  observation/image, observation/wrist_image (HWC, rotated / resized by the client), observation/state,
              prompt; RTC also prev_chunk_left_over (unconsumed original actions, model units) and inference_delay_steps
    response: actions (environment units), original_actions (model units, the 32 dimensions of openpi), server_timing
The model is libero_pro/scripts/openpi_sample_server.SamplePolicy: openpi's own transforms and sampler, N samples per
request with a shared prefix, RTC of the paper (Eq. 2 with the Jacobian, beta = --max-guidance-weight, default 5).

* noise is deterministic per request (hash of state, prompt, leftover length and --seed), so configurations share it;
  sample 0 does not depend on N, so AAC and its constant control execute the same first sample;
* --mode rtc: with a leftover, guidance toward it with the weights of the RTC paper (1 for i < d, then exponential,
  0 from the mask end on); the mask ends at --rtc-execution-horizon (default --actions-per-chunk), cut to the leftover;
* --adaptive --rule aac_paper --n-samples N: AAC as in the authors' code (libero_pro/aac.py), the step
  s = clip(h* - d, s_min, s_max) set through the length s + d of the returned chunk (KEEP_CHUNK=1, the runner's
  --prefetch-threshold d); in --mode plain (sync) d = 0, in --mode rtc with --no-guidance naive switching;
* --option-budget CONFIG --budget-traces DIR: the option budget of budget_options.py (runner strategy
  learned_budget_queue). Before generation a pair (s, r) is chosen from deployable features and held for option_ticks;
  r sets the RTC mask of this generation (mask end d + r, r = -1: no guidance, naive), s the next request. The full
  H-chunk is returned. openpi has no prefix hook here, so the residual block is empty and no latent is logged.

    cd ../libero_pro/third_party/openpi && .venv/bin/python ../../../libero_dynamic/openpi_dynamic_server.py --port 8124
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIBERO_PRO = HERE.parent / "libero_pro"
sys.path.insert(0, str(LIBERO_PRO))
sys.path.insert(0, str(LIBERO_PRO / "scripts"))

ACTION_DIM = 7
EPISODES_KEPT = 64  # option state of recent episodes; workers run at most a few at once per server
log = logging.getLogger("openpi_dynamic_server")


def request_noise(obs: dict, seed: int, n: int, horizon: int, dim: int) -> np.ndarray:
    """(N, H, A) noise determined by the request content; row 0 is the same for every N."""
    h = hashlib.sha256()
    h.update(np.asarray(obs["observation/state"], dtype=np.float32).tobytes())
    h.update(str(obs.get("prompt", "")).encode())
    h.update(str(len(np.asarray(obs.get("prev_chunk_left_over", [])))).encode())
    h.update(str(seed).encode())
    rng = np.random.default_rng(int.from_bytes(h.digest()[:8], "little"))
    return rng.standard_normal((n, horizon, dim), dtype=np.float32)


def paper_weights(delay: int, mask_end: int, horizon: int) -> np.ndarray:
    """(H,) weights of the RTC paper, as libero_pro/scripts/eval_openpi.paper_weights: 1 for i < d,
    c (e^c - 1) / (e - 1) for d <= i < end with c = (end - i) / (end - d + 1), then 0."""
    row = np.zeros(horizon, dtype=np.float32)
    row[:delay] = 1.0
    for i in range(delay, mask_end):
        c = (mask_end - i) / (mask_end - delay + 1)
        row[i] = c * (math.exp(c) - 1) / (math.e - 1)
    return row


class DynamicPolicy:
    """policy.infer of openpi's WebsocketPolicyServer: one LIBERO-Dynamic request -> one chunk."""

    def __init__(self, sample_policy, horizon: int, action_dim: int, args):
        self.model, self.horizon, self.dim, self.args = sample_policy, horizon, action_dim, args
        self.apc = int(args.actions_per_chunk)
        self.exec_h = int(args.rtc_execution_horizon or self.apc)
        if not 1 <= self.apc <= horizon or not 1 <= self.exec_h <= horizon:
            raise ValueError(f"--actions-per-chunk and --rtc-execution-horizon must be in 1..H = {horizon}")
        self.log = open(args.log_jsonl, "a") if args.log_jsonl else None
        self.option_budget, self.episodes = None, OrderedDict()
        if args.option_budget:
            from budget_options import OptionRuntime

            self.option_budget = OptionRuntime(args.option_budget, args.budget_traces)
            if self.option_budget.horizon != horizon or self.apc != horizon:
                raise ValueError(f"the option budget needs H = {horizon} in its config and --actions-per-chunk {horizon}")

    @property
    def metadata(self) -> dict:
        a = self.args
        return {"policy": "openpi_pi05_libero", "horizon": self.horizon, "actions_per_chunk": self.apc,
                "rtc_conditioning_supported": a.mode == "rtc", "rtc_execution_horizon": self.exec_h,
                "max_guidance_weight": a.max_guidance_weight, "n_samples": a.n_samples, "adaptive": a.adaptive,
                "rule": a.rule, "no_guidance": a.no_guidance, "s_min": a.s_min, "s_max": a.s_max, "seed": a.seed,
                "option_budget": a.option_budget}

    def guidance(self, obs: dict, delay: int, mask_end: int | None = None) -> dict:
        """RTC fields of openpi_sample_server, or {} (no leftover, plain mode or --no-guidance).
        mask_end overrides --rtc-execution-horizon (the option budget's d + r)."""
        left = np.asarray(obs.get("prev_chunk_left_over", []), dtype=np.float32)
        if self.args.mode != "rtc" or self.args.no_guidance or left.size == 0:
            return {}
        left = left.reshape(-1, left.shape[-1])[: self.horizon]
        target = np.zeros((self.horizon, self.dim), dtype=np.float32)
        target[: len(left), : left.shape[1]] = left
        end = min(self.exec_h if mask_end is None else mask_end, len(left))
        weights = paper_weights(min(delay, end), end, self.horizon)
        return {"rtc_target": target, "rtc_weights": weights, "rtc_max_weight": self.args.max_guidance_weight,
                "rtc_jacobian": True}

    def step(self, normalized: np.ndarray, actions: np.ndarray, delay: int) -> tuple[int, int]:
        """AAC: (h*, s) with s = clip(h* - d, s_min, s_max); the chunk returned is s + d long."""
        from aac import paper_h_star

        a = self.args
        h = int(paper_h_star(normalized[..., :ACTION_DIM], actions))
        s_min = a.s_min if a.s_min is not None else max(delay, 1)
        s_max = max(min(a.s_max, self.horizon - delay), s_min)
        return h, int(np.clip(h - delay, s_min, s_max))

    def episode_state(self, obs: dict) -> dict:
        """Option state of this request's episode; the first request of an episode (no leftover) starts it afresh."""
        key = str(obs["budget_episode"])
        if len(np.asarray(obs.get("prev_chunk_left_over", []))) == 0 or key not in self.episodes:
            self.episodes[key] = {}
        self.episodes.move_to_end(key)
        while len(self.episodes) > EPISODES_KEPT:
            self.episodes.popitem(last=False)
        return self.episodes[key]

    def infer_option(self, obs: dict) -> dict:
        """One chunk whose (s, r) the option budget chose before generation; the full H-chunk is returned."""
        started = time.monotonic()
        delay = int(obs.get("inference_delay_steps", 0) or 0)
        raw = np.asarray(obs.get("prev_chunk_left_over", []), dtype=np.float32)
        raw = raw.reshape(-1, raw.shape[-1]) if raw.size else np.zeros((0, ACTION_DIM), np.float32)
        decision = self.option_budget.choose(obs, self.episode_state(obs), raw[:, :ACTION_DIM], delay, self.horizon,
                                             None, np.zeros(0, np.float32))
        element = {k: obs[k] for k in ("observation/image", "observation/wrist_image", "observation/state", "prompt")}
        element["noise"] = request_noise(obs, self.args.seed, 1, self.horizon, self.dim)
        rtc = {} if decision["r"] < 0 else self.guidance(obs, delay, delay + decision["r"])
        reply = self.model.infer(element | rtc)
        normalized = np.asarray(reply["normalized_action"], dtype=np.float32)
        actions = np.asarray(reply["actions"], dtype=np.float32)
        plan = {"guided": bool(rtc), "s": decision["s"], "r": decision["r"], "len": self.horizon}
        infer_ms = (time.monotonic() - started) * 1000.0
        if self.log is not None:
            self.log.write(json.dumps({"prompt": str(obs.get("prompt", "")), "delay": delay, "infer_ms": round(infer_ms, 2),
                                       "model_ms": round(float(reply["policy_timing"]["infer_ms"]), 2), **plan}) + "\n")
            self.log.flush()
        return {"actions": actions[0], "original_actions": normalized[0], "plan": plan, "budget": decision,
                "server_timing": {"infer_ms": infer_ms}}

    def infer(self, obs: dict) -> dict:
        if self.option_budget is not None:
            return self.infer_option(obs)
        a = self.args
        started = time.monotonic()
        delay = int(obs.get("inference_delay_steps", 0) or 0)
        element = {k: obs[k] for k in ("observation/image", "observation/wrist_image", "observation/state", "prompt")}
        element["noise"] = request_noise(obs, a.seed, a.n_samples, self.horizon, self.dim)
        rtc = self.guidance(obs, delay)
        reply = self.model.infer(element | rtc)
        normalized = np.asarray(reply["normalized_action"], dtype=np.float32)
        actions = np.asarray(reply["actions"], dtype=np.float32)
        length, plan = self.apc, {"guided": bool(rtc)}
        if a.adaptive:
            h, s = self.step(normalized, actions, delay)
            length = min(s + delay, self.horizon)
            plan.update(h_star=h, s=s)
        plan["len"] = length
        infer_ms = (time.monotonic() - started) * 1000.0
        if self.log is not None:
            self.log.write(json.dumps({"prompt": str(obs.get("prompt", "")), "delay": delay, "infer_ms": round(infer_ms, 2),
                                       "model_ms": round(float(reply["policy_timing"]["infer_ms"]), 2), **plan}) + "\n")
            self.log.flush()
        return {"actions": actions[0, :length], "original_actions": normalized[0, :length], "plan": plan,
                "server_timing": {"infer_ms": infer_ms}}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8124)
    p.add_argument("--mode", choices=["plain", "rtc"], default="plain")
    p.add_argument("--actions-per-chunk", type=int, default=10)
    p.add_argument("--rtc-execution-horizon", type=int, default=None)
    p.add_argument("--max-guidance-weight", type=float, default=5.0, help="beta; 5 as in the RTC paper")
    p.add_argument("--n-samples", type=int, default=1)
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--rule", choices=["aac_paper"], default="aac_paper")
    p.add_argument("--no-guidance", action="store_true", help="--mode rtc (the runner's queue) without RTC: naive")
    p.add_argument("--s-min", type=int, default=None)
    p.add_argument("--s-max", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-jsonl", default=None, help="append per-request timing and plan here")
    p.add_argument("--option-budget", default=None, help="JSON config of budget_options.OptionRuntime")
    p.add_argument("--budget-traces", default=None, help="directory of the option decisions (with --option-budget)")
    a = p.parse_args(argv)
    if a.adaptive and a.n_samples < 2:
        p.error("--adaptive needs --n-samples > 1")
    if a.adaptive and a.mode == "rtc" and not a.no_guidance:
        p.error("AAC is without RTC: --mode plain, or --mode rtc --no-guidance")
    if a.n_samples > 1 and not a.adaptive:
        p.error("--n-samples > 1 is only for --adaptive")
    if a.option_budget and (a.mode != "rtc" or a.adaptive or a.no_guidance or not a.budget_traces):
        p.error("--option-budget needs --mode rtc (the runner's queue), one sample, no --no-guidance and --budget-traces")
    return a


def main() -> None:
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config
    from openpi_sample_server import CHECKPOINT, CONFIG, SamplePolicy

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    args = parse_args()
    config = _config.get_config(CONFIG)
    policy = policy_config.create_trained_policy(config, CHECKPOINT)
    horizon, dim = config.model.action_horizon, config.model.action_dim
    dynamic = DynamicPolicy(SamplePolicy(policy, horizon, dim), horizon, dim, args)
    log.info("serving on %s:%d %s", args.host, args.port, dynamic.metadata)
    server = websocket_policy_server.WebsocketPolicyServer(policy=dynamic, host=args.host, port=args.port,
                                                           metadata=policy.metadata | dynamic.metadata)
    server.serve_forever()


if __name__ == "__main__":
    main()
