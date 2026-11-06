"""Option budget inside the frozen pi0.5 server: a pair (s,r) is chosen before denoising and held for option_ticks.

Called after the prefix pass, before the first denoising step, so the decision sees the fresh prefix latent and the
old plan's residual, and one chunk is generated per request. Per-connection `state` holds the episode's history.
Modes: explore (uniform), epsilon (greedy + uniform), policy (greedy), marginal (state-independent action
frequencies), fixed (any constant pair, on or off the grid), rule (pair by the gripper phase of the robot state).
Random numbers depend on the request content only, not on the order in which workers reach a shared server.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from budget_features import HORIZON, actor_features, gray, gripper_width
from budget_privileged import N_OBJECTS, PRIV_DIM

MODES = ("explore", "epsilon", "policy", "marginal", "fixed", "rule")


def request_rng(seed, obs, step):
    h = hashlib.sha256()
    h.update(f"{seed}|{obs.get('budget_episode', '')}|{obs.get('prompt', '')}|{step}".encode())
    h.update(np.asarray(obs["observation/state"], np.float32).tobytes())
    return np.random.default_rng(int.from_bytes(h.digest()[:8], "little"))


def check_pair(pair, delay, horizon=HORIZON):
    s, r = pair
    if not max(1, delay) <= s <= horizon - delay or r < 0 or s + r + delay > horizon:
        raise ValueError(f"pair {pair} is not repeatable: require d <= s <= H-d and s+r+d <= H")


class OptionRuntime:
    def __init__(self, config_path, trace_dir):
        self.config = json.loads(Path(config_path).read_text())
        c = self.config
        if c["mode"] not in MODES:
            raise ValueError(f"unknown budget mode {c['mode']}")
        self.mode, self.delay, self.option_ticks = c["mode"], int(c["delay"]), int(c["option_ticks"])
        self.pairs = [tuple(p) for p in c["pairs"]]
        for pair in self.pairs:
            check_pair(pair, self.delay)
        self.seed, self.max_steps = int(c["seed"]), int(c["max_steps"])
        self.fixed = tuple(c["fixed"]) if c.get("fixed") else None
        self.rule = {k: tuple(c["rule"][k]) for k in ("open", "closed")} if c.get("rule") else None
        for pair in (self.fixed, *(self.rule.values() if self.rule else ())):
            if pair is not None:
                check_pair(pair, self.delay)
        self.width = float(c.get("rule_width", .07))
        self.epsilon = float(c.get("epsilon", 0.))
        self.marginal = np.asarray(c["marginal"], np.float64) if c.get("marginal") else None
        self.policy = None
        if self.mode in ("epsilon", "policy"):
            from budget_q import load_policy

            self.policy = load_policy(c["policy"])
            if self.policy.pairs != self.pairs:
                raise ValueError("policy grid differs from the configured pairs")
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def index(self, pair):
        return self.pairs.index(pair) if pair in self.pairs else -1

    def select(self, x, latent, obs, step):
        """-> (action index or -1, probability of the choice, pair)"""
        n = len(self.pairs)
        if self.mode == "fixed":
            return self.index(self.fixed), 1., self.fixed
        if self.mode == "rule":
            pair = self.rule["closed" if gripper_width(x) < self.width else "open"]
            return self.index(pair), 1., pair
        rng = request_rng(self.seed, obs, step)
        if self.mode == "explore":
            action = int(rng.integers(n))
            return action, 1. / n, self.pairs[action]
        if self.mode == "marginal":
            action = int(rng.choice(n, p=self.marginal))
            return action, float(self.marginal[action]), self.pairs[action]
        greedy = int(self.policy.greedy(x, latent)[0])
        if self.mode == "policy":
            return greedy, 1., self.pairs[greedy]
        action = int(rng.integers(n)) if rng.random() < self.epsilon else greedy
        return action, (1 - self.epsilon) * (action == greedy) + self.epsilon / n, self.pairs[action]

    def choose(self, obs, state, left, delay, horizon, residual, latent):
        started = time.perf_counter()
        if horizon != HORIZON or delay != self.delay:
            raise ValueError(f"budget H/d {HORIZON}/{self.delay} does not match the request {horizon}/{delay}")
        uid, step = str(obs["budget_id"]), int(obs["budget_step"])
        if not uid.isalnum():
            raise ValueError("invalid budget trace id")
        priv = np.asarray(obs["budget_priv"], np.float32).reshape(PRIV_DIM)
        if priv[N_OBJECTS] <= 0:
            raise ValueError("the environment exposes no object poses; the value target would be meaningless")
        left = np.asarray(left, np.float32).reshape(-1, 7)
        robot = np.asarray(obs["observation/state"], np.float32)
        frames = gray(obs["observation/image"]), gray(obs["observation/wrist_image"])
        decision = "option" not in state or step >= state["option"]["end"]
        if decision:
            x = actor_features(robot, state.get("budget_robot"), left, delay, step, self.max_steps, frames,
                               state.get("budget_frames"), residual, state.get("budget_pair", (0, 0)))
            action, prob, pair = self.select(x, latent, obs, step)
            state["option"] = {"pair": pair, "end": step + self.option_ticks}
            state["budget_pair"] = pair
            np.savez_compressed(self.trace_dir / f"{uid}.npz", x=x, priv=priv, action=action, prob=prob, step=step,
                                pair=np.asarray(pair), latent=np.asarray(latent, np.float16).ravel())
        s, r = state["option"]["pair"]
        # Only the first request of an episode has too short a plan: check_pair covers every later one.
        r = min(r, max(0, len(left) - delay))
        state.update(budget_frames=frames, budget_robot=robot.copy())
        return {"id": uid, "s": int(s), "r": int(r), "step": step, "decision": bool(decision),
                "budget_elapsed_s": time.perf_counter() - started}
