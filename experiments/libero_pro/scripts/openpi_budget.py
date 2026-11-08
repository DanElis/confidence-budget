"""Budget of the replanning step and the RTC window on openpi (H = 10): eval_openpi.py --budget CONFIG.

Port of the option budget of experiments/libero_dynamic (budget_v2.py) to LIBERO-Pro. An action is a pair (s, r): the
step s, and the RTC window r of the next chunks (RTC of the paper with the Jacobian, the first d actions frozen, soft
weights up to d + r, beta = 5); r = -1 is naive switching, without guidance. A config with "grid" (a list of s) is the
naive budget of 07.11.2026, every r = -1; one with "pairs" ([[s, r], ...]) chooses both. The pair is chosen right after
a chunk is generated (the decision sees the new chunk) and held for an option of option_ticks control steps, as in v2.
The r chosen with chunk k applies from chunk k + 1 on: when chunk k + 1 is generated, the old plan is chunk k, and
d + r <= H - s keeps the window inside it.

Modes: explore (uniform over the pairs, logged propensity), fixed (a constant pair, on or off the grid), marginal (given
frequencies, independent of the state: the control at equal frequencies), policy (greedy Q of openpi_budget_fit.py,
an ensemble of budget_q.QNet). Random numbers depend on (seed, episode, step) only, never on the order of requests.

Features (FEATURES, all deployable on the robot): robot state, episode time, the new chunk (first action, its change
over the chunk, first gripper flip), its disagreement with the old plan over their overlap, camera motion since the
previous request, the previous s.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from planner import Plan  # noqa: E402

GRID = (2, 4, 6, 8)
NAIVE = -1  # r of a pair without guidance
MODES = ("explore", "fixed", "marginal", "policy")
ACTION_DIM = 7
GRIPPER = 6
RTC_MAX_WEIGHT = 5.0  # beta of the RTC paper
FEATURES = ([f"state_{i}" for i in range(8)] + ["time"] + [f"first_{i}" for i in range(ACTION_DIM)]
            + [f"change_{i}" for i in range(ACTION_DIM)]
            + ["gripper_flip", "has_old", "divergence", "divergence_gripper", "motion_agent", "motion_wrist", "prev_s"])
FEATURE_DIM = len(FEATURES)


def gray(image) -> np.ndarray:
    """Coarse grayscale in [0, 1]: enough for how much the scene moved between two requests."""
    return np.asarray(image)[::8, ::8].mean(-1, dtype=np.float32) / 255.0


def gripper_flip(actions: np.ndarray) -> float:
    """Index of the first sign change of the gripper command in the chunk, / H; 1 if there is none."""
    g = np.sign(np.asarray(actions)[:, GRIPPER])
    change = np.flatnonzero(g != g[0])
    return float(change[0]) / len(g) if len(change) else 1.0


def features(state, plan, actions, old_rest, step, max_steps, frames, prev_frames, prev_s, horizon) -> np.ndarray:
    """plan (H, >=7) model units, actions (H, 7) environment units, old_rest: the old plan from this step on (model
    units) or None at the first request; frames / prev_frames: (agent, wrist) gray images or None."""
    p = np.asarray(plan, np.float32)[:, :ACTION_DIM]
    if old_rest is None:
        old = [0.0, 0.0, 0.0]
    else:
        n = min(len(old_rest), len(p))
        gap = np.abs(p[:n] - np.asarray(old_rest, np.float32)[:n, :ACTION_DIM])
        old = [1.0, float(gap.mean()), float(gap[:, GRIPPER].mean())]
    if prev_frames is None:
        motion = [0.0, 0.0]
    else:
        motion = [float(np.abs(a - b).mean()) for a, b in zip(frames, prev_frames)]
    x = np.concatenate((np.asarray(state, np.float32).reshape(8), [step / max_steps], p[0], p[-1] - p[0],
                        [gripper_flip(actions)], old, motion, [prev_s / horizon]))
    return x.astype(np.float32)


def check_stride(s: int, delay: int, horizon: int) -> None:
    if not max(delay, 1) <= s <= horizon - delay:
        raise ValueError(f"step {s} is not feasible: need max(d, 1) <= s <= H - d, d={delay}, H={horizon}")


def check_pair(pair, delay: int, horizon: int) -> None:
    s, r = pair
    check_stride(s, delay, horizon)
    if r != NAIVE and (r < 0 or delay < 1 or delay + r > horizon - s):
        raise ValueError(f"pair {pair} is not feasible: RTC needs d >= 1 and d + r <= H - s, d={delay}, H={horizon}")


def parse_pair(text) -> tuple[int, int]:
    """'s' or 's:r' or [s, r] -> (s, r); a bare s is naive."""
    if isinstance(text, (list, tuple)):
        return int(text[0]), int(text[1])
    s, _, r = str(text).partition(":")
    return int(s), int(r) if r else NAIVE


def config_pairs(config: dict) -> list:
    if "pairs" in config:
        return [parse_pair(p) for p in config["pairs"]]
    return [(int(s), NAIVE) for s in config.get("grid", GRID)]


def normalize_pairs(pairs) -> list:
    """Policies fitted before the RTC pairs (07.11.2026) saved the naive grid as (s, 0)."""
    pairs = [(int(s), int(r)) for s, r in pairs]
    return [(s, NAIVE) for s, _ in pairs] if all(r == 0 for _, r in pairs) else pairs


@dataclass(frozen=True)
class GivenStride:
    """The pair chosen by the infer function (the budget runtime) comes as the chunk's signal: s, or (s, r).
    mask_end of the next generation is d + r, or -1 for naive (no guidance)."""

    horizon: int

    def __call__(self, signal, n_valid, delay) -> Plan:
        values = np.asarray(signal).ravel()
        s = int(values[0])
        r = int(values[1]) if len(values) > 1 else NAIVE
        return Plan(stride=s, mask_end=delay + r if r != NAIVE else -1)

    @property
    def needs_entropy(self) -> bool:
        return False


def load_qpolicy(path):
    sys.path.insert(0, str(ROOT.parent / "libero_dynamic"))
    from budget_q import load_policy

    policy = load_policy(path)
    policy.pairs = normalize_pairs(policy.pairs)
    return policy


class BudgetRuntime:
    """Per-episode options. decide() is called once per generated chunk, in step order within an episode."""

    def __init__(self, config: dict, seed: int, horizon: int, delay: int):
        self.config = config
        self.mode = config["mode"]
        if self.mode not in MODES:
            raise ValueError(f"unknown budget mode {self.mode}")
        self.pairs = config_pairs(config)
        self.option_ticks, self.max_steps = int(config["option_ticks"]), int(config["max_steps"])
        self.horizon, self.delay, self.seed = horizon, delay, int(seed)
        for pair in self.pairs:
            check_pair(pair, delay, horizon)
        self.fixed = None
        if self.mode == "fixed":
            self.fixed = parse_pair(config["fixed"])
            check_pair(self.fixed, delay, horizon)
        self.marginal = None
        if self.mode == "marginal":
            self.marginal = np.asarray(config["marginal"], np.float64)
            if len(self.marginal) != len(self.pairs) or abs(self.marginal.sum() - 1) > 1e-6 or (self.marginal < 0).any():
                raise ValueError("marginal: one probability per grid pair, summing to 1")
        self.policy = None
        if self.mode == "policy":
            self.policy = load_qpolicy(config["policy"])
            if self.policy.pairs != self.pairs:
                raise ValueError(f"policy grid {self.policy.pairs} differs from {self.pairs}")
        self.episodes: dict = {}

    @property
    def uses_rtc(self) -> bool:
        pairs = self.pairs + ([self.fixed] if self.fixed is not None else [])
        return any(r != NAIVE for _, r in pairs)

    def select(self, x, episode: int, step: int) -> tuple[int, float, tuple[int, int]]:
        """-> (grid index or -1, probability of the choice, (s, r))"""
        n = len(self.pairs)
        if self.mode == "fixed":
            index = self.pairs.index(self.fixed) if self.fixed in self.pairs else -1
            return index, 1.0, self.fixed
        rng = np.random.default_rng([self.seed, int(episode), int(step)])
        if self.mode == "explore":
            a = int(rng.integers(n))
            return a, 1.0 / n, self.pairs[a]
        if self.mode == "marginal":
            a = int(rng.choice(n, p=self.marginal))
            return a, float(self.marginal[a]), self.pairs[a]
        a = int(self.policy.greedy(x)[0])
        return a, 1.0, self.pairs[a]

    def decide(self, episode: int, step: int, element: dict, plan, actions) -> dict:
        state = self.episodes.get(episode)
        if step == 0 or state is None:
            state = self.episodes[episode] = {}
        frames = (gray(element["observation/image"]), gray(element["observation/wrist_image"]))
        old_rest = None
        if "plan" in state:  # the active chunk is the previous request's: it took over d <= s steps after it
            old_rest = state["plan"][step - state["step"]:]
        x = features(element["observation/state"], plan, actions, old_rest, step, self.max_steps, frames,
                     state.get("frames"), state.get("s", 0), self.horizon)
        decision = "end" not in state or step >= state["end"]
        if decision:
            a, prob, (s, r) = self.select(x, episode, step)
            state.update(action=a, prob=prob, s=s, r=r, end=step + self.option_ticks)
        state.update(plan=np.asarray(plan, np.float32), step=step, frames=frames)
        return {"x": x, "action": state["action"], "prob": state["prob"], "s": state["s"], "r": state["r"],
                "decision": float(decision)}


def make_budget_infer(client, prompt: str, horizon: int, runtime: BudgetRuntime, make_element=None, rtc_weights=None,
                      max_weight: float = RTC_MAX_WEIGHT):
    """One request per environment to openpi_sample_server.py (it returns the plan in model units). With an RTC pair,
    the old plan (the loop's leftover, model units) is the target, weights rtc_weights(d, mask_end), the Jacobian on.
    make_element / rtc_weights: eval_openpi.make_element / paper_weights (passed in when eval_openpi runs as __main__)."""
    import torch

    from closed_loop import Chunk
    from selection import expand_noise

    if make_element is None:
        from eval_openpi import make_element
    if rtc_weights is None:
        from eval_openpi import paper_weights

        def rtc_weights(delay, mask_end):
            return paper_weights(delay, mask_end, horizon)

    def infer(observations, noise, leftover, delay, mask_end, episodes, step) -> Chunk:
        rows, plans, records = [], [], []
        for i, (obs, row, episode) in enumerate(zip(observations, noise, episodes)):
            element = make_element(obs, prompt) | {"noise": expand_noise(row[None], 1).numpy()}
            end = -1 if mask_end is None else int(mask_end[i])
            if leftover is not None and delay > 0 and end >= delay:  # the previous option chose an RTC window
                element |= {"rtc_target": leftover[i].numpy(), "rtc_weights": rtc_weights(delay, end),
                            "rtc_max_weight": max_weight, "rtc_jacobian": True}
            reply = client.infer(element)
            if "normalized_action" not in reply:
                raise RuntimeError("the budget needs the plan in model units: serve with scripts/openpi_sample_server.py")
            actions = np.asarray(reply["actions"], dtype=np.float32)
            if actions.ndim == 2:
                actions = actions[None]
            if actions.shape != (1, horizon, ACTION_DIM):
                raise RuntimeError(f"server returned actions {actions.shape}, expected {(1, horizon, ACTION_DIM)}")
            plan = np.asarray(reply["normalized_action"], dtype=np.float32)[0]
            record = runtime.decide(int(episode), step, element, plan, actions[0])  # episode: task * 1000 + init state
            record["guided"] = float("rtc_target" in element)
            records.append(record)
            rows.append(actions[0]), plans.append(plan)
        extras = {f"budget_{k}": np.stack([np.asarray(r[k], np.float32) for r in records])
                  for k in ("x", "action", "prob", "decision", "r", "guided")}
        return Chunk(torch.from_numpy(np.stack(plans)), np.stack(rows),
                     signal=np.array([[r["s"], r["r"]] for r in records]), extras=extras)

    infer.wants_context = True
    return infer


def write_config(path: Path, **config) -> None:
    Path(path).write_text(json.dumps(config, indent=1))
