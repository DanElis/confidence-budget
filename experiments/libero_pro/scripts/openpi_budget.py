"""Budget of the replanning step on openpi (H = 10, naive switching, no RTC): eval_openpi.py --budget CONFIG.

Port of the option budget of experiments/libero_dynamic (budget_v2.py) to LIBERO-Pro. With naive switching only s
matters, so an action is a step s of GRID. It is chosen right after a chunk is generated (the decision sees the new
chunk) and held for an option of option_ticks control steps, as in v2: few decisions per episode, each with an effect
on the outcome that a value model can resolve.

Modes: explore (uniform over the grid, logged propensity), fixed (a constant s, on or off the grid), marginal (given
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
MODES = ("explore", "fixed", "marginal", "policy")
ACTION_DIM = 7
GRIPPER = 6
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


@dataclass(frozen=True)
class GivenStride:
    """The step chosen by the infer function (the budget runtime) comes as the chunk's signal."""

    horizon: int

    def __call__(self, signal, n_valid, delay) -> Plan:
        s = int(signal)
        return Plan(stride=s, mask_end=self.horizon - s)

    @property
    def needs_entropy(self) -> bool:
        return False


def load_qpolicy(path):
    sys.path.insert(0, str(ROOT.parent / "libero_dynamic"))
    from budget_q import load_policy

    return load_policy(path)


class BudgetRuntime:
    """Per-episode options. decide() is called once per generated chunk, in step order within an episode."""

    def __init__(self, config: dict, seed: int, horizon: int, delay: int):
        self.config = config
        self.mode = config["mode"]
        if self.mode not in MODES:
            raise ValueError(f"unknown budget mode {self.mode}")
        self.grid = tuple(int(s) for s in config.get("grid", GRID))
        self.option_ticks, self.max_steps = int(config["option_ticks"]), int(config["max_steps"])
        self.horizon, self.delay, self.seed = horizon, delay, int(seed)
        for s in self.grid:
            check_stride(s, delay, horizon)
        self.fixed = None
        if self.mode == "fixed":
            self.fixed = int(config["fixed"])
            check_stride(self.fixed, delay, horizon)
        self.marginal = None
        if self.mode == "marginal":
            self.marginal = np.asarray(config["marginal"], np.float64)
            if len(self.marginal) != len(self.grid) or abs(self.marginal.sum() - 1) > 1e-6 or (self.marginal < 0).any():
                raise ValueError("marginal: one probability per grid step, summing to 1")
        self.policy = None
        if self.mode == "policy":
            self.policy = load_qpolicy(config["policy"])
            if [s for s, _ in self.policy.pairs] != list(self.grid):
                raise ValueError(f"policy grid {self.policy.pairs} differs from {self.grid}")
        self.episodes: dict = {}

    def select(self, x, episode: int, step: int) -> tuple[int, float, int]:
        """-> (grid index or -1, probability of the choice, s)"""
        n = len(self.grid)
        if self.mode == "fixed":
            index = self.grid.index(self.fixed) if self.fixed in self.grid else -1
            return index, 1.0, self.fixed
        rng = np.random.default_rng([self.seed, int(episode), int(step)])
        if self.mode == "explore":
            a = int(rng.integers(n))
            return a, 1.0 / n, self.grid[a]
        if self.mode == "marginal":
            a = int(rng.choice(n, p=self.marginal))
            return a, float(self.marginal[a]), self.grid[a]
        a = int(self.policy.greedy(x)[0])
        return a, 1.0, self.grid[a]

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
            a, prob, s = self.select(x, episode, step)
            state.update(action=a, prob=prob, s=s, end=step + self.option_ticks)
        state.update(plan=np.asarray(plan, np.float32), step=step, frames=frames)
        return {"x": x, "action": state["action"], "prob": state["prob"], "s": state["s"], "decision": float(decision)}


def make_budget_infer(client, prompt: str, horizon: int, runtime: BudgetRuntime, make_element=None):
    """One request per environment to openpi_sample_server.py (it returns the plan in model units).
    make_element: eval_openpi.make_element (passed in when eval_openpi runs as __main__)."""
    import torch

    from closed_loop import Chunk
    from selection import expand_noise

    if make_element is None:
        from eval_openpi import make_element

    def infer(observations, noise, leftover, delay, mask_end, episodes, step) -> Chunk:
        rows, plans, records = [], [], []
        for obs, row, episode in zip(observations, noise, episodes):
            element = make_element(obs, prompt) | {"noise": expand_noise(row[None], 1).numpy()}
            reply = client.infer(element)
            if "normalized_action" not in reply:
                raise RuntimeError("the budget needs the plan in model units: serve with scripts/openpi_sample_server.py")
            actions = np.asarray(reply["actions"], dtype=np.float32)
            if actions.ndim == 2:
                actions = actions[None]
            if actions.shape != (1, horizon, ACTION_DIM):
                raise RuntimeError(f"server returned actions {actions.shape}, expected {(1, horizon, ACTION_DIM)}")
            plan = np.asarray(reply["normalized_action"], dtype=np.float32)[0]
            records.append(runtime.decide(int(episode), step, element, plan, actions[0]))  # task * 1000 + init state
            rows.append(actions[0]), plans.append(plan)
        extras = {f"budget_{k}": np.stack([np.asarray(r[k], np.float32) for r in records])
                  for k in ("x", "action", "prob", "decision")}
        return Chunk(torch.from_numpy(np.stack(plans)), np.stack(rows), signal=np.array([r["s"] for r in records]),
                     extras=extras)

    infer.wants_context = True
    return infer


def write_config(path: Path, **config) -> None:
    Path(path).write_text(json.dumps(config, indent=1))
