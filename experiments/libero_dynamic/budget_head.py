"""Small joint (stride, soft RTC length) actor/critic; no robot-policy weights."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

HORIZON = 50
VISUAL_DIM = 1024  # two pooled ResNet-18 camera features
INPUT_DIM = 2 * VISUAL_DIM + HORIZON * 8 + 2 * 8 + 5


def action_pairs(strides, masks, delay, horizon=HORIZON):
    pairs = [(int(s), int(r)) for s in strides for r in masks]
    if not pairs or len(set(pairs)) != len(pairs):
        raise ValueError("empty or duplicate action grid")
    if any(not max(1, delay) <= s <= horizon - delay or not 0 <= r <= horizon - delay for s, r in pairs):
        raise ValueError("require d <= s <= H-d and 0 <= r <= H-d")
    if 0 not in masks:
        raise ValueError("r=0 is required for the initial request")
    return pairs


def valid_actions(pairs, left_len, delay):
    # r belongs to the OLD plan; the selected s belongs to the NEW plan.
    available = max(0, left_len - delay)
    return np.array([r <= available for s, r in pairs], dtype=bool)


def features(visual, previous_visual, state, old_state, leftover, age, delay, step, max_steps=600):
    visual = np.asarray(visual, np.float32).reshape(VISUAL_DIM)
    previous = visual if previous_visual is None else np.asarray(previous_visual, np.float32)
    tail = np.zeros((HORIZON, 7), np.float32)
    leftover = np.asarray(leftover, np.float32).reshape(-1, 7)
    n = min(len(leftover), HORIZON)
    tail[:n] = leftover[:n]
    present = np.arange(HORIZON) < n
    state = np.asarray(state, np.float32).reshape(8)
    old = state if old_state is None else np.asarray(old_state, np.float32).reshape(8)
    x = np.concatenate((visual, visual - previous, tail.ravel(), present.astype(np.float32),
                        state, old, [age / HORIZON, delay / HORIZON, step / max_steps,
                                     n / HORIZON, float(previous_visual is not None)])).astype(np.float32)
    if x.shape != (INPUT_DIM,) or not np.isfinite(x).all():
        raise ValueError("invalid budget features")
    return x


class BudgetHead(nn.Module):
    def __init__(self, n_actions, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(INPUT_DIM), nn.Linear(INPUT_DIM, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)

    def forward(self, x, valid):
        h = self.trunk(x)
        logits = self.actor(h).masked_fill(~valid, -1e9)
        return Categorical(logits=logits), self.critic(h).squeeze(-1)


def save_head(path, model, config, optimizer=None, step=0):
    torch.save({"model": model.state_dict(), "config": config, "step": step,
                "optimizer": None if optimizer is None else optimizer.state_dict()}, path)


def load_head(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = BudgetHead(len(ckpt["config"]["pairs"]))
    model.load_state_dict(ckpt["model"])
    return model, ckpt
