"""Low-dimensional deployable input of the option budget, computed before the current chunk is denoised.

Blocks (BLOCKS gives their slices; ablations zero one of them):
  robot     fresh state (8), per-tick eef and gripper-width change since the old plan's observation (4)
  plan      old plan geometry over index windows relative to d, first gripper transition, length and age
  motion    per-tick mean absolute frame difference since the old plan's observation: agent camera 4x4 grid + max,
            wrist camera mean + max, presence flag. Pooled ImageNet features barely change when a small object moves.
  residual  old plan's denoising residual under the fresh prefix (selection.plan_residual) over three windows + flag
  context   t/T, d/H, previous pair
H defaults to 50 (LeRobot pi0.5); openpi's pi05_libero passes horizon=10. The dimension does not depend on H.
"""
from __future__ import annotations

import numpy as np

HORIZON = 50
BLOCKS = {"robot": (0, 12), "plan": (12, 31), "motion": (31, 51), "residual": (51, 55), "context": (55, 59)}
ACTOR_DIM = 59
GRIPPER_FLIP = 1.


def gray(image):
    a = np.asarray(image)
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f"expected an HWC image, got {a.shape}")
    return a.astype(np.float32).mean(-1) / 255.


def _cells(a, grid):
    return np.array([c.mean() for row in np.array_split(a, grid, 0) for c in np.array_split(row, grid, 1)], np.float32)


def motion_features(frames, previous, age):
    out = np.zeros(20, np.float32)
    if previous is None or age <= 0:
        return out
    agent, wrist = (np.abs(a - b) / age for a, b in zip(frames, previous))
    cells = _cells(agent, 4)
    out[:16], out[16] = cells, cells.max()
    out[17:19] = wrist.mean(), _cells(wrist, 4).max()
    out[19] = 1.
    return out


def _windows(delay, horizon=HORIZON):
    return (0, delay), (delay, delay + 4), (delay + 4, delay + 12), (delay + 12, delay + 24), (delay + 24, max(horizon, delay + 24))


def plan_features(left, delay, age, horizon=HORIZON):
    left = np.asarray(left, np.float32).reshape(-1, 7)
    out = np.zeros(19, np.float32)
    for i, (start, end) in enumerate(_windows(delay, horizon)):
        part = left[start:end]
        if len(part):
            out[3 * i:3 * i + 3] = np.abs(part[:, :3]).mean(), np.abs(part[:, 3:6]).mean(), part[:, 6].mean()
    future = left[delay:, 6]
    if len(future):
        flips = np.flatnonzero(np.abs(future - future[0]) > GRIPPER_FLIP)
        if len(flips):
            out[15:17] = 1., flips[0] / horizon
    out[17:19] = len(left) / horizon, age / horizon
    return out


def residual_features(residual, delay, left_len):
    out = np.zeros(4, np.float32)
    if residual is None or left_len <= delay:
        return out
    r = np.asarray(residual, np.float64)[:left_len]
    for i, (start, end) in enumerate(((delay, delay + 4), (delay + 4, delay + 12), (delay + 12, left_len))):
        part = r[start:end]
        if len(part):
            out[i] = np.sqrt(part.mean())
    out[3] = 1.
    return out


def robot_features(state, previous, age):
    state = np.asarray(state, np.float32).reshape(8)
    out = np.zeros(12, np.float32)
    out[:8] = state
    if previous is not None and age > 0:
        previous = np.asarray(previous, np.float32).reshape(8)
        out[8:11] = (state[:3] - previous[:3]) / age
        out[11] = ((state[6] - state[7]) - (previous[6] - previous[7])) / age
    return out


def gripper_width(x):
    return x[..., 6] - x[..., 7]


def actor_features(state, previous_state, left, delay, step, max_steps, frames, previous_frames, residual, previous_pair,
                   horizon=HORIZON):
    left = np.asarray(left, np.float32).reshape(-1, 7)
    age = horizon - len(left) if len(left) else 0
    context = np.array([step / max_steps, delay / horizon, previous_pair[0] / horizon, previous_pair[1] / horizon], np.float32)
    x = np.concatenate((robot_features(state, previous_state, age), plan_features(left, delay, age, horizon),
                        motion_features(frames, previous_frames, age), residual_features(residual, delay, len(left)),
                        context)).astype(np.float32)
    if x.shape != (ACTOR_DIM,) or not np.isfinite(x).all():
        raise ValueError("invalid option budget features")
    return x
