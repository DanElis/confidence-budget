"""AAC [13] as in the authors' code: Adaptive-Action-Chunking/libero-pro@1c83fa7, action_optimization/action_entropy_pi05.py,
select_chunk_size(method="gaussian_bernoulli", move_th=3), with the executed sample 0 (--chunk_id_selector 0).

Per action index: Gaussian entropy of translation and of rotation (3-DOF blocks, normalised actions), Bernoulli entropy
of the binarised gripper (environment actions, close = value >= 0). h* = argmax_h (E_{h+1} - E_h) over the mean entropy
of the first h actions (Eq. 5), at least 2 and at least xi, the shortest prefix of sample 0 whose motion exceeds move_th
(Eq. 6-10; environment actions). If no prefix does, xi = H, as in their find_first_idx.
"""

from __future__ import annotations

import math

import numpy as np

SAMPLES = 20  # N of the paper's main experiments
MOVE_TH = 3.0  # alpha, "set to 3 in all experiments"
GRIPPER_WEIGHT = 0.2  # wgrip of action_magnitude


def gaussian_entropy(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """(N, H, D) -> (H,) entropy of a Gaussian with the samples' covariance + eps * I per action index."""
    n, _, d = x.shape
    centered = x - x.mean(0, keepdims=True)
    cov = np.einsum("nti,ntj->tij", centered, centered) / (n - 1) + eps * np.eye(d)
    return 0.5 * (d * math.log(2 * math.pi * math.e) + np.linalg.slogdet(cov)[1])


def bernoulli_entropy(gripper: np.ndarray) -> np.ndarray:
    """(N, H) binary -> (H,)"""
    p = np.clip(gripper.mean(0), 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def action_entropy(normalized: np.ndarray, env_actions: np.ndarray) -> np.ndarray:
    """normalized, env_actions (N, H, >= 7) -> total entropy per action index (H,)."""
    return (gaussian_entropy(normalized[..., :3]) + gaussian_entropy(normalized[..., 3:6])
            + bernoulli_entropy((env_actions[..., 6] >= 0).astype(np.float64)))


def entropy_horizon(total: np.ndarray) -> int:
    """Eq. 5 with their lower bound of 2."""
    chunk_mean = np.cumsum(total) / np.arange(1, len(total) + 1)
    return max(int(np.argmax(np.diff(chunk_mean))) + 1, 2)


def _axisangle2quat(vec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vec))
    if math.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])
    return np.r_[vec / angle * math.sin(angle / 2), math.cos(angle / 2)]


def _quat_multiply(q1: np.ndarray, q0: np.ndarray) -> np.ndarray:
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    return np.array((x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0, -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0, -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0))


def _rotation_angle(q: np.ndarray) -> float:
    """Norm of the axis-angle of a unit quaternion (their quat2axisangle: identity gives 0)."""
    w = float(np.clip(q[3], -1.0, 1.0))
    return 0.0 if math.isclose(math.sqrt(1.0 - w * w), 0.0) else 2.0 * math.acos(w)


def motion_floor(action: np.ndarray, move_th: float = MOVE_TH) -> int:
    """xi of one chunk of environment actions (H, >= 7): the shortest k >= 2 with translation + rotation + gripper > move_th."""
    horizon = len(action)
    position = np.cumsum(action[:, :3], axis=0)
    gripper = (action[:, 6] >= 0).astype(int)
    q = _quat_multiply(_axisangle2quat(action[0, 3:6]), np.array([0.0, 0.0, 0.0, 1.0]))
    for k in range(2, horizon + 1):
        q = _quat_multiply(_axisangle2quat(action[k - 1, 3:6]), q)
        toggled = int(np.any(np.diff(gripper[:k])))
        if np.linalg.norm(position[k - 1]) + _rotation_angle(q) + GRIPPER_WEIGHT * toggled > move_th:
            return k
    return horizon


def paper_h_star(normalized: np.ndarray, env_actions: np.ndarray, move_th: float = MOVE_TH) -> int:
    """h* from N samples of one chunk, (N, H, >= 7) each; row 0 is the executed sample."""
    return max(entropy_horizon(action_entropy(normalized, env_actions)), motion_floor(env_actions[0], move_th))
