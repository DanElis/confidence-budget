"""Confidence signal of Xu et al., "Knowing When to Stop" (arXiv:2609.00908).

For every action token j of a chunk, the action expert's attention to the
vision-language tokens is summed over layers, renormalised over the valid
vision-language tokens and turned into an entropy E_j. A sustained high-entropy
plateau marks the end of the reliable horizon h*.

Indices are 0-based here; the paper writes the same rule 1-based.
"""

from __future__ import annotations

import math

import numpy as np

ZONE_FROZEN = 0  # j < d: actions executed during inference, weight 1
ZONE_RAMP = 1  # d <= j < mask_end: soft attachment to the previous chunk
ZONE_TAIL = 2  # j >= mask_end: free generation
ZONE_NAMES = {ZONE_FROZEN: "frozen", ZONE_RAMP: "ramp", ZONE_TAIL: "tail"}


def mask_zones(horizon: int, delay: int, mask_end: int) -> np.ndarray:
    """Label each action index by its zone in the RTC mask."""
    if not 0 <= delay <= mask_end <= horizon:
        raise ValueError(f"need 0 <= delay <= mask_end <= horizon, got {delay}, {mask_end}, {horizon}")
    zones = np.full(horizon, ZONE_TAIL, dtype=np.int8)
    zones[:mask_end] = ZONE_RAMP
    zones[:delay] = ZONE_FROZEN
    return zones


def smooth(entropy: np.ndarray, k: int) -> np.ndarray:
    """Forward moving average: out[j] = mean(entropy[j : j + k]), length H - k + 1."""
    if k < 1 or k > entropy.shape[-1]:
        raise ValueError(f"window k={k} must be in [1, {entropy.shape[-1]}]")
    kernel = np.ones(k) / k
    return np.convolve(entropy, kernel, mode="valid")


def plateau_horizon(
    entropy: np.ndarray,
    n_valid: int,
    k: int = 5,
    eta: float = 0.95,
    tau: float = 0.01,
) -> tuple[int, bool]:
    """Reliable horizon h* (number of actions) and whether a plateau was found.

    A window starting at j is a plateau when every smoothed entropy in
    bar_E[j : j + k] is at least eta * log(n_valid) and the mean of the first
    differences over the same window is below tau in absolute value. Following
    the paper, h* = j + 1 + k for the earliest such j (the 1-based min(C) + k);
    without a plateau h* = H.

    Args:
        entropy: per-action entropy in nats, shape (H,).
        n_valid: number of valid vision-language tokens; log(n_valid) is the
            maximum attainable entropy.
        k: smoothing window (paper: 5 for H = 50, 3 for H = 10).
        eta: high-entropy threshold as a fraction of log(n_valid) (paper: 0.95).
        tau: stability threshold on the mean entropy change, nats (paper: 0.01).
    """
    entropy = np.asarray(entropy, dtype=np.float64)
    horizon = entropy.shape[0]
    if horizon < 2 * k:
        raise ValueError(f"horizon {horizon} is too short for window k={k}")
    bar = smooth(entropy, k)  # length H - k + 1
    diff = np.diff(bar)  # length H - k
    level = eta * math.log(n_valid)
    for j in range(horizon - 2 * k + 1):
        high = bar[j : j + k].min() >= level
        stable = abs(diff[j : j + k].mean()) < tau
        if high and stable:
            return min(j + 1 + k, horizon), True
    return horizon, False
