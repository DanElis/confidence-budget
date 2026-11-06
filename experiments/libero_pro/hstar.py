"""Reliable horizon h* from the per-action cross-attention entropy of one chunk.

Rules (all 0-based internally; h* is a number of actions in [1, H]):

* ``absolute``  (the rule of [15]) - first window of ``k`` smoothed values that
  are all >= eta * log(n_valid) with mean slope below ``tau``; h* = j + 1 + k.
* ``relative``  - the smoothed curve is rescaled to [0, 1] within the chunk;
  h* = first j where it reaches ``q`` and stays there for ``k`` steps, + 1.
* ``rise``      - h* = position of the largest increase of the smoothed
  curve (like AAC's argmax of entropy growth), at least ``min_h``.
* ``above``     - h* = first j where the smoothed signal reaches ``eta`` (an absolute level, e.g.
  for the dispersion across noise samples, as in AAC), at least ``min_h``.
* ``const``     - h* = ``c`` regardless of the entropy (baseline for calibration).
* ``aac``       - AAC [13] Eq. 5: h* = argmax_h (D_{h+1} - D_h), D_h = mean of the signal over the first h
  actions (no smoothing, no threshold), at least ``min_h``.

If no rule fires, h* = H.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

RULES = ("absolute", "relative", "rise", "above", "const", "aac")


@dataclass(frozen=True)
class HStarRule:
    kind: str = "relative"
    k: int = 5
    eta: float = 0.95  # absolute
    tau: float = 0.01  # absolute
    q: float = 0.7  # relative
    min_h: int = 5  # rise
    c: int = 25  # const

    def __post_init__(self):
        if self.kind not in RULES:
            raise ValueError(f"unknown h* rule {self.kind!r}, choose from {RULES}")
        if self.k < 1:
            raise ValueError("k must be >= 1")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HStarRule":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def __call__(self, entropy: np.ndarray, n_valid: int) -> int:
        return hstar(entropy, n_valid, self)


def smooth(x: np.ndarray, k: int) -> np.ndarray:
    """Trailing-free moving average, same length as ``x`` (shrinking window at the end)."""
    x = np.asarray(x, dtype=np.float64)
    if k <= 1:
        return x
    c = np.concatenate([[0.0], np.cumsum(x)])
    n = len(x)
    idx = np.arange(n)
    hi = np.minimum(idx + k, n)
    return (c[hi] - c[idx]) / (hi - idx)


def hstar(entropy: np.ndarray, n_valid: int, rule: HStarRule) -> int:
    e = np.asarray(entropy, dtype=np.float64)
    horizon = len(e)
    if rule.kind == "const":
        return int(min(max(rule.c, 1), horizon))
    bar = smooth(e, rule.k)
    if rule.kind == "absolute":
        level = rule.eta * math.log(max(n_valid, 2))
        diff = np.diff(bar)
        for j in range(horizon - 2 * rule.k + 1):
            if bar[j : j + rule.k].min() >= level and abs(diff[j : j + rule.k].mean()) < rule.tau:
                return int(min(j + 1 + rule.k, horizon))
        return horizon
    if rule.kind == "relative":
        lo, hi = bar.min(), bar.max()
        if hi - lo < 1e-9:
            return horizon
        z = (bar - lo) / (hi - lo)
        above = z >= rule.q
        for j in range(horizon - rule.k + 1):
            if above[j : j + rule.k].all():
                return int(max(j + 1, 1))
        return horizon
    if rule.kind == "above":
        hits = np.flatnonzero(bar >= rule.eta)
        j = int(hits[0]) if hits.size else horizon
        return int(min(max(j, rule.min_h), horizon))
    if rule.kind == "aac":
        prefix_mean = np.cumsum(e) / np.arange(1, horizon + 1)
        j = int(np.argmax(np.diff(prefix_mean))) + 1
        return int(min(max(j, rule.min_h), horizon))
    if rule.kind == "rise":
        grow = np.diff(bar)
        j = int(np.argmax(grow)) + 1
        return int(min(max(j, rule.min_h), horizon))
    raise AssertionError(rule.kind)
