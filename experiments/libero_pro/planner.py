"""Execution planners: after each chunk, choose the replanning step s and the next soft mask.

The plan for chunk k is made when chunk k is generated:

* ``stride``  - s_k, actions of chunk k executed before the next inference starts;
* ``mask_end`` - end of the RTC soft mask used when chunk k+1 is generated, in
  leftover coordinates (0 = first action aligned with that inference start):
  the first d leftover actions are frozen, the next r are softly attached, so
  mask_end = d + r.

``FixedPlanner`` is ordinary RTC (constant s, mask to H - s). ``BudgetPlanner``
is the confidence budget: with h* from the chunk's entropy,

    rho = h* - 2d - r_min,   s = d + round(lambda * rho),   r = r_min + rho - (s - d)

so that s + r <= h* - d (the next mask ends inside chunk k's reliable part),
s >= d and r >= r_min; if rho < 0, s = d and r = r_min. lambda is fixed or
computed per chunk from the entropy (``EntropyLambda``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hstar import HStarRule


@dataclass(frozen=True)
class Plan:
    stride: int
    mask_end: int
    h_star: int | None = None
    lam: float | None = None


@dataclass(frozen=True)
class FixedPlanner:
    horizon: int
    stride: int
    mask_end: int | None = None  # default H - s

    def __call__(self, entropy: np.ndarray | None, n_valid: int | None, delay: int) -> Plan:
        end = self.horizon - self.stride if self.mask_end is None else self.mask_end
        return Plan(self.stride, end)

    @property
    def needs_entropy(self) -> bool:
        return False


@dataclass(frozen=True)
class EntropyLambda:
    """lambda from how unusual the entropy of the chunk's early actions is.

    z = (mean normalised entropy of actions [0, zone) - mu) / sigma, where mu and
    sigma describe chunks without conflict (from calibration). High z: the chunk
    disagrees with the plan it was attached to, so trust the old plan less, i.e.
    short mask and long own execution (lambda -> 1). lambda = clip(0.5 + z / (2 z_max), 0, 1).
    """

    mu: float
    sigma: float
    zone: int = 10
    z_max: float = 2.0

    def __call__(self, entropy: np.ndarray, n_valid: int) -> float:
        e = np.asarray(entropy, dtype=np.float64)[: self.zone] / math.log(max(n_valid, 2))
        z = (float(e.mean()) - self.mu) / max(self.sigma, 1e-9)
        return float(np.clip(0.5 + z / (2 * self.z_max), 0.0, 1.0))


@dataclass(frozen=True)
class BudgetPlanner:
    horizon: int
    rule: HStarRule
    r_min: int = 5
    lam: float | None = 0.5  # fixed lambda; None means use ``lam_fn``
    lam_fn: EntropyLambda | None = None
    s_min: int = 1

    def __post_init__(self):
        if (self.lam is None) == (self.lam_fn is None):
            raise ValueError("give exactly one of lam and lam_fn")
        if self.lam is not None and not 0.0 <= self.lam <= 1.0:
            raise ValueError("lam must be in [0, 1]")

    @property
    def needs_entropy(self) -> bool:
        return True

    def __call__(self, entropy: np.ndarray, n_valid: int, delay: int) -> Plan:
        h = self.rule(entropy, n_valid)
        lam = self.lam if self.lam is not None else self.lam_fn(entropy, n_valid)
        return budget_plan(h, delay, self.r_min, lam, self.horizon, self.s_min)


def budget_plan(h_star: int, delay: int, r_min: int, lam: float, horizon: int, s_min: int = 1) -> Plan:
    d = delay
    s_floor = max(d, s_min)
    rho = h_star - 2 * d - r_min
    if rho < 0:
        s, r = s_floor, r_min
    else:
        extra = int(round(lam * rho))
        s = max(d + extra, s_floor)
        r = r_min + rho - extra
    # keep the schedule feasible whatever h* says: s + d <= H and d + r <= H - s
    s = min(s, horizon - d)
    r = max(0, min(r, horizon - s - d))
    return Plan(stride=int(s), mask_end=int(d + r), h_star=int(h_star), lam=float(lam))
