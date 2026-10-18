import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hstar import HStarRule, hstar, smooth  # noqa: E402
from planner import BudgetPlanner, EntropyLambda, FixedPlanner, budget_plan  # noqa: E402

H = 50


def ramp(flat_from=20, low=0.6, high=0.66, n=500):
    """Normalised entropy rising linearly then flat, in nats."""
    e = np.concatenate([np.linspace(low, high, flat_from), np.full(H - flat_from, high)])
    return e * math.log(n)


def test_smooth_keeps_length():
    x = np.arange(10.0)
    assert smooth(x, 3).shape == (10,)
    assert smooth(x, 3)[0] == 1.0 and smooth(x, 3)[-1] == 9.0


def test_rules():
    e = ramp(20)
    assert hstar(e, 500, HStarRule("const", c=17)) == 17
    assert hstar(e, 500, HStarRule("absolute", eta=0.95)) == H  # never that high on LIBERO
    h_abs = hstar(e, 500, HStarRule("absolute", eta=0.65, tau=0.01, k=5))
    assert 15 <= h_abs <= 30
    h_rel = hstar(e, 500, HStarRule("relative", q=0.9, k=5))
    assert 15 <= h_rel <= 20
    jump = np.concatenate([np.full(12, 3.0), np.full(H - 12, 4.0)])
    assert hstar(jump, 500, HStarRule("rise", k=1, min_h=5)) == 12
    assert hstar(np.full(H, 3.0), 500, HStarRule("relative")) == H


def test_rule_roundtrip_and_validation():
    r = HStarRule("relative", q=0.6, k=3)
    assert HStarRule.from_dict(r.to_dict()) == r
    with pytest.raises(ValueError):
        HStarRule("nope")


@pytest.mark.parametrize("h,d,rmin,lam", [(40, 4, 5, 0.0), (40, 4, 5, 1.0), (40, 4, 5, 0.5), (12, 4, 5, 0.5), (50, 0, 5, 0.3)])
def test_budget_plan_respects_budget(h, d, rmin, lam):
    p = budget_plan(h, d, rmin, lam, H)
    s, r = p.stride, p.mask_end - d
    assert s >= max(d, 1) and r >= 0 and s + d <= H and p.mask_end <= H - s
    if h - 2 * d - rmin >= 0:
        assert s + r == h - d and r >= rmin
    else:
        assert s == max(d, 1) and r == rmin


def test_lambda_extremes():
    assert budget_plan(40, 4, 5, 1.0, H).stride == 40 - 4 - 5  # all slack into s
    assert budget_plan(40, 4, 5, 0.0, H).stride == 4  # replan as soon as possible, long mask


def test_planners():
    fixed = FixedPlanner(H, 10)
    assert fixed(None, None, 4).stride == 10 and fixed(None, None, 4).mask_end == 40
    e = ramp(20)
    bp = BudgetPlanner(H, HStarRule("const", c=30), r_min=5, lam=1.0)
    plan = bp(e, 500, 4)
    assert plan.h_star == 30 and plan.stride == 30 - 8 - 5 + 4
    with pytest.raises(ValueError):
        BudgetPlanner(H, HStarRule("const"), lam=None)


def test_entropy_lambda_goes_up_with_unusual_entropy():
    n = 500
    lam = EntropyLambda(mu=0.62, sigma=0.01, zone=10, z_max=2.0)
    normal = np.full(H, 0.62 * math.log(n))
    high = np.full(H, 0.66 * math.log(n))
    low = np.full(H, 0.58 * math.log(n))
    assert lam(normal, n) == pytest.approx(0.5)
    assert lam(high, n) == 1.0 and lam(low, n) == 0.0
