"""The aac h* rule (AAC [13] Eq. 5) against the formula written out."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hstar import HStarRule, hstar  # noqa: E402


def aac_ref(x, min_h):
    mean = {h: x[:h].mean() for h in range(1, len(x) + 1)}
    gains = {h: mean[h + 1] - mean[h] for h in range(1, len(x))}
    return min(max(max(gains, key=lambda h: (gains[h], -h)), min_h), len(x))


def test_aac_rule_matches_formula():
    rng = np.random.default_rng(0)
    for min_h in (1, 5):
        rule = HStarRule("aac", min_h=min_h)
        for _ in range(100):
            x = rng.uniform(0, 1, 50)
            assert hstar(x, 0, rule) == aac_ref(x, min_h)
    assert hstar(np.r_[np.full(10, 0.1), np.full(40, 1.0)], 0, HStarRule("aac", min_h=1)) == 10
