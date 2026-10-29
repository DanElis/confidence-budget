import math

import numpy as np
import pytest
import torch

from h1.attention import summarize_attention
from h1.signal import ZONE_FROZEN, ZONE_RAMP, ZONE_TAIL, mask_zones, plateau_horizon, smooth


def test_mask_zones():
    zones = mask_zones(10, 2, 6)
    assert zones.tolist() == [ZONE_FROZEN] * 2 + [ZONE_RAMP] * 4 + [ZONE_TAIL] * 4
    with pytest.raises(ValueError):
        mask_zones(10, 7, 6)


def test_smooth_is_moving_average():
    x = np.arange(10, dtype=float)
    assert np.allclose(smooth(x, 3), np.arange(1, 9))


def test_plateau_found_where_entropy_saturates():
    n_valid, horizon, k = 500, 50, 5
    top = math.log(n_valid)
    entropy = np.concatenate([np.linspace(0.5 * top, top, 20), np.full(30, top)])
    h_star, found = plateau_horizon(entropy, n_valid, k=k)
    assert found
    # smoothed curve first stays above 0.95 log N with zero slope from window start j0 = 19
    bar = smooth(entropy, k)
    j0 = next(j for j in range(horizon - 2 * k + 1)
              if bar[j:j + k].min() >= 0.95 * top and abs(np.diff(bar)[j:j + k].mean()) < 0.01)
    assert h_star == j0 + 1 + k


def test_no_plateau_returns_horizon():
    entropy = np.linspace(0.0, 1.0, 50)  # never reaches 0.95 log N
    assert plateau_horizon(entropy, 500) == (50, False)


def test_plateau_needs_two_windows():
    with pytest.raises(ValueError):
        plateau_horizon(np.zeros(8), 10, k=5)


def test_summarize_attention_ignores_padding():
    attention = torch.ones(1, 3, 6)
    valid = torch.tensor([[True, True, True, True, False, False]])
    attention[..., 4:] = 100.0  # mass on padding must be dropped
    entropy, share = summarize_attention(attention, valid, n_nonlang=2)
    assert torch.allclose(entropy, torch.full((1, 3), math.log(4)))
    assert torch.allclose(share, torch.full((1, 3), 0.5))
