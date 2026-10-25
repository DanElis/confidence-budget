"""selection.py, the old plan passed to infer by closed_loop, adaptive mask_r and the K-sample CLI checks."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from closed_loop import Chunk, LoopConfig, run_batch  # noqa: E402
from hstar import HStarRule  # noqa: E402
from planner import AdaptivePlanner, FixedPlanner  # noqa: E402
from selection import agreement, conf_weights, dispersion, expand_noise, select_index  # noqa: E402

H = 10


def test_expand_noise_keeps_row_zero_and_is_deterministic():
    noise = torch.randn(3, H, 4)
    a, b = expand_noise(noise, 4), expand_noise(noise, 4)
    assert a.shape == (12, H, 4)
    torch.testing.assert_close(a[::4], noise)
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a[1], a[2])


def test_dispersion():
    samples = torch.zeros(2, 3, H, 2)
    samples[1, :, :, 0] = torch.tensor([0.0, 1.0, 2.0])[:, None]
    d = dispersion(samples)
    torch.testing.assert_close(d[0], torch.zeros(H))
    torch.testing.assert_close(d[1], torch.full((H,), 0.5))  # std 1 on one of two dims


def test_conf_weights():
    rtc_w = torch.tensor([[1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0, 0, 0, 0]])
    end = torch.tensor([6])
    disp = torch.linspace(0, 0.9, H)[None]
    torch.testing.assert_close(conf_weights("rtc", rtc_w, disp, 2, end, 0.3), rtc_w)
    torch.testing.assert_close(conf_weights("rtc_x_conf", rtc_w, disp, 2, end, float("inf")), rtc_w)
    w = conf_weights("rtc_x_conf", rtc_w, disp, 2, end, 0.3)
    torch.testing.assert_close(w[0, :2], torch.ones(2))
    torch.testing.assert_close(w[0, 2:], rtc_w[0, 2:] * torch.exp(-disp[0, 2:] / 0.3))
    pure = conf_weights("conf", rtc_w, disp, 2, end, 0.3)
    torch.testing.assert_close(pure[0, :2], torch.ones(2))
    torch.testing.assert_close(pure[0, 2:6], torch.exp(-disp[0, 2:6] / 0.3))
    assert (pure[0, 6:] == 0).all()
    torch.testing.assert_close(conf_weights("conf", rtc_w, None, 2, end, 0.3), rtc_w)  # first chunk: no old plan


def test_select_index():
    g = torch.Generator().manual_seed(0)
    leftover = torch.randn(2, H, 3, generator=g)
    samples = torch.randn(2, 6, H, 3, generator=g) * 3
    samples[0, 4] = leftover[0] + 0.01
    samples[1, 1] = leftover[1] - 0.01
    w = torch.ones(2, H)
    assert select_index("first", samples, leftover, w).tolist() == [0, 0]
    assert select_index("coherent", samples, leftover, w).tolist() == [4, 1]
    assert select_index("coherent", samples, None, None).tolist() == [0, 0]  # first chunk
    only_last = torch.zeros(2, H)
    only_last[:, -1] = 1
    s2 = samples.clone()
    s2[0, 2, -1] = leftover[0, -1]
    assert select_index("coherent", s2, leftover, only_last)[0] == 2  # weights decide where to compare
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    xy = np.insert(np.stack([np.cos(angles), np.sin(angles)], 1), 2, [0.0, 0.0], axis=0)  # sample 2 at the centre
    ring = torch.zeros(2, 6, H, 3)
    ring[..., :2] = torch.as_tensor(xy, dtype=torch.float32)[None, :, None, :]
    assert select_index("medoid", ring, leftover, w).tolist() == [2, 2]


def test_agreement():
    g = torch.Generator().manual_seed(1)
    samples = torch.randn(1, 8, H, 3, generator=g) * 0.1
    mean = samples.mean(1)
    far = mean.clone()
    far[0, 4:] += 5.0  # the old plan leaves the cloud from index 4
    z_near = agreement(samples, mean, torch.tensor([H]))
    z_far = agreement(samples, far, torch.tensor([7]))
    assert float(z_near.max()) < 1e-5
    assert (z_far[0, 4:7] > 10).all() and (z_far[0, :4] < 1e-5).all()
    assert (z_far[0, 7:] == 0).all()  # past the end of the old plan: no evidence
    rule = HStarRule("above", eta=2.0, k=1, min_h=1)
    assert rule(z_far[0].numpy(), 0) == 4 and rule(z_near[0].numpy(), 0) == H
    assert agreement(samples, None, None).abs().sum() == 0


def test_adaptive_mask_r():
    rule = HStarRule("const", c=30)
    plan = AdaptivePlanner(50, rule, s_min=5, mask_r=5)(np.zeros(50), 0, 8)
    assert plan.stride == 22 and plan.mask_end == 13
    long = AdaptivePlanner(50, rule, s_min=5)(np.zeros(50), 0, 8)
    assert long.mask_end == 50 - 22
    capped = AdaptivePlanner(50, HStarRule("const", c=50), s_min=5, s_max=25, mask_r=5)(np.zeros(50), 0, 8)
    assert capped.stride == 25 and capped.mask_end == 13


# ---------------------------------------------------------------- closed loop with a dispersion-returning infer

class Env:
    def __init__(self):
        self.t = 0

    def reset(self):
        self.t = 0
        return {"step": np.array([0.0])}, {}

    def step(self, action):
        self.t += 1
        return {"step": np.array([float(self.t)])}, 0.0, False, False, {}


def disp_infer(calls):
    def infer(observations, noise, leftover, delay, mask_end, **kw):
        t = int(observations[0]["step"][0])
        calls.append({"t": t, "leftover": leftover, "mask_end": mask_end, **kw})
        n = len(observations)
        j = torch.arange(H, dtype=torch.float32)
        model = torch.stack([torch.full((H,), float(t)), j], -1).expand(n, H, 2).clone()
        disp = np.tile(t + np.arange(H) / 100.0, (n, 1))  # identifies the chunk and the index
        return Chunk(model, model.numpy().copy(), dispersion=disp)
    return infer


@pytest.mark.parametrize("guidance", [False, True])
def test_old_plan_is_passed_with_dispersion(guidance):
    d, s = 2, 4
    calls = []
    run_batch([Env()], disp_infer(calls), LoopConfig(H, d, guidance, 13), FixedPlanner(H, s), [0], (H, 2))
    assert calls[0]["leftover"] is None and "leftover_disp" not in calls[0]
    for c in calls[1:]:  # chunks at t = 4, 8, 12, each sees the chunk started s steps before
        prev, off = c["t"] - s, s
        assert c["leftover"] is not None  # also without guidance
        np.testing.assert_allclose(c["leftover"][0, : H - off, 0], prev)
        np.testing.assert_allclose(c["leftover"][0, : H - off, 1], np.arange(off, H))
        np.testing.assert_allclose(c["leftover_disp"][0, : H - off], prev + np.arange(off, H) / 100.0, rtol=1e-6)
        assert (c["leftover_disp"][0, H - off :] == 0).all()
        assert int(c["leftover_len"][0]) == H - off
        assert int(c["mask_end"][0]) == H - s


def test_old_infer_without_dispersion_is_unchanged():
    calls = []

    def infer(observations, noise, leftover, delay, mask_end):  # no **kw: would fail if extras were passed
        calls.append(leftover)
        n = len(observations)
        return Chunk(torch.zeros(n, H, 2), np.zeros((n, H, 2)))

    run_batch([Env()], infer, LoopConfig(H, 2, False, 9), FixedPlanner(H, 4), [0], (H, 2))
    assert all(x is None for x in calls)  # naive without dispersion: no leftover, as before


def test_sample_cli_checks():
    from eval_libero import check_sample_args, parse_args

    ok = ["--out", "x", "--n-samples", "4", "--select", "coherent", "--guidance", "lerobot"]
    check_sample_args(parse_args(ok))
    check_sample_args(parse_args(ok + ["--planner", "adaptive", "--signal", "agreement",
                                       "--hstar-rule", "kind=above,eta=2,k=1,min_h=1"]))
    for bad in (["--out", "x", "--select", "coherent"],  # K = 1
                ok + ["--planner", "adaptive", "--hstar-rule", "kind=above,eta=2"],  # entropy signal with K > 1
                ["--out", "x", "--n-samples", "4", "--mask-mode", "rtc_x_conf"],  # mask without guidance
                ok + ["--record-entropy"]):
        with pytest.raises(SystemExit):
            check_sample_args(parse_args(bad))
