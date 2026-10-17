"""Guidance against a linear denoiser, where the vector-Jacobian product is known in closed form."""

import pytest
import torch
from lerobot.policies.rtc.modeling_rtc import RTCProcessor

from h1.guidance import ExactRTCProcessor, guidance_weight, make_rtc_config

H, A, REAL, D, END = 10, 6, 4, 2, 6


@pytest.fixture()
def linear():
    g = torch.Generator().manual_seed(0)
    mix = torch.randn(A, A, generator=g) * 0.3  # mixes real and padded dimensions
    bias = torch.randn(1, H, A, generator=g)
    return mix, bias, (lambda x: x @ mix.T + bias)


def _inputs():
    g = torch.Generator().manual_seed(1)
    return torch.randn(1, H, A, generator=g), torch.randn(1, H - 3, REAL, generator=g)


def test_exact_guidance_matches_closed_form(linear):
    mix, _bias, denoiser = linear
    x_t, prev = _inputs()
    time = 0.6
    proc = ExactRTCProcessor(make_rtc_config("exp", 5.0, END))
    out = proc.denoise_step(x_t, prev, D, time, denoiser, execution_horizon=END)

    v = denoiser(x_t)
    x1 = x_t - time * v
    target = torch.zeros(1, H, A)
    target[:, : prev.shape[1], :REAL] = prev
    w = proc.get_prefix_weights(D, END, H).view(1, H, 1).expand(1, H, A).clone()
    w[..., REAL:] = 0
    err = (target - x1) * w
    # x1 = x (I - t M^T) + const  =>  VJP(err) = err (I - t M)
    correction = err @ (torch.eye(A) - time * mix)
    expected = v - guidance_weight(time, 5.0) * correction
    assert torch.allclose(out, expected, atol=1e-5)

    # without masking the padded dimensions, their error would leak through the Jacobian
    w_all = proc.get_prefix_weights(D, END, H).view(1, H, 1).expand(1, H, A)
    leaky = v - guidance_weight(time, 5.0) * (((target - x1) * w_all) @ (torch.eye(A) - time * mix))
    assert not torch.allclose(out[..., :REAL], leaky[..., :REAL], atol=1e-3)


def test_lerobot_guidance_drops_the_jacobian(linear):
    _mix, _bias, denoiser = linear
    x_t, prev = _inputs()
    cfg = make_rtc_config("exp", 5.0, END)
    exact = ExactRTCProcessor(cfg).denoise_step(x_t, prev, D, 0.6, denoiser, execution_horizon=END)
    approx = RTCProcessor(cfg).denoise_step(x_t, prev, D, 0.6, denoiser, execution_horizon=END)
    assert not torch.allclose(exact, approx, atol=1e-3)


def test_no_previous_chunk_is_plain_denoising(linear):
    _mix, _bias, denoiser = linear
    x_t, _prev = _inputs()
    proc = ExactRTCProcessor(make_rtc_config("exp", 5.0, END))
    assert torch.equal(proc.denoise_step(x_t, None, D, 0.6, denoiser), denoiser(x_t))


def test_guidance_weight_is_clipped():
    assert guidance_weight(1.0, 10.0).item() == pytest.approx(10.0)  # pure noise: clip
    assert guidance_weight(0.5, 100.0).item() == pytest.approx(2.0)  # (1-tau)/(tau r^2) at tau = 0.5
