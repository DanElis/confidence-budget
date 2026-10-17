"""RTC guidance variants on top of LeRobot's ``RTCProcessor``.

LeRobot computes the velocity ``v_t`` before ``x_t`` requires a gradient, so
``x1 = x_t - time * v_t`` depends on ``x_t`` only through the identity term and
the "correction" equals the weighted error itself: the network Jacobian of the
paper's pseudoinverse guidance (Black et al., 2025, Eq. 2) is dropped.

``ExactRTCProcessor`` restores the vector-Jacobian product through the
denoiser. It also restricts the error to the real action dimensions: LeRobot
pads the previous chunk with zeros up to ``max_action_dim``, which is harmless
when the Jacobian is the identity but would leak padding errors into the real
dimensions through a full Jacobian.
"""

from __future__ import annotations

import torch

from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor

GUIDANCE_MODES = ("exact", "lerobot")


def guidance_weight(time: float, max_weight: float) -> torch.Tensor:
    """Clipped PiGDM weight min(beta, (1 - tau) / (tau * r_tau^2)), tau = 1 - time."""
    tau = torch.as_tensor(1.0 - time, dtype=torch.float64)
    max_w = torch.as_tensor(max_weight, dtype=torch.float64)
    one_minus = (1 - tau) ** 2
    inv_r2 = (one_minus + tau**2) / one_minus
    c = torch.nan_to_num((1 - tau) / tau, posinf=max_weight)
    weight = torch.nan_to_num(c * inv_r2, posinf=max_weight)
    return torch.minimum(weight, max_w).float()


class ExactRTCProcessor(RTCProcessor):
    """RTC with the Jacobian of the denoiser in the guidance term."""

    def denoise_step(
        self,
        x_t,
        prev_chunk_left_over,
        inference_delay,
        time,
        original_denoise_step_partial,
        execution_horizon=None,
    ):
        if prev_chunk_left_over is None:
            return original_denoise_step_partial(x_t)

        squeezed = x_t.ndim < 3
        if squeezed:
            x_t = x_t.unsqueeze(0)
        prev = prev_chunk_left_over if prev_chunk_left_over.ndim == 3 else prev_chunk_left_over.unsqueeze(0)

        if execution_horizon is None:
            execution_horizon = self.rtc_config.execution_horizon
        execution_horizon = min(execution_horizon, prev.shape[1])

        batch, horizon, action_dim = x_t.shape
        real_dim = min(prev.shape[2], action_dim)
        padded = torch.zeros(batch, horizon, action_dim, device=x_t.device, dtype=x_t.dtype)
        steps = min(prev.shape[1], horizon)
        padded[:, :steps, :real_dim] = prev[:, :steps, :real_dim].to(device=x_t.device, dtype=x_t.dtype)

        weights = self.get_prefix_weights(inference_delay, execution_horizon, horizon).to(x_t.device)
        weights = weights.view(1, horizon, 1).expand(batch, horizon, action_dim).clone()
        weights[:, :, real_dim:] = 0.0

        with torch.enable_grad():
            x = x_t.detach().clone().requires_grad_(True)
            v_t = original_denoise_step_partial(x)
            x1 = x - time * v_t
            err = (padded - x1) * weights.to(x1.dtype)
            (correction,) = torch.autograd.grad(x1, x, grad_outputs=err.detach())

        weight = guidance_weight(time, self.rtc_config.max_guidance_weight).to(x_t.device)
        result = v_t.detach() - weight.to(v_t.dtype) * correction.to(v_t.dtype)
        return result.squeeze(0) if squeezed else result


def make_rtc_config(schedule: str, max_guidance_weight: float, execution_horizon: int) -> RTCConfig:
    schedules = {
        "exp": RTCAttentionSchedule.EXP,
        "linear": RTCAttentionSchedule.LINEAR,
        "ones": RTCAttentionSchedule.ONES,
        "zeros": RTCAttentionSchedule.ZEROS,
    }
    if schedule not in schedules:
        raise ValueError(f"unknown schedule {schedule!r}, choose from {sorted(schedules)}")
    return RTCConfig(
        enabled=True,
        mode="guided",
        prefix_attention_schedule=schedules[schedule],
        max_guidance_weight=max_guidance_weight,
        execution_horizon=execution_horizon,
    )


def install_rtc(policy, mode: str, rtc_config: RTCConfig) -> RTCProcessor:
    """Enable guided RTC on a LeRobot policy with the chosen guidance variant."""
    if mode not in GUIDANCE_MODES:
        raise ValueError(f"unknown guidance mode {mode!r}, choose from {GUIDANCE_MODES}")
    policy.config.rtc_config = rtc_config
    processor_cls = ExactRTCProcessor if mode == "exact" else RTCProcessor
    processor = processor_cls(rtc_config)
    policy.rtc_processor = processor
    policy.model.rtc_processor = processor
    return processor
