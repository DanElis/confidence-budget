"""K noise samples per inference: confidence mask, choice of the executed sample, agreement of old and new plans.

All functions work on the policy's normalised action space and only the real action dimensions.

* ``conf_weights``  - RTC soft mask modulated by the confidence of the OLD plan: c_i = exp(-D_i / sigma), D_i = the
  dispersion of the previous chunk's samples at leftover index i. ``rtc_x_conf``: w_RTC(i) * c_i after the frozen
  part; ``conf``: c_i inside the RTC window [d, mask_end).
* ``select_index``  - ``coherent``: the sample closest to the old leftover under the mask (BID backward loss);
  ``medoid``: the sample closest to the other samples (control without the old plan); ``first``: sample 0.
* ``agreement``     - per index, how far the old leftover lies from the cloud of new samples:
  z_i = mean over action dims of |leftover - mean| / std, 0 where the old plan has ended. With the ``above`` h* rule
  (eta = z threshold) h* is the first index where the policy no longer supports the old plan.
* ``plan_residual`` - fresh-frame check of the old plan: noise it to flow time t, x_t = t*eps + (1-t)*Y (LeRobot's
  convention, noise at t = 1), take one velocity of the policy under the NEW observation and compare the denoised
  estimate x_t - t*v with Y per index. Small = the policy would still produce the old plan.
* ``transition_weights`` - RTC mask with an exponential transition of length L that grows with the deviation of the
  old plan near the stitch: W_j = 1 (j < d), exp(-(j - d) / L) (d <= j < leftover end), 0 after;
  L = clip(L0 * (delta / delta_ref)^p, L_min, L_max), delta = sqrt(mean residual over [d, d + window)). p = 0 is a
  constant L0 (the paired control).
* ``expand_noise``  - (B, H, A) -> (B*K, H, A); row 0 of each group is the episode's noise, the others are drawn
  deterministically from it (as experiments/robotwin), so every configuration sees the same samples.
"""

from __future__ import annotations

import torch

MASK_MODES = ("rtc", "rtc_x_conf", "conf")
SELECTS = ("first", "coherent", "medoid")


def expand_noise(noise: torch.Tensor, k: int) -> torch.Tensor:
    out = []
    for row in noise:
        out.append(row)
        seed = int(abs(float(row.flatten()[:16].double().sum())) * 1e9) % (2**62)
        g = torch.Generator().manual_seed(seed)
        out += [torch.randn(row.shape, generator=g, dtype=row.dtype) for _ in range(k - 1)]
    return torch.stack(out)


def joint_batch(noise_k: torch.Tensor, leftover: torch.Tensor, mask_end: torch.Tensor, weights: torch.Tensor,
                k: int):
    """Rows of one policy call with K unguided and K guided samples per environment (same K noises in both halves).

    noise_k (n*K, H, A) as from expand_noise, leftover (n, H, A), mask_end (n,), weights (n, H) ->
    noise, leftover, mask_end, weights for n*2K rows ordered [env 0: K unguided, K guided | env 1: ...];
    unguided rows get weight 0, so the guidance correction vanishes and they are plain samples.
    """
    n = leftover.shape[0]
    noise = noise_k.view(n, k, *noise_k.shape[1:])
    noise2 = torch.cat([noise, noise], dim=1).reshape(n * 2 * k, *noise_k.shape[1:])
    w = torch.cat([torch.zeros(n, k, weights.shape[1], dtype=weights.dtype, device=weights.device),
                   weights[:, None].expand(n, k, weights.shape[1])], dim=1).reshape(n * 2 * k, weights.shape[1])
    return (noise2, leftover.repeat_interleave(2 * k, 0), mask_end.repeat_interleave(2 * k, 0), w)


def split_joint(out: torch.Tensor, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(n*2K, H, A) output of a joint_batch call -> unguided (n, K, H, A), guided (n, K, H, A)."""
    out = out.reshape(n, 2 * k, *out.shape[1:])
    return out[:, :k], out[:, k:]


def dispersion(samples: torch.Tensor) -> torch.Tensor:
    """(B, K, H, A) -> (B, H): std across samples, mean over action dims."""
    return samples.std(dim=1, unbiased=True).mean(dim=-1)


def conf_weights(mode: str, rtc_w: torch.Tensor, leftover_disp: torch.Tensor | None, delay: int,
                 mask_end: torch.Tensor, sigma: float) -> torch.Tensor:
    """rtc_w (B, H) RTC weights for start ``delay`` and end ``mask_end`` (B,) -> weights of ``mode``."""
    if mode == "rtc" or leftover_disp is None:
        return rtc_w
    idx = torch.arange(rtc_w.shape[1], device=rtc_w.device)
    conf = torch.exp(-leftover_disp.to(rtc_w) / sigma)
    frozen = idx < delay
    if mode == "rtc_x_conf":
        return torch.where(frozen, rtc_w, rtc_w * conf)
    window = idx[None] < mask_end.to(rtc_w.device)[:, None]
    return torch.where(frozen, torch.ones_like(rtc_w), torch.where(window, conf, torch.zeros_like(conf)))


def select_index(mode: str, samples: torch.Tensor, leftover: torch.Tensor | None,
                 weights: torch.Tensor | None) -> torch.Tensor:
    """samples (B, K, H, A), leftover (B, H, A), weights (B, H) -> (B,) index of the sample to execute."""
    b = samples.shape[0]
    if mode == "first" or (mode == "coherent" and leftover is None):
        return torch.zeros(b, dtype=torch.long)
    if mode == "coherent":
        err = torch.linalg.norm(samples - leftover[:, None].to(samples), dim=-1)  # (B, K, H)
        score = (err * weights[:, None].to(samples)).sum(-1)
    else:
        err = torch.linalg.norm(samples[:, :, None] - samples[:, None], dim=-1).mean(-1)  # (B, K, K)
        score = err.sum(-1)
    return score.argmin(dim=1).cpu()


def agreement(samples: torch.Tensor, leftover: torch.Tensor | None, leftover_len: torch.Tensor | None,
              eps: float = 1e-2) -> torch.Tensor:
    """samples (B, K, H, A), leftover (B, H, A), leftover_len (B,) -> z (B, H); zeros without a leftover."""
    b, _, h, _ = samples.shape
    if leftover is None:
        return torch.zeros(b, h)
    mean, std = samples.mean(1), samples.std(1, unbiased=True)
    z = ((leftover.to(samples) - mean).abs() / (std + eps)).mean(-1)  # (B, H)
    valid = torch.arange(h, device=z.device)[None] < leftover_len.to(z.device)[:, None]
    return torch.where(valid, z, torch.zeros_like(z)).cpu()


def plan_residual(velocity_fn, leftover: torch.Tensor, noise: torch.Tensor, t: float) -> torch.Tensor:
    """leftover (R, H, A) old plan, noise (R, H, A_max), velocity_fn(x_t, time (R,)) -> (R, H, A_max) under the new
    observation -> r (R, H): mean squared error of the denoised estimate over the A real dimensions."""
    a = leftover.shape[-1]
    y = torch.zeros_like(noise)
    y[..., :a] = leftover.to(noise)
    x_t = t * noise + (1 - t) * y
    v = velocity_fn(x_t, torch.full((noise.shape[0],), t, dtype=torch.float32, device=noise.device))
    est = x_t - t * v.to(x_t)
    return ((est[..., :a] - y[..., :a]) ** 2).mean(-1).float()


def transition_weights(residual, delay: int, leftover_len: int, horizon: int, l0: float, delta_ref: float = 1.0,
                       power: float = 1.0, l_min: float = 1.0, l_max: float = 50.0, window: int = 5):
    """residual (H,) of the old plan -> (weights (H,), L, delta); delta is nan when no leftover action follows the
    frozen part (then L = L0, and the weights after it are 0 anyway)."""
    r = torch.as_tensor(residual, dtype=torch.float64)
    near = r[delay: min(delay + window, leftover_len)]
    delta = float(near.mean().sqrt()) if len(near) else float("nan")
    if power == 0 or not len(near):
        length = float(l0)
    else:
        length = float(l0 * (max(delta, 1e-12) / delta_ref) ** power)
    length = min(max(length, l_min), l_max)
    j = torch.arange(horizon, dtype=torch.float64)
    w = torch.where(j < delay, torch.ones_like(j), torch.exp(-(j - delay) / length))
    w[j >= max(leftover_len, delay)] = 0.0
    return w.float(), length, delta
