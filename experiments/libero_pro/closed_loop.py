"""Closed-loop chunked execution with simulated inference delay and per-chunk planning.

The simulator does not advance while the policy computes, so latency is
simulated by bookkeeping, as in the RTC paper's simulated benchmarks:

* the first chunk of an episode is computed synchronously (the robot waits);
* inference for chunk k+1 starts s_k steps after inference for chunk k started,
  on the observation of that step; the robot keeps executing chunk k for
  ``delay`` (d) more steps, then switches to chunk k+1 at its index d (its
  first d actions are already in the past);
* with guidance, chunk k+1 is generated with RTC toward chunk k's leftover
  (indices s_k onward), frozen for d actions and softly attached up to the
  mask end chosen when chunk k was planned.

s_k and the mask are chosen by a planner from chunk k (see planner.py):
constant for ordinary RTC, from the entropy for the confidence budget. Each
environment of a batch follows its own schedule; environments that need a
chunk at the same step are batched into one policy call.

``delay = 0`` without guidance is ordinary synchronous chunking; ``delay > 0``
without guidance is naive asynchronous switching.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class LoopConfig:
    horizon: int  # H, actions per chunk
    delay: int  # d, simulated inference latency in control steps (0 = synchronous)
    guidance: bool  # RTC guidance toward the previous chunk
    max_steps: int
    seed: int = 0

    def __post_init__(self):
        if not 0 <= self.delay < self.horizon:
            raise ValueError(f"need 0 <= delay < horizon, got d={self.delay}, H={self.horizon}")


@dataclass
class Chunk:
    model: torch.Tensor  # (B, H, A) policy-space actions; leftovers for guidance are sliced from it
    actions: np.ndarray  # (B, H, A) environment actions
    entropy: np.ndarray | None = None  # (B, H) cross-attention entropy (last denoising step)
    n_valid: np.ndarray | None = None  # (B,) number of valid prefix tokens (entropy normaliser)
    # (B, H) per-action signal the planner reads instead of the entropy (e.g. dispersion across noise samples)
    signal: np.ndarray | None = None
    # name -> (B, ...) arrays stored in the chunk log when keep_entropy (e.g. "dispersion", "plan")
    extras: dict = field(default_factory=dict)
    # (B, H) dispersion across noise samples. When an infer returns it, the loop keeps it with the chunk and passes
    # the next infer the old plan even without guidance: leftover, mask_end, plus keyword arguments
    # leftover_disp (B, H) (aligned with the leftover, zero-padded) and leftover_len (B,)
    dispersion: np.ndarray | None = None


# infer(observations, noise (B,H,A_max), leftover (B,H,A) zero-padded | None, delay, mask_end (B,) | None
#       [, leftover_disp=..., leftover_len=...]) -> Chunk
Infer = Callable[..., Chunk]


@dataclass
class BatchResult:
    success: np.ndarray  # (B,) bool
    length: np.ndarray  # (B,) steps until success or the end
    actions: np.ndarray  # (B, max_steps, A) executed actions, NaN after the episode ended
    switch_steps: list[list[int]]  # per env: control steps at which a new chunk took over
    n_inference: np.ndarray  # (B,) policy calls per env
    chunks: list[dict] = field(default_factory=list)  # per generated chunk: env, step, plan, entropy


def chunk_noise(seed: int, episode_ids: Sequence[int], step: int, shape: tuple[int, int]) -> torch.Tensor:
    """(B, H, A_max) noise, deterministic per (seed, episode, step) so conditions share it."""
    out = []
    for episode in episode_ids:
        g = torch.Generator().manual_seed((seed * 1_000_003 + int(episode) * 10_007 + step) % (2**63))
        out.append(torch.randn(shape, generator=g, dtype=torch.float32))
    return torch.stack(out)


def run_batch(
    envs: Sequence,
    infer: Infer,
    cfg: LoopConfig,
    planner,
    episode_ids: Sequence[int],
    noise_shape: tuple[int, int],
    keep_entropy: bool = False,
) -> BatchResult:
    """Roll out one episode in each env; each env follows the planner's schedule."""
    n, horizon, d = len(envs), cfg.horizon, cfg.delay
    # Environments with step_async / step_wait (and reset_async / reset_wait) are driven concurrently.
    parallel = all(hasattr(env, "step_async") and hasattr(env, "step_wait") for env in envs)
    if parallel and all(hasattr(env, "reset_async") for env in envs):
        for env in envs:
            env.reset_async()
        observations = [env.reset_wait()[0] for env in envs]
    else:
        observations = [env.reset()[0] for env in envs]
    done = np.zeros(n, dtype=bool)
    success = np.zeros(n, dtype=bool)
    length = np.full(n, cfg.max_steps, dtype=np.int64)
    executed = None
    n_inference = np.zeros(n, dtype=np.int64)
    switch_steps: list[list[int]] = [[] for _ in range(n)]
    chunks: list[dict] = []

    active_model: list[torch.Tensor | None] = [None] * n
    active_actions: list[np.ndarray | None] = [None] * n
    active_start = np.zeros(n, dtype=np.int64)
    active_disp: list[np.ndarray | None] = [None] * n
    pending: list[tuple | None] = [None] * n  # (model, actions, start, switch_at, dispersion)
    next_start = np.zeros(n, dtype=np.int64)
    next_mask = np.zeros(n, dtype=np.int64)

    def generate(t: int, due: list[int]) -> None:
        first = active_model[due[0]] is None
        with_disp = not first and active_disp[due[0]] is not None
        leftover, mask, extra = None, None, {}
        if (cfg.guidance or with_disp) and not first:
            leftover = torch.zeros(len(due), horizon, active_model[due[0]].shape[-1])
            for i, b in enumerate(due):
                rest = active_model[b][t - active_start[b] :]
                leftover[i, : rest.shape[0]] = rest.detach().float().cpu()
            mask = torch.as_tensor(next_mask[due], dtype=torch.long)
        if with_disp:
            disp = torch.zeros(len(due), horizon)
            for i, b in enumerate(due):
                rest = active_disp[b][t - active_start[b] :]
                disp[i, : rest.shape[0]] = torch.as_tensor(rest, dtype=torch.float32)
            extra = {"leftover_disp": disp,
                     "leftover_len": torch.as_tensor([horizon - (t - active_start[b]) for b in due])}
        noise = chunk_noise(cfg.seed, [episode_ids[b] for b in due], t, noise_shape)
        chunk = infer([observations[b] for b in due], noise, leftover, 0 if first else d, mask, **extra)
        for i, b in enumerate(due):
            entropy = None if chunk.entropy is None else np.asarray(chunk.entropy[i])
            n_valid = None if chunk.n_valid is None else int(chunk.n_valid[i])
            signal = entropy if chunk.signal is None else np.asarray(chunk.signal[i])
            plan = planner(signal, n_valid, d)
            switch = t if first else t + d
            disp_i = None if chunk.dispersion is None else np.asarray(chunk.dispersion[i], dtype=np.float32)
            pending[b] = (chunk.model[i].detach(), np.asarray(chunk.actions[i]), t, switch, disp_i)
            next_start[b] = t + plan.stride
            next_mask[b] = plan.mask_end
            n_inference[b] += 1
            record = {"env": b, "step": t, "stride": plan.stride, "mask_end": plan.mask_end,
                      "h_star": plan.h_star, "lam": plan.lam}
            if keep_entropy and entropy is not None:
                record["entropy"] = entropy.astype(np.float32).tolist()
                record["n_valid"] = n_valid
            if keep_entropy:
                for name, values in chunk.extras.items():
                    record[name] = np.asarray(values[i], dtype=np.float32).round(5).tolist()
            chunks.append(record)

    def activate(t: int) -> None:
        for b in range(n):
            if pending[b] is not None and pending[b][3] == t:
                active_model[b], active_actions[b], active_start[b], _, active_disp[b] = pending[b]
                pending[b] = None
                switch_steps[b].append(t)

    for t in range(cfg.max_steps):
        if done.all():
            break
        activate(t)  # a chunk started d steps ago takes over before the next inference reads its leftover
        due = [b for b in range(n) if not done[b] and next_start[b] == t]
        if due:
            generate(t, due)
            activate(t)  # synchronous chunks (first chunk, d = 0) take over immediately

        running = np.flatnonzero(~done)
        actions_t = {}
        for b in running:
            action = active_actions[b][t - active_start[b]]
            if executed is None:
                executed = np.full((n, cfg.max_steps, action.shape[-1]), np.nan, dtype=np.float32)
            actions_t[b] = action
            if parallel:
                envs[b].step_async(action)  # remote simulators step concurrently
        for b in running:
            step = envs[b].step_wait() if parallel else envs[b].step(actions_t[b])
            observations[b], _reward, terminated, truncated, info = step
            executed[b, t] = actions_t[b]
            if info.get("is_success", False):
                success[b], done[b], length[b] = True, True, t + 1
            elif terminated or truncated:
                done[b], length[b] = True, t + 1

    return BatchResult(success, length, executed, switch_steps, n_inference, chunks)


def stack_observations(observations: list):
    """Stack a list of (nested) observation dicts of numpy arrays along a new batch axis."""
    first = observations[0]
    if isinstance(first, dict):
        return {key: stack_observations([obs[key] for obs in observations]) for key in first}
    return np.stack([np.asarray(obs) for obs in observations])


def switch_jumps(actions: np.ndarray, switch_steps: Sequence[Sequence[int]]) -> tuple[float, float]:
    """Mean L2 action change at chunk switches vs at all other steps (NaN-aware). Ignores step 0."""
    diff = np.linalg.norm(np.diff(actions, axis=1), axis=-1)  # (B, T-1), diff[:, t-1] = a_t - a_{t-1}
    mask = np.zeros(diff.shape, dtype=bool)
    for b, steps in enumerate(switch_steps):
        for t in steps:
            if 1 <= t <= diff.shape[1]:
                mask[b, t - 1] = True
    finite = np.isfinite(diff)
    at_switch = diff[mask & finite]
    other = diff[~mask & finite]
    return (float(at_switch.mean()) if at_switch.size else float("nan"),
            float(other.mean()) if other.size else float("nan"))


def sparc(speed: np.ndarray, fs: float, padlevel: int = 4, fc: float = 10.0, amp_th: float = 0.05) -> float:
    """Spectral arc length of a speed profile (Balasubramanian et al., 2015). Closer to 0 = smoother."""
    speed = np.asarray(speed, dtype=np.float64)
    speed = speed[np.isfinite(speed)]
    if speed.size < 4 or np.allclose(speed, 0):
        return float("nan")
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + padlevel))
    freq = np.arange(0, fs, fs / nfft)
    mag = np.abs(np.fft.fft(speed, nfft))
    mag = mag / mag.max()
    sel = freq <= fc
    freq, mag = freq[sel], mag[sel]
    above = np.flatnonzero(mag >= amp_th)
    if above.size == 0:
        return float("nan")
    freq, mag = freq[: above[-1] + 1], mag[: above[-1] + 1]
    return float(-np.sum(np.sqrt((np.diff(freq) / (freq[-1] - freq[0])) ** 2 + np.diff(mag) ** 2)))
