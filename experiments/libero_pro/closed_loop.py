"""Closed-loop chunked execution with simulated inference delay.

The simulator does not advance while the policy computes, so latency is
simulated by bookkeeping, as in the RTC paper's simulated benchmarks:

* the first chunk of an episode is computed synchronously (the robot waits);
* every ``stride`` steps (``s``) inference starts on the current observation;
  the robot keeps executing the previous chunk for ``delay`` more steps
  (``d``), then switches to the new chunk at index ``d`` (its first ``d``
  actions are already in the past);
* with guidance, the new chunk is generated with RTC toward the previous
  chunk's leftover (indices ``s`` onward, aligned with the inference start).

``delay = 0`` without guidance is ordinary synchronous chunking; ``delay > 0``
without guidance is naive asynchronous switching. All sub-environments of a
batch share the schedule, so inference is batched.

The policy and the environments are injected, so the schedule is testable
without a simulator or a model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class LoopConfig:
    horizon: int  # H, actions per chunk
    stride: int  # s, steps between inference starts
    delay: int  # d, simulated inference latency in control steps (0 = synchronous)
    guidance: bool  # RTC guidance toward the previous chunk
    max_steps: int
    mask_end: int | None = None  # end of the RTC soft mask; default H - s
    seed: int = 0

    def __post_init__(self):
        h, s, d = self.horizon, self.stride, self.delay
        if not (1 <= s and 0 <= d <= s and s + d <= h):
            raise ValueError(f"need 1 <= stride, 0 <= delay <= stride, stride + delay <= horizon; got s={s}, d={d}, H={h}")
        if self.mask_end is None:
            self.mask_end = h - s
        if not d <= self.mask_end <= h - s:
            raise ValueError(f"need delay <= mask_end <= horizon - stride, got mask_end={self.mask_end}")


@dataclass
class Chunk:
    model: torch.Tensor  # (B, H, A) policy-space actions; leftovers for guidance are sliced from it
    actions: np.ndarray  # (B, H, A) environment actions
    entropy: np.ndarray | None = None  # (B, H) cross-attention entropy (last denoising step), optional


# infer(observations, noise, leftover, delay, mask_end) -> Chunk
Infer = Callable[[list, torch.Tensor, torch.Tensor | None, int, int], Chunk]


@dataclass
class BatchResult:
    success: np.ndarray  # (B,) bool
    length: np.ndarray  # (B,) steps until success or the end
    actions: np.ndarray  # (B, max_steps, A) executed actions, NaN after the episode ended
    switch_steps: list[int]  # control steps at which a new chunk took over
    n_inference: int
    chunk_steps: list[int] = field(default_factory=list)  # inference start step of each logged chunk
    chunk_entropy: list[np.ndarray] = field(default_factory=list)  # (B, H) per chunk, if available


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
    episode_ids: Sequence[int],
    noise_shape: tuple[int, int],
) -> BatchResult:
    """Roll out one episode in each env with the shared chunk schedule."""
    n = len(envs)
    observations = [env.reset()[0] for env in envs]
    done = np.zeros(n, dtype=bool)
    success = np.zeros(n, dtype=bool)
    length = np.full(n, cfg.max_steps, dtype=np.int64)
    executed = None
    switch_steps: list[int] = [0]
    chunk_steps: list[int] = []
    chunk_entropy: list[np.ndarray] = []

    def call(step: int, leftover, delay: int) -> Chunk:
        chunk = infer(observations, chunk_noise(cfg.seed, episode_ids, step, noise_shape), leftover, delay, cfg.mask_end)
        chunk_steps.append(step)
        if chunk.entropy is not None:
            chunk_entropy.append(np.asarray(chunk.entropy, dtype=np.float32))
        return chunk

    active, active_start = call(0, None, 0), 0
    n_inference = 1
    pending, pending_start, switch_at = None, None, None
    next_start = cfg.stride

    for t in range(cfg.max_steps):
        if done.all():
            break
        if t == next_start:
            leftover = active.model[:, t - active_start :] if cfg.guidance else None
            pending, pending_start, switch_at = call(t, leftover, cfg.delay), t, t + cfg.delay
            n_inference += 1
            next_start = t + cfg.stride
        if pending is not None and t == switch_at:
            active, active_start, pending = pending, pending_start, None
            switch_steps.append(t)

        action = active.actions[:, t - active_start]
        if executed is None:
            executed = np.full((n, cfg.max_steps, action.shape[-1]), np.nan, dtype=np.float32)
        for b in np.flatnonzero(~done):
            observations[b], _reward, terminated, truncated, info = envs[b].step(action[b])
            executed[b, t] = action[b]
            if info.get("is_success", False):
                success[b], done[b], length[b] = True, True, t + 1
            elif terminated or truncated:
                done[b], length[b] = True, t + 1

    return BatchResult(
        success=success,
        length=length,
        actions=executed,
        switch_steps=switch_steps,
        n_inference=n_inference,
        chunk_steps=chunk_steps,
        chunk_entropy=chunk_entropy,
    )


def stack_observations(observations: list):
    """Stack a list of (nested) observation dicts of numpy arrays along a new batch axis."""
    first = observations[0]
    if isinstance(first, dict):
        return {key: stack_observations([obs[key] for obs in observations]) for key in first}
    return np.stack([np.asarray(obs) for obs in observations])


def switch_jumps(actions: np.ndarray, switch_steps: Sequence[int]) -> tuple[float, float]:
    """Mean L2 action change at chunk switches vs at all other steps (NaN-aware). Ignores step 0."""
    diff = np.linalg.norm(np.diff(actions, axis=1), axis=-1)  # (B, T-1), diff[:, t-1] = a_t - a_{t-1}
    mask = np.zeros(diff.shape[1], dtype=bool)
    for t in switch_steps:
        if 1 <= t <= diff.shape[1]:
            mask[t - 1] = True
    at_switch = diff[:, mask]
    other = diff[:, ~mask]
    mean = lambda x: float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")  # noqa: E731
    return mean(at_switch), mean(other)
