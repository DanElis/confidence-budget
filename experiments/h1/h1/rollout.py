"""Offline asynchronous rollout over one demonstration episode.

Inference for chunk k starts at frame t_k = t_0 + k * s. Under RTC the robot
has executed s actions of the previous chunk since its inference started, so
the previous chunk's leftover from index s onward is aligned with frame t_k;
its first d actions are frozen (inference delay) and the next ones are
softly attached up to ``mask_end``.

At every frame two chunks are generated from the same observation and the same
noise: one freely (no guidance) and one with RTC guidance. The pair shares the
previous leftover so both can be scored against it. The first frame only seeds
the chain and is not recorded.

The observation source and the generator are injected, so the bookkeeping can
be tested without a model or a dataset.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Observation:
    batch: dict
    gt_raw: np.ndarray  # (H, A) future demonstration actions, raw units
    gt_pad: np.ndarray  # (H,) True where the future runs past the episode end


@dataclass
class Generation:
    pred: torch.Tensor  # (H, A) chunk in the model's normalised action space
    pred_raw: np.ndarray  # (H, A) chunk in raw units
    entropy: np.ndarray  # (S, H) per denoising step
    nonlang_share: np.ndarray  # (S, H)
    n_valid: int


@dataclass
class RolloutConfig:
    horizon: int
    delay: int
    stride: int
    mask_end: int
    prev_source: str = "rtc"  # chain from the previous RTC chunk ("rtc") or free chunk ("free")
    seed: int = 0

    def __post_init__(self):
        h, d, s, m = self.horizon, self.delay, self.stride, self.mask_end
        if not 0 <= d <= s <= h - d:
            raise ValueError(f"need 0 <= delay <= stride <= horizon - delay, got d={d}, s={s}, H={h}")
        if not d <= m <= h - s:
            raise ValueError(f"need delay <= mask_end <= horizon - stride, got mask_end={m}")
        if self.prev_source not in ("rtc", "free"):
            raise ValueError(f"prev_source must be 'rtc' or 'free', got {self.prev_source!r}")


@dataclass
class ChunkRecord:
    episode: int
    frame: int
    step: int  # k, index of the chunk in the episode sequence (k >= 1)
    condition: str  # "free" or "rtc"
    pred: np.ndarray
    pred_raw: np.ndarray
    gt_raw: np.ndarray
    gt_pad: np.ndarray
    prev: np.ndarray  # (H, A) previous leftover aligned to this chunk, NaN past H - s
    entropy: np.ndarray
    nonlang_share: np.ndarray
    n_valid: int
    meta: dict = field(default_factory=dict)


def chunk_frames(length: int, stride: int, offset: int, max_chunks: int | None = None) -> list[int]:
    """Frames t_0 = offset, offset + stride, ... inside an episode of ``length`` frames."""
    frames = list(range(offset, length, stride))
    if max_chunks is not None:
        frames = frames[:max_chunks]
    return frames


def frame_noise(seed: int, episode: int, frame: int, shape: tuple[int, ...]) -> torch.Tensor:
    """Deterministic standard-normal noise shared by the free and RTC chunks of one frame."""
    generator = torch.Generator().manual_seed((seed * 1_000_003 + episode * 10_007 + frame) % (2**63))
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def run_episode(
    episode: int,
    frames: list[int],
    get_observation: Callable[[int], Observation],
    generate: Callable[[dict, torch.Tensor, torch.Tensor | None], Generation],
    noise_shape: tuple[int, ...],
    cfg: RolloutConfig,
) -> list[ChunkRecord]:
    """Generate paired free/RTC chunks along ``frames`` (spaced by ``cfg.stride``)."""
    if any(b - a != cfg.stride for a, b in zip(frames, frames[1:], strict=False)):
        raise ValueError("frames must be spaced exactly by the stride")
    records: list[ChunkRecord] = []
    chain: torch.Tensor | None = None
    for k, frame in enumerate(frames):
        obs = get_observation(frame)
        noise = frame_noise(cfg.seed, episode, frame, noise_shape)
        free = generate(obs.batch, noise, None)
        if chain is None:
            chain = free.pred
            continue

        leftover = chain[cfg.stride :]  # (H - s, A), aligned with this frame
        rtc = generate(obs.batch, noise, leftover)

        prev = np.full(free.pred.shape, np.nan, dtype=np.float32)
        prev[: leftover.shape[0]] = leftover.detach().float().cpu().numpy()
        for condition, gen in (("free", free), ("rtc", rtc)):
            records.append(
                ChunkRecord(
                    episode=episode,
                    frame=frame,
                    step=k,
                    condition=condition,
                    pred=gen.pred.detach().float().cpu().numpy(),
                    pred_raw=np.asarray(gen.pred_raw, dtype=np.float32),
                    gt_raw=np.asarray(obs.gt_raw, dtype=np.float32),
                    gt_pad=np.asarray(obs.gt_pad, dtype=bool),
                    prev=prev,
                    entropy=np.asarray(gen.entropy, dtype=np.float32),
                    nonlang_share=np.asarray(gen.nonlang_share, dtype=np.float32),
                    n_valid=int(gen.n_valid),
                )
            )
        chain = rtc.pred if cfg.prev_source == "rtc" else free.pred
    return records


def stack_records(records: list[ChunkRecord]) -> dict[str, np.ndarray]:
    """Arrays for ``np.savez``: one row per chunk."""
    if not records:
        raise ValueError("no records to stack")
    return {
        "episode": np.array([r.episode for r in records], dtype=np.int64),
        "frame": np.array([r.frame for r in records], dtype=np.int64),
        "step": np.array([r.step for r in records], dtype=np.int64),
        "is_rtc": np.array([r.condition == "rtc" for r in records], dtype=bool),
        "pred": np.stack([r.pred for r in records]),
        "pred_raw": np.stack([r.pred_raw for r in records]),
        "gt_raw": np.stack([r.gt_raw for r in records]),
        "gt_pad": np.stack([r.gt_pad for r in records]),
        "prev": np.stack([r.prev for r in records]),
        "entropy": np.stack([r.entropy for r in records]),
        "nonlang_share": np.stack([r.nonlang_share for r in records]),
        "n_valid": np.array([r.n_valid for r in records], dtype=np.int64),
    }
