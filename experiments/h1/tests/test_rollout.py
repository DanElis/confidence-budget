"""Bookkeeping of the offline rollout with a fake generator: alignment, pairing, noise."""

import numpy as np
import pytest
import torch

from h1.rollout import (
    Generation,
    Observation,
    RolloutConfig,
    chunk_frames,
    frame_noise,
    run_episode,
    stack_records,
)

H, A, S, D = 50, 7, 25, 5


def _fake(calls):
    def get_observation(frame):
        return Observation(batch={"frame": frame}, gt_raw=np.full((H, A), frame, float), gt_pad=np.zeros(H, bool))

    def generate(batch, noise, leftover):
        calls.append((batch["frame"], noise.clone(), None if leftover is None else leftover.clone()))
        # chunk value encodes frame and condition: free = frame, rtc = frame + 0.5
        value = batch["frame"] + (0.0 if leftover is None else 0.5)
        pred = torch.full((H, A), float(value)) + torch.arange(H)[:, None] * 1e-3
        return Generation(pred, pred.numpy(), np.zeros((10, H)), np.zeros((10, H)), 500)

    return get_observation, generate


@pytest.mark.parametrize("source,offset", [("rtc", 0.5), ("free", 0.0)])
def test_leftover_is_previous_chunk_shifted_by_stride(source, offset):
    calls = []
    get_observation, generate = _fake(calls)
    cfg = RolloutConfig(H, D, S, H - S, prev_source=source)
    frames = chunk_frames(130, S, 3)
    records = run_episode(0, frames, get_observation, generate, (1, H, 32), cfg)

    assert frames == [3, 28, 53, 78, 103, 128]
    assert len(records) == 2 * (len(frames) - 1)
    rtc_calls = [c for c in calls if c[2] is not None]
    for k, (frame, _noise, leftover) in enumerate(rtc_calls, start=1):
        prev_frame = frames[k - 1]
        # the first frame only seeds the chain with the free chunk
        prev_value = prev_frame + (offset if k > 1 else 0.0)
        expected = prev_value + torch.arange(S, H)[:, None] * 1e-3
        assert frame == frames[k]
        assert leftover.shape == (H - S, A)
        assert torch.allclose(leftover, expected.expand(H - S, A))


def test_pairs_share_noise_previous_and_target():
    calls = []
    get_observation, generate = _fake(calls)
    cfg = RolloutConfig(H, D, S, H - S)
    records = run_episode(4, chunk_frames(80, S, 0), get_observation, generate, (1, H, 32), cfg)
    free_calls = {c[0]: c[1] for c in calls if c[2] is None}
    for c in calls:
        if c[2] is not None:
            assert torch.equal(c[1], free_calls[c[0]])
    for free, rtc in zip(records[::2], records[1::2], strict=True):
        assert (free.condition, rtc.condition) == ("free", "rtc")
        assert free.frame == rtc.frame
        assert np.array_equal(free.prev, rtc.prev, equal_nan=True)
        assert np.isnan(free.prev[H - S :]).all() and not np.isnan(free.prev[: H - S]).any()
    data = stack_records(records)
    assert data["pred"].shape == (len(records), H, A)
    assert data["is_rtc"].tolist() == [False, True] * (len(records) // 2)


def test_noise_is_deterministic_and_frame_specific():
    a = frame_noise(0, 1, 10, (1, H, 32))
    assert torch.equal(a, frame_noise(0, 1, 10, (1, H, 32)))
    assert not torch.equal(a, frame_noise(0, 1, 11, (1, H, 32)))
    assert not torch.equal(a, frame_noise(0, 2, 10, (1, H, 32)))


def test_config_rejects_impossible_schedules():
    with pytest.raises(ValueError):
        RolloutConfig(H, 30, 25, 25)  # d > s
    with pytest.raises(ValueError):
        RolloutConfig(H, 5, 25, 30)  # mask past the leftover


def test_stale_previous_chunk_is_planned_from_older_observation():
    calls, observed = [], []
    get_observation, generate = _fake(calls)

    def get_obs_logged(frame):
        observed.append(frame)
        return get_observation(frame)

    cfg = RolloutConfig(H, D, S, H - S, prev_source="stale", stale_frames=10)
    frames = chunk_frames(130, S, 40)  # 40, 65, 90, 115
    records = run_episode(0, frames, get_obs_logged, generate, (1, H, 32), cfg)
    assert len(records) == 2 * (len(frames) - 1)
    # for frame t the stale plan comes from frame t - s - 10
    for frame in frames[1:]:
        assert frame - S - 10 in observed
    rtc_calls = [c for c in calls if c[2] is not None]
    for (frame, _noise, leftover) in rtc_calls:
        source = frame - S - 10
        expected = source + torch.arange(S, H)[:, None] * 1e-3  # free chunk of the source frame, shifted by s
        assert torch.allclose(leftover, expected.expand(H - S, A))


def test_shift_offsets_leftover_but_not_gripper():
    calls = []
    get_observation, generate = _fake(calls)
    base = RolloutConfig(H, D, S, H - S)
    shifted = RolloutConfig(H, D, S, H - S, prev_shift=0.5)
    run_episode(0, chunk_frames(80, S, 0), get_observation, generate, (1, H, 32), base)
    plain = [c[2] for c in calls if c[2] is not None]
    calls.clear()
    records = run_episode(0, chunk_frames(80, S, 0), get_observation, generate, (1, H, 32), shifted)
    moved = [c[2] for c in calls if c[2] is not None]
    for a, b in zip(plain[:1], moved[:1]):  # first leftover comes from the same free chunk in both runs
        delta = (b - a)[0]
        assert torch.allclose(delta[:-1].abs(), torch.full((A - 1,), 0.5))
        assert delta[-1] == 0
        assert torch.allclose(b - a, delta.expand_as(b))  # constant over time
    rtc = [r for r in records if r.condition == "rtc"][0]
    assert np.allclose(rtc.prev[: H - S], moved[0].numpy())


def test_stale_frames_need_stale_source():
    with pytest.raises(ValueError):
        RolloutConfig(H, D, S, H - S, stale_frames=10)
