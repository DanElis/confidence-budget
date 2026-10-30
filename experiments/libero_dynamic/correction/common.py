"""Shared pieces of the correction head (A2C2-style, arXiv 2509.23224) on LIBERO-Dynamic.

Image conventions (checked against LIBERO-Dynamic @b66cb6b and its training release):
* the release's LeRobot v2.1 videos (pi05_raw) are the simulator cameras at 256x256, already rotated by 180 degrees
  (generate_dynamic_demos.py: ``obs[key][::-1, ::-1]``), the same rotation the benchmark client applies
  (eval_dynamic_libero_style.OpenPIWebsocketPolicy._preprocess_image) before resize_with_pad to 224;
* the state is eef pos + axis-angle + gripper qpos (8), as the client sends it;
* actions are in environment space, gripper -1 open / +1 close.
So pi0.5 gets ``client_image(frame)`` in both the offline replay and closed loop, and the head gets
``head_image(frame)`` of the rotated 256 frame (the correction server rotates the raw simulator frame first).

Async schedule of the runner's rtc_queue (realtime_eval/control.py, fixed latency, d = ceil(latency * 20)):
a request at tick t_r returns a chunk of K actions at tick t_r + d; its first d actions are dropped and the queue is
replaced, so it executes indices j = d .. d + s - 1 at ticks t_r + j, where s is the time to the next request
(with --prefetch-threshold 10 >= K - d the next request goes out at the arrival tick: s = d; with a lower threshold
q, s = d + max(0, K - d - q)). The leftover sent with the next request is the unexecuted original chunk at that
moment, chunk[s:K]. ``request_schedule`` reproduces this along a demonstration; tests/test_correction.py checks it
against the runner's own classes.
"""

from __future__ import annotations

import numpy as np

REPO = "rsj2003/libero-dynamic-strict-nonbenchmark-1800"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
CAMERAS = ("observation.images.image", "observation.images.wrist_image")  # agentview, wrist
HEAD_SIZE = 128
CLIENT_SIZE = 224
ACTION_DIM = 7
STATE_DIM = 8


def client_image(frame: np.ndarray, size: int = CLIENT_SIZE) -> np.ndarray:
    """openpi_client.image_tools.resize_with_pad for a square frame (PIL bilinear, no padding needed), uint8."""
    from PIL import Image

    frame = np.asarray(frame, dtype=np.uint8)
    if frame.shape[:2] == (size, size):
        return frame
    if frame.shape[0] != frame.shape[1]:
        raise ValueError(f"expected a square frame, got {frame.shape}")
    return np.asarray(Image.fromarray(frame).resize((size, size), resample=Image.BILINEAR))


def head_image(frame: np.ndarray, size: int = HEAD_SIZE) -> np.ndarray:
    """Rotated 256x256 RGB frame -> the head's input (size x size, area interpolation), uint8."""
    import cv2

    frame = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))
    if frame.shape[:2] == (size, size):
        return frame
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def request_schedule(length: int, delay: int, steps, offset: int = 0) -> list[tuple[int, int]]:
    """Requests (t_r, s) of the async RTC process along an episode of ``length`` ticks, first request at ``offset``.

    ``steps`` is an int (constant s) or a callable k -> s_k. Request k executes chunk indices [delay, delay + s_k)
    at ticks t_r + j; requests whose chunk would arrive after the episode end are dropped."""
    if delay < 0:
        raise ValueError("delay must be >= 0")
    out, t, k = [], int(offset), 0
    while t + delay < length:
        s = int(steps(k) if callable(steps) else steps)
        if s < max(delay, 1):
            raise ValueError(f"step {s} < delay {delay}: the runner cannot request while a chunk is in flight")
        out.append((t, s))
        t += s
        k += 1
    return out


def executed(length: int, t_r: int, s: int, delay: int) -> range:
    """Ticks at which the chunk requested at t_r is executed (j = t - t_r)."""
    return range(t_r + delay, min(t_r + delay + s, length))
