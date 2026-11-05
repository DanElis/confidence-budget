"""The openpi client of eval_openpi.py: observation, protocol, its place in the closed loop."""

import math
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from closed_loop import LoopConfig, run_batch  # noqa: E402
from eval_openpi import (  # noqa: E402
    check_guidance,
    check_samples,
    load_progress,
    make_element,
    make_infer,
    make_rtc_weights,
    pack_array,
    paper_weights,
    parse_args,
    progress_key,
    quat2axisangle,
    save_progress,
    unpack_array,
)
from planner import AdaptivePlanner, FixedPlanner, GivenHStar  # noqa: E402

H = 10


def observation():
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    half_turn_z = [0.0, 0.0, 1.0, 0.0]
    return {"pixels": {"image": image, "image2": image + 1},
            "robot_state": {"eef": {"pos": np.array([1.0, 2.0, 3.0]), "quat": np.array(half_turn_z)},
                            "gripper": {"qpos": np.array([0.04, -0.04])}}}


def test_quat2axisangle():
    np.testing.assert_allclose(quat2axisangle(np.array([0.0, 0.0, 1.0, 0.0])), [0, 0, math.pi])
    np.testing.assert_allclose(quat2axisangle(np.array([0.0, 0.0, 0.0, 1.0])), [0, 0, 0])
    np.testing.assert_allclose(quat2axisangle(np.array([0.0, 0.0, 0.0, 1.0 + 1e-9])), [0, 0, 0])  # rounding above 1


def test_element_rotates_images_and_builds_the_state():
    obs = observation()
    e = make_element(obs, "pick up the milk")
    np.testing.assert_array_equal(e["observation/image"], obs["pixels"]["image"][::-1, ::-1])
    np.testing.assert_array_equal(e["observation/wrist_image"], obs["pixels"]["image2"][::-1, ::-1])
    np.testing.assert_allclose(e["observation/state"], [1, 2, 3, 0, 0, math.pi, 0.04, -0.04])
    assert e["prompt"] == "pick up the milk" and e["observation/image"].dtype == np.uint8


class Env:
    def reset(self):
        return observation(), {}

    def step(self, action):
        self.last = action
        return observation(), 0.0, False, False, {}


class Client:
    def __init__(self, horizon=H):
        self.horizon, self.calls = horizon, 0

    def infer(self, element):
        assert set(element) == {"observation/image", "observation/wrist_image", "observation/state", "prompt", "noise"}
        self.calls += 1
        return {"actions": np.full((self.horizon, 7), self.calls, dtype=np.float64)}


def test_loop_executes_the_server_actions_as_returned():
    client = Client()
    res = run_batch([Env()], make_infer(client, "task", H), LoopConfig(H, 0, False, 12), FixedPlanner(H, 5), [0], (H, 7))
    assert [c["step"] for c in res.chunks] == [0, 5, 10] and client.calls == 3
    np.testing.assert_array_equal(res.actions[0, :, 0], [1] * 5 + [2] * 5 + [3] * 2)


def test_wrong_horizon_is_an_error_not_a_failed_episode():
    with pytest.raises(RuntimeError, match="expected"):
        run_batch([Env()], make_infer(Client(50), "task", H), LoopConfig(H, 0, False, 12), FixedPlanner(H, 5), [0], (H, 7))


def test_args():
    base = ["--suites", "pos:x0.2", "--out", "x"]
    ok = base + ["--stride", "5"]
    assert parse_args(ok).horizon == 10 and parse_args(ok).resolution == 256
    assert parse_args(base + ["--signal", "aac", "--n-samples", "20"]).stride is None
    assert parse_args(base + ["--signal", "aac", "--n-samples", "20", "--delay", "5"]).delay == 5
    assert parse_args(base + ["--stride", "6", "--delay", "4"]).stride == 6
    for bad in (base + ["--stride", "11"], base, base + ["--signal", "aac"],  # beyond H; no step; one sample
                base + ["--signal", "aac", "--n-samples", "20", "--stride", "5"],
                base + ["--stride", "3", "--delay", "4"], base + ["--stride", "7", "--delay", "4"],  # s < d; s + d > H
                base + ["--signal", "aac", "--n-samples", "20", "--delay", "6"]):
        with pytest.raises(SystemExit):
            parse_args(bad)


@pytest.mark.parametrize(("delay", "starts", "strides"), [(2, [0, 5, 10], [5, 5, 5]), (4, [0, 4, 8], [4, 4, 4])])
def test_aac_with_delay_keeps_the_old_chunk_for_d_steps(delay, starts, strides):
    client = SampleClient()
    infer = make_infer(client, "task", H, 20, aac=True)
    res = run_batch([Env()], infer, LoopConfig(H, delay, False, 12, 1), AdaptivePlanner(H, GivenHStar(), s_min=1), [0], (H, 32))
    assert [c["step"] for c in res.chunks] == starts and [c["stride"] for c in res.chunks] == strides  # s = max(7 - d, d)
    assert res.switch_steps[0][:2] == [0, starts[1] + delay]
    assert np.isfinite(res.actions[0]).all()  # the plan never runs out


class SampleClient:
    """As openpi_sample_server.py: the reply follows the noise, deterministically."""

    def __init__(self):
        self.noises = []

    def infer(self, element):
        noise = element["noise"]
        self.noises.append(noise)
        jump = np.zeros((len(noise), H, 7))
        jump[:, 7:] = noise[:, :1, :7]  # the samples start to disagree at index 7: Eq. 5 gives h* = 7
        env = np.zeros((len(noise), H, 7))
        env[..., 0] = 1.0  # xi = 4
        env[:, :, 1] = noise[:, 0, 0, None]
        return {"actions": env, "normalized_action": np.concatenate((jump, np.zeros((len(noise), H, 25))), axis=-1)}


def test_aac_runs_chunks_for_h_star_and_executes_sample_0():
    client = SampleClient()
    infer = make_infer(client, "task", H, 20, aac=True)
    res = run_batch([Env()], infer, LoopConfig(H, 0, False, 12, 1), AdaptivePlanner(H, GivenHStar(), s_min=1), [0], (H, 32))
    assert [c["step"] for c in res.chunks] == [0, 7] and [c["h_star"] for c in res.chunks] == [7, 7]
    assert all(n.shape == (20, H, 32) for n in client.noises)
    np.testing.assert_array_equal(res.actions[0, :7, 1], np.full(7, client.noises[0][0, 0, 0]))


def test_constant_step_sends_sample_0_of_aac():
    aac, const = SampleClient(), SampleClient()
    plan = AdaptivePlanner(H, GivenHStar(), s_min=1)
    run_batch([Env()], make_infer(aac, "t", H, 20, aac=True), LoopConfig(H, 0, False, 8, 1), plan, [3], (H, 32))
    run_batch([Env()], make_infer(const, "t", H), LoopConfig(H, 0, False, 8, 1), FixedPlanner(H, 7), [3], (H, 32))
    assert [len(n) for n in const.noises] == [1, 1]
    for a, c in zip(aac.noises, const.noises):
        np.testing.assert_array_equal(a[0], c[0])


def test_progress_round_trip_and_settings_guard(tmp_path):
    args = parse_args(["--suites", "pos:x0.2", "--out", "x", "--stride", "5"])
    key = progress_key(args)
    path = tmp_path / "suite.progress.json"
    assert load_progress(path, key) == ([], [], [], 0.0)
    save_progress(path, key, [{"task_id": 0}], [{"b": 1}], [{"c": 2}], 12.5)
    assert load_progress(path, key) == ([{"task_id": 0}], [{"b": 1}], [{"c": 2}], 12.5)
    assert not path.with_name(path.name + ".tmp").exists()
    other = progress_key(parse_args(["--suites", "pos:x0.2", "--out", "x", "--stride", "6"]))
    with pytest.raises(SystemExit):
        load_progress(path, other)
    assert progress_key(parse_args(["--suites", "pos:x0.2", "--out", "y", "--stride", "5", "--port", "9"])) == key


def test_check_samples():
    good = SampleClient()
    good.infer = lambda e: {"normalized_action": e["noise"]}
    assert check_samples(good, 4, H) == 0

    same = SampleClient()
    same.infer = lambda e: {"normalized_action": np.zeros_like(e["noise"])}
    with pytest.raises(RuntimeError, match="coincide"):
        check_samples(same, 4, H)

    def batch_dependent(e):
        return {"normalized_action": e["noise"] + len(e["noise"])}

    bad = SampleClient()
    bad.infer = batch_dependent
    with pytest.raises(RuntimeError, match="differs"):
        check_samples(bad, 4, H)


class RtcClient:
    """Each reply: the environment actions hold the call number, the plan in model units is distinct per call and index."""

    def __init__(self):
        self.elements = []

    def infer(self, element):
        self.elements.append(element)
        call = len(self.elements)
        plan = np.full((1, H, 32), float(call), dtype=np.float32)
        plan[0, :, 0] += np.arange(H) / 100
        return {"actions": np.full((1, H, 7), call, dtype=np.float64), "normalized_action": plan}


def fake_weights(delay, mask_end):
    return np.array([delay, mask_end] + [0] * (H - 2), dtype=np.float32)


@pytest.mark.parametrize(("stride", "mask_end", "left"), [(4, None, 6), (5, None, 5), (4, 5, 6)])
def test_rtc_sends_the_old_plan_in_model_units(stride, mask_end, left):
    client = RtcClient()
    infer = make_infer(client, "task", H, rtc_weights=fake_weights, max_weight=7.0)
    res = run_batch([Env()], infer, LoopConfig(H, 4, True, 2 * stride + 1, 1), FixedPlanner(H, stride, mask_end), [0], (H, 32))
    first, second, third = client.elements
    assert "rtc_target" not in first  # the first chunk has no old plan
    plan = np.full((H, 32), 1.0, dtype=np.float32)
    plan[:, 0] += np.arange(H) / 100
    np.testing.assert_array_equal(second["rtc_target"][:left], plan[stride:])
    assert (second["rtc_target"][left:] == 0).all()  # past the end of the old plan
    np.testing.assert_array_equal(second["rtc_weights"][:2], [4, H - stride if mask_end is None else mask_end])
    assert second["rtc_max_weight"] == 7.0
    # s >= d: chunk 2 has taken over by the third request, so it is the old plan there
    np.testing.assert_array_equal(third["rtc_target"][0, 1:], np.full(31, 2.0, dtype=np.float32))
    # the robot executes environment actions, and chunk 1 keeps running for d steps after the second request
    np.testing.assert_array_equal(res.actions[0, : stride + 4, 0], np.full(stride + 4, 1.0))
    assert res.actions[0, stride + 4, 0] == 2.0


@pytest.mark.parametrize(("end", "soft"), [(6, [0.3682, 0.0768]), (5, [0.2036])])
def test_paper_weights(end, soft):
    # c = 2/3, 1/3 at end = 6; c = 1/2 at end = 5; weight = c (e^c - 1) / (e - 1)
    w = paper_weights(4, end, H)
    assert w.shape == (H, 32)
    np.testing.assert_allclose(w[:4, :7], np.ones((4, 7)))
    np.testing.assert_allclose(w[4:end, 0], soft, atol=1e-3)
    assert (w[end:] == 0).all() and (w[:, 7:] == 0).all()  # past the old plan; the model's padding dimensions


def test_paper_mode_sends_the_jacobian_flag_and_beta_5():
    base = ["--suites", "pos:x0.2", "--out", "x", "--stride", "4", "--delay", "4"]
    paper = parse_args(base + ["--guidance", "paper"])
    assert paper.max_guidance_weight == 5.0
    assert parse_args(base + ["--guidance", "lerobot"]).max_guidance_weight == 10.0
    assert progress_key(paper)["guidance"] == "paper"

    client = RtcClient()
    infer = make_infer(client, "t", H, rtc_weights=lambda d, e: paper_weights(d, e, H), max_weight=5.0, jacobian=True)
    run_batch([Env()], infer, LoopConfig(H, 4, True, 5, 1), FixedPlanner(H, 4), [0], (H, 32))
    assert client.elements[1]["rtc_jacobian"] is True and client.elements[1]["rtc_max_weight"] == 5.0
    assert client.elements[1]["rtc_weights"].shape == (H, 32)


def test_rtc_and_naive_share_the_noise():
    rtc, naive = RtcClient(), RtcClient()
    plan = FixedPlanner(H, 4)
    run_batch([Env()], make_infer(rtc, "t", H, rtc_weights=fake_weights), LoopConfig(H, 4, True, 9, 1), plan, [3], (H, 32))
    run_batch([Env()], make_infer(naive, "t", H), LoopConfig(H, 4, False, 9, 1), plan, [3], (H, 32))
    assert len(rtc.elements) == len(naive.elements)
    for a, b in zip(rtc.elements, naive.elements):
        np.testing.assert_array_equal(a["noise"], b["noise"])
        assert "rtc_target" not in b


def test_rtc_args_and_progress_key():
    base = ["--suites", "pos:x0.2", "--out", "x"]
    rtc = base + ["--stride", "4", "--delay", "4", "--guidance", "lerobot"]
    assert parse_args(rtc).guidance == "lerobot" and parse_args(rtc + ["--mask-end", "5"]).mask_end == 5
    for bad in (base + ["--stride", "5", "--guidance", "lerobot"],  # no delay
                base + ["--signal", "aac", "--n-samples", "20", "--delay", "4", "--guidance", "lerobot"],
                base + ["--stride", "4", "--delay", "4", "--mask-end", "5"],  # a mask without RTC
                rtc + ["--mask-end", "3"], rtc + ["--mask-end", "7"]):  # mask_end < d; mask_end > H - s
        with pytest.raises(SystemExit):
            parse_args(bad)
    plain = progress_key(parse_args(base + ["--stride", "4", "--delay", "4"]))
    assert "guidance" not in plain  # runs without RTC resume from the progress files they already wrote
    assert progress_key(parse_args(rtc))["guidance"] == "lerobot"
    assert progress_key(parse_args(rtc)) != progress_key(parse_args(rtc + ["--mask-end", "5"]))


def test_rtc_weights_freeze_d_actions_and_end_at_the_mask():
    pytest.importorskip("lerobot")
    w = make_rtc_weights("exp", 10.0, H)(4, 6)
    assert w.shape == (H,)
    np.testing.assert_array_equal(w[:4], np.ones(4))
    np.testing.assert_array_equal(w[6:], np.zeros(4))
    assert ((w[4:6] >= 0) & (w[4:6] <= 1)).all() and w[4] >= w[5]
    np.testing.assert_array_equal(make_rtc_weights("exp", 10.0, H)(4, 4), [1, 1, 1, 1, 0, 0, 0, 0, 0, 0])


@pytest.mark.parametrize(("t", "weight"), [(1.0, 10.0), (0.5, 2.0), (0.1, 9.1111)])
def test_guidance_weight_table_shared_with_the_server(t, weight):
    """The same table is in test_openpi_sample_server.py, which runs in openpi's venv without LeRobot."""
    pytest.importorskip("lerobot")
    from h1.guidance import guidance_weight

    assert float(guidance_weight(t, 10.0)) == pytest.approx(weight, abs=1e-3)


class GuidanceServer:
    def __init__(self, frozen_error=0.0, ignore=False, zero_shift=0.0):
        self.frozen_error, self.ignore, self.zero_shift = frozen_error, ignore, zero_shift

    def infer(self, element):
        free = element["noise"].copy()
        if "rtc_target" not in element or self.ignore:
            return {"normalized_action": free}
        w = element["rtc_weights"][None, :, None]
        if not w.any():
            return {"normalized_action": free + self.zero_shift}
        return {"normalized_action": w * (element["rtc_target"][None] + self.frozen_error) + (1 - w) * free}


def test_check_guidance():
    weights = lambda delay, end: (np.arange(H) < delay).astype(np.float32)  # noqa: E731
    assert check_guidance(GuidanceServer(frozen_error=0.1), weights, 4, H, 10.0) == (0.0, pytest.approx(0.1, abs=1e-4))
    with pytest.raises(RuntimeError, match="frozen"):
        check_guidance(GuidanceServer(ignore=True), weights, 4, H, 10.0)  # openpi's stock server
    with pytest.raises(RuntimeError, match="zero weights"):
        check_guidance(GuidanceServer(zero_shift=0.1), weights, 4, H, 10.0)


def test_eval_libero_resolution_stays_lerobots_by_default():
    from eval_libero import parse_args as libero_args

    assert libero_args(["--out", "x"]).resolution is None
    assert libero_args(["--out", "x", "--resolution", "256"]).resolution == 256


def serve(handler):
    from websockets.sync.server import serve as ws_serve

    server = ws_serve(handler, "127.0.0.1", 0, compression=None, max_size=None)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.socket.getsockname()[1]


def test_protocol_round_trip():
    pytest.importorskip("websockets")
    msgpack = pytest.importorskip("msgpack")
    from eval_openpi import OpenPiClient

    def handler(ws):  # as openpi's server: metadata first, then a reply with timings per observation
        ws.send(msgpack.packb({"m": 1}, default=pack_array))
        obs = msgpack.unpackb(ws.recv(), object_hook=unpack_array)
        assert obs["prompt"] == "task" and obs["observation/image"].shape == (4, 4, 3)
        ws.send(msgpack.packb({"actions": np.ones((H, 7)), "server_timing": {"infer_ms": 1.0}}, default=pack_array))
        ws.recv()

    server, port = serve(handler)
    try:
        client = OpenPiClient("127.0.0.1", port)
        assert client.metadata == {"m": 1}
        reply = client.infer(make_element(observation(), "task"))
        np.testing.assert_array_equal(reply["actions"], np.ones((H, 7)))
        client.close()
    finally:
        server.shutdown()


def test_server_traceback_raises():
    pytest.importorskip("websockets")
    msgpack = pytest.importorskip("msgpack")
    from eval_openpi import OpenPiClient

    def handler(ws):
        ws.send(msgpack.packb({}, default=pack_array))
        ws.recv()
        ws.send("Traceback: boom")

    server, port = serve(handler)
    try:
        client = OpenPiClient("127.0.0.1", port)
        with pytest.raises(RuntimeError, match="boom"):
            client.infer(make_element(observation(), "task"))
    finally:
        server.shutdown()
