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
    check_samples,
    load_progress,
    make_element,
    make_infer,
    pack_array,
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
