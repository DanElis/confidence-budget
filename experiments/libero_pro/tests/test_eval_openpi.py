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
from eval_openpi import make_element, make_infer, pack_array, parse_args, quat2axisangle, unpack_array  # noqa: E402
from planner import FixedPlanner  # noqa: E402

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
        assert set(element) == {"observation/image", "observation/wrist_image", "observation/state", "prompt"}
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
    ok = ["--suites", "pos:x0.2", "--out", "x", "--stride", "5"]
    assert parse_args(ok).horizon == 10 and parse_args(ok).resolution == 256
    with pytest.raises(SystemExit):
        parse_args(["--suites", "pos:x0.2", "--out", "x", "--stride", "11"])  # beyond H


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
