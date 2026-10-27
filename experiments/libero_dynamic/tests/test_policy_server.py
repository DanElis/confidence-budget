"""Wire format and request handling of policy_server.py (no model).

    cd experiments/libero_pro && uv run --with websockets --with msgpack python -m pytest -q ../libero_dynamic/tests
"""

import sys
from pathlib import Path

import msgpack
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import policy_server as ps  # noqa: E402


def test_pack_unpack_roundtrip():
    msg = {"actions": np.arange(20, dtype=np.float32).reshape(10, 2), "x": np.float32(1.5), "s": "ok"}
    raw = msgpack.Packer(default=ps.pack_array).pack(msg)
    back = msgpack.unpackb(raw, object_hook=ps.unpack_array)
    np.testing.assert_array_equal(back["actions"], msg["actions"])
    assert back["x"] == np.float32(1.5) and back["s"] == "ok"


def test_as_image_hwc_uint8_to_chw_unit():
    img = np.full((224, 224, 3), 255, dtype=np.uint8)
    t = ps.as_image(img)
    assert t.shape == (3, 224, 224) and float(t.max()) == 1.0


def obs(state, left=None, prompt="pick"):
    o = {"observation/state": np.asarray(state, dtype=np.float32), "prompt": prompt}
    if left is not None:
        o["prev_chunk_left_over"] = np.asarray(left, dtype=np.float32)
    return o


def test_request_noise_is_deterministic_and_depends_on_content():
    a = ps.request_noise(obs([0.1] * 8), 0, 50, 32)
    torch.testing.assert_close(a, ps.request_noise(obs([0.1] * 8), 0, 50, 32))
    assert a.shape == (1, 50, 32)
    for other in (obs([0.2] * 8), obs([0.1] * 8, prompt="place"), obs([0.1] * 8, left=np.zeros((3, 7)))):
        assert not torch.allclose(a, ps.request_noise(other, 0, 50, 32))
    assert not torch.allclose(a, ps.request_noise(obs([0.1] * 8), 1, 50, 32))


def test_leftover_tensor_pads_to_horizon():
    left, n = ps.leftover_tensor(obs([0] * 8, left=np.ones((4, 7))), 50)
    assert n == 4 and left.shape == (1, 50, 7)
    assert float(left[0, :4].sum()) == 28 and float(left[0, 4:].abs().sum()) == 0
    assert ps.leftover_tensor(obs([0] * 8), 50) == (None, 0)


def test_cli_checks():
    assert ps.parse_args([]).mode == "plain"
    ps.parse_args(["--mode", "rtc", "--n-samples", "4", "--select", "coherent"])
    for bad in (["--n-samples", "4"], ["--select", "coherent"], ["--mode", "rtc", "--mask-mode", "rtc_x_conf"]):
        with pytest.raises(SystemExit):
            ps.parse_args(bad)


def make_planner(**kw):
    """Server with only what plan_step needs (no model)."""
    from hstar import HStarRule

    args = ps.parse_args(["--mode", "rtc", "--n-samples", "4", "--adaptive", *kw.pop("argv", [])])
    srv = ps.Server.__new__(ps.Server)
    srv.args, srv.horizon = args, 50
    srv.rule = HStarRule("above", eta=args.eta, k=1, min_h=1)
    return srv


def test_plan_step_from_agreement():
    srv = make_planner(argv=["--eta", "3", "--s-max", "20"])
    z = np.zeros(50)
    z[12:] = 5.0  # the old plan leaves the cloud at index 12
    assert srv.plan_step(z, 4) == (12, 8)  # s = h* - d
    assert srv.plan_step(np.zeros(50), 4) == (50, 20)  # always agrees: s_max
    z2 = np.zeros(50)
    z2[2:] = 5.0
    assert srv.plan_step(z2, 4) == (2, 4)  # disagrees early: s_min = d
    assert srv.plan_step(None, 4) == (50, 4)  # first chunk of an episode: s_min
    srv2 = make_planner(argv=["--s-min", "6"])
    assert srv2.plan_step(z2, 4)[1] == 6


def test_adaptive_cli_checks():
    with pytest.raises(SystemExit):
        ps.parse_args(["--mode", "rtc", "--adaptive"])  # needs K > 1


def test_aac_h_star_matches_formula():
    rng = np.random.default_rng(0)
    for _ in range(50):
        disp = rng.uniform(0, 1, 50)
        mean = {h: disp[:h].mean() for h in range(1, 51)}
        gains = {h: mean[h + 1] - mean[h] for h in range(1, 50)}
        want = max(gains, key=lambda h: (gains[h], -h))
        assert ps.aac_h_star(disp) == want
    flat_then_jump = np.r_[np.full(10, 0.1), np.full(40, 1.0)]
    assert ps.aac_h_star(flat_then_jump) == 10
    assert ps.aac_h_star(flat_then_jump, min_h=20) == 20


def test_plan_step_aac():
    srv = make_planner(argv=["--rule", "aac", "--s-max", "20"])
    disp = np.r_[np.full(10, 0.1), np.full(40, 1.0)]
    assert srv.plan_step(None, 4, disp) == (10, 6)  # works on the first chunk too: needs only the new samples
    assert srv.plan_step(np.zeros(50), 4, np.full(50, 0.2))[1] == 4  # flat profile: h* = 1, s = s_min
