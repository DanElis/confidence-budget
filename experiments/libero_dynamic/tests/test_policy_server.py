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
    assert srv.plan_step(np.zeros(50), 4, np.full(50, 0.25))[1] == 4  # flat (exact in binary): h* = 1, s = s_min


# ---------------------------------------------------------------- transition mask

from selection import plan_residual, transition_weights  # noqa: E402


def wants(target):
    """Velocity of a policy that denoises everything to ``target``: x_t - t * v = target."""
    return lambda x_t, time: (x_t - target) / time[:, None, None]


def test_plan_residual_is_zero_for_the_wanted_plan_and_grows_with_the_deviation():
    torch.manual_seed(0)
    noise, plan = torch.randn(1, 50, 32), torch.randn(1, 50, 7)
    target = torch.zeros(1, 50, 32)
    target[..., :7] = plan
    torch.testing.assert_close(plan_residual(wants(target), plan, noise, 0.5), torch.zeros(1, 50))
    target[0, 20:, :7] += 0.5  # the new frame wants something else from index 20 on
    r = plan_residual(wants(target), plan, noise, 0.5)
    assert float(r[0, :20].abs().max()) < 1e-10
    torch.testing.assert_close(r[0, 20:], torch.full((30,), 0.25))
    target[0, :, 7:] = 3.0  # padded dimensions are ignored
    torch.testing.assert_close(plan_residual(wants(target), plan, noise, 0.5)[0, 20:], torch.full((30,), 0.25))


def test_transition_weights_shape_and_length():
    r = torch.zeros(50)
    r[4:9] = 0.04  # delta = 0.2 near the stitch (d = 4, window 5)
    w, length, delta = transition_weights(r, 4, 30, 50, l0=6, delta_ref=0.1, power=1)
    assert delta == pytest.approx(0.2) and length == pytest.approx(12)  # twice the reference: twice as long
    assert torch.all(w[:4] == 1) and float(w[4]) == 1.0
    torch.testing.assert_close(w[4:30], torch.exp(-torch.arange(26.0) / 12))
    assert torch.all(w[30:] == 0)  # no old plan beyond the leftover
    assert transition_weights(r, 4, 30, 50, l0=6, delta_ref=0.4, power=1)[1] == pytest.approx(3)  # closer: shorter
    assert transition_weights(r, 4, 30, 50, l0=6, delta_ref=0.1, power=0)[1] == 6  # control: constant
    assert transition_weights(r, 4, 30, 50, l0=6, delta_ref=1e-6, power=1, l_max=20)[1] == 20
    assert transition_weights(r * 0, 4, 30, 50, l0=6, delta_ref=0.1, power=1, l_min=1)[1] == 1
    w, length, delta = transition_weights(r, 4, 3, 50, l0=6)  # leftover ends inside the frozen part
    assert np.isnan(delta) and length == 6 and torch.all(w[4:] == 0) and torch.all(w[:4] == 1)


class FakeRTC:
    weight_override = None


def make_transition_server(monkeypatch, argv=()):
    import shared_prefix

    args = ps.parse_args(["--mode", "rtc", "--mask-mode", "transition", "--device", "cpu", *argv])
    srv = ps.Server.__new__(ps.Server)
    srv.args, srv.horizon, srv.a_max, srv.apc, srv.exec_h = args, 50, 32, 10, 10
    srv.policy, srv.rtc, srv.log = None, FakeRTC(), None
    calls = []

    def fake_predict(policy, batch, repeats, noise, before_integrate=None, **kw):
        new = torch.zeros(1, 50, 32)
        new[0, :, :7] = batch["want"]
        if before_integrate is not None:
            before_integrate(wants(new))
        calls.append(dict(kw, weights=None if srv.rtc.weight_override is None else srv.rtc.weight_override.clone()))
        return batch["want"][None].clone()

    monkeypatch.setattr(shared_prefix, "predict_chunk_shared", fake_predict)
    return srv, calls


def test_transition_uses_the_kept_plan_and_lengthens_for_a_far_new_plan(monkeypatch):
    srv, calls = make_transition_server(monkeypatch, ["--trans-len", "6", "--trans-ref", "0.1"])
    state = {}
    first = torch.zeros(50, 7)
    chunk, plan = srv.infer_transition({"want": first}, torch.randn(1, 50, 32), None, 0, 4, torch.tensor([10]), state)
    assert plan == {"sel": 0, "len": 10} and calls[-1]["weights"] is None and torch.equal(state["full"], first)
    # the client executed 7 of the 10 returned actions; its leftover is the last 3, the old plan is full[7:]
    left, left_len = ps.leftover_tensor({"prev_chunk_left_over": first[7:10].numpy()}, 50)
    near = torch.zeros(50, 7)
    _, p_near = srv.infer_transition({"want": near}, torch.randn(1, 50, 32), left, left_len, 4, torch.tensor([10]),
                                     dict(state))
    assert calls[-1]["prev_chunk_left_over"].shape == (1, 50, 7)
    w = calls[-1]["weights"][0]
    assert torch.all(w[:4] == 1) and torch.all(w[43:] == 0) and float(w[42]) > 0  # 43 actions of old plan
    assert p_near["L"] == 1.0 and p_near["delta"] == 0  # the new frame agrees: shortest transition
    far = torch.zeros(50, 7)
    far[:, 0] = 0.2 * np.sqrt(7)  # RMS deviation 0.2 = twice the reference
    _, p_far = srv.infer_transition({"want": far}, torch.randn(1, 50, 32), left, left_len, 4, torch.tensor([10]),
                                    dict(state))
    assert p_far["delta"] == pytest.approx(0.2, rel=1e-4) and p_far["L"] == pytest.approx(12, rel=1e-3)
    assert srv.rtc.weight_override is None


def test_transition_cli_checks():
    ps.parse_args(["--mode", "rtc", "--mask-mode", "transition", "--trans-power", "0"])
    for bad in (["--mask-mode", "transition"], ["--mode", "rtc", "--mask-mode", "transition", "--n-samples", "4"],
                ["--mode", "rtc", "--mask-mode", "transition", "--fresh-t", "1"]):
        with pytest.raises(SystemExit):
            ps.parse_args(bad)
