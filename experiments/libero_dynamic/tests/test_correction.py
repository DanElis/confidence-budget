"""Correction head pipeline without a model or simulator (libero_pro venv):

    cd experiments/libero_pro && uv run --with websockets --with msgpack python -m pytest -q ../libero_dynamic/tests

The runner tests use the benchmark's own realtime_eval classes from third_party/LIBERO-Dynamic (setup.sh) and are
skipped without the checkout.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "correction"))
sys.path.insert(1, str(ROOT))
import common  # noqa: E402
import replay  # noqa: E402
import train_head  # noqa: E402
from head import CorrectionHead, save_head  # noqa: E402

DYN_SCRIPTS = ROOT / "third_party" / "LIBERO-Dynamic" / "scripts"
needs_runner = pytest.mark.skipif(not (DYN_SCRIPTS / "realtime_eval" / "control.py").exists(),
                                  reason="benchmark checkout missing (setup.sh)")


# ---------------------------------------------------------------- schedule

def test_request_schedule_constant_step_covers_every_tick_once():
    sched = common.request_schedule(60, 4, 4)
    assert sched[:3] == [(0, 4), (4, 4), (8, 4)]
    ticks = [t for t_r, s in sched for t in common.executed(60, t_r, s, 4)]
    assert ticks == list(range(4, 60))


def test_request_schedule_variable_step_and_offset():
    steps = [5, 7, 4, 12]
    sched = common.request_schedule(40, 4, lambda k: steps[k % 4], offset=2)
    assert [t for t, _ in sched] == [2, 7, 14, 18, 30, 35]
    with pytest.raises(ValueError):
        common.request_schedule(40, 4, 3)


def test_images():
    frame = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    assert common.client_image(frame).shape == (224, 224, 3)
    assert common.head_image(frame).shape == (128, 128, 3) and common.head_image(frame).dtype == np.uint8


# ---------------------------------------------------------------- equivalence with the runner's queue

def code(c: int, i: np.ndarray) -> np.ndarray:
    """Action value encoding chunk number c and index i (inside [-1, 1], so the strategy's clip keeps it)."""
    return (c * 100 + np.asarray(i)) * 1e-4


class FakeClient:
    def __init__(self, ctrl_ref, log, K):
        self.ctrl_ref, self.log, self.K, self.n = ctrl_ref, log, K, 0

    def infer(self, request):
        c = self.n
        self.n += 1
        self.log.append({"step": self.ctrl_ref[0].submit_step_idx, "left": np.asarray(request["prev_chunk_left_over"]),
                         "scale": request.get("rtc_scale", 1.0), "delay": request["inference_delay_steps"]})
        full = np.repeat(code(c, np.arange(50))[:, None], 7, 1).astype(np.float32)
        return {"actions": full[: self.K], "original_actions": full[: self.K], "full_actions": full,
                "latent": np.full(2048, c, np.float16)}


class FakePolicy:
    """The parts of OpenPIWebsocketPolicy that the controllers and the latent proxy use."""
    supports_rtc_conditioning = True
    action_dim = 7

    def __init__(self, client, K):
        self.client, self.replan_steps = client, K

    def _observation(self, obs, language):
        return {"prompt": language}

    def _actions_from_result(self, result):
        return np.asarray(result["actions"], np.float32)[: self.replan_steps]

    def get_action_chunk_rtc(self, obs, language, *, prev_chunk_left_over, inference_delay_steps):
        r = self.client.infer({"prev_chunk_left_over": prev_chunk_left_over, "inference_delay_steps": inference_delay_steps})
        return {"actions": r["actions"], "original_actions": r["original_actions"]}


def runner_modules():
    sys.path.insert(0, str(DYN_SCRIPTS))
    from realtime_eval import control
    from realtime_eval.strategies import rtc_queue

    name = "realtime_eval.strategies.corrected_rtc_queue"
    spec = importlib.util.spec_from_file_location(name, ROOT / "correction" / "corrected_rtc_queue.py")
    corrected = importlib.util.module_from_spec(spec)
    sys.modules[name] = corrected
    spec.loader.exec_module(corrected)
    return control, rtc_queue, corrected


def simulate(control, cls, T, K, latency=0.175, threshold=10):
    log, ref = [], [None]
    client = FakeClient(ref, log, K)
    ctrl = cls(policy=FakePolicy(client, K), language="task", control_freq=20, mode="rtc_queue",
               prefetch_threshold=threshold, action_chunk_size=K, chunk_size_threshold=None,
               latency_sampler=control.LatencySampler(source="fixed", fixed_seconds=latency))
    ref[0] = ctrl
    ctrl.reset()
    actions = control.ActionRuntime(7, empty_action="hold")
    executed, originals = [], []
    obs = {"agentview_image": np.zeros((256, 256, 3), np.uint8), "robot0_eye_in_hand_image": np.zeros((256, 256, 3), np.uint8),
           "robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0, 0, 0, 1.0]), "robot0_gripper_qpos": np.zeros(2)}
    for step in range(T):
        pkt = control.ObservationPacket(episode_id="e", step_idx=step, sim_time_s=step / 20, language="task", obs=obs,
                                        queue_len=actions.queue_len, last_action=actions.last_action)
        ctrl.tick(pkt, actions)
        a, orig, *_ = actions.next_action()
        executed.append(float(a[0]))
        originals.append(float(orig[0]))
    ctrl.close()
    return ctrl, log, np.array(executed), np.array(originals)


@needs_runner
def test_offline_schedule_matches_rtc_queue_k10():
    control, rtc_queue, _ = runner_modules()
    T, K, d = 80, 10, 4
    _, log, executed, _ = simulate(control, rtc_queue.RtcQueueController, T, K)
    sched = common.request_schedule(T, d, 4)
    assert [x["step"] for x in log] == [t for t, _ in sched][: len(log)]
    assert all(x["delay"] == d for x in log)
    for k, (t_r, s) in enumerate(sched):
        for t in common.executed(T, t_r, s, d):
            assert executed[t] == pytest.approx(code(k, t - t_r)), (k, t)
        if k:  # leftover = the previous chunk's rest [s_prev:K] (model space)
            prev_s = sched[k - 1][1]
            np.testing.assert_allclose(log[k]["left"][:, 0], code(k - 1, np.arange(prev_s, K)), atol=1e-7)


@needs_runner
def test_offline_schedule_matches_lower_threshold():
    """--prefetch-threshold q < K - d: s = d + K - d - q (the adaptive variant's base trigger, K = 16, q = 4)."""
    control, rtc_queue, _ = runner_modules()
    T, K, d, q = 90, 16, 4, 4
    _, log, executed, _ = simulate(control, rtc_queue.RtcQueueController, T, K, threshold=q)
    sched = common.request_schedule(T, d, d + K - d - q)
    assert [x["step"] for x in log] == [t for t, _ in sched][: len(log)]
    for k, (t_r, s) in enumerate(sched):
        for t in common.executed(T, t_r, s, d):
            assert executed[t] == pytest.approx(code(k, t - t_r))


@needs_runner
def test_corrected_queue_replaces_only_the_executed_action(monkeypatch):
    control, _, corrected = runner_modules()
    requests = []

    class FakeCorr:
        def infer(self, req):
            requests.append(req)
            return {"delta": np.full(7, 0.5, np.float32)}

    monkeypatch.setattr(corrected, "_client", lambda host, port: FakeCorr())
    monkeypatch.setattr(corrected, "_state", lambda obs: np.zeros(8, np.float32))
    T, K, d = 60, 10, 4
    ctrl, log, executed, originals = simulate(control, corrected.CorrectedRtcQueueController, T, K)
    sched = common.request_schedule(T, d, 4)
    n = 0
    for k, (t_r, s) in enumerate(sched):
        for t in common.executed(T, t_r, s, d):
            assert executed[t] == pytest.approx(code(k, t - t_r) + 0.5)
            assert originals[t] == pytest.approx(code(k, t - t_r))
            req = requests[n]
            assert req["index"] == t - t_r and float(req["latent"][0]) == k
            assert req["chunk"][t - t_r, 0] == pytest.approx(code(k, t - t_r))
            n += 1
        if k:  # RTC leftover is the uncorrected plan
            np.testing.assert_allclose(log[k]["left"][:, 0], code(k - 1, np.arange(4, K)), atol=1e-7)
    assert n == len(requests) == T - d
    summ = ctrl.summary()
    assert summ["correction_calls"] == T - d
    assert summ["correction_abs_mean"] == pytest.approx(0.5 * np.sqrt(7), rel=1e-5)


@needs_runner
def test_corrected_queue_delay_scale_and_trigger(monkeypatch):
    control, _, corrected = runner_modules()

    class FakeCorr:
        def infer(self, req):
            return {"delta": np.full(7, 0.5, np.float32)}

    monkeypatch.setattr(corrected, "_client", lambda host, port: FakeCorr())
    monkeypatch.setattr(corrected, "_state", lambda obs: np.zeros(8, np.float32))
    monkeypatch.setenv("CORR_DELAY_TICKS", "1")
    monkeypatch.setenv("CORR_SCALE_REF", "0.66")
    monkeypatch.setenv("CORR_SCALE_MIN", "0.25")
    _, log, executed, _ = simulate(control, corrected.CorrectedRtcQueueController, 40, 10)
    assert log[0]["scale"] == 1.0 and log[2]["scale"] == pytest.approx(0.66 / (0.5 * np.sqrt(7)))
    # (a): base trigger at queue <= 4 with K = 16 gives s = 12; the correction sum triggers earlier
    monkeypatch.delenv("CORR_SCALE_REF")
    monkeypatch.setenv("CORR_QUEUE_THRESHOLD", "4")
    monkeypatch.setenv("CORR_TRIGGER", str(0.5 * np.sqrt(7) * 2.5))  # reached after 3 corrected ticks
    monkeypatch.setenv("CORR_S_MIN", "1")
    ctrl, log, _, _ = simulate(control, corrected.CorrectedRtcQueueController, 60, 16)
    steps = [x["step"] for x in log]
    # first chunk arrives at 4, corrections at ticks 4, 5, 6 reach the trigger, request at 7 (instead of 12);
    # later the sum is reached during the delay, so the next request goes out at the arrival (s = d)
    assert steps[1] - steps[0] == 7
    assert all(b - a == 4 for a, b in zip(steps[1:], steps[2:]))
    assert ctrl.summary()["early_requests"] > 0


# ---------------------------------------------------------------- head, server, training data

def tiny_head(**kw):
    return CorrectionHead(image_size=64, dim=64, heads=4, layers=1, ff=128, latent_dim=32, pretrained=False, **kw)


def head_inputs(b=3, size=64, tail=False):
    g = torch.Generator().manual_seed(0)
    args = (torch.randint(0, 255, (b, 2, size, size, 3), dtype=torch.uint8, generator=g), torch.randn(b, 8),
            torch.randn(b, 7), torch.randn(b, 50, 7), torch.tensor([4, 5, 7][:b]), torch.randn(b, 32))
    kw = {}
    if tail:
        mask = torch.zeros(b, 16, dtype=torch.bool)
        mask[:, 6:] = True
        kw = {"tail": torch.randn(b, 16, 7), "tail_mask": mask}
    return args, kw


@pytest.mark.parametrize("tail", [False, True])
def test_head_starts_at_zero_and_trains(tail):
    model = tiny_head(tail=16 if tail else 0)
    args, kw = head_inputs(tail=tail)
    out = model(*args, **kw)
    assert out.shape == (3, 7) and float(out.abs().max()) == 0.0
    loss = ((model.forward_norm(*args, **kw) - 1.0) ** 2).mean()
    loss.backward()
    assert model.out[-1].weight.grad.abs().sum() > 0


def test_full_size_head_parameter_count():
    n = sum(p.numel() for p in CorrectionHead(pretrained=False).parameters())
    assert 28e6 < n < 36e6


def test_corrector_roundtrip(tmp_path):
    from correction_server import Corrector

    model = tiny_head()
    save_head(model, tmp_path / "h.pt", step=1)
    corr = Corrector(str(tmp_path / "h.pt"), device="cpu")
    req = {"image": np.zeros((256, 256, 3), np.uint8), "wrist_image": np.zeros((256, 256, 3), np.uint8),
           "state": np.zeros(8), "chunk": np.zeros((50, 7), np.float32), "index": 5, "latent": np.zeros(32, np.float16)}
    delta = corr.correct(req)
    assert delta.shape == (7,) and delta.dtype == np.float32


def fake_data(tmp_path, size=16):
    """Two episodes (one val) and one replay file with s = d = 4, K = 10."""
    eps = [{"start": 0, "length": 20, "split": "train", "speed": 0.06, "task": "a"},
           {"start": 20, "length": 15, "split": "val", "speed": 0.03, "task": "b"}]
    (tmp_path / "episodes.json").write_text(json.dumps(eps))
    n = 35
    actions = np.random.default_rng(0).normal(size=(n, 7)).astype(np.float32)
    np.savez(tmp_path / "demo.npz", actions=actions, states=np.zeros((n, 8), np.float32))
    np.memmap(tmp_path / f"frames_{size}.u8", dtype=np.uint8, mode="w+", shape=(n, 2, size, size, 3)).flush()
    rec = {k: [] for k in ("ep", "t", "s", "scale", "prev", "chunk", "latent")}
    for e, ep in enumerate(eps):
        first = len(rec["ep"])
        for k, (t, s) in enumerate(common.request_schedule(ep["length"], 4, 4)):
            rec["ep"].append(e), rec["t"].append(t), rec["s"].append(s), rec["scale"].append(1.0)
            rec["prev"].append(first + k - 1 if k else -1)
            rec["chunk"].append(np.full((50, 7), len(rec["ep"]), np.float32))
            rec["latent"].append(np.zeros(2048, np.float16))
    np.savez(tmp_path / "replay_x.npz", **{k: np.asarray(v) for k, v in rec.items()},
             attrs=json.dumps({"delay": 4, "K": 10}))
    return eps, actions


def test_training_samples_targets_and_tail(tmp_path):
    eps, actions = fake_data(tmp_path)
    rep = train_head.Replays(tmp_path, ["x"], eps)
    tr, va = rep.samples("train"), rep.samples("val")
    assert sorted(tr[:, 1].tolist()) == list(range(4, 20)) and sorted(va[:, 1].tolist()) == list(range(4, 15))
    assert set((tr[:, 1] - rep.t[tr[:, 0]]).tolist()) == {4, 5, 6, 7} and (tr[:, 2] == tr[:, 1] - rep.t[tr[:, 0]]).all()
    ds = train_head.SampleSet(rep, tr, tmp_path / "frames_16.u8", 35, 16, actions, np.zeros((35, 8), np.float32),
                              tail=True, shift=2)
    i = int(np.nonzero(tr[:, 0] == 1)[0][0])  # second request of episode 0
    item = ds[i]
    k, t, j = tr[i]
    np.testing.assert_allclose(item["target"].numpy(), actions[t] - rep.chunk[k][j])
    assert item["images"].shape == (2, 16, 16, 3)
    assert int((~item["tail_mask"]).sum()) == 6 and float(item["tail"][0, 0]) == rep.chunk[0][0, 0]


def test_replay_helpers():
    class A:
        delay, step, step_range, max_offset, scale_range = 4, 4, [4, 12], 3, [0.25, 1.0]

    plan = replay.episode_plan({"length": 100}, A, np.random.default_rng(0))
    assert all(4 <= s <= 12 and 0.25 <= f <= 1.0 for _, s, f in plan) and plan[0][0] <= 3
    rows = [{"x": torch.zeros(1, 3), "task": ["a"]}, {"x": torch.ones(1, 3), "task": ["b"]}]
    out = replay.collate(rows)
    assert out["x"].shape == (2, 3) and out["task"] == ["a", "b"]


def test_replay_merge_offsets_prev(tmp_path):
    def shard(name, n, prev):
        np.savez(tmp_path / name, ep=np.zeros(n, np.int32), t=np.zeros(n, np.int32), s=np.zeros(n, np.int16),
                 scale=np.ones(n, np.float32), prev=np.asarray(prev, np.int32), chunk=np.zeros((n, 50, 7), np.float32),
                 latent=np.zeros((n, 4), np.float16))
        return tmp_path / name
    replay.merge([shard("a.npz", 2, [-1, 0]), shard("b.npz", 3, [-1, 0, 1])], tmp_path / "m.npz", {"K": 10})
    z = np.load(tmp_path / "m.npz")
    assert z["prev"].tolist() == [-1, 0, -1, 2, 3] and json.loads(str(z["attrs"]))["K"] == 10


def test_policy_server_return_latent_cli():
    import policy_server as ps

    assert ps.parse_args(["--mode", "rtc", "--return-latent"]).return_latent
    with pytest.raises(SystemExit):
        ps.parse_args(["--mode", "rtc", "--return-latent", "--n-samples", "2"])
