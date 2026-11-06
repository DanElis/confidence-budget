"""AAC as in the authors' code (aac.py) and its place in the closed loop."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from aac import action_entropy, bernoulli_entropy, entropy_horizon, gaussian_entropy, motion_floor, paper_h_star  # noqa: E402
from closed_loop import Chunk, LoopConfig, run_batch  # noqa: E402
from planner import AdaptivePlanner, GivenHStar  # noqa: E402
from step_controls import controls, main  # noqa: E402

H = 12


def test_gaussian_entropy_matches_the_covariance_formula():
    x = np.random.default_rng(0).normal(size=(20, H, 3))
    want = [0.5 * (3 * math.log(2 * math.pi * math.e) + np.linalg.slogdet(np.cov(x[:, t], rowvar=False) + 1e-6 * np.eye(3))[1])
            for t in range(H)]
    np.testing.assert_allclose(gaussian_entropy(x), want)


def test_bernoulli_entropy():
    g = np.zeros((4, 2))
    g[:2, 1] = 1
    h = bernoulli_entropy(g)
    assert h[0] == pytest.approx(0, abs=1e-6) and h[1] == pytest.approx(math.log(2))


def test_action_entropy_sums_translation_rotation_and_gripper():
    g = np.random.default_rng(1)
    norm, env = g.normal(size=(20, H, 7)), g.normal(size=(20, H, 7))
    want = gaussian_entropy(norm[..., :3]) + gaussian_entropy(norm[..., 3:6]) + bernoulli_entropy((env[..., 6] >= 0) * 1.0)
    np.testing.assert_allclose(action_entropy(norm, env), want)


def test_entropy_horizon_is_equation_5_with_floor_2():
    assert entropy_horizon(np.r_[np.zeros(7), np.ones(H - 7)]) == 7
    assert entropy_horizon(np.r_[np.zeros(1), np.ones(H - 1)]) == 2  # argmax at h = 1: their lower bound
    rng = np.random.default_rng(0)
    for _ in range(20):
        e = rng.normal(size=H)
        mean = {h: e[:h].mean() for h in range(1, H + 1)}
        gains = {h: mean[h + 1] - mean[h] for h in range(1, H)}
        assert entropy_horizon(e) == max(max(gains, key=lambda h: (gains[h], -h)), 2)


def actions(**columns):
    a = np.zeros((H, 7))
    for i, value in columns.items():
        a[:, int(i[1:])] = value
    return a


def test_motion_floor():
    assert motion_floor(actions(a0=0.5), 3.0) == 7  # 0.5 k > 3
    assert motion_floor(actions(a0=1.0, a1=1.0), 3.0) == 3  # the norm of the sum: sqrt(2) k > 3
    assert motion_floor(actions(a3=0.45), 3.0) == 7  # rotations about one axis add up
    assert motion_floor(np.tile([0, 0, 0, 0.1, 0, 0, 1.0], (H, 1)), 3.0) == H  # never: H, as their find_first_idx
    alternating = actions(a0=np.r_[1.0, -1.0, 1.0, -1.0, np.zeros(H - 4)])
    assert motion_floor(alternating, 0.5) == 3  # end to end: 0 at k = 2, the path length is not counted
    toggle = actions(a6=np.r_[-1.0, -1.0, 1.0, np.ones(H - 3)])
    assert motion_floor(toggle, 0.1) == 3 and motion_floor(toggle, 0.2) == H  # weight 0.2, strictly above move_th


def samples_with_jump(at, n=20):
    pattern = np.random.default_rng(0).normal(size=(n, 7))
    norm = np.zeros((n, H, 7))
    norm[:, at:] = pattern[:, None]
    env = np.zeros((n, H, 7))
    env[..., 0] = 1.0  # the same motion in every sample: xi = 4
    return norm, env


def test_paper_h_star_takes_the_larger_of_entropy_and_motion():
    norm, env = samples_with_jump(7)
    assert paper_h_star(norm, env, 3.0) == 7
    norm, env = samples_with_jump(2)
    assert paper_h_star(norm, env, 3.0) == 4


class Env:
    def reset(self):
        return {}, {}

    def step(self, action):
        return {}, 0.0, False, False, {}


@pytest.mark.parametrize(("delay", "starts", "strides"), [(0, [0, 5], [5, 5]), (2, [0, 3, 6], [3, 3, 3])])
def test_loop_runs_chunks_for_h_star_minus_delay(delay, starts, strides):
    def infer(observations, noise, leftover, d, mask_end):
        n = len(observations)
        return Chunk(torch.zeros(n, H, 2), np.zeros((n, H, 2)), signal=np.full(n, 5))

    res = run_batch([Env()], infer, LoopConfig(H, delay, False, 9), AdaptivePlanner(H, GivenHStar(), s_min=1), [0], (H, 2))
    assert [c["step"] for c in res.chunks] == starts and [c["stride"] for c in res.chunks] == strides
    assert all(c["h_star"] == 5 for c in res.chunks)


def test_aac_cli_checks():
    from eval_libero import check_sample_args, make_planner, parse_args

    ok = ["--out", "x", "--n-samples", "20", "--planner", "adaptive", "--signal", "aac"]
    check_sample_args(parse_args(ok))
    planner = make_planner(parse_args(ok), 50)  # no threshold, no config
    assert planner(np.asarray(9), None, 4).stride == 5 and planner(np.asarray(9), None, 0).stride == 9
    assert planner(np.asarray(2), None, 4).stride == 4  # s >= d
    for bad in (["--out", "x", "--signal", "aac", "--planner", "adaptive"],  # K = 1
                ok + ["--guidance", "lerobot"], ok + ["--select", "coherent"],
                ["--out", "x", "--n-samples", "20", "--signal", "aac", "--stride", "5"]):
        with pytest.raises(SystemExit):
            check_sample_args(parse_args(bad))


def test_step_controls(tmp_path, capsys):
    assert controls([7, 7, 7, 8]) == [7]  # 7.25: within 5%
    assert controls([7, 8]) == [7, 8]  # 7.5: more than 5% from either, bracket the rate
    assert controls([8] * 3) == [8]
    pro = tmp_path / "pro"
    pro.mkdir()
    (pro / "suite_chunks.json").write_text(json.dumps([{"stride": 10}, {"stride": 12}]))
    (pro / "suite.json").write_text("{}")
    dyn = tmp_path / "dyn"
    dyn.mkdir()
    (dyn / "signals_0.jsonl").write_text('{"s": 7}\n{"s": 8}\n')
    main([str(pro)])
    main([str(dyn)])
    assert capsys.readouterr().out.split() == ["11", "7", "8"]
