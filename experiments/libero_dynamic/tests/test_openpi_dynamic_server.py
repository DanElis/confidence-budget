"""openpi_dynamic_server.py: the LIBERO-Dynamic protocol over openpi's pi05_libero, with a fake model."""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "libero_pro" / "scripts"))
from openpi_dynamic_server import DynamicPolicy, paper_weights, parse_args, request_noise  # noqa: E402

H, A = 10, 32


def obs(left=None, delay=0):
    o = {"observation/image": np.zeros((8, 8, 3), np.uint8), "observation/wrist_image": np.zeros((8, 8, 3), np.uint8),
         "observation/state": np.arange(8, dtype=np.float32), "prompt": "pick up the milk", "extra_field": 1}
    if left is not None:
        o |= {"prev_chunk_left_over": left, "inference_delay_steps": delay}
    return o


class FakeModel:
    """As openpi_sample_server.SamplePolicy: actions (N, H, 7), normalized_action (N, H, A), policy_timing."""

    def __init__(self):
        self.elements = []

    def infer(self, element):
        self.elements.append(element)
        n = len(element["noise"])
        normalized = np.repeat(np.arange(H, dtype=np.float32)[None, :, None], n, 0) * np.ones((n, H, A), np.float32)
        actions = normalized[..., :7].copy()
        actions[:, :, 1] += element["noise"][:, :, 0]  # samples disagree everywhere
        return {"actions": actions, "normalized_action": normalized, "policy_timing": {"infer_ms": 1.0}}


def test_noise_is_fixed_per_request_and_sample_0_does_not_depend_on_n():
    one, many = request_noise(obs(), 0, 1, H, A), request_noise(obs(), 0, 20, H, A)
    np.testing.assert_array_equal(one[0], many[0])
    assert not np.array_equal(request_noise(obs(), 1, 1, H, A), one)  # the seed changes it
    assert not np.array_equal(request_noise(obs(np.zeros((4, A))), 0, 1, H, A), one)  # so does the leftover length


def test_paper_weights_match_eval_openpi():
    from eval_openpi import paper_weights as libero_pro_weights

    for d, end in ((2, 2), (2, 6), (3, 10)):
        np.testing.assert_allclose(paper_weights(d, end, H), libero_pro_weights(d, end, H)[:, 0], rtol=1e-6)


def test_plain_request_is_one_unguided_sample_cut_to_the_chunk_length():
    model = FakeModel()
    reply = DynamicPolicy(model, H, A, parse_args(["--actions-per-chunk", "8"])).infer(obs(np.ones((4, A)), 2))
    element = model.elements[0]
    assert set(element) == {"observation/image", "observation/wrist_image", "observation/state", "prompt", "noise"}
    assert element["noise"].shape == (1, H, A) and reply["actions"].shape == (8, 7)
    assert reply["original_actions"].shape == (8, A) and not reply["plan"]["guided"]


def test_rtc_guides_toward_the_leftover_with_the_paper_weights_cut_to_it():
    model = FakeModel()
    policy = DynamicPolicy(model, H, A, parse_args(["--mode", "rtc"]))
    first = policy.infer(obs())  # the first chunk of an episode: no leftover, no guidance
    assert not first["plan"]["guided"] and "rtc_target" not in model.elements[0]
    left = np.full((6, A), 0.5, np.float32)
    policy.infer(obs(left, 2))
    e = model.elements[1]
    np.testing.assert_array_equal(e["rtc_target"][:6], left)
    np.testing.assert_array_equal(e["rtc_target"][6:], 0.0)
    np.testing.assert_allclose(e["rtc_weights"], paper_weights(2, 6, H))  # the mask end is cut to the leftover
    assert e["rtc_jacobian"] and e["rtc_max_weight"] == 5.0


def test_aac_returns_s_plus_d_actions():
    model = FakeModel()
    policy = DynamicPolicy(model, H, A, parse_args(["--mode", "rtc", "--no-guidance", "--adaptive", "--n-samples", "20",
                                                    "--s-min", "2"]))
    reply = policy.infer(obs(np.ones((5, A), np.float32), 2))
    assert "rtc_target" not in model.elements[0] and model.elements[0]["noise"].shape == (20, H, A)
    s = reply["plan"]["s"]
    assert 2 <= s <= H - 2 and len(reply["actions"]) == reply["plan"]["len"] == s + 2


def test_args():
    with pytest.raises(SystemExit):
        parse_args(["--adaptive"])  # one sample
    with pytest.raises(SystemExit):
        parse_args(["--mode", "rtc", "--adaptive", "--n-samples", "20"])  # AAC with RTC
    with pytest.raises(SystemExit):
        parse_args(["--n-samples", "4"])  # samples without AAC
    with pytest.raises(ValueError):
        DynamicPolicy(FakeModel(), H, A, parse_args(["--actions-per-chunk", "16"]))  # longer than H
