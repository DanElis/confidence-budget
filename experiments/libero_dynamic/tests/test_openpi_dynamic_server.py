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


def budget_obs(step, episode="e", left=None, delay=3):
    from budget_privileged import N_OBJECTS, PRIV_DIM

    o = obs(np.zeros((0, A), np.float32) if left is None else left, delay)  # the queue always sends both fields
    o.update(budget_id=f"{episode}x{step}", budget_step=step, budget_episode=episode,
             budget_priv=np.eye(PRIV_DIM, dtype=np.float32)[N_OBJECTS] * 3)
    return o


def budget_policy(tmp_path, model, **config):
    import json

    path = tmp_path / "budget.json"
    path.write_text(json.dumps({"mode": "fixed", "pairs": [[3, -1], [3, 2], [5, -1], [5, 2]], "delay": 3, "seed": 0,
                                "option_ticks": 40, "max_steps": 600, "horizon": H, "fixed": [5, 2], **config}))
    return DynamicPolicy(model, H, A, parse_args(["--mode", "rtc", "--option-budget", str(path),
                                                  "--budget-traces", str(tmp_path / "traces")]))


def test_option_budget_sets_the_rtc_window_before_generation_and_returns_the_full_chunk(tmp_path):
    model = FakeModel()
    policy = budget_policy(tmp_path, model)
    first = policy.infer(budget_obs(0))
    assert "rtc_target" not in model.elements[0] and len(first["actions"]) == H and first["original_actions"].shape == (H, A)
    assert first["budget"]["s"] == 5 and first["budget"]["horizon"] == H and first["budget"]["decision"]
    left = np.full((5, A), .5, np.float32)
    second = policy.infer(budget_obs(5, left=left))
    e = model.elements[1]
    np.testing.assert_allclose(e["rtc_weights"], paper_weights(3, 5, H))  # mask end d + r = 5
    np.testing.assert_array_equal(e["rtc_target"][:5], left)
    assert second["plan"] == {"guided": True, "s": 5, "r": 2, "len": H} and not second["budget"]["decision"]
    assert (tmp_path / "traces/ex0.npz").exists() and not (tmp_path / "traces/ex5.npz").exists()


def test_naive_pair_has_no_guidance_and_episodes_keep_their_own_options(tmp_path):
    model = FakeModel()
    policy = budget_policy(tmp_path, model, fixed=[3, -1])
    policy.infer(budget_obs(0, "a"))
    policy.infer(budget_obs(0, "b"))
    out = policy.infer(budget_obs(3, "a", left=np.ones((7, A), np.float32)))
    assert "rtc_target" not in model.elements[-1] and out["budget"]["r"] == -1 and not out["budget"]["decision"]
    again = policy.infer(budget_obs(0, "a"))  # a new episode with the same key starts afresh
    assert again["budget"]["decision"]


def test_option_budget_arguments():
    with pytest.raises(SystemExit):
        parse_args(["--option-budget", "c.json", "--budget-traces", "t"])  # plain mode
    with pytest.raises(SystemExit):
        parse_args(["--mode", "rtc", "--option-budget", "c.json"])  # no traces
    assert parse_args(["--mode", "rtc", "--option-budget", "c.json", "--budget-traces", "t"]).option_budget == "c.json"
