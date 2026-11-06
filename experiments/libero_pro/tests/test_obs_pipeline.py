"""Stacked LIBERO-like observations go through LeRobot's env preprocessing as in lerobot-eval."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from closed_loop import stack_observations  # noqa: E402

pytest.importorskip("lerobot")


def raw_obs(seed):
    rng = np.random.default_rng(seed)
    return {
        "pixels": {"image": rng.integers(0, 255, (360, 360, 3), dtype=np.uint8),
                   "image2": rng.integers(0, 255, (360, 360, 3), dtype=np.uint8)},
        "robot_state": {
            "eef": {"pos": rng.normal(size=3), "quat": np.array([0.0, 0.0, 0.0, 1.0]), "mat": np.eye(3)},
            "gripper": {"qpos": rng.normal(size=2), "qvel": rng.normal(size=2)},
            "joints": {"pos": rng.normal(size=7), "vel": rng.normal(size=7)},
        },
    }


def test_stacked_observations_match_lerobot_eval_format():
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.constants import OBS_STATE

    env_pre, _ = LiberoEnv(task="libero_object").get_env_processors()
    obs = preprocess_observation(stack_observations([raw_obs(0), raw_obs(1), raw_obs(2)]))
    obs["task"] = ["pick up the butter and place it in the basket"] * 3
    out = env_pre(obs)
    assert out[OBS_STATE].shape == (3, 8)
    assert out["observation.images.image"].shape == (3, 3, 360, 360)
    assert out["observation.images.image2"].shape == (3, 3, 360, 360)
