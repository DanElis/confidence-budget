"""MimicGen environments (robosuite 1.4.1) as closed_loop.run_batch environments, for eval_openpi.py --env mimicgen.

The environment is built from the env_args of the MimicGen dataset (data.attrs["env_args"]: env_name and env_kwargs), as
robomimic does, with the cameras of our LeRobot dataset: agentview and robot0_eye_in_hand, square images.

Scenes. A scene is a seed: numpy's global generator is seeded right before env.reset(), so the object placement and the
robot's initialisation noise (both drawn from it in robosuite 1.4) are the same in every run. init_hash, a CRC of the
simulator state after the reset, goes to each episode's record: equal hashes of an item across runs show the pairing
(`python mimicgen_env.py --check` builds one scene twice and compares). A scene set is a name (the "suite"): its scenes
are scene_seed(set, task, init state); a "task" is only a block of scenes, all with the same task and prompt.

Observation -> request as our dataset was made (and as openpi's LIBERO client): both images rotated by 180 degrees
(raw[::-1, ::-1]), state = eef position, eef axis-angle, gripper qpos (8). Actions: OSC_POSE deltas (7), executed as given.
Success: env._check_success() after each step; the loop stops at success or at max_steps.
"""

from __future__ import annotations

import json
import math
import os
import zlib
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

DEFAULT_HDF5 = Path("~/data/mimicgen/core/threading_d0.hdf5").expanduser()
PROMPT = "thread the needle through the ring"  # the task string of our LeRobot dataset (MimicGen has no language)
CAMERAS = ("agentview", "robot0_eye_in_hand")
RESOLUTION = 224  # our dataset's images
HORIZON = 50  # action_horizon of vdb_pi05_threading_d0
MAX_STEPS = 400  # default limit of an episode; demos of threading_d0 last 225 steps on average
SUITE_PREFIX = "mimicgen_"
N_BLOCKS = 10  # "tasks" per scene set by default, as LIBERO's 10 tasks
BLOCK = 1000  # scene index = block * BLOCK + init state, as the episode ids of eval_openpi.py


def suite_name(item: str) -> str:
    """A scene set of the command line -> its suite name. mg:y1 or pos:y0.1 (the stage scripts' level names reused
    as labels of disjoint scene sets: in MimicGen they are not difficulty levels) -> mimicgen_y1, mimicgen_y0.1."""
    for prefix in ("mg:", "pos:"):
        if item.startswith(prefix):
            return SUITE_PREFIX + item[len(prefix):]
    return item if item.startswith(SUITE_PREFIX) else SUITE_PREFIX + item


def resolve_suites(spec: str) -> list[str]:
    return list(dict.fromkeys(suite_name(x.strip()) for x in spec.split(",") if x.strip()))


def scene_seed(suite: str, task: int, init_state: int) -> int:
    """Deterministic seed of a scene; different sets give unrelated seeds."""
    return (zlib.crc32(suite.encode()) * 1_000_003 + int(task) * BLOCK + int(init_state)) % (2**31 - 1)


def load_env_meta(hdf5: Path) -> dict:
    import h5py

    with h5py.File(Path(hdf5).expanduser(), "r") as f:
        return json.loads(f["data"].attrs["env_args"])


def env_kwargs(env_meta: dict, resolution: int = RESOLUTION) -> dict:
    kwargs = dict(env_meta["env_kwargs"])
    kwargs.update(has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, camera_names=list(CAMERAS),
                  camera_heights=resolution, camera_widths=resolution, camera_depths=False, ignore_done=True,
                  reward_shaping=False)
    return kwargs


def register_envs() -> None:
    """MimicGen's robosuite environments register themselves when their modules are imported."""
    import mimicgen  # noqa: F401
    import mimicgen.envs.robosuite.threading  # noqa: F401


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """(x, y, z, w) -> axis * angle, as _quat2axisangle of openpi's LIBERO client."""
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = math.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return np.asarray(quat[:3]) * 2.0 * math.acos(w) / den


def make_element(obs: dict, prompt: str) -> dict:
    """A robosuite observation as openpi's LiberoInputs reads it (the layout of our LeRobot dataset)."""
    return {
        "observation/image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
        "observation/wrist_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
        "observation/state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ).astype(np.float64),
        "prompt": str(prompt),
    }


def probe_element(resolution: int = RESOLUTION) -> dict:
    image = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    return {"observation/image": image, "observation/wrist_image": image, "observation/state": np.zeros(8),
            "prompt": PROMPT}


def state_hash(env) -> int:
    """CRC of the simulator state rounded to 1e-6: the same scene gives the same hash."""
    state = np.round(np.asarray(env.sim.get_state().flatten(), dtype=np.float64), 6) + 0.0  # + 0.0: no -0.0
    return zlib.crc32(state.tobytes())


class MimicGenEnv:
    """One episode of one scene: reset() -> (obs, info), step(a) -> (obs, reward, terminated, truncated, info)."""

    def __init__(self, env_meta: dict, seed: int, resolution: int = RESOLUTION):
        import robosuite

        register_envs()
        self.seed = int(seed)
        self.env = robosuite.make(env_meta["env_name"], **env_kwargs(env_meta, resolution))
        self.init_hash = None

    def reset(self):
        np.random.seed(self.seed)  # placement and robot initialisation noise of robosuite 1.4 use the global generator
        obs = self.env.reset()
        self.init_hash = state_hash(self.env)
        return obs, {}

    def step(self, action):
        obs, reward, _done, info = self.env.step(np.asarray(action, dtype=np.float64))
        info = dict(info, is_success=bool(self.env._check_success()))
        return obs, reward, False, False, info  # the loop ends the episode at success or max_steps

    def close(self) -> None:
        self.env.close()


def check_scene(hdf5: Path, seed: int, steps: int = 5) -> tuple[int, int]:
    """Builds a scene in two environments: equal reset hashes, and equal hashes after the same steps (deterministic
    physics). Raises otherwise. -> (reset hash, hash after the steps)."""
    meta = load_env_meta(hdf5)
    hashes = []
    for _ in range(2):
        env = MimicGenEnv(meta, seed)
        try:
            obs, _ = env.reset()
            for _ in range(steps):
                obs, *_ = env.step(np.zeros(7))
            element = make_element(obs, PROMPT)
            for key in ("observation/image", "observation/wrist_image"):
                if element[key].shape != (RESOLUTION, RESOLUTION, 3) or element[key].dtype != np.uint8:
                    raise RuntimeError(f"{key}: {element[key].shape} {element[key].dtype}")
            hashes.append((env.init_hash, state_hash(env.env)))
        finally:
            env.close()
    if hashes[0] != hashes[1]:
        raise RuntimeError(f"scene {seed} is not reproducible: {hashes}")
    other = MimicGenEnv(meta, seed + 1)
    try:
        other.reset()
        if other.init_hash == hashes[0][0]:
            raise RuntimeError("two seeds gave the same scene: the placement does not use numpy's global generator")
    finally:
        other.close()
    return hashes[0]


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Check that a MimicGen scene is reproducible.")
    p.add_argument("--check", action="store_true")
    p.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    p.add_argument("--seed", type=int, default=scene_seed(suite_name("mg:check"), 0, 0))
    a = p.parse_args()
    meta = load_env_meta(a.hdf5)
    print("env:", meta["env_name"], json.dumps(env_kwargs(meta), default=str))
    if a.check:
        print("scene check ok: reset hash %d, after steps %d" % check_scene(a.hdf5, a.seed))
