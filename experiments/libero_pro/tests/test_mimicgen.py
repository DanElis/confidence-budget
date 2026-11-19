"""MimicGen through eval_openpi.py --env mimicgen: scenes, observation, arguments, the environment over a stub robosuite."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT.parent / "mimicgen"))
import mimicgen_env as mg  # noqa: E402
from closed_loop import LoopConfig, run_batch  # noqa: E402
from eval_openpi import make_infer, parse_args, progress_key  # noqa: E402
from planner import FixedPlanner  # noqa: E402

H = mg.HORIZON


def test_suite_names_and_scene_seeds():
    assert mg.suite_name("mg:y1") == "mimicgen_y1" and mg.suite_name("pos:x0.2") == "mimicgen_x0.2"
    assert mg.suite_name("mimicgen_z") == "mimicgen_z"
    assert mg.resolve_suites("mg:y1, pos:y0.2,mg:y1") == ["mimicgen_y1", "mimicgen_y0.2"]
    a = mg.scene_seed("mimicgen_y1", 3, 7)
    assert a == mg.scene_seed("mimicgen_y1", 3, 7) and 0 <= a < 2**31 - 1
    seeds = {mg.scene_seed(s, t, i) for s in ("mimicgen_y1", "mimicgen_x1") for t in range(10) for i in range(50)}
    assert len(seeds) == 2 * 10 * 50  # sets and blocks do not share scenes


def robosuite_obs(fill=0):
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3) + fill
    return {"agentview_image": image, "robot0_eye_in_hand_image": image + 1,
            "robot0_eef_pos": np.array([1.0, 2.0, 3.0]), "robot0_eef_quat": np.array([0.0, 0.0, 1.0, 0.0]),
            "robot0_gripper_qpos": np.array([0.04, -0.04])}


def test_make_element():
    obs = robosuite_obs()
    e = mg.make_element(obs, mg.PROMPT)
    assert np.array_equal(e["observation/image"], obs["agentview_image"][::-1, ::-1])
    assert np.array_equal(e["observation/wrist_image"], obs["robot0_eye_in_hand_image"][::-1, ::-1])
    assert e["observation/image"].flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(e["observation/state"], [1, 2, 3, 0, 0, np.pi, 0.04, -0.04], atol=1e-6)
    assert e["prompt"] == mg.PROMPT
    probe = mg.probe_element(224)
    assert probe["observation/image"].shape == (224, 224, 3) and probe["observation/state"].shape == (8,)


def test_env_kwargs_keep_the_dataset_env_and_set_the_cameras():
    meta = {"env_name": "Threading_D0", "env_kwargs": {"robots": ["Panda"], "control_freq": 20,
                                                       "camera_names": ["agentview"], "camera_heights": 84}}
    kw = mg.env_kwargs(meta, 224)
    assert kw["robots"] == ["Panda"] and kw["control_freq"] == 20
    assert kw["camera_names"] == ["agentview", "robot0_eye_in_hand"] and kw["camera_heights"] == kw["camera_widths"] == 224
    assert kw["use_camera_obs"] and kw["has_offscreen_renderer"] and not kw["has_renderer"] and kw["ignore_done"]
    assert meta["env_kwargs"]["camera_heights"] == 84  # the dataset's meta is not changed


def test_args_defaults_and_bounds(monkeypatch):
    base = ["--suites", "mg:y1", "--out", "x", "--env", "mimicgen"]
    a = parse_args(base + ["--stride", "10"])
    assert (a.horizon, a.resolution, a.max_steps, a.mg_tasks) == (50, 224, 400, 10)
    assert a.mg_hdf5 == mg.DEFAULT_HDF5
    assert parse_args(base + ["--stride", "48", "--delay", "2"]).stride == 48  # s <= H - d with H = 50
    with pytest.raises(SystemExit):
        parse_args(base + ["--stride", "49", "--delay", "2"])
    assert parse_args(base + ["--stride", "10", "--max-steps", "300", "--mg-tasks", "2"]).max_steps == 300
    lib = parse_args(["--suites", "pos:x0.2", "--out", "x", "--stride", "5"])
    assert (lib.env, lib.horizon, lib.resolution) == ("libero", 10, 256)
    with pytest.raises(SystemExit):
        parse_args(["--suites", "pos:x0.2", "--out", "x", "--stride", "5", "--mg-tasks", "2"])
    key = progress_key(a)
    assert key["env"] == "mimicgen" and key["mg_hdf5"] == str(mg.DEFAULT_HDF5) and key["horizon"] == 50
    assert "env" not in progress_key(lib)  # progress files of LIBERO runs stay valid
    monkeypatch.setattr("eval_openpi.EVAL_ENV", "mimicgen")  # the profile's EVAL_ENV: the stage scripts' calls unchanged
    staged = parse_args(["--suites", "pos:x0.2", "--out", "x", "--stride", "5", "--mg-tasks", "2"])
    assert (staged.env, staged.horizon, staged.mg_tasks) == ("mimicgen", 50, 2)


class StubRobosuiteEnv:
    """Placement from numpy's global generator, as robosuite 1.4; success when the summed x action exceeds 1 (strictly)."""

    def __init__(self, **kwargs):
        self.kwargs, self.total, self.closed = kwargs, 0.0, False
        self.sim = types.SimpleNamespace(get_state=lambda: types.SimpleNamespace(flatten=lambda: self.state))

    def reset(self):
        self.state = np.random.uniform(-1, 1, size=5)
        self.total = 0.0
        return robosuite_obs()

    def step(self, action):
        self.total += float(action[0])
        self.state = self.state + 0.01
        return robosuite_obs(), 0.0, False, {}

    def _check_success(self):
        return self.total > 1.0

    def close(self):
        self.closed = True


@pytest.fixture
def stub_robosuite(monkeypatch):
    made = []

    def make(name, **kwargs):
        env = StubRobosuiteEnv(**kwargs)
        made.append((name, env))
        return env

    monkeypatch.setitem(sys.modules, "robosuite", types.SimpleNamespace(make=make))
    monkeypatch.setattr(mg, "register_envs", lambda: None)
    return made


META = {"env_name": "Threading_D0", "env_kwargs": {"robots": ["Panda"]}}


def test_env_scene_is_reproducible_and_seeds_differ(stub_robosuite):
    a, b, c = (mg.MimicGenEnv(META, s) for s in (5, 5, 6))
    for env in (a, b, c):
        np.random.seed(123)  # whatever ran before: reset seeds the generator itself
        env.reset()
    assert a.init_hash == b.init_hash != c.init_hash
    assert stub_robosuite[0][0] == "Threading_D0" and stub_robosuite[0][1].kwargs["camera_heights"] == 224
    obs, reward, terminated, truncated, info = a.step(np.zeros(7))
    assert not terminated and not truncated and info["is_success"] is False
    a.close()
    assert stub_robosuite[0][1].closed


class FakeClient:
    def __init__(self):
        self.prompts = []

    def infer(self, element):
        self.prompts.append(element["prompt"])
        assert element["observation/state"].shape == (8,)
        actions = np.zeros((1, H, 7), dtype=np.float32)
        actions[..., 0] = 0.125  # exact in float32: the sum is exactly 1 after 8 steps
        return {"actions": actions, "normalized_action": np.zeros((1, H, 32), dtype=np.float32)}


def test_closed_loop_over_stub(stub_robosuite):
    client = FakeClient()
    infer = make_infer(client, mg.PROMPT, H, element_fn=mg.make_element)
    envs = [mg.MimicGenEnv(META, mg.scene_seed("mimicgen_y1", 0, i)) for i in range(2)]
    res = run_batch(envs, infer, LoopConfig(H, 2, False, 30), FixedPlanner(H, 10), [0, 1], (H, 32))
    assert res.success.all() and (res.length == 9).all()  # 0.125 per step: the sum passes 1 at the 9th step
    assert set(client.prompts) == {mg.PROMPT}
    assert all(env.init_hash is not None for env in envs)
