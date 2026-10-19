#!/usr/bin/env python
"""One RoboTwin 2.0 simulator behind a socket; runs in RoboTwin's own Python environment.

Started by remote_env.RemoteEnv (cwd = RoboTwin root). Commands (see wire.py for the format):
    {"cmd": "make", "task": name, "cameras": [...]}        -> {"step_lim": int}
    {"cmd": "reset", "seed": int, "max_tries": int}        -> {"obs", "seed", "step_lim", "instruction"}
    {"cmd": "step", "action": (14,) array}                 -> {"obs", "success", "done", "step"}
    {"cmd": "close"}
Errors come back as {"error": message}; the server keeps running.

Episode setup mirrors RoboTwin's scripts/eval_policy_xpolicylab.py: task_config demo_clean,
eval_mode=True (per-task step limits from _eval_step_limit.yml), is_test=True, a seed that
raises UnStableError is skipped for seed + 1 (deterministic, so every method sees the same
scenes). No expert check (that needs CuRobo); stepping in joint space uses mplib only, so
CuRobo is replaced by a stub when it is not installed. Observation and setup code follow
lerobot/envs/robotwin.py (Apache-2.0).

--fake serves synthetic observations without RoboTwin (for tests).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import io
import os
import sys
import time
import types
from multiprocessing.connection import Listener

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wire  # noqa: E402

ACTION_DIM = 14
AUTHKEY = b"robotwin-confidence-budget"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--address", required=True, help="unix socket path")
    p.add_argument("--fake", action="store_true")
    p.add_argument("--fake-step-lim", type=int, default=60)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------------------------
# CuRobo stub: RoboTwin imports CuRobo at module level and builds CuRobo planners when a robot
# is loaded, but joint-space evaluation never calls them.
# ---------------------------------------------------------------------------------------------


class _Anything:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Anything()

    def __getattr__(self, name):
        return _Anything()


class _NoCuroboPlanner:
    """Stands in for RoboTwin's CuroboPlanner. Keeps the parts that do not need CuRobo
    (gripper interpolation, copied from envs/robot/planner.py); motion planning raises."""

    def __init__(self, *args, **kwargs):
        pass

    def plan_grippers(self, now_val, target_val):
        num_step = 200
        return {"num_step": num_step, "per_step": (target_val - now_val) / num_step,
                "result": np.linspace(now_val, target_val, num_step)}

    def update_point_cloud(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        raise RuntimeError(f"CuRobo planner is not available (called .{name}); "
                           "end-effector control and expert checks need CuRobo installed")


def _stub_getattr(name: str):
    if name.startswith("__"):  # module dunders (__file__, __path__, ...) must stay missing for inspect & co
        raise AttributeError(name)
    return _Anything


def install_curobo_stub() -> bool:
    """Make RoboTwin importable without CuRobo.

    envs/robot/robot.py does ``from .planner import CuroboPlanner``, and planner.py defines that class
    only if its CuRobo imports succeed. So: register stub ``curobo`` modules, load the package
    ``envs.robot`` without running its __init__, import ``envs.robot.planner`` alone, set its
    CuroboPlanner to a stub, and only then run the package __init__ (which imports robot.py).
    """
    if importlib.util.find_spec("curobo") is not None:
        return False
    for name in ["curobo", "curobo.types", "curobo.types.math", "curobo.types.robot", "curobo.wrap",
                 "curobo.wrap.reacher", "curobo.wrap.reacher.motion_gen", "curobo.util"]:
        mod = types.ModuleType(name)
        mod.__getattr__ = _stub_getattr
        sys.modules[name] = mod
    sys.modules["curobo.util"].logger = _Anything()

    import envs

    pkg_dir = os.path.join(os.path.dirname(os.path.abspath(envs.__file__)), "robot")
    if "envs.robot" not in sys.modules:
        spec = importlib.util.spec_from_file_location("envs.robot", os.path.join(pkg_dir, "__init__.py"),
                                                      submodule_search_locations=[pkg_dir])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["envs.robot"] = pkg
        envs.robot = pkg
        with contextlib.redirect_stdout(io.StringIO()):  # planner.py prints a long warning when CuRobo is off
            planner_mod = importlib.import_module("envs.robot.planner")
        planner_mod.CuroboPlanner = _NoCuroboPlanner
        spec.loader.exec_module(pkg)
    sys.modules["envs.robot.planner"].CuroboPlanner = _NoCuroboPlanner
    sys.modules["envs.robot.robot"].CuroboPlanner = _NoCuroboPlanner
    return True


# ---------------------------------------------------------------------------------------------
# Real RoboTwin
# ---------------------------------------------------------------------------------------------


def setup_kwargs(task_name: str) -> dict:
    import yaml
    from envs import CONFIGS_PATH

    with open(os.path.join(CONFIGS_PATH, "demo_clean.yml"), encoding="utf-8") as f:
        args = yaml.safe_load(f)
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), encoding="utf-8") as f:
        embodiments = yaml.safe_load(f)
    emb = args.get("embodiment", ["aloha-agilex"])
    if len(emb) != 1:
        raise ValueError(f"only single dual-arm embodiments are supported, got {emb}")
    robot_file = embodiments[emb[0]]["file_path"]
    args["left_robot_file"] = args["right_robot_file"] = robot_file
    args["dual_arm_embodied"] = True
    for side in ("left", "right"):
        with open(os.path.join(args[f"{side}_robot_file"], "config.yml"), encoding="utf-8") as f:
            args[f"{side}_embodiment_config"] = yaml.safe_load(f)
    with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), encoding="utf-8") as f:
        cams = yaml.safe_load(f)
    head = args["camera"]["head_camera_type"]
    args["head_camera_h"], args["head_camera_w"] = cams[head]["h"], cams[head]["w"]
    data_type = args.setdefault("data_type", {})
    data_type["rgb"] = data_type["qpos"] = True
    args.update(render_freq=0, task_name=task_name, task_config="demo_clean", eval_mode=True,
                eval_video_log=False, collect_data=False)
    return args


class RealSim:
    def __init__(self):
        self.stub = install_curobo_stub()
        from envs.utils.create_actor import UnStableError

        self.UnStableError = UnStableError
        self.task_name = None
        self.env = None
        self.args = None
        self.cameras = ["head_camera", "left_camera", "right_camera"]
        self.episodes = 0
        self.live = False

    def make(self, task: str, cameras) -> dict:
        if task != self.task_name:
            self.close_episode()
            module = importlib.import_module(f"envs.{task}")
            self.env = getattr(module, task)()
            self.task_name, self.args = task, setup_kwargs(task)
        if cameras:
            self.cameras = list(cameras)
        return {"step_lim": self.step_limit()}

    def step_limit(self) -> int:
        import yaml
        from envs import CONFIGS_PATH

        with open(os.path.join(CONFIGS_PATH, "_eval_step_limit.yml"), encoding="utf-8") as f:
            return int(yaml.safe_load(f).get(self.task_name, 1000))

    def close_episode(self) -> None:
        if self.env is not None and self.live:
            with contextlib.suppress(Exception):
                self.env.close_env(clear_cache=(self.episodes % int(self.args.get("clear_cache_freq", 5)) == 0))
        self.live = False

    def reset(self, seed: int, max_tries: int) -> dict:
        self.close_episode()
        last = None
        for s in range(seed, seed + max_tries):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.env.setup_demo(now_ep_num=self.episodes, seed=s, is_test=True, **self.args)
            except self.UnStableError as e:
                last = e
                with contextlib.suppress(Exception):
                    self.env.close_env()
                continue
            self.live = True
            self.episodes += 1
            instruction = self.task_name.replace("_", " ")  # what lerobot's RoboTwin wrapper feeds the policy
            return {"obs": self.obs(), "seed": s, "step_lim": int(self.env.step_lim or self.step_limit()),
                    "instruction": instruction}
        raise RuntimeError(f"no stable seed in [{seed}, {seed + max_tries}): {last}")

    def obs(self) -> dict:
        raw = self.env.get_obs()
        images = {}
        for cam in self.cameras:
            img = (raw.get("observation", {}).get(cam) or {}).get("rgb")
            img = np.zeros((240, 320, 3), np.uint8) if img is None else np.asarray(img, dtype=np.uint8)[..., :3]
            images[cam] = img
        vec = (raw.get("joint_action") or {}).get("vector")
        state = np.zeros(ACTION_DIM, np.float32) if vec is None else np.asarray(vec, np.float32).ravel()[:ACTION_DIM]
        return {"pixels": images, "agent_pos": state}

    def step(self, action: np.ndarray) -> dict:
        with contextlib.redirect_stdout(io.StringIO()):  # take_action prints a progress line per step
            self.env.take_action(np.asarray(action, dtype=np.float64))
        success = bool(self.env.eval_success)
        steps = int(self.env.take_action_cnt)
        done = success or (self.env.step_lim is not None and steps >= self.env.step_lim)
        return {"obs": self.obs(), "success": success, "done": bool(done), "step": steps}


# ---------------------------------------------------------------------------------------------
# Fake simulator for tests: succeeds after 20 + seed % 30 steps when seed is even, else never.
# ---------------------------------------------------------------------------------------------


class FakeSim:
    def __init__(self, step_lim: int):
        self.step_lim, self.t, self.seed, self.task = step_lim, 0, 0, None
        self.cameras = ["head_camera", "left_camera", "right_camera"]
        self.state = np.zeros(ACTION_DIM, np.float32)

    def make(self, task, cameras):
        self.task = task
        if cameras:
            self.cameras = list(cameras)
        return {"step_lim": self.step_lim}

    def obs(self):
        rng = np.random.default_rng(self.seed * 1000 + self.t)
        return {"pixels": {c: rng.integers(0, 255, (24, 32, 3), dtype=np.uint8) for c in self.cameras},
                "agent_pos": self.state.copy()}

    def reset(self, seed, max_tries):
        if seed % 7 == 6:  # "unstable" seeds are skipped like in RoboTwin
            seed += 1
        self.seed, self.t = seed, 0
        self.state = np.zeros(ACTION_DIM, np.float32)
        return {"obs": self.obs(), "seed": seed, "step_lim": self.step_lim, "instruction": f"fake {self.task}"}

    def step(self, action):
        self.t += 1
        self.state = np.asarray(action, np.float32)
        success = self.seed % 2 == 0 and self.t >= 20 + self.seed % 30
        return {"obs": self.obs(), "success": bool(success), "done": bool(success or self.t >= self.step_lim),
                "step": self.t}


def serve(args) -> None:
    sim = FakeSim(args.fake_step_lim) if args.fake else RealSim()
    with contextlib.suppress(FileNotFoundError):
        os.unlink(args.address)
    with Listener(args.address, family="AF_UNIX", authkey=AUTHKEY) as listener:
        print(f"ready {args.address}", flush=True)
        with listener.accept() as conn:
            while True:
                try:
                    msg = wire.recv(conn)
                except (EOFError, ConnectionResetError):
                    break
                cmd = msg.get("cmd")
                if cmd == "close":
                    wire.send(conn, {"ok": True})
                    break
                started = time.time()
                try:
                    if cmd == "make":
                        out = sim.make(msg["task"], msg.get("cameras"))
                    elif cmd == "reset":
                        out = sim.reset(int(msg["seed"]), int(msg.get("max_tries", 50)))
                    elif cmd == "step":
                        out = sim.step(msg["action"])
                    else:
                        out = {"error": f"unknown command {cmd!r}"}
                except Exception as e:  # keep serving; the client decides what an error means
                    import traceback

                    traceback.print_exc()
                    out = {"error": f"{type(e).__name__}: {e}"}
                out["seconds"] = time.time() - started
                wire.send(conn, out)
    with contextlib.suppress(FileNotFoundError):
        os.unlink(args.address)


if __name__ == "__main__":
    serve(parse_args())
    sys.stdout.flush()
    os._exit(0)  # skip SAPIEN / Vulkan teardown, which can hang
