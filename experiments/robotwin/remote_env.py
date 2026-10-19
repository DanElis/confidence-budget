"""Client side of robotwin_server.py: a gym-like env per simulator process.

RemoteEnv spawns ``<robotwin_python> robotwin_server.py`` in the RoboTwin checkout and talks to it
over a unix socket. It implements reset / step and the step_async / step_wait (reset_async /
reset_wait) pairs that closed_loop.run_batch uses to step all simulators of a batch concurrently.

Episode identity: ``set_episode(task, seed)`` before ``reset``; the server skips unstable seeds
deterministically (seed, seed + 1, ...) and reports the seed it used.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile
import time
import uuid
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wire  # noqa: E402

SERVER = Path(__file__).resolve().parent / "robotwin_server.py"
AUTHKEY = b"robotwin-confidence-budget"


class SimulatorError(RuntimeError):
    pass


class RemoteEnv:
    def __init__(self, python: str, robotwin_root: str | Path | None, log_path: str | Path | None = None,
                 fake: bool = False, fake_step_lim: int = 60, start_timeout: float = 600.0,
                 cameras=("head_camera", "left_camera", "right_camera")):
        self.cameras = list(cameras)
        self.address = os.path.join(tempfile.gettempdir(), f"robotwin-{uuid.uuid4().hex[:12]}.sock")
        cmd = [python, str(SERVER), "--address", self.address]
        if fake:
            cmd += ["--fake", "--fake-step-lim", str(fake_step_lim)]
        env = dict(os.environ)
        if robotwin_root is not None:
            env["PYTHONPATH"] = f"{robotwin_root}{os.pathsep}{env.get('PYTHONPATH', '')}"
        for key in ("VIRTUAL_ENV", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT"):  # do not leak our venv
            env.pop(key, None)
        self.log = open(log_path, "ab") if log_path else subprocess.DEVNULL  # noqa: SIM115
        self.proc = subprocess.Popen(cmd, cwd=str(robotwin_root) if robotwin_root else None, env=env,
                                     stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.time() + start_timeout
        self.conn = None
        self.task = None
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise SimulatorError(f"simulator server exited with code {self.proc.returncode}; see {log_path}")
            if os.path.exists(self.address):
                with contextlib.suppress(ConnectionRefusedError, FileNotFoundError):
                    self.conn = Client(self.address, family="AF_UNIX", authkey=AUTHKEY)
                    break
            time.sleep(0.2)
        if self.conn is None:
            self.close()
            raise SimulatorError(f"simulator server did not start within {start_timeout:.0f} s; see {log_path}")
        self.task, self.seed, self.used_seed, self.step_lim, self.instruction = None, 0, None, None, None
        self.last_obs = None
        self.sim_seconds = 0.0
        self._pending = None

    # -- protocol -----------------------------------------------------------------------------
    @property
    def alive(self) -> bool:
        return self.conn is not None and self.proc.poll() is None

    def _send(self, msg: dict) -> None:
        if not self.alive:
            raise SimulatorError("simulator server is not running")
        wire.send(self.conn, msg)

    def _call(self, msg: dict) -> dict:
        self._send(msg)
        return self._reply(msg["cmd"])

    def _reply(self, cmd: str) -> dict:
        out = wire.recv(self.conn)
        self.sim_seconds += float(out.get("seconds", 0.0))
        if "error" in out:
            raise SimulatorError(f"{cmd}: {out['error']}")
        return out

    # -- gym-like API -------------------------------------------------------------------------
    def set_episode(self, task: str, seed: int) -> None:
        if task != self.task:
            self.step_lim = int(self._call({"cmd": "make", "task": task, "cameras": self.cameras})["step_lim"])
            self.task = task
        self.seed = int(seed)

    def reset(self, **_):
        self.reset_async()
        return self.reset_wait()

    def reset_async(self) -> None:
        self._send({"cmd": "reset", "seed": self.seed, "max_tries": 50})

    def reset_wait(self):
        out = self._reply("reset")
        self.used_seed, self.step_lim, self.instruction = int(out["seed"]), int(out["step_lim"]), out["instruction"]
        self.last_obs = out["obs"]
        return self.last_obs, {"seed": self.used_seed}

    def step(self, action):
        self.step_async(action)
        return self.step_wait()

    def step_async(self, action) -> None:
        self._send({"cmd": "step", "action": np.asarray(action, dtype=np.float64)})
        self._pending = True

    def step_wait(self):
        self._pending = None
        try:
            out = self._reply("step")
        except SimulatorError as e:  # a simulator error ends the episode as a failure
            return self.last_obs, 0.0, True, False, {"is_success": False, "error": str(e)}
        except (EOFError, OSError) as e:  # the server died: fail this episode; the next reset raises
            self.conn = None
            return self.last_obs, 0.0, True, False, {"is_success": False, "error": f"server died: {e!r}"}
        self.last_obs = out["obs"]
        success = bool(out["success"])
        return self.last_obs, float(success), success, bool(out["done"]) and not success, \
            {"is_success": success, "step": int(out["step"])}

    def close(self) -> None:
        if getattr(self, "conn", None) is not None:
            with contextlib.suppress(Exception):
                wire.send(self.conn, {"cmd": "close"})
                wire.recv(self.conn)
            with contextlib.suppress(Exception):
                self.conn.close()
            self.conn = None
        if getattr(self, "proc", None) is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.log not in (None, subprocess.DEVNULL):
            with contextlib.suppress(Exception):
                self.log.close()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.address)

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()
