"""Simulator object state for the value target only: logged with each request, never an input of the deployed actor.

Runs in the simulator venv (numpy only). The target object is not visible to the controller, so objects are chosen
by the words of the task instruction, with all objects as a recorded fallback.
"""
from __future__ import annotations

import re

import numpy as np

PRIV_DIM = 12
N_OBJECTS = 9  # index of the number of observed objects; 0 means that the environment exposed no object poses
LIFTED_M = .03


def _words(text):
    return set(re.findall(r"[a-z]+", str(text).lower()))


class ObjectTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.first, self.last, self.last_step = {}, {}, None

    def __call__(self, obs, language, step):
        positions = {k[:-4]: np.asarray(v, np.float64) for k, v in obs.items()
                     if k.endswith("_pos") and not k.startswith("robot") and "_to_" not in k and np.shape(v) == (3,)}
        grip = np.asarray(obs.get("robot0_gripper_qpos", np.zeros(2)), np.float64)
        eef = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), np.float64)
        out = np.zeros(PRIV_DIM, np.float32)
        out[7:9] = grip[0] - grip[-1], eef[2]
        if not positions:
            return out
        words = _words(language)
        names = [n for n in sorted(positions) if _words(n) & words]
        fallback = not names
        names = names or sorted(positions)
        dt = max(int(step) - self.last_step, 1) if self.last_step is not None else 1
        speed = max((np.linalg.norm(p - self.last.get(n, p)) / dt for n, p in positions.items()), default=0.)
        for name, p in positions.items():
            self.first.setdefault(name, p.copy())
        self.last, self.last_step = {n: p.copy() for n, p in positions.items()}, int(step)
        p = np.stack([positions[n] for n in names])
        first = np.stack([self.first[n] for n in names])
        to_eef = np.linalg.norm(p - eef, axis=1)
        lift = p[:, 2] - first[:, 2]
        pair = [np.linalg.norm(p[i] - p[j]) for i in range(len(p)) for j in range(i)]
        out[:7] = (to_eef.min(), to_eef.mean(), lift.max(), np.mean(lift > LIFTED_M),
                   np.linalg.norm((p - first)[:, :2], axis=1).max(), speed, min(pair, default=0.))
        out[9:12] = len(positions), float(fallback), to_eef[int(lift.argmax())]
        return out
