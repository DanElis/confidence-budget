#!/usr/bin/env python
"""Where does a RoboTwin step spend its time? Run with RoboTwin's Python from the RoboTwin root:

    cd third_party/RoboTwin
    PYTHONPATH=$PWD .venv/bin/python ../../scripts/profile_sim.py --task place_a2b_left --steps 20

Times scene setup, get_obs (camera rendering) and take_action (TOPP + physics) separately, with a
hold-still action, then prints the top functions of take_action by cumulative time.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402

import robotwin_server as S  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="place_a2b_left")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=100000)
    args = p.parse_args()
    import torch

    print(f"threads: torch {torch.get_num_threads()}, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, "
          f"cpus {os.cpu_count()}")
    t = time.time()
    sim = S.RealSim()
    sim.make(args.task, None)
    print(f"import + make: {time.time() - t:.1f} s")
    t = time.time()
    out = sim.reset(args.seed, 50)
    print(f"reset (scene setup + first obs): {time.time() - t:.1f} s, seed {out['seed']}, step_lim {out['step_lim']}")

    hold = out["obs"]["agent_pos"].astype(np.float64)
    obs_t, act_t = [], []
    prof = cProfile.Profile()
    for _ in range(args.steps):
        t = time.time()
        prof.enable()
        sim.env.take_action(hold)
        prof.disable()
        act_t.append(time.time() - t)
        t = time.time()
        sim.obs()
        obs_t.append(time.time() - t)
    print(f"take_action: mean {np.mean(act_t):.3f} s, median {np.median(act_t):.3f} s")
    print(f"get_obs:     mean {np.mean(obs_t):.3f} s, median {np.median(obs_t):.3f} s")
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(18)
    print(s.getvalue())


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
