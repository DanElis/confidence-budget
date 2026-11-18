#!/usr/bin/env python
"""Request time of the openpi server for one client, without and with RTC of the paper (Jacobian), and the delay in
control steps it implies: d = ceil(median ms / control period). The simulated delay of the closed loop is then this d.

    python ../mimicgen/latency.py --port 8900 --delay 2 --stride 10 --out runs/mimicgen/latency.txt
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "libero_pro" / "scripts"))
sys.path.insert(0, str(HERE.parent / "libero_pro"))
import mimicgen_env as mg  # noqa: E402
from eval_openpi import MODEL_ACTION_DIM, OpenPiClient, paper_weights  # noqa: E402


def timed(client, element, n: int) -> list[float]:
    out = []
    for _ in range(n):
        started = time.monotonic()
        client.infer(element)
        out.append((time.monotonic() - started) * 1000)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--requests", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5, help="requests left out: the first ones compile")
    p.add_argument("--control-hz", type=float, default=20.0, help="control_freq of the environment")
    p.add_argument("--delay", type=int, default=2, help="d of the RTC weights of the guided requests")
    p.add_argument("--stride", type=int, default=10, help="s of the RTC weights (mask to H - s)")
    p.add_argument("--horizon", type=int, default=mg.HORIZON)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    client = OpenPiClient(a.host, a.port)
    rng = np.random.default_rng(0)
    element = mg.probe_element() | {"noise": rng.standard_normal((1, a.horizon, MODEL_ACTION_DIM), dtype=np.float32)}
    reply = client.infer(element)
    if np.asarray(reply["actions"]).shape[-2] != a.horizon:
        raise SystemExit(f"server returned {np.asarray(reply['actions']).shape}, expected H = {a.horizon}")
    plan = np.asarray(reply["normalized_action"], dtype=np.float32)[0]
    guided = element | {"rtc_target": plan, "rtc_weights": paper_weights(a.delay, a.horizon - a.stride, a.horizon),
                        "rtc_max_weight": 5.0, "rtc_jacobian": True}
    period = 1000.0 / a.control_hz
    lines = []
    for name, el in (("plain", element), ("rtc_paper", guided)):
        timed(client, el, a.warmup)
        ms = np.array(timed(client, el, a.requests))
        med = float(np.median(ms))
        lines.append(f"{name}: median {med:.1f} ms, p10 {np.quantile(ms, 0.1):.1f}, p90 {np.quantile(ms, 0.9):.1f} "
                     f"over {a.requests}; period {period:.0f} ms -> d = {max(1, math.ceil(med / period))}")
    client.close()
    text = "\n".join(lines)
    print(text)
    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
