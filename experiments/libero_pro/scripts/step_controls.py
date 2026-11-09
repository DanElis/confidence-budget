#!/usr/bin/env python
"""Constant steps to compare a finished adaptive run with: the integer whose request rate is within 5% of the run's,
else the two integers around it (the comparison at equal rate is then bracketed, not exact).

    python scripts/step_controls.py RUN    # prints e.g. "7" or "7 8"

RUN is an eval_libero.py run (*_chunks.json) or a LIBERO-Dynamic run of policy_server.py (signals_*.jsonl).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

EQUAL_RATE = 0.05


def load_steps(run: Path) -> list[int]:
    rows = [c for f in sorted(run.glob("*_chunks.json")) for c in json.loads(f.read_text())]
    rows += [json.loads(line) for f in sorted(run.glob("signals_*.jsonl")) for line in f.read_text().splitlines()
             if line.strip()]
    return [s for s in (r.get("stride", r.get("s")) for r in rows) if s is not None]


def controls(steps: list[int]) -> list[int]:
    mean = sum(steps) / len(steps)  # T / number of requests: the request rate is 1 / mean
    nearest = round(mean)
    if abs(nearest / mean - 1) <= EQUAL_RATE:
        return [nearest]
    return [math.floor(mean), math.ceil(mean)]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    args = p.parse_args(argv)
    steps = load_steps(args.run)
    if not steps:
        sys.exit(f"no planned steps in {args.run}")
    print(*controls(steps))


if __name__ == "__main__":
    main()
