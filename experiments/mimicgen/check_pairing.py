#!/usr/bin/env python
"""Pairing check of MimicGen runs: every (suite, task, init state) must have the same init_hash (the simulator state
after the reset) in every run that has it. Prints the number of items checked; exits 1 on a mismatch.

    python ../mimicgen/check_pairing.py runs/mimicgen/mg_sync_s10 runs/mimicgen/mg_naive_s10_d2 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def hashes(run: Path) -> dict:
    out = {}
    for f in sorted(run.glob("mimicgen_*.json")):
        if f.name.endswith("_chunks.json"):
            continue
        r = json.loads(f.read_text())
        for e in r.get("episodes", []):
            if "init_hash" in e:
                out[(r["suite"], e["task_id"], e["init_state"])] = e["init_hash"]
    return out


def check(runs: list[Path]) -> tuple[int, list[str]]:
    seen: dict = {}
    problems = []
    for run in runs:
        for key, h in hashes(run).items():
            first = seen.setdefault(key, (run, h))
            if first[1] != h:
                problems.append(f"{key}: {first[0].name} {first[1]} != {run.name} {h}")
    return len(seen), problems


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", type=Path, nargs="+")
    a = p.parse_args()
    n, problems = check(a.runs)
    for line in problems[:20]:
        print("MISMATCH", line)
    print(f"pairing: {n} items over {len(a.runs)} runs, {len(problems)} mismatches")
    sys.exit(1 if problems or n == 0 else 0)


if __name__ == "__main__":
    main()
