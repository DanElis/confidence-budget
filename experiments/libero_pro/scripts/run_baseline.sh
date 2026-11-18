#!/usr/bin/env bash
# Synchronous PI0.5 baseline on LIBERO-Object and the LIBERO-PRO position levels.
#
# Published reference (AAC, GeoAAC; PI0.5, fixed horizon 5): 53.2% at x0.2,
# 30.9% averaged over x0.2-x0.4. Reproducing it checks the whole setup before
# any asynchronous or RTC run.
#
# Usage (from experiments/libero_pro):
#   bash scripts/run_baseline.sh            # 20 episodes per task, about 2-3 h on one A100
#   EPISODES=50 bash scripts/run_baseline.sh
set -u
cd "$(dirname "$0")/.."
mkdir -p runs
EPISODES=${EPISODES:-20}
SUITES=${SUITES:-"libero_object,pos:x0.1,pos:x0.2,pos:x0.3,pos:x0.4,pos:x0.5"}
for s in ${STRIDES:-5 10}; do
  uv run python scripts/eval_libero.py --suites "$SUITES" --episodes-per-task "$EPISODES" \
    --stride "$s" --delay 0 --guidance none --out "runs/sync_s$s" 2>&1 | tee -a "runs/sync_s$s.log"
done
