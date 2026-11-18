#!/usr/bin/env bash
# Stage 1: synchronous, naive asynchronous and RTC with a fixed step on the tuning levels (y).
# Picks the best fixed step s* for RTC at d = 4. About 3-4 h with JOBS=4.
#   bash scripts/stages/stage1_async_baselines.sh
# Then for the other delays (after choosing S_BEST):
#   S_BEST=10 DELAYS="2 8" bash scripts/stages/stage1_async_baselines.sh
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh
D=${D:-4}
STRIDES=${STRIDES:-"5 10 15 20"}
if [[ -z "${S_BEST:-}" ]]; then
  for s in 5 10; do run_eval "y_sync_s$s" "$Y_LEVELS" --stride "$s" --delay 0 --guidance none; done
  for s in $STRIDES; do
    run_eval "y_naive_s${s}_d$D" "$Y_LEVELS" --stride "$s" --delay "$D" --guidance none
    run_eval "y_rtc_s${s}_d$D"   "$Y_LEVELS" --stride "$s" --delay "$D" --guidance lerobot
  done
  finish
  $PY scripts/compare.py --ref "$RUNS/y_sync_s5" "$RUNS"/y_sync_s10 "$RUNS"/y_naive_s*_d$D "$RUNS"/y_rtc_s*_d$D \
    --suites "$(suites_csv "$Y_LEVELS")" --out "$RUNS/stage1_d$D.md"
else
  for d in ${DELAYS:-2 8}; do
    run_eval "y_naive_s${S_BEST}_d$d" "$Y_LEVELS" --stride "$S_BEST" --delay "$d" --guidance none
    run_eval "y_rtc_s${S_BEST}_d$d"   "$Y_LEVELS" --stride "$S_BEST" --delay "$d" --guidance lerobot
  done
  finish
  $PY scripts/compare.py --ref "$RUNS/y_sync_s5" "$RUNS"/y_naive_s${S_BEST}_d* "$RUNS"/y_rtc_s${S_BEST}_d* \
    --suites "$(suites_csv "$Y_LEVELS")" --out "$RUNS/stage1_delays.md"
fi
