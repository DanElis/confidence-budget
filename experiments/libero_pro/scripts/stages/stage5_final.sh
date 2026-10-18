#!/usr/bin/env bash
# Stage 5: final evaluation on the report levels (libero_object, x0.1-x0.4), 50 init states per task.
#   S_BEST=10 S_SYNC=5 LAM_BEST=0.5 ZMAX_BEST=2 bash scripts/stages/stage5_final.sh
# About 4-5 h per method and delay with JOBS=1; JOBS=4-5 divides that.
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh
: "${S_BEST:?}"; : "${S_SYNC:?}"; : "${LAM_BEST:?}"; : "${ZMAX_BEST:?}"
EPISODES=${EPISODES_FINAL:-50}
HSTAR=${HSTAR:-$RUNS/hstar.json}
R_MIN=${R_MIN:-5}
LAMCFG="$RUNS/lam_entropy_z$ZMAX_BEST.json"
run_eval "x_sync_s$S_SYNC" "$X_LEVELS" --stride "$S_SYNC" --delay 0 --guidance none
for d in ${DELAYS:-4 2 8}; do
  run_eval "x_naive_s${S_BEST}_d$d" "$X_LEVELS" --stride "$S_BEST" --delay "$d" --guidance none
  run_eval "x_rtc_s${S_BEST}_d$d"   "$X_LEVELS" --stride "$S_BEST" --delay "$d" --guidance lerobot
  run_eval "x_budget_entropy_d$d"   "$X_LEVELS" --planner budget --hstar-config "$HSTAR" --lam entropy \
    --lam-config "$LAMCFG" --r-min "$R_MIN" --delay "$d" --guidance lerobot
  run_eval "x_budget_lam${LAM_BEST}_d$d" "$X_LEVELS" --planner budget --hstar-config "$HSTAR" --lam "$LAM_BEST" \
    --r-min "$R_MIN" --delay "$d" --guidance lerobot
  if [[ $d == 4 ]]; then  # the two extremes of the budget only at the main delay
    for lam in 0 1; do
      run_eval "x_budget_lam${lam}_d$d" "$X_LEVELS" --planner budget --hstar-config "$HSTAR" --lam "$lam" \
        --r-min "$R_MIN" --delay "$d" --guidance lerobot
    done
  fi
done
finish
for d in ${DELAYS:-4 2 8}; do
  $PY scripts/compare.py --ref "$RUNS/x_rtc_s${S_BEST}_d$d" "$RUNS/x_sync_s$S_SYNC" "$RUNS/x_naive_s${S_BEST}_d$d" \
    "$RUNS"/x_budget_*_d$d --suites "$(suites_csv "$X_LEVELS")" --out "$RUNS/final_d$d.md"
done
