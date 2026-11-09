#!/usr/bin/env bash
# Stage 3: confidence budget with fixed lambda on the tuning levels; compare with RTC (s*).
#   S_BEST=10 bash scripts/stages/stage3_lambda_grid.sh
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh
: "${S_BEST:?set S_BEST to the best RTC step from stage 1}"
D=${D:-4}
HSTAR=${HSTAR:-$RUNS/hstar.json}
R_MIN=${R_MIN:-5}
for lam in ${LAMBDAS:-0 0.25 0.5 0.75 1}; do
  run_eval "y_budget_lam${lam}_d$D" "$Y_LEVELS" --planner budget --hstar-config "$HSTAR" --lam "$lam" \
    --r-min "$R_MIN" --delay "$D" --guidance lerobot
done
finish
$PY scripts/compare.py --ref "$RUNS/y_rtc_s${S_BEST}_d$D" "$RUNS"/y_budget_lam*_d$D \
  --suites "$(suites_csv "$Y_LEVELS")" --out "$RUNS/stage3_d$D.md"
