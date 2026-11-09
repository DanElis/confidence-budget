#!/usr/bin/env bash
# Stage 4: lambda from the entropy of the chunk's early actions (H2); compare with RTC and the best fixed lambda.
#   S_BEST=10 LAM_BEST=0.5 bash scripts/stages/stage4_lambda_entropy.sh
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh
: "${S_BEST:?set S_BEST}"; : "${LAM_BEST:?set LAM_BEST to the best fixed lambda from stage 3}"
D=${D:-4}
HSTAR=${HSTAR:-$RUNS/hstar.json}
R_MIN=${R_MIN:-5}
for zmax in ${ZMAX:-1 2 3}; do
  python3 - "$HSTAR" "$RUNS/lam_entropy_z$zmax.json" "$zmax" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))["lambda_entropy"]
cfg["z_max"] = float(sys.argv[3])
json.dump({"lambda_entropy": cfg}, open(sys.argv[2], "w"), indent=2)
PY
  run_eval "y_budget_entropy_z${zmax}_d$D" "$Y_LEVELS" --planner budget --hstar-config "$HSTAR" --lam entropy \
    --lam-config "$RUNS/lam_entropy_z$zmax.json" --r-min "$R_MIN" --delay "$D" --guidance lerobot
done
finish
$PY scripts/compare.py --ref "$RUNS/y_rtc_s${S_BEST}_d$D" "$RUNS/y_budget_lam${LAM_BEST}_d$D" "$RUNS"/y_budget_entropy_z*_d$D \
  --suites "$(suites_csv "$Y_LEVELS")" --out "$RUNS/stage4_d$D.md"
