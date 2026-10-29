#!/usr/bin/env bash
# Stage 6: ablations on the report levels (20 init states per task by default).
#   * h* = best constant instead of the entropy rule (does the signal matter?)
#   * exact guidance (VJP) instead of LeRobot's
#   * r_min 3 and 10
#   S_BEST=10 LAM_BEST=0.5 ZMAX_BEST=2 bash scripts/stages/stage6_ablations.sh
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh
: "${S_BEST:?}"; : "${LAM_BEST:?}"; : "${ZMAX_BEST:?}"
EPISODES=${EPISODES_ABL:-20}
D=${D:-4}
HSTAR=${HSTAR:-$RUNS/hstar.json}
LAMCFG="$RUNS/lam_entropy_z$ZMAX_BEST.json"
python3 - "$HSTAR" "$RUNS/hstar_const.json" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
json.dump({"rule": cfg["families"]["const"]["rule"]}, open(sys.argv[2], "w"), indent=2)
PY
run_eval "abl_const_hstar_d$D" "$X_LEVELS" --planner budget --hstar-config "$RUNS/hstar_const.json" --lam entropy \
  --lam-config "$LAMCFG" --delay "$D" --guidance lerobot
run_eval "abl_rtc_exact_d$D" "$X_LEVELS" --stride "$S_BEST" --delay "$D" --guidance exact
run_eval "abl_budget_exact_d$D" "$X_LEVELS" --planner budget --hstar-config "$HSTAR" --lam entropy \
  --lam-config "$LAMCFG" --delay "$D" --guidance exact
for r in 3 10; do
  run_eval "abl_rmin${r}_d$D" "$X_LEVELS" --planner budget --hstar-config "$HSTAR" --lam entropy \
    --lam-config "$LAMCFG" --r-min "$r" --delay "$D" --guidance lerobot
done
finish
$PY scripts/compare.py --ref "$RUNS/x_budget_entropy_d$D" "$RUNS"/abl_*_d$D "$RUNS/x_rtc_s${S_BEST}_d$D" \
  --suites "$(suites_csv "$X_LEVELS")" --out "$RUNS/ablations_d$D.md"
