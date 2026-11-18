#!/usr/bin/env bash
# Stage 2: choose the h* rule.
#   (a) offline on the H1 runs (no GPU, minutes);
#   (b) one closed-loop RTC run with entropy recording on the tuning levels;
#   (c) calibrate again with the closed-loop check -> runs/hstar.json (rule + lambda_entropy).
#   S_BEST=10 bash scripts/stages/stage2_calibrate_hstar.sh
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh
: "${S_BEST:?set S_BEST to the best RTC step from stage 1}"
D=${D:-4}
H1=${H1_RUNS:-../h1/runs}
CONFLICT=$(ls "$H1"/h1_stale*_lerobot.npz "$H1"/h1_shift*_lerobot.npz 2>/dev/null | tr '\n' ' ')
$PY scripts/calibrate_hstar.py "$H1/h1_lerobot.npz" --conflict $CONFLICT --out "$RUNS/hstar_offline.json"
run_eval "y_rtc_s${S_BEST}_d${D}_entropy" "$Y_LEVELS" --stride "$S_BEST" --delay "$D" --guidance lerobot --record-entropy
finish
$PY scripts/calibrate_hstar.py "$H1/h1_lerobot.npz" --conflict $CONFLICT \
  --closed-loop "$RUNS/y_rtc_s${S_BEST}_d${D}_entropy" --out "$RUNS/hstar.json"
