#!/usr/bin/env bash
# All experiments in one go, without waiting for choices between stages.
#
# 1. h* rule and entropy-lambda parameters from the offline H1 runs (minutes, no GPU).
# 2. The whole configuration grid on the tuning levels (y, EPISODES_Y per task) AND on the
#    report levels (libero_object + x, EPISODES_X per task), all launched into one pool of
#    $JOBS parallel processes. Tuning runs are queued first.
# 3. scripts/select_report.py then picks s*, lambda*, z_max* on y only and writes the final
#    tables on x for the picked configurations (choice fixed in advance, so it stays fair).
#
# Grid at each delay in $DELAYS (default 4):
#   sync s=5,10 (delay 0) | naive s=5,10,15,20 | RTC s=5,10,15,20 |
#   budget lambda=0,0.25,0.5,0.75,1 | budget lambda from entropy, z_max=1,2,3
# Ablations on x (EPISODES_ABL per task): h* = best constant, exact guidance (RTC s=10 and budget),
# r_min = 3, 10 (budget, entropy lambda, z_max=2).
#
# Usage (from experiments/libero_pro):
#   tmux new -s all
#   bash scripts/stages/run_all.sh                    # ~21 h with JOBS=5 (~45k episodes)
#   EPISODES_X=20 bash scripts/stages/run_all.sh      # ~11 h, wider confidence intervals on x
#   DELAYS="2 4 8" bash scripts/stages/run_all.sh     # also the other delays (~2.5x longer)
# Re-running skips finished suites. Progress: runs/all.log; result: runs/report.md
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh
JOBS=${JOBS:-5}
EPISODES_Y=${EPISODES_Y:-10}
EPISODES_X=${EPISODES_X:-50}
EPISODES_ABL=${EPISODES_ABL:-20}
DELAYS=${DELAYS:-4}
STRIDES="5 10 15 20"
LAMBDAS="0 0.25 0.5 0.75 1"
ZMAXES="1 2 3"
H1=${H1_RUNS:-../h1/runs}
HSTAR=$RUNS/hstar.json
main() {
# 1. calibration ------------------------------------------------------------------
if [[ ! -f $HSTAR ]]; then
  CONFLICT=$(ls "$H1"/h1_stale*_lerobot.npz "$H1"/h1_shift*_lerobot.npz 2>/dev/null | tr '\n' ' ')
  $PY scripts/calibrate_hstar.py "$H1/h1_lerobot.npz" --conflict $CONFLICT --out "$HSTAR" || exit 1
fi
for z in $ZMAXES; do
  python3 - "$HSTAR" "$RUNS/lam_entropy_z$z.json" "$z" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))["lambda_entropy"]
cfg["z_max"] = float(sys.argv[3])
json.dump({"lambda_entropy": cfg}, open(sys.argv[2], "w"), indent=2)
PY
done
python3 - "$HSTAR" "$RUNS/hstar_const.json" <<'PY'
import json, sys
json.dump({"rule": json.load(open(sys.argv[1]))["families"]["const"]["rule"]}, open(sys.argv[2], "w"), indent=2)
PY

# 2. grid -------------------------------------------------------------------------
grid() {  # grid PREFIX LEVELS EPISODES
  local p=$1 levels=$2 ep=$3 d s lam z
  for s in 5 10; do EPISODES=$ep run_eval "${p}_sync_s$s" "$levels" --stride "$s" --delay 0 --guidance none; done
  for d in $DELAYS; do
    for s in $STRIDES; do
      EPISODES=$ep run_eval "${p}_naive_s${s}_d$d" "$levels" --stride "$s" --delay "$d" --guidance none
      EPISODES=$ep run_eval "${p}_rtc_s${s}_d$d"   "$levels" --stride "$s" --delay "$d" --guidance lerobot
    done
    for lam in $LAMBDAS; do
      EPISODES=$ep run_eval "${p}_budget_lam${lam}_d$d" "$levels" --planner budget --hstar-config "$HSTAR" \
        --lam "$lam" --delay "$d" --guidance lerobot
    done
    for z in $ZMAXES; do
      EPISODES=$ep run_eval "${p}_budget_entropy_z${z}_d$d" "$levels" --planner budget --hstar-config "$HSTAR" \
        --lam entropy --lam-config "$RUNS/lam_entropy_z$z.json" --delay "$d" --guidance lerobot
    done
  done
}
grid y "$Y_LEVELS" "$EPISODES_Y"
grid x "$X_LEVELS" "$EPISODES_X"

for d in $DELAYS; do
  ABL=(--planner budget --lam entropy --lam-config "$RUNS/lam_entropy_z2.json" --delay "$d")
  EPISODES=$EPISODES_ABL run_eval "abl_const_hstar_d$d" "$X_LEVELS" "${ABL[@]}" --hstar-config "$RUNS/hstar_const.json" --guidance lerobot
  EPISODES=$EPISODES_ABL run_eval "abl_budget_exact_d$d" "$X_LEVELS" "${ABL[@]}" --hstar-config "$HSTAR" --guidance exact
  EPISODES=$EPISODES_ABL run_eval "abl_rtc_exact_s10_d$d" "$X_LEVELS" --stride 10 --delay "$d" --guidance exact
  for r in 3 10; do
    EPISODES=$EPISODES_ABL run_eval "abl_rmin${r}_d$d" "$X_LEVELS" "${ABL[@]}" --hstar-config "$HSTAR" --r-min "$r" --guidance lerobot
  done
done
finish

# 3. selection on y, report on x ---------------------------------------------------
$PY scripts/select_report.py --runs "$RUNS" --delays $DELAYS --out "$RUNS/report.md"
}

main 2>&1 | tee -a "$RUNS/all.log"
