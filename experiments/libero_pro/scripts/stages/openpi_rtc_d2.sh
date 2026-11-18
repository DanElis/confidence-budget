#!/usr/bin/env bash
# Constant RTC of the paper (Jacobian, beta = 5) on openpi's pi05_libero, H = 10, d = 2: does RTC help at all at d = 2?
# y0.1-y0.3, seed 1, the scenes and noise of the naive constant runs openpi_budget_const_s<s>_d2 (openpi_budget.sh).
# A pair (s, r): step s, the first d actions frozen, soft weights up to d + r (--mask-end d + r), d + r <= H - s.
#
#   openpi_budget_rtc_s<s>_r<r>_d2   for PAIRS (default 2:2 2:6 4:2 4:4 5:3 6:2)
#
# Fixed before the run. For each pair: RTC against naive with the same s, paired Δ of success over y0.1-y0.3, CI at
# 1 - 0.05 / (number of pairs) (Bonferroni). The best constant pair per s (ties to naive) goes to
# runs/openpi_budget_rtc_d2/constants.json: it is the constant reference of openpi_budget_rtc.sh.
#
# Usage (from experiments/libero_pro, after openpi_budget.sh):
#   tmux new -d -s rtc_d2 "bash scripts/stages/openpi_rtc_d2.sh"
# PAIRS="2:2 2:6 4:2 4:4 5:3 6:2" EPISODES_PER_TASK=20 SEED=1 PORT=8840 JOBS=6 by default; re-running skips finished suites.
JOBS=${JOBS:-6}
SUITE_TIMEOUT=${SUITE_TIMEOUT:-12h}
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh && source scripts/stages/openpi_lib.sh
EPISODES=${EPISODES_PER_TASK:-20}
DELAY=${DELAY:-2}
PAIRS=${PAIRS:-"2:2 2:6 4:2 4:4 5:3 6:2"}
SEED=${SEED:-1}
PORT=${PORT:-8840}
LEVELS=${LEVELS:-"pos:y0.1 pos:y0.2 pos:y0.3"}
SERVER_LOG=openpi_rtc_d2
OUT=$RUNS/openpi_budget_rtc_d$DELAY
FIT="$PY scripts/openpi_budget_fit.py"

main() {
trap 'trap - INT TERM; stop_servers; kill 0' INT TERM
trap stop_servers EXIT
echo "$(date '+%F %T') openpi_rtc_d2: pairs=$PAIRS d=$DELAY EPISODES=$EPISODES seed=$SEED levels=$LEVELS"
git log -1 --oneline
mkdir -p "$OUT"
local p s r runs=() strides=""
for p in $PAIRS; do
  s=${p%%:*}; r=${p##*:}
  (( DELAY + r <= 10 - s )) || { echo "pair $p: need d + r <= H - s"; exit 1; }
  strides+=" $s"
done
for s in $(tr ' ' '\n' <<< "$strides" | sort -nu); do
  [[ -f $RUNS/openpi_budget_const_s${s}_d$DELAY/libero_object_pos_y0.3.json ]] || \
    { echo "no naive run openpi_budget_const_s${s}_d$DELAY: run openpi_budget.sh first"; exit 1; }
  runs+=("$RUNS/openpi_budget_const_s${s}_d$DELAY")
done
start_servers || exit 1
p=${PAIRS%% *}
smoke_eval openpi_rtc_d2 --stride "${p%%:*}" --delay "$DELAY" --guidance paper --mask-end $(( DELAY + ${p##*:} )) || exit 1

for p in $PAIRS; do
  s=${p%%:*}; r=${p##*:}
  runs+=("$RUNS/openpi_budget_rtc_s${s}_r${r}_d$DELAY")
  per_level "openpi_budget_rtc_s${s}_r${r}_d$DELAY" --stride "$s" --delay "$DELAY" --guidance paper \
    --mask-end $(( DELAY + r ))
done
finish
stop_servers
$FIT constants "${runs[@]}" --delay "$DELAY" --level "$(awk -v n="$(wc -w <<< "$PAIRS")" 'BEGIN { print 1 - 0.05 / n }')" \
  --out "$OUT" || exit 1
echo "$(date '+%F %T') done: $OUT/constants.md"
}

main 2>&1 | tee -a "$RUNS/openpi_rtc_d2.log"
