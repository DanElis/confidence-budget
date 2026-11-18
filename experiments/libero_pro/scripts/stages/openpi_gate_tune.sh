#!/usr/bin/env bash
# Adaptive RTC on openpi, step 1 of 2: the threshold, on the tuning levels pos:y0.1-y0.3 (the report is on x, seed 2).
# H = 10, d = 4, s = 5, RTC of the paper (--guidance paper).
#
# Per request the chunk is generated without guidance first; D = mean |new - old plan| over their overlap. The request is
# guided (a second call, same noise) if D < tau; otherwise the unguided chunk runs, exactly as naive.
#
#   openpi_gate_cal_s5_d4        tau = 0: naive execution with D logged; its D gives tau for guided shares 25/50/75%
#   openpi_gate_q<NN>_s5_d4      the gate at each of those tau
#
# Fixed before the run (gate_summary.py --select): the highest mean success over y0.1-y0.3; within 1 pp of it the larger
# guided share wins. The chosen "run tau" goes to runs/openpi_gate_tau.txt for openpi_gate_report.sh. The shares are
# those of naive states: under the gate they shift, the table in runs/openpi_gate_tune_report.md gives the real ones.
#
# Usage (from experiments/libero_pro):  tmux new -d -s gate_tune "bash scripts/stages/openpi_gate_tune.sh"
# EPISODES_PER_TASK=20 SEED=1 PORT=8800 SHARES="0.25 0.5 0.75" by default; re-running skips finished suites.
SUITE_TIMEOUT=${SUITE_TIMEOUT:-12h}
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh && source scripts/stages/openpi_lib.sh
EPISODES=${EPISODES_PER_TASK:-20}
STRIDE=${STRIDE:-5}
DELAY=${DELAY:-4}
SEED=${SEED:-1}
PORT=${PORT:-8800}
LEVELS=${LEVELS:-"pos:y0.1 pos:y0.2 pos:y0.3"}
SHARES=${SHARES:-"0.25 0.5 0.75"}
SERVER_LOG=openpi_gate_tune
REPORT=$RUNS/openpi_gate_tune_report.md
TAU_FILE=$RUNS/openpi_gate_tau.txt
RTC=(--stride "$STRIDE" --delay "$DELAY" --guidance paper)

main() {
trap 'trap - INT TERM; stop_servers; kill 0' INT TERM
trap stop_servers EXIT
echo "$(date '+%F %T') openpi_gate_tune: s=$STRIDE d=$DELAY shares=$SHARES EPISODES=$EPISODES seed=$SEED levels=$LEVELS"
git log -1 --oneline
start_servers || exit 1
smoke_eval openpi_gate "${RTC[@]}" --rtc-gate divergence --rtc-tau 0.1 || exit 1

local cal="openpi_gate_cal_s${STRIDE}_d$DELAY" taus q tau i=0 runs=()
per_level "$cal" "${RTC[@]}" --rtc-gate divergence --rtc-tau 0
finish
taus=$($PY scripts/gate_summary.py "$RUNS/$cal" --tau-for $SHARES) || exit 1
read -ra taus <<< "$taus"
echo "$(date '+%F %T') shares $SHARES -> tau ${taus[*]}"
for q in $SHARES; do
  tau=${taus[$i]}; i=$(( i + 1 ))
  local name="openpi_gate_q$(percent "$q")_s${STRIDE}_d$DELAY"
  runs+=("$RUNS/$name")
  per_level "$name" "${RTC[@]}" --rtc-gate divergence --rtc-tau "$tau"
done
finish
stop_servers

$PY scripts/gate_summary.py "$RUNS/$cal" "${runs[@]}" > "$REPORT"
echo >> "$REPORT"
$PY scripts/compare.py --ref "$RUNS/$cal" "$RUNS/$cal" "${runs[@]}" >> "$REPORT"
$PY scripts/gate_summary.py "${runs[@]}" --select > "$TAU_FILE" || exit 1
echo "$(date '+%F %T') chosen: $(cat "$TAU_FILE")"
echo "$(date '+%F %T') done: $REPORT"
}

main 2>&1 | tee -a "$RUNS/openpi_gate_tune.log"
