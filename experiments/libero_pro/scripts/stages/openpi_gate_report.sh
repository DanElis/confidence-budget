#!/usr/bin/env bash
# Adaptive RTC on openpi, step 2 of 2: the report on pos:x0.1-x0.3 with seed 2 (seed 1 of x suggested the idea), with
# tau from openpi_gate_tune.sh. H = 10, d = 4, s = 5, all runs with the same noise.
#
#   openpi_naive_s5_d4_seed2         naive
#   openpi_rtc_paper_s5_d4_seed2     RTC of the paper on every request
#   openpi_gate_s5_d4_seed2          the divergence gate at the chosen tau
#   openpi_gate_random_s5_d4_seed2   guide at random with the gate's realized share: the main control
#
# Fixed before the run. Main comparison: gate against random at the same guided share, paired Δ of success over
# x0.1-x0.3, 95% bootstrap CI above zero. Secondary (multiple comparisons): gate against naive (success, |Δa| at the
# switch), against RTC on every request. The simulated d does not include the second call of a guided request; the
# chunk log has the calls per request.
#
# Usage (from experiments/libero_pro, after openpi_gate_tune.sh):
#   tmux new -d -s gate_report "bash scripts/stages/openpi_gate_report.sh"
# TAU=... overrides runs/openpi_gate_tau.txt. EPISODES_PER_TASK=20 SEED=2 PORT=8810 by default.
SUITE_TIMEOUT=${SUITE_TIMEOUT:-12h}
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh && source scripts/stages/openpi_lib.sh
EPISODES=${EPISODES_PER_TASK:-20}
STRIDE=${STRIDE:-5}
DELAY=${DELAY:-4}
SEED=${SEED:-2}
PORT=${PORT:-8810}
LEVELS=${LEVELS:-"pos:x0.1 pos:x0.2 pos:x0.3"}
SERVER_LOG=openpi_gate_report
REPORT=$RUNS/openpi_gate_report.md
TAU_FILE=$RUNS/openpi_gate_tau.txt
BASE=(--stride "$STRIDE" --delay "$DELAY")
TAG="s${STRIDE}_d${DELAY}_seed$SEED"

main() {
trap 'trap - INT TERM; stop_servers; kill 0' INT TERM
trap stop_servers EXIT
local tau=${TAU:-}
if [[ -z $tau ]]; then
  [[ -f $TAU_FILE ]] || { echo "no $TAU_FILE: run openpi_gate_tune.sh first, or give TAU"; exit 1; }
  tau=$(awk '{ print $2 }' "$TAU_FILE")
fi
echo "$(date '+%F %T') openpi_gate_report: tau=$tau s=$STRIDE d=$DELAY EPISODES=$EPISODES seed=$SEED levels=$LEVELS"
git log -1 --oneline
start_servers || exit 1
smoke_eval openpi_gate_report "${BASE[@]}" --guidance paper --rtc-gate random --rtc-share 0.5 || exit 1

local naive="openpi_naive_$TAG" rtc="openpi_rtc_paper_$TAG" gate="openpi_gate_$TAG" random="openpi_gate_random_$TAG" share
per_level "$naive" "${BASE[@]}"
per_level "$rtc" "${BASE[@]}" --guidance paper
per_level "$gate" "${BASE[@]}" --guidance paper --rtc-gate divergence --rtc-tau "$tau"
finish
share=$($PY scripts/gate_summary.py "$RUNS/$gate" --print share) || exit 1
echo "$(date '+%F %T') gate guided share $share: random control at that share"
per_level "$random" "${BASE[@]}" --guidance paper --rtc-gate random --rtc-share "$share"
finish
stop_servers

{
  echo "tau = $tau, guided share of the gate = $share"
  echo
  $PY scripts/gate_summary.py "$RUNS/$gate" "$RUNS/$random"
  echo
  $PY scripts/compare.py --ref "$RUNS/$random" "$RUNS/$random" "$RUNS/$gate"
  echo
  $PY scripts/compare.py --ref "$RUNS/$naive" "$RUNS/$naive" "$RUNS/$gate" "$RUNS/$rtc" "$RUNS/$random"
} > "$REPORT"
echo "$(date '+%F %T') done: $REPORT"
}

main 2>&1 | tee -a "$RUNS/openpi_gate_report.log"
