#!/usr/bin/env bash
# Confirmation of the step budget on openpi (stage openpi_budget.sh, test of 07.11.2026): the SAME frozen policies, no
# refit, on new scenes and new noise. x0.1-x0.3 with init states 20..39 (the test used 0..19) and noise seed 3
# (the test used 2). H = 10, d = 2, naive switching.
#
#   openpi_budget_confirm_fixed_s<r>_d2      constant r
#   openpi_budget_confirm_learned_s<r>_d2    runs/openpi_budget_d2/policy_s<r>.json, as in the test
#   openpi_budget_confirm_marginal_s<r>_d2   runs/openpi_budget_d2/marginal_s<r>.json, as in the test
#   openpi_budget_confirm_fixed_s8_d2        the best constant on y
#
# Fixed before the run, as in the test. PRIMARY for each r in REF_STRIDES: learned against constant r at equal rate
# (within 5%), paired Δ of success over x0.1-x0.3, 97.5% bootstrap CI above zero (Bonferroni over the two r).
# Secondary: against marginal at r, against s = 8. Confirmed only if the primary holds for the r it is claimed for.
#
# Usage (from experiments/libero_pro, after openpi_budget.sh):
#   tmux new -d -s budget_confirm "bash scripts/stages/openpi_budget_confirm.sh"
# REF_STRIDES="6 5" FIRST_EPISODE=20 SEED=3 EPISODES_PER_TASK=20 PORT=8830 JOBS=6 by default; re-running skips
# finished suites.
JOBS=${JOBS:-6}
SUITE_TIMEOUT=${SUITE_TIMEOUT:-12h}
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh && source scripts/stages/openpi_lib.sh
EPISODES=${EPISODES_PER_TASK:-20}
DELAY=${DELAY:-2}
REF_STRIDES=${REF_STRIDES:-"6 5"}
BEST=${BEST:-8}
FIRST_EPISODE=${FIRST_EPISODE:-20}
SEED=${SEED:-3}
PORT=${PORT:-8830}
LEVELS=${LEVELS:-"pos:x0.1 pos:x0.2 pos:x0.3"}
SERVER_LOG=openpi_budget_confirm
B=$RUNS/openpi_budget_d$DELAY
FIT="$PY scripts/openpi_budget_fit.py"
NAME=openpi_budget_confirm

run() {  # run NAME ARGS...
  local name=$1; shift
  per_level "$name" --delay "$DELAY" --first-episode "$FIRST_EPISODE" "$@"
}

main() {
trap 'trap - INT TERM; stop_servers; kill 0' INT TERM
trap stop_servers EXIT
echo "$(date '+%F %T') $NAME: refs=$REF_STRIDES best=$BEST init states $FIRST_EPISODE..$(( FIRST_EPISODE + EPISODES - 1 )) seed=$SEED levels=$LEVELS"
git log -1 --oneline
local r f n_refs level
for r in $REF_STRIDES; do
  for f in "$B/policy_s$r.json" "$B/marginal_s$r.json"; do
    [[ -f $f ]] || { echo "no $f: run openpi_budget.sh first"; exit 1; }
    echo "$(date '+%F %T') frozen $f: $(sha256sum < "$f" | cut -c1-16)"
  done
  f=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["policy"])' "$B/policy_s$r.json")
  echo "$(date '+%F %T') frozen $f: $(sha256sum < "$f" | cut -c1-16)"
done
start_servers || exit 1
# the init states past the first 20 must exist: one episode of the last one, as a run would
$OPENPI_PY scripts/eval_openpi.py --suites "${LEVELS%% *}" --task-ids 0 --episodes-per-task 1 --batch-size 1 \
    --first-episode $(( FIRST_EPISODE + EPISODES - 1 )) --seed "$SEED" --port "$PORT" --delay "$DELAY" \
    --budget "$B/policy_s${REF_STRIDES%% *}.json" --out "$RUNS/${NAME}_smoke" > "$RUNS/${NAME}_smoke.log" 2>&1 \
  || { echo "$(date '+%F %T') SMOKE FAILED"; tail -20 "$RUNS/${NAME}_smoke.log"; exit 1; }
echo "$(date '+%F %T') smoke: $(grep -o 'success [0-9.]*%' "$RUNS/${NAME}_smoke.log" | tail -1)"

run "${NAME}_fixed_s${BEST}_d$DELAY" --stride "$BEST"
for r in $REF_STRIDES; do
  run "${NAME}_fixed_s${r}_d$DELAY" --stride "$r"
  run "${NAME}_learned_s${r}_d$DELAY" --budget "$B/policy_s$r.json"
  run "${NAME}_marginal_s${r}_d$DELAY" --budget "$B/marginal_s$r.json"
done
finish
stop_servers

n_refs=$(wc -w <<< "$REF_STRIDES")
level=$(awk -v n="$n_refs" 'BEGIN { print 1 - 0.05 / n }')
for r in $REF_STRIDES; do
  $FIT report --fixed "$RUNS/${NAME}_fixed_s${r}_d$DELAY" --learned "$RUNS/${NAME}_learned_s${r}_d$DELAY" \
    --marginal "$RUNS/${NAME}_marginal_s${r}_d$DELAY" --best "$RUNS/${NAME}_fixed_s${BEST}_d$DELAY" --level "$level" \
    --out "$RUNS/${NAME}_report_s$r.md" || exit 1
  echo "$(date '+%F %T') done: $RUNS/${NAME}_report_s$r.md"
done
}

main 2>&1 | tee -a "$RUNS/$NAME.log"
