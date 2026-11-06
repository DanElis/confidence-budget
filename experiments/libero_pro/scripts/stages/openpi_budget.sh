#!/usr/bin/env bash
# Step budget on openpi's pi05_libero (H = 10), d = 2, naive switching: the option budget of LIBERO-Dynamic (v2) on
# LIBERO-Pro, with s in GRID held for OPTION_TICKS steps (scripts/openpi_budget.py, scripts/openpi_budget_fit.py).
#
#   0 smoke     2 tasks x 2 episodes of explore, a 2-fold fit, policy / marginal / calibrate on them
#   1 headroom  constant s of GRID on y, seed 1            openpi_budget_const_s<s>_d2
#   2 explore   uniform s per option on y, EXPLORE_SEEDS   openpi_budget_explore_seed<k>_d2
#   3 fit       V, lambda-returns, Q (CPU)                 runs/openpi_budget_d2/fit/fit.md
#   4 val       the policy on y, seed 2; one rate recalibration if off by > 5%
#   5 test      x, seed 2: fixed s*, learned, marginal     runs/openpi_budget_report.md
#
# Fixed before the run:
#   headroom: the cross-fitted per-scene best s over the best constant s (init states split by parity) >= 2 pp,
#             else STOP before any budget episode. s* = the best constant on y: the reference and the target rate.
#   fit:      V out-of-fold AUROC >= 0.65 at steps 50..200, and the doubly robust value of state dependence at equal
#             action frequencies with CI low > 0; else STOP before validation and test.
#   test:     PRIMARY learned against constant s* at equal rate (within 5%), paired Δ of success over x0.1-x0.3,
#             95% bootstrap CI above zero. Secondary: learned against marginal (same frequencies, state-independent).
# A stop writes runs/openpi_budget_d2/stopped.txt; FORCE=1 continues for inspection only.
#
# Usage (from experiments/libero_pro, after setup_openpi.sh):
#   tmux new -d -s budget "bash scripts/stages/openpi_budget.sh"
# EPISODES_PER_TASK=20 GRID="2 4 6 8" OPTION_TICKS=40 EXPLORE_SEEDS="101 102" PORT=8820 JOBS=6 by default; re-running
# skips finished suites and keeps a finished fit.
JOBS=${JOBS:-6}   # two clients per server: the simulator and the server take turns
SUITE_TIMEOUT=${SUITE_TIMEOUT:-12h}
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh && source scripts/stages/openpi_lib.sh
EPISODES=${EPISODES_PER_TASK:-20}
DELAY=${DELAY:-2}
GRID=${GRID:-"2 4 6 8"}
OPTION_TICKS=${OPTION_TICKS:-40}
EXPLORE_SEEDS=${EXPLORE_SEEDS:-"101 102"}
MAX_STEPS=${MAX_STEPS:-280}
PORT=${PORT:-8820}
FORCE=${FORCE:-0}
TUNE_LEVELS="pos:y0.1 pos:y0.2 pos:y0.3"
TEST_LEVELS="pos:x0.1 pos:x0.2 pos:x0.3"
LEVELS=$TUNE_LEVELS   # one server per level slot: the same three servers serve y and x
SEED=1
SERVER_LOG=openpi_budget
B=$RUNS/openpi_budget_d$DELAY
FIT="$PY scripts/openpi_budget_fit.py"
REPORT=$RUNS/openpi_budget_report.md

run_set() {  # run_set NAME SEED "LEVELS" ARGS...
  local name=$1; SEED=$2; LEVELS=$3; shift 3
  per_level "$name" --delay "$DELAY" "$@"
}

config() {  # config FILE MODE [ARGS...]
  local file=$1 mode=$2; shift 2
  $FIT config --out "$file" --mode "$mode" --grid $GRID --option-ticks "$OPTION_TICKS" --max-steps "$MAX_STEPS" "$@"
}

stop() {
  echo "$1" > "$B/stopped.txt"
  echo "$(date '+%F %T') STOP: $1"
  [[ $FORCE == 1 ]] && { echo "$(date '+%F %T') FORCE=1: continuing for inspection"; return 0; }
  exit 0
}

smoke() {  # every code path once, on 4 episodes; the numbers mean nothing
  local s=$RUNS/openpi_budget_smoke log=$RUNS/openpi_budget_smoke.log
  mkdir -p "$s"
  {
    config "$s/explore.json" explore &&
    $OPENPI_PY scripts/eval_openpi.py --suites pos:y0.1 --task-ids 0 1 --episodes-per-task 2 --batch-size 2 --seed 1 \
        --port "$PORT" --delay "$DELAY" --budget "$s/explore.json" --out "$s/explore" &&
    $FIT fit "$s/explore" --out "$s/fit" --grid $GRID --option-ticks "$OPTION_TICKS" --target-rate 0.25 --folds 2 &&
    config "$s/policy.json" policy --policy "$s/fit/policy.pt" &&
    $OPENPI_PY scripts/eval_openpi.py --suites pos:y0.1 --task-ids 0 --episodes-per-task 2 --batch-size 2 --seed 2 \
        --port "$PORT" --delay "$DELAY" --budget "$s/policy.json" --out "$s/policy" &&
    $FIT calibrate "$s/policy" --policy "$s/fit/policy.pt" --target-rate 0.9 --out "$s/fit/policy_recal.pt" &&
    config "$s/marginal.json" marginal --marginal "$($FIT marginal "$s/policy" --grid $GRID)" &&
    $OPENPI_PY scripts/eval_openpi.py --suites pos:y0.1 --task-ids 0 --episodes-per-task 2 --batch-size 2 --seed 2 \
        --port "$PORT" --delay "$DELAY" --budget "$s/marginal.json" --out "$s/marginal"
  } > "$log" 2>&1 || { echo "$(date '+%F %T') SMOKE FAILED"; tail -30 "$log"; return 1; }
  echo "$(date '+%F %T') smoke: explore $(grep -o 'success [0-9.]*%' "$log" | head -1), rates" \
       "$($FIT rate "$s/explore") / $($FIT rate "$s/policy") / $($FIT rate "$s/marginal")"
}

main() {
trap 'trap - INT TERM; stop_servers; kill 0' INT TERM
trap stop_servers EXIT
echo "$(date '+%F %T') openpi_budget: d=$DELAY grid=$GRID option=$OPTION_TICKS explore=$EXPLORE_SEEDS EPISODES=$EPISODES JOBS=$JOBS"
git log -1 --oneline
mkdir -p "$B"
rm -f "$B/stopped.txt"
start_servers || exit 1
smoke || exit 1

# 1 headroom
local s const=()
for s in $GRID; do
  const+=("$RUNS/openpi_budget_const_s${s}_d$DELAY")
  run_set "openpi_budget_const_s${s}_d$DELAY" 1 "$TUNE_LEVELS" --stride "$s"
done
finish
local pass s_star rate_star out
out=$($FIT headroom "${const[@]}" --out "$B") || exit 1
read -r pass s_star rate_star <<< "$out"
echo "$(date '+%F %T') headroom: pass=$pass s*=$s_star rate=$rate_star ($B/headroom.md)"
[[ $pass == 1 ]] || stop "headroom below 2 pp: see $B/headroom.md; no budget episode spent"

# 2 explore
local k explored=()
config "$B/explore.json" explore
for k in $EXPLORE_SEEDS; do
  explored+=("$RUNS/openpi_budget_explore_seed${k}_d$DELAY")
  run_set "openpi_budget_explore_seed${k}_d$DELAY" "$k" "$TUNE_LEVELS" --budget "$B/explore.json"
done
finish

# 3 fit: the offline target is 1/s* in options; validation matches the realized rate of s*
pass=$($FIT fit "${explored[@]}" --out "$B/fit" --grid $GRID --option-ticks "$OPTION_TICKS" \
       --target-rate "$(awk -v s="$s_star" 'BEGIN { print 1 / s }')") || exit 1
echo "$(date '+%F %T') fit: pass=$pass ($B/fit/fit.md)"
[[ $pass == 1 ]] || stop "offline gates failed: see $B/fit/fit.md; no validation or test episode spent"

# 4 validation
local val=openpi_budget_val_d$DELAY policy
config "$B/policy.json" policy --policy "$B/fit/policy.pt"
run_set "$val" 2 "$TUNE_LEVELS" --budget "$B/policy.json"
finish
policy=$($FIT calibrate "$RUNS/$val" --policy "$B/fit/policy.pt" --target-rate "$rate_star" \
         --out "$B/fit/policy_recal.pt") || exit 1
if [[ $policy != "$B/fit/policy.pt" ]]; then
  echo "$(date '+%F %T') validation rate $($FIT rate "$RUNS/$val") vs $rate_star: recalibrated"
  config "$B/policy.json" policy --policy "$policy"
  val=openpi_budget_val_recal_d$DELAY
  run_set "$val" 2 "$TUNE_LEVELS" --budget "$B/policy.json"
  finish
fi
echo "$(date '+%F %T') validation: rate $($FIT rate "$RUNS/$val") vs $rate_star of s*"
config "$B/marginal.json" marginal --marginal "$($FIT marginal "$RUNS/$val" --grid $GRID)"

# 5 test
local fixed=openpi_budget_test_fixed_s${s_star}_d$DELAY learned=openpi_budget_test_learned_d$DELAY
local control=openpi_budget_test_marginal_d$DELAY
run_set "$fixed" 2 "$TEST_LEVELS" --stride "$s_star"
run_set "$learned" 2 "$TEST_LEVELS" --budget "$B/policy.json"
run_set "$control" 2 "$TEST_LEVELS" --budget "$B/marginal.json"
finish
stop_servers
$FIT report --fixed "$RUNS/$fixed" --learned "$RUNS/$learned" --marginal "$RUNS/$control" --headroom "$B/headroom.json" \
  --out "$REPORT" || exit 1
echo "$(date '+%F %T') done: $REPORT"
}

main 2>&1 | tee -a "$RUNS/openpi_budget.log"
