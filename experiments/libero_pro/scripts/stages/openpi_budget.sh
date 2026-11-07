#!/usr/bin/env bash
# Step budget on openpi's pi05_libero (H = 10), d = 2, naive switching: the option budget of LIBERO-Dynamic (v2) on
# LIBERO-Pro, with s in GRID held for OPTION_TICKS steps (scripts/openpi_budget.py, scripts/openpi_budget_fit.py).
#
#   0 smoke     2 tasks x 2 episodes of explore, a 2-fold fit, policy / marginal / calibrate / retarget on them
#   1 headroom  constant s of GRID and REF_STRIDES on y, seed 1          openpi_budget_const_s<s>_d2
#   2 explore   uniform s per option on y, EXPLORE_SEEDS                 openpi_budget_explore_seed<k>_d2
#   3 fit       V, lambda-returns, Q (CPU), one Q for every reference     runs/openpi_budget_d2/fit/fit.md
#   4 val       per reference step r: the policy at the rate of constant r on y, seed 2; one recalibration if > 5% off
#   5 test      x, seed 2: constant r, learned at r, marginal at r for each r; the best constant s* once
#
# REF_STRIDES empty (the first run, 06.11.2026): the reference is the best constant s* on y, and the headroom gate
# (cross-fitted per-scene best s over s* >= 2 pp) stops the run. It stopped: s* = 8 = H - d, the lowest possible rate,
# where a budget at equal rate can only be s = 8 itself.
#
# REF_STRIDES="6 5" (07.11.2026): a budget spends more calls than s*, and is asked where to spend them. The headroom is
# reported, not a gate. Fixed before the run:
#   fit:   V out-of-fold AUROC >= 0.65 at steps 50..200, and the doubly robust value of state dependence at equal action
#          frequencies, at the rate of the first reference, with CI low > 0; else STOP before validation and test.
#   test:  PRIMARY for each r: learned at the rate of constant r against constant r (rate within 5%), paired Δ of
#          success over x0.1-x0.3, CI above zero at 1 - 0.05 / (number of r) (Bonferroni). Secondary: against
#          marginal at r; against s*: a budget that spends more calls is of use only if it also beats s*.
# A stop writes runs/openpi_budget_d2/stopped.txt; FORCE=1 continues for inspection only.
#
# Usage (from experiments/libero_pro, after setup_openpi.sh):
#   tmux new -d -s budget "REF_STRIDES='6 5' bash scripts/stages/openpi_budget.sh"
# EPISODES_PER_TASK=20 GRID="2 4 6 8" OPTION_TICKS=40 EXPLORE_SEEDS="101 102" PORT=8820 JOBS=6 by default; re-running
# skips finished suites and keeps a finished fit.
JOBS=${JOBS:-6}   # two clients per server: the simulator and the server take turns
SUITE_TIMEOUT=${SUITE_TIMEOUT:-12h}
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh && source scripts/stages/openpi_lib.sh
EPISODES=${EPISODES_PER_TASK:-20}
DELAY=${DELAY:-2}
GRID=${GRID:-"2 4 6 8"}
REF_STRIDES=${REF_STRIDES:-}
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

inverse() { awk -v s="$1" 'BEGIN { print 1 / s }'; }

smoke() {  # every code path once, on 4 episodes; the numbers mean nothing
  local s=$RUNS/openpi_budget_smoke log=$RUNS/openpi_budget_smoke.log
  mkdir -p "$s"
  {
    config "$s/explore.json" explore &&
    $OPENPI_PY scripts/eval_openpi.py --suites pos:y0.1 --task-ids 0 1 --episodes-per-task 2 --batch-size 2 --seed 1 \
        --port "$PORT" --delay "$DELAY" --budget "$s/explore.json" --out "$s/explore" &&
    $FIT fit "$s/explore" --out "$s/fit" --grid $GRID --option-ticks "$OPTION_TICKS" --target-rate 0.25 --folds 2 &&
    $FIT retarget "$s/explore" --policy "$s/fit/policy.pt" --target-rate 0.2 --out "$s/fit/policy_retarget.pt" &&
    config "$s/policy.json" policy --policy "$s/fit/policy.pt" &&
    $OPENPI_PY scripts/eval_openpi.py --suites pos:y0.1 --task-ids 0 --episodes-per-task 2 --batch-size 2 --seed 2 \
        --port "$PORT" --delay "$DELAY" --budget "$s/policy.json" --out "$s/policy" &&
    $FIT calibrate "$s/policy" --policy "$s/fit/policy.pt" --target-rate 0.9 --out "$s/fit/policy_recal.pt" &&
    config "$s/marginal.json" marginal --marginal "$($FIT marginal "$s/policy" --grid $GRID)" &&
    $OPENPI_PY scripts/eval_openpi.py --suites pos:y0.1 --task-ids 0 --episodes-per-task 2 --batch-size 2 --seed 2 \
        --port "$PORT" --delay "$DELAY" --budget "$s/marginal.json" --out "$s/marginal" &&
    $FIT report --fixed "$s/marginal" --learned "$s/policy" --marginal "$s/marginal" --best "$s/explore" --level 0.975 \
        --out "$s/report.md"
  } > "$log" 2>&1 || { echo "$(date '+%F %T') SMOKE FAILED"; tail -30 "$log"; return 1; }
  echo "$(date '+%F %T') smoke: explore $(grep -o 'success [0-9.]*%' "$log" | head -1), rates" \
       "$($FIT rate "$s/explore") / $($FIT rate "$s/policy") / $($FIT rate "$s/marginal")"
}

main() {
trap 'trap - INT TERM; stop_servers; kill 0' INT TERM
trap stop_servers EXIT
echo "$(date '+%F %T') openpi_budget: d=$DELAY grid=$GRID refs='$REF_STRIDES' option=$OPTION_TICKS explore=$EXPLORE_SEEDS EPISODES=$EPISODES JOBS=$JOBS"
git log -1 --oneline
mkdir -p "$B"
rm -f "$B/stopped.txt"
start_servers || exit 1
smoke || exit 1

# 1 headroom: constant s of the grid and of the references, on y
local s grid_runs=()
for s in $(echo $GRID $REF_STRIDES | tr ' ' '\n' | sort -nu); do
  run_set "openpi_budget_const_s${s}_d$DELAY" 1 "$TUNE_LEVELS" --stride "$s"
done
for s in $GRID; do grid_runs+=("$RUNS/openpi_budget_const_s${s}_d$DELAY"); done
finish
local pass s_star rate_star out refs
out=$($FIT headroom "${grid_runs[@]}" --out "$B") || exit 1
read -r pass s_star rate_star <<< "$out"
echo "$(date '+%F %T') headroom: pass=$pass s*=$s_star rate=$rate_star ($B/headroom.md)"
if [[ -z $REF_STRIDES ]]; then
  refs=$s_star
  [[ $pass == 1 ]] || stop "headroom below 2 pp: see $B/headroom.md; no budget episode spent"
else
  refs=$REF_STRIDES
  echo "$(date '+%F %T') references $refs: the headroom is reported, not a gate"
fi
local n_refs level
n_refs=$(wc -w <<< "$refs")
level=$(awk -v n="$n_refs" 'BEGIN { print 1 - 0.05 / n }')

# 2 explore
local k explored=()
config "$B/explore.json" explore
for k in $EXPLORE_SEEDS; do
  explored+=("$RUNS/openpi_budget_explore_seed${k}_d$DELAY")
  run_set "openpi_budget_explore_seed${k}_d$DELAY" "$k" "$TUNE_LEVELS" --budget "$B/explore.json"
done
finish

# 3 fit, at the rate of the first reference (in options, 1/r); the gates are judged there
local first=${refs%% *}
pass=$($FIT fit "${explored[@]}" --out "$B/fit" --grid $GRID --option-ticks "$OPTION_TICKS" \
       --target-rate "$(inverse "$first")") || exit 1
echo "$(date '+%F %T') fit: pass=$pass ($B/fit/fit.md)"
[[ $pass == 1 ]] || stop "offline gates failed: see $B/fit/fit.md; no validation or test episode spent"

# 4 validation per reference: the same Q, its multiplier for the rate of constant r
local r val policy target
for r in $refs; do
  target=$($FIT rate "$RUNS/openpi_budget_const_s${r}_d$DELAY") || exit 1
  policy=$($FIT retarget "${explored[@]}" --policy "$B/fit/policy.pt" --target-rate "$(inverse "$r")" \
           --out "$B/fit/policy_s$r.pt") || exit 1
  config "$B/policy_s$r.json" policy --policy "$policy"
  val=openpi_budget_val_s${r}_d$DELAY
  run_set "$val" 2 "$TUNE_LEVELS" --budget "$B/policy_s$r.json"
done
finish
for r in $refs; do
  target=$($FIT rate "$RUNS/openpi_budget_const_s${r}_d$DELAY") || exit 1
  val=openpi_budget_val_s${r}_d$DELAY
  policy=$($FIT calibrate "$RUNS/$val" --policy "$B/fit/policy_s$r.pt" --target-rate "$target" \
           --out "$B/fit/policy_s${r}_recal.pt") || exit 1
  if [[ $policy != "$B/fit/policy_s$r.pt" ]]; then
    echo "$(date '+%F %T') r=$r: validation rate $($FIT rate "$RUNS/$val") vs $target: recalibrated"
    config "$B/policy_s$r.json" policy --policy "$policy"
    run_set "openpi_budget_val_recal_s${r}_d$DELAY" 2 "$TUNE_LEVELS" --budget "$B/policy_s$r.json"
  fi
done
finish
for r in $refs; do
  val=openpi_budget_val_s${r}_d$DELAY
  [[ -d $RUNS/openpi_budget_val_recal_s${r}_d$DELAY ]] && val=openpi_budget_val_recal_s${r}_d$DELAY
  echo "$(date '+%F %T') r=$r validation: rate $($FIT rate "$RUNS/$val") vs $($FIT rate "$RUNS/openpi_budget_const_s${r}_d$DELAY")"
  config "$B/marginal_s$r.json" marginal --marginal "$($FIT marginal "$RUNS/$val" --grid $GRID)"
done

# 5 test
run_set "openpi_budget_test_fixed_s${s_star}_d$DELAY" 2 "$TEST_LEVELS" --stride "$s_star"
for r in $refs; do
  run_set "openpi_budget_test_fixed_s${r}_d$DELAY" 2 "$TEST_LEVELS" --stride "$r"
  run_set "openpi_budget_test_learned_s${r}_d$DELAY" 2 "$TEST_LEVELS" --budget "$B/policy_s$r.json"
  run_set "openpi_budget_test_marginal_s${r}_d$DELAY" 2 "$TEST_LEVELS" --budget "$B/marginal_s$r.json"
done
finish
stop_servers
for r in $refs; do
  $FIT report --fixed "$RUNS/openpi_budget_test_fixed_s${r}_d$DELAY" --learned "$RUNS/openpi_budget_test_learned_s${r}_d$DELAY" \
    --marginal "$RUNS/openpi_budget_test_marginal_s${r}_d$DELAY" --best "$RUNS/openpi_budget_test_fixed_s${s_star}_d$DELAY" \
    --headroom "$B/headroom.json" --level "$level" --out "$RUNS/openpi_budget_report_s$r.md" || exit 1
  echo "$(date '+%F %T') done: $RUNS/openpi_budget_report_s$r.md"
done
}

main 2>&1 | tee -a "$RUNS/openpi_budget.log"
