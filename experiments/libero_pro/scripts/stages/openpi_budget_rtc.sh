#!/usr/bin/env bash
# Learned budget of (s, r) on openpi's pi05_libero, H = 10, d = 2: the step s and the RTC window r of the paper
# (Jacobian, beta = 5), r = -1 naive, held for an option of OPTION_TICKS steps (scripts/openpi_budget.py). The pipeline
# of openpi_budget.sh (REF_STRIDES, 07.11.2026); run after openpi_rtc_d2.sh, whose constants.json gives the constant
# reference pair for each reference step.
#
#   0 smoke     2 tasks x 2 episodes of explore over the pairs, a 2-fold fit, policy / marginal on them
#   1 explore   uniform pair per option on y, EXPLORE_SEEDS                      openpi_budget_rtc_explore_seed<k>_d2
#   2 fit       V, lambda-returns, Q over the pairs; data: these runs and the naive explore of openpi_budget.sh, whose
#               decisions are the pairs (s, naive) with their own propensities
#   3 val       per reference step q: the multiplier for the rate of constant q on y, seed 2; one recalibration
#   4 test      x, seed 2, init 0..19: learned, marginal, the constant reference pair at q
#   5 confirm   x, seed 3, init 20..39: the same, frozen
#
# Fixed before the run, for each q in REF_STRIDES (CI at 1 - 0.05 / their number, Bonferroni), in test and in confirm:
#   PRIMARY: learned (s, r) against the best constant pair with s = q on y (openpi_rtc_d2.sh; naive if no RTC pair beats
#            it), equal rate within 5%, paired Δ of success over x0.1-x0.3, CI above zero.
#   Secondary: against marginal at q; against learned s only (naive, openpi_budget.sh, same episodes): does r add to the
#            choice of s?; against s = 8.
# The offline gates are reported in fit.md; the run goes on whatever they say (as the run of 07.11.2026, FORCE=1).
#
# Usage (from experiments/libero_pro):
#   tmux new -d -s budget_rtc "bash scripts/stages/openpi_rtc_d2.sh && bash scripts/stages/openpi_budget_rtc.sh"
# PAIR_GRID="2:-1 2:2 2:4 4:-1 4:2 4:4 6:-1 6:2 8:-1" REF_STRIDES="6 5" OPTION_TICKS=40 EXPLORE_SEEDS="201 202"
# PORT=8850 JOBS=6 by default; re-running skips finished suites and keeps a finished fit.
JOBS=${JOBS:-6}
SUITE_TIMEOUT=${SUITE_TIMEOUT:-12h}
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh && source scripts/stages/openpi_lib.sh
EPISODES=${EPISODES_PER_TASK:-20}
DELAY=${DELAY:-2}
PAIR_GRID=${PAIR_GRID:-"2:-1 2:2 2:4 4:-1 4:2 4:4 6:-1 6:2 8:-1"}
REF_STRIDES=${REF_STRIDES:-"6 5"}
BEST=${BEST:-8}
OPTION_TICKS=${OPTION_TICKS:-40}
EXPLORE_SEEDS=${EXPLORE_SEEDS:-"201 202"}
NAIVE_EXPLORE_SEEDS=${NAIVE_EXPLORE_SEEDS:-"101 102"}
MAX_STEPS=${MAX_STEPS:-280}
PORT=${PORT:-8850}
TUNE_LEVELS="pos:y0.1 pos:y0.2 pos:y0.3"
TEST_LEVELS="pos:x0.1 pos:x0.2 pos:x0.3"
LEVELS=$TUNE_LEVELS
SEED=1
SERVER_LOG=openpi_budget_rtc
B=$RUNS/openpi_budget_rtc_d$DELAY
FIT="$PY scripts/openpi_budget_fit.py"
N=openpi_budget_rtc

run_set() {  # run_set NAME SEED "LEVELS" ARGS...
  local name=$1; SEED=$2; LEVELS=$3; shift 3
  per_level "$name" --delay "$DELAY" "$@"
}

config() {  # config FILE MODE [ARGS...]
  local file=$1 mode=$2; shift 2
  $FIT config --out "$file" --mode "$mode" --grid $PAIR_GRID --option-ticks "$OPTION_TICKS" --max-steps "$MAX_STEPS" "$@"
}

inverse() { awk -v s="$1" 'BEGIN { print 1 / s }'; }

reference() {  # reference Q -> "s r run": the best constant pair with s = Q on y
  python3 -c 'import json, sys; b = json.load(open(sys.argv[1]))["best"][sys.argv[2]]; print(b["s"], b["r"], b["run"])' \
    "$B/constants.json" "$1"
}

fixed_args() {  # fixed_args S R -> the eval arguments of the constant pair
  if (( $2 < 0 )); then echo "--stride $1"; else echo "--stride $1 --guidance paper --mask-end $(( DELAY + $2 ))"; fi
}

smoke() {
  local s=$RUNS/${N}_smoke log=$RUNS/${N}_smoke.log
  mkdir -p "$s"
  {
    config "$s/explore.json" explore &&
    $OPENPI_PY scripts/eval_openpi.py --suites pos:y0.1 --task-ids 0 1 --episodes-per-task 2 --batch-size 2 --seed 1 \
        --port "$PORT" --delay "$DELAY" --budget "$s/explore.json" --out "$s/explore" &&
    $FIT fit "$s/explore" --out "$s/fit" --grid $PAIR_GRID --option-ticks "$OPTION_TICKS" --target-rate 0.25 --folds 2 &&
    config "$s/policy.json" policy --policy "$s/fit/policy.pt" &&
    $OPENPI_PY scripts/eval_openpi.py --suites pos:y0.1 --task-ids 0 --episodes-per-task 2 --batch-size 2 --seed 2 \
        --port "$PORT" --delay "$DELAY" --budget "$s/policy.json" --out "$s/policy" &&
    config "$s/marginal.json" marginal --marginal "$($FIT marginal "$s/policy" --grid $PAIR_GRID)" &&
    $OPENPI_PY scripts/eval_openpi.py --suites pos:y0.1 --task-ids 0 --episodes-per-task 2 --batch-size 2 --seed 2 \
        --port "$PORT" --delay "$DELAY" --budget "$s/marginal.json" --out "$s/marginal"
  } > "$log" 2>&1 || { echo "$(date '+%F %T') SMOKE FAILED"; tail -30 "$log"; return 1; }
  echo "$(date '+%F %T') smoke: $(grep -o 'RTC check.*' "$log" | head -1); guided share of explore" \
       "$(python3 -c 'import json, glob, sys; c = [x for f in glob.glob(sys.argv[1] + "/*_chunks.json") for x in json.load(open(f))]; print(round(sum(x["budget_guided"] for x in c) / len(c), 3))' "$s/explore")"
}

stage() {  # stage TAG SEED FIRST_EPISODE: learned, marginal and the constant reference for every q, on x
  local tag=$1 seed=$2 first=$3 q rs rr rrun fixed
  local extra=(); (( first > 0 )) && extra=(--first-episode "$first")
  for q in $REF_STRIDES; do
    read -r rs rr rrun <<< "$(reference "$q")"
    if (( rr < 0 )); then  # naive is the reference: the runs of openpi_budget.sh on the same episodes
      fixed=openpi_budget_${tag}_fixed_s${q}_d$DELAY
    else
      fixed=${N}_${tag}_fixed_s${q}_r${rr}_d$DELAY
    fi
    # shellcheck disable=SC2046
    run_set "$fixed" "$seed" "$TEST_LEVELS" ${extra[@]+"${extra[@]}"} $(fixed_args "$q" "$rr")
    run_set "${N}_${tag}_learned_s${q}_d$DELAY" "$seed" "$TEST_LEVELS" ${extra[@]+"${extra[@]}"} --budget "$B/policy_s$q.json"
    run_set "${N}_${tag}_marginal_s${q}_d$DELAY" "$seed" "$TEST_LEVELS" ${extra[@]+"${extra[@]}"} --budget "$B/marginal_s$q.json"
  done
  finish
  for q in $REF_STRIDES; do
    read -r rs rr rrun <<< "$(reference "$q")"
    fixed=openpi_budget_${tag}_fixed_s${q}_d$DELAY
    (( rr >= 0 )) && fixed=${N}_${tag}_fixed_s${q}_r${rr}_d$DELAY
    $FIT report --fixed "$RUNS/$fixed" --learned "$RUNS/${N}_${tag}_learned_s${q}_d$DELAY" \
      --marginal "$RUNS/${N}_${tag}_marginal_s${q}_d$DELAY" --best "$RUNS/openpi_budget_${tag}_fixed_s${BEST}_d$DELAY" \
      --prior "$RUNS/openpi_budget_${tag}_learned_s${q}_d$DELAY" --level "$LEVEL" --out "$RUNS/${N}_${tag}_report_s$q.md" \
      || exit 1
    echo "$(date '+%F %T') done: $RUNS/${N}_${tag}_report_s$q.md"
  done
}

main() {
trap 'trap - INT TERM; stop_servers; kill 0' INT TERM
trap stop_servers EXIT
echo "$(date '+%F %T') $N: d=$DELAY pairs=$PAIR_GRID refs=$REF_STRIDES option=$OPTION_TICKS explore=$EXPLORE_SEEDS JOBS=$JOBS"
git log -1 --oneline
[[ -f $B/constants.json ]] || { echo "no $B/constants.json: run openpi_rtc_d2.sh first"; exit 1; }
local q k tag f
for tag in test confirm; do  # the runs of openpi_budget.sh / openpi_budget_confirm.sh on the same episodes
  for f in "openpi_budget_${tag}_fixed_s${BEST}_d$DELAY" $(for q in $REF_STRIDES; do echo "openpi_budget_${tag}_learned_s${q}_d$DELAY"; done); do
    [[ -f $RUNS/$f/libero_object_pos_x0.3.json ]] || { echo "no $RUNS/$f: run openpi_budget.sh and openpi_budget_confirm.sh first"; exit 1; }
  done
done
for q in $REF_STRIDES; do echo "$(date '+%F %T') q=$q: constant reference (s r run) $(reference "$q")"; done
LEVEL=$(awk -v n="$(wc -w <<< "$REF_STRIDES")" 'BEGIN { print 1 - 0.05 / n }')
start_servers || exit 1
smoke || exit 1

# 1 explore over the pairs
local explored=()
for k in $NAIVE_EXPLORE_SEEDS; do explored+=("$RUNS/openpi_budget_explore_seed${k}_d$DELAY"); done
config "$B/explore.json" explore
for k in $EXPLORE_SEEDS; do
  explored+=("$RUNS/${N}_explore_seed${k}_d$DELAY")
  run_set "${N}_explore_seed${k}_d$DELAY" "$k" "$TUNE_LEVELS" --budget "$B/explore.json"
done
finish

# 2 fit, gates judged at the rate of the first reference
local first=${REF_STRIDES%% *} pass
pass=$($FIT fit "${explored[@]}" --out "$B/fit" --grid $PAIR_GRID --option-ticks "$OPTION_TICKS" \
       --target-rate "$(inverse "$first")") || exit 1
echo "$(date '+%F %T') fit: pass=$pass ($B/fit/fit.md); the run goes on either way"

# 3 validation per reference step
local rs rr rrun target policy val
for q in $REF_STRIDES; do
  policy=$($FIT retarget "${explored[@]}" --policy "$B/fit/policy.pt" --target-rate "$(inverse "$q")" \
           --out "$B/fit/policy_s$q.pt") || exit 1
  config "$B/policy_s$q.json" policy --policy "$policy"
  run_set "${N}_val_s${q}_d$DELAY" 2 "$TUNE_LEVELS" --budget "$B/policy_s$q.json"
done
finish
for q in $REF_STRIDES; do
  read -r rs rr rrun <<< "$(reference "$q")"
  target=$($FIT rate "$rrun") || exit 1
  val=${N}_val_s${q}_d$DELAY
  policy=$($FIT calibrate "$RUNS/$val" --policy "$B/fit/policy_s$q.pt" --target-rate "$target" \
           --out "$B/fit/policy_s${q}_recal.pt") || exit 1
  if [[ $policy != "$B/fit/policy_s$q.pt" ]]; then
    echo "$(date '+%F %T') q=$q: validation rate $($FIT rate "$RUNS/$val") vs $target: recalibrated"
    config "$B/policy_s$q.json" policy --policy "$policy"
    run_set "${N}_val_recal_s${q}_d$DELAY" 2 "$TUNE_LEVELS" --budget "$B/policy_s$q.json"
  fi
done
finish
for q in $REF_STRIDES; do
  read -r rs rr rrun <<< "$(reference "$q")"
  val=${N}_val_s${q}_d$DELAY
  [[ -d $RUNS/${N}_val_recal_s${q}_d$DELAY ]] && val=${N}_val_recal_s${q}_d$DELAY
  echo "$(date '+%F %T') q=$q validation: rate $($FIT rate "$RUNS/$val") vs $($FIT rate "$rrun") of the reference"
  config "$B/marginal_s$q.json" marginal --marginal "$($FIT marginal "$RUNS/$val" --grid $PAIR_GRID)"
done

# 4 test and 5 confirm
stage test 2 0
stage confirm 3 20
stop_servers
}

main 2>&1 | tee -a "$RUNS/$N.log"
