#!/usr/bin/env bash
# AAC [13] as in the authors' code (aac.py) on LIBERO-Pro, without RTC: nothing to tune, so on the report levels.
#
#   d = 0   AAC proper: synchronous. Paper, PI0.5: x0.2 57.4, x0.3 35.3 against fixed 5: 53.2, 29.9.
#   d = 8   AAC with delay and naive switching, s = max(h* - d, d): our extension, the paper has no latency.
# For each d: AAC with K = 20, then the constant step of the same request rate with the same K and sample 0
# (step_controls.py, the two neighbouring integers if none is within 5%).
# Main comparison fixed in advance, at each d: AAC against that control, paired Δ with 95% bootstrap CI; two d: Bonferroni.
# Frame: fixed s = 5, K = 1, d = 0, the paper's baseline. Our H = 50; the paper does not state it for PI0.5.
#
# Usage (from experiments/libero_pro):  tmux new -s aac "bash scripts/stages/pure_aac.sh"
#   DELAYS="0 8" K=20 AAC_EPISODES=20 SEED=1 BATCH_K=5 JOBS=4 by default; re-running skips finished suites.
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh
EPISODES=${AAC_EPISODES:-20}  # lib.sh already sets EPISODES (10)
DELAYS=${DELAYS:-"0 8"}
K=${K:-20}
SEED=${SEED:-1}
BATCH_K=${BATCH_K:-5}
LEVELS=${LEVELS:-"pos:x0.1,pos:x0.2,pos:x0.3"}
export MIN_FREE_MB=${MIN_FREE_K:-24000}   # K samples per environment: batch BATCH_K * K sequences per policy call
KARGS=(--n-samples "$K" --batch-size "$BATCH_K" --shared-prefix --guidance none --seed "$SEED")
AAC=(--planner adaptive --signal aac)
REPORT=$RUNS/pure_aac_report.md

smoke() {
  local out=$RUNS/aac_smoke d
  mkdir -p "$out"
  for d in $DELAYS; do
    [[ -f $out/d$d/libero_object_pos_x0.2.json ]] && continue
    if ! $PY scripts/eval_libero.py --suites pos:x0.2 --task-ids 0 --episodes-per-task 2 "${KARGS[@]}" --batch-size 2 \
        "${AAC[@]}" --delay "$d" --out "$out/d$d" > "$out/d$d.log" 2>&1; then
      echo "$(date '+%F %T') SMOKE FAILED: d=$d, see $out/d$d.log"; tail -20 "$out/d$d.log"; return 1
    fi
    echo "$(date '+%F %T') smoke d=$d ok: $(grep -o 'success [0-9.]*%' "$out/d$d.log" | tail -1)"
  done
}

main() {
trap 'trap - INT TERM; kill 0' INT TERM
echo "$(date '+%F %T') pure_aac: delays=$DELAYS K=$K EPISODES=$EPISODES seed=$SEED levels=$LEVELS"
git log -1 --oneline
smoke || exit 1

MIN_FREE_MB=18000 run_eval aac_base_s5 "$LEVELS" --stride 5 --delay 0 --guidance none --seed "$SEED"
local d
for d in $DELAYS; do run_eval "aac_d$d" "$LEVELS" "${AAC[@]}" --delay "$d" "${KARGS[@]}"; done
finish

: > "$REPORT"
local s steps
for d in $DELAYS; do
  steps=$($PY scripts/step_controls.py "$RUNS/aac_d$d") || exit 1
  echo "$(date '+%F %T') d=$d: constant control s = $steps"
  for s in $steps; do run_eval "aac_const_s${s}_d$d" "$LEVELS" --stride "$s" --delay "$d" "${KARGS[@]}"; done
done
finish
for d in $DELAYS; do
  steps=$($PY scripts/step_controls.py "$RUNS/aac_d$d") || exit 1
  for s in $steps; do
    local runs=("$RUNS/aac_const_s${s}_d$d" "$RUNS/aac_d$d")
    (( d == 0 )) && runs+=("$RUNS/aac_base_s5")
    $PY scripts/compare.py --ref "$RUNS/aac_const_s${s}_d$d" "${runs[@]}" >> "$REPORT"
  done
done
echo "$(date '+%F %T') done: $REPORT"
}

main 2>&1 | tee -a "$RUNS/pure_aac.log"
