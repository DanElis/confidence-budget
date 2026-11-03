#!/usr/bin/env bash
# Adaptive step alone on LIBERO-Pro: naive switching with delay, no RTC guidance, no selection, no mask.
#
# 1. Constant step S with K samples (sample 0 executed, dispersion logged): the control and the calibration data.
# 2. eta of the dispersion rule from that run (calibrate_step.py), so that the request rate matches S. No outcomes used.
# 3. Adaptive step s = clip(h* - d, S_MIN, S_MAX), same K and noise.
# 4. If the achieved mean s rounds to another integer, the constant control at that s as well.
# Main comparison fixed in advance: pa_ada against the constant control with the closest request rate, paired Δ with
# 95% bootstrap CI, requests per episode within 5%. Tuning levels only: confirm a positive result on x and a new seed.
#
# Usage (from experiments/libero_pro):  tmux new -s pure "bash scripts/stages/pure_adaptive.sh"
#   D=8 S=12 S_MIN=$D S_MAX=25 K=4 EPISODES=10 SEED=0 JOBS=4 by default; re-running skips finished suites.
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh
D=${D:-8}
S=${S:-12}
S_MIN=${S_MIN:-$D}
S_MAX=${S_MAX:-25}
K=${K:-4}
SEED=${SEED:-0}
BATCH_K=${BATCH_K:-5}
LEVELS=${LEVELS:-$Y_LEVELS}
export MIN_FREE_MB=${MIN_FREE_K:-20000}   # K samples per environment: batch BATCH_K * K sequences per policy call
KARGS=(--n-samples "$K" --batch-size "$BATCH_K" --shared-prefix --guidance none --delay "$D" --seed "$SEED")
CONST=pa_const_s${S}_d$D
ADA=pa_ada_s${S}_d$D
ADA_ARGS=(--planner adaptive --signal dispersion --s-min "$S_MIN" --s-max "$S_MAX")

smoke() {
  local out=$RUNS/pa_smoke name args
  mkdir -p "$out"
  for name in const ada; do
    [[ -f $out/$name/libero_object_pos_y0.1.json ]] && continue
    [[ $name == const ]] && args=(--stride "$S") || args=("${ADA_ARGS[@]}" --hstar-rule "kind=above,eta=0.05,k=3,min_h=1")
    if ! $PY scripts/eval_libero.py --suites pos:y0.1 --task-ids 0 --episodes-per-task 2 "${KARGS[@]}" --batch-size 2 \
        "${args[@]}" --out "$out/$name" > "$out/$name.log" 2>&1; then
      echo "$(date '+%F %T') SMOKE FAILED: $name, see $out/$name.log"; tail -20 "$out/$name.log"; return 1
    fi
    echo "$(date '+%F %T') smoke $name ok: $(grep -o 'success [0-9.]*%' "$out/$name.log" | tail -1)"
  done
}

main() {
trap 'trap - INT TERM; kill 0' INT TERM
echo "$(date '+%F %T') pure_adaptive: d=$D s=$S [$S_MIN, $S_MAX] K=$K EPISODES=$EPISODES seed=$SEED levels=$LEVELS"
git log -1 --oneline
smoke || exit 1

run_eval "$CONST" "$LEVELS" --stride "$S" "${KARGS[@]}"
finish
local step=$RUNS/$CONST/step.json
[[ -f $step ]] || $PY scripts/calibrate_step.py "$RUNS/$CONST" --target "$S" --delay "$D" --s-min "$S_MIN" --s-max "$S_MAX" \
  --out "$step" || exit 1
run_eval "$ADA" "$LEVELS" "${ADA_ARGS[@]}" --hstar-config "$step" "${KARGS[@]}"
finish

local got runs=("$RUNS/$CONST" "$RUNS/$ADA")
got=$($PY scripts/calibrate_step.py "$RUNS/$ADA" --achieved) || exit 1
echo "$(date '+%F %T') achieved mean s = $got (target $S)"
if (( got != S )); then
  run_eval "pa_const_s${got}_d$D" "$LEVELS" --stride "$got" "${KARGS[@]}"
  finish
  runs+=("$RUNS/pa_const_s${got}_d$D")
fi
local frame
for frame in "cs_naive_s10_d$D" cs_sync_s10; do [[ -d $RUNS/$frame ]] && runs+=("$RUNS/$frame"); done
$PY scripts/compare.py --ref "$RUNS/$CONST" "${runs[@]}" --out "$RUNS/pure_adaptive_report.md"
if (( got != S )); then
  $PY scripts/compare.py --ref "$RUNS/pa_const_s${got}_d$D" "$RUNS/pa_const_s${got}_d$D" "$RUNS/$ADA" \
    >> "$RUNS/pure_adaptive_report.md"
fi
echo "$(date '+%F %T') done: $RUNS/pure_adaptive_report.md"
}

main 2>&1 | tee -a "$RUNS/pure_adaptive.log"
