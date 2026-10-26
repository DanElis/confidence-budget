#!/usr/bin/env bash
# Confirmation of the night_coherence winner on the report levels: coherent sample + confidence-weighted RTC mask.
#
# Chosen on the tuning levels (y, seed 0, runs/cs_cohconf1_d8): sigma = 1, conf_ref = 0.03716 (median dispersion of
# the night's smoke), K = 4, s = 10, mask to d + 5, d = 8. Nothing is tuned here.
# Report levels (libero_object, x0.1-x0.3), new noise seed 1, EPISODES per task, one call per inference with a shared
# prefix (--shared-prefix --sample-call joint: K unguided + K guided rows, latency ~1.03x of a plain call per robot).
#
# Main comparison fixed in advance: cf_cohconf1 vs cf_rtck (RTC with the same K samples, sample 0), paired Δ with
# 95% bootstrap CI. Frame: cf_rtc (K = 1), cf_naive, cf_sync.
#
# Usage (from experiments/libero_pro):  tmux new -s confirm "bash scripts/stages/confirm_conf.sh"
#   EPISODES=20 JOBS=4 by default (~10 h); re-running skips finished suites; report at any time:
#   uv run python scripts/compare.py --ref runs/cf_rtck_s10_d8 runs/cf_* --out runs/confirm_report.md
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh
EPISODES=${EPISODES:-20}
SEED=${SEED:-1}
D=${D:-8}
K=${K:-4}
SIGMA=${SIGMA:-1}
CONF_REF=${CONF_REF:-0.03716}
BATCH_K=${BATCH_K:-5}
MIN_FREE_K=${MIN_FREE_K:-20000}
LEVELS=$X_LEVELS
S=10
M=$((D + 5))
KARGS=(--n-samples "$K" --batch-size "$BATCH_K" --shared-prefix --sample-call joint)

main() {
trap 'trap - INT TERM; kill 0' INT TERM
echo "$(date '+%F %T') confirm_conf: d=$D K=$K sigma=$SIGMA conf_ref=$CONF_REF EPISODES=$EPISODES seed=$SEED levels=$LEVELS"
git log -1 --oneline

# smoke of the exact code path (1 task, 2 episodes); stop if it fails
SMOKE=$RUNS/cf_smoke
if [[ ! -f $SMOKE/libero_object.json ]]; then
  echo "$(date '+%F %T') smoke"
  if ! $PY scripts/eval_libero.py --suites libero_object --task-ids 0 --episodes-per-task 2 --batch-size 2 \
      --stride $S --mask-end $M --delay "$D" --guidance lerobot --seed "$SEED" --n-samples "$K" --shared-prefix \
      --sample-call joint --select coherent --mask-mode rtc_x_conf --conf-sigma "$SIGMA" --conf-ref "$CONF_REF" \
      --out "$SMOKE" > "$SMOKE.log" 2>&1; then
    echo "$(date '+%F %T') SMOKE FAILED, see $SMOKE.log"; tail -20 "$SMOKE.log"; exit 1
  fi
  echo "$(date '+%F %T') smoke ok: $(grep -o 'success [0-9.]*%' "$SMOKE.log" | tail -1)"
fi

# main comparison first, then the frame
MIN_FREE_MB=$MIN_FREE_K run_eval "cf_rtck_s${S}_d$D" "$LEVELS" --stride $S --mask-end $M --delay "$D" \
  --guidance lerobot --seed "$SEED" "${KARGS[@]}"
MIN_FREE_MB=$MIN_FREE_K run_eval "cf_cohconf1_s${S}_d$D" "$LEVELS" --stride $S --mask-end $M --delay "$D" \
  --guidance lerobot --seed "$SEED" "${KARGS[@]}" --select coherent --mask-mode rtc_x_conf \
  --conf-sigma "$SIGMA" --conf-ref "$CONF_REF"
run_eval "cf_rtc_s${S}_m${M}_d$D" "$LEVELS" --stride $S --mask-end $M --delay "$D" --guidance lerobot --seed "$SEED"
run_eval "cf_naive_s${S}_d$D" "$LEVELS" --stride $S --delay "$D" --guidance none --seed "$SEED"
run_eval cf_sync_s10 "$LEVELS" --stride 10 --delay 0 --guidance none --seed "$SEED"
finish

$PY scripts/compare.py --ref "$RUNS/cf_rtck_s${S}_d$D" "$RUNS/cf_cohconf1_s${S}_d$D" "$RUNS/cf_rtc_s${S}_m${M}_d$D" \
  "$RUNS/cf_naive_s${S}_d$D" "$RUNS/cf_sync_s10" --out "$RUNS/confirm_report.md"
}

main 2>&1 | tee -a "$RUNS/confirm_conf.log"
