#!/usr/bin/env bash
# Adaptive step alone on LIBERO-Dynamic: the runner's rtc_queue (requests by chunk length, chunk replaced after d ticks)
# with an unguided server, so naive switching; no coherent sample, no mask. Same schedule machinery as the earlier
# adaptive runs (ada_const_s7_b2_w6), so comparing with them shows what RTC adds.
#
# 1. Constant step S with K samples (sample 0 executed, dispersion logged): the control and the calibration data.
# 2. eta of the dispersion rule from that run (libero_pro/scripts/calibrate_step.py), so that the request rate matches
#    S. No outcomes used.
# 3. Adaptive step s = clip(h* - d, S_MIN, S_MAX), same K.
# 4. If the achieved mean s rounds to another integer, the constant control at that s as well.
# Main comparison fixed in advance: pure_ada against the constant control with the closest request rate, v = 0.06,
# 600 pairs, CI of Δ above zero, requests per episode within 5%. From experiments/libero_dynamic:
#
#   LIMIT=2 RUNS=runs/pure_smoke bash run_pure_adaptive.sh        # smoke
#   tmux new -d -s pure "bash run_pure_adaptive.sh 2>&1 | tee -a runs/pure_adaptive.log"
#
# Environment: as run_dynamic.sh (SERVERS, WORKERS, PORT, LIMIT, RUNS) plus SPEEDS (0.06), LATENCY (0.175: d = 4),
# D (4, must match LATENCY), S (7), S_MIN (D), S_MAX (20), N_SAMPLES (4), FRAME (earlier runs for the report, if present).
# Finished speeds are skipped by run_dynamic.sh, so the script can be restarted.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS=${RUNS:-$HERE/runs}
mkdir -p "$RUNS" && RUNS="$(cd "$RUNS" && pwd)"  # absolute before the cd below (a relative RUNS is from the caller)
export RUNS
cd "$HERE"
export SPEEDS=${SPEEDS:-0.06}
LATENCY=${LATENCY:-0.175} D=${D:-4} S=${S:-7} S_MIN=${S_MIN:-$D} S_MAX=${S_MAX:-20} N_SAMPLES=${N_SAMPLES:-4}
FRAME=${FRAME:-ada_const_s7_b2_w6}
export KEEP_CHUNK=1 RUNNER_ARGS="--prefetch-threshold $D"  # the chunk of length s + d sets the step
K=$((S_MIN + D))  # the runner's minimum chunk length
COMMON=(--n-samples "$N_SAMPLES" --no-guidance --adaptive --rule disp)
CONST=pure_const_s$S
ADA=pure_ada_s$S
calibrate() { (cd ../libero_pro && uv run python scripts/calibrate_step.py "$@"); }

git log -1 --oneline
bash run_dynamic.sh "$CONST" rtc "$K" "$LATENCY" "${COMMON[@]}" --s-min "$S" --s-max "$S" || exit 1
STEP=$RUNS/$CONST/step.json
[[ -f $STEP ]] || calibrate "$RUNS/$CONST" --target "$S" --delay "$D" --s-min "$S_MIN" --s-max "$S_MAX" --out "$STEP" || exit 1
ETA=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["rule"]["eta"])' "$STEP") || exit 1
echo "$(date '+%F %T') --eta $ETA (dispersion of $CONST, rate of s = $S)"
bash run_dynamic.sh "$ADA" rtc "$K" "$LATENCY" "${COMMON[@]}" --eta "$ETA" --s-min "$S_MIN" --s-max "$S_MAX" || exit 1

GOT=$(calibrate "$RUNS/$ADA" --achieved) || exit 1
echo "$(date '+%F %T') achieved mean s = $GOT (target $S)"
REPORT=("$RUNS/$CONST" "$RUNS/$ADA")
if (( GOT != S )); then
  bash run_dynamic.sh "pure_const_s$GOT" rtc "$K" "$LATENCY" "${COMMON[@]}" --s-min "$GOT" --s-max "$GOT" || exit 1
  REPORT+=("$RUNS/pure_const_s$GOT")
fi
for f in $FRAME; do [[ -d $RUNS/$f ]] && REPORT+=("$RUNS/$f"); done
python3 report_dynamic.py "${REPORT[@]}" --ref "$RUNS/$CONST" --speeds $SPEEDS --out "$RUNS/pure_adaptive_report.md"
if (( GOT != S )); then
  python3 report_dynamic.py "$RUNS/pure_const_s$GOT" "$RUNS/$ADA" --ref "$RUNS/pure_const_s$GOT" --speeds $SPEEDS \
    | tee -a "$RUNS/pure_adaptive_report.md"
fi
