#!/usr/bin/env bash
# AAC [13] as in the authors' code (libero_pro/aac.py) on LIBERO-Dynamic, without RTC:
#
#   aac_sync  AAC proper: the runner's sync_hold (the robot holds during inference, the scene moves), plain server.
#   aac_d$D   AAC with delay: the runner's rtc_queue with an unguided server, so naive switching, s = max(h* - d, d).
#             Our extension, the paper has no latency. The same queue as the earlier adaptive runs.
# Each against the constant step of the same request rate with the same K and sample 0 (libero_pro/scripts/step_controls.py,
# the two neighbouring integers if none is within 5%). The chunk length sets the step, hence KEEP_CHUNK.
# Main comparison fixed in advance, for each of the two: AAC against that control, v = 0.06, 600 pairs, Bonferroni for two.
# From experiments/libero_dynamic:
#
#   LIMIT=2 RUNS=runs/aac_smoke bash run_pure_aac.sh        # smoke
#   tmux new -d -s aac "bash run_pure_aac.sh 2>&1 | tee -a runs/pure_aac.log"
#
# Environment: as run_dynamic.sh (SERVERS, WORKERS, PORT, LIMIT, RUNS) plus SPEEDS (0.06), LATENCY (0.175, one for all),
# D (4: must match LATENCY at 20 Hz), N_SAMPLES (20). Finished speeds are skipped, so the script can be restarted.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS=${RUNS:-$HERE/runs}
mkdir -p "$RUNS" && RUNS="$(cd "$RUNS" && pwd)"  # absolute before the cd below (a relative RUNS is from the caller)
export RUNS
cd "$HERE"
export SPEEDS=${SPEEDS:-0.06} KEEP_CHUNK=1
LATENCY=${LATENCY:-0.175} D=${D:-4} N_SAMPLES=${N_SAMPLES:-20}
COMMON=(--n-samples "$N_SAMPLES" --adaptive --rule aac_paper)
REPORT=$RUNS/pure_aac_report.md
controls() { (cd ../libero_pro && uv run python scripts/step_controls.py "$1"); }

git log -1 --oneline
: > "$REPORT"

pair() {  # pair NAME STRATEGY K S_MIN RUNNER_ARGS [server args]
  local name=$1 strategy=$2 k=$3 s_min=$4 runner=$5 s steps; shift 5
  RUNNER_ARGS=$runner bash run_dynamic.sh "$name" "$strategy" "$k" "$LATENCY" "${COMMON[@]}" "$@" --s-min "$s_min" --s-max 50 || exit 1
  steps=$(controls "$RUNS/$name") || exit 1
  echo "$(date '+%F %T') $name: constant control s = $steps"
  for s in $steps; do
    RUNNER_ARGS=$runner bash run_dynamic.sh "${name}_const_s$s" "$strategy" "$k" "$LATENCY" "${COMMON[@]}" "$@" --s-min "$s" --s-max "$s" || exit 1
    python3 report_dynamic.py "$RUNS/${name}_const_s$s" "$RUNS/$name" --ref "$RUNS/${name}_const_s$s" --speeds $SPEEDS | tee -a "$REPORT"
  done
}

pair aac_sync sync 2 1 ""
pair "aac_d$D" rtc $((2 * D)) "$D" "--prefetch-threshold $D" --no-guidance
