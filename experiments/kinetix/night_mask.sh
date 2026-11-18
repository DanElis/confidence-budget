#!/usr/bin/env bash
# Fresh-state RTC mask selection BEFORE generation; fixed s=4, d=2, H=8.
# Small pilot defaults because deterministic Kinetix GPU physics was ~54x slower.
# Usage: bash night_mask.sh diagnose|smoke|night
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=${1:-night}
if (( $# > 0 )); then shift; fi
export OUT=${OUT:-$HERE/runs/learned_mask_before_d2_s4}
export TRAIN_HOURS=${TRAIN_HOURS:-1} ITERATIONS=${ITERATIONS:-8}
export TRAIN_ENVS=${TRAIN_ENVS:-8} EVAL_ENVS=${EVAL_ENVS:-16} MODEL_SEEDS=${MODEL_SEEDS:-0}
exec bash "$HERE/night_learned.sh" "$MODE" --mask-before --strides 4 --reference-stride 4 \
  --call-cost 0 --levels grasp_easy catapult --val-seeds 1000 --test-seeds 2000 "$@"
