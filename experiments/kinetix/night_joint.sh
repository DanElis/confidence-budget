#!/usr/bin/env bash
# Joint (s_k, r_k) BEFORE inference, from fresh state and old leftover only.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=${1:-night}
if (( $# > 0 )); then shift; fi
export OUT=${OUT:-$HERE/runs/learned_joint_before_d2}
export TRAIN_HOURS=${TRAIN_HOURS:-1} ITERATIONS=${ITERATIONS:-8}
export TRAIN_ENVS=${TRAIN_ENVS:-8} EVAL_ENVS=${EVAL_ENVS:-16} MODEL_SEEDS=${MODEL_SEEDS:-0}
exec bash "$HERE/night_learned.sh" "$MODE" --mask-before --strides 2 4 --reference-stride 2 \
  --call-cost 0.1 --levels grasp_easy catapult --val-seeds 1000 --test-seeds 2000 "$@"
