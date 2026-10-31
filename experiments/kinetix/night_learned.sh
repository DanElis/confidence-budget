#!/usr/bin/env bash
# One GPU, no parallel model processes. Run from tmux; stage outputs are atomic/restartable.
#   bash night_learned.sh smoke
#   bash night_learned.sh night
# Controls: GPU=0 OUT=runs/learned_d2 MIN_FREE_MB=32000 WAIT_GPU_SECONDS=7200
#           TRAIN_HOURS=4 ITERATIONS=40 TRAIN_ENVS=32 EVAL_ENVS=256 MODEL_SEEDS="0 1 2"
# Extra Python args can follow the mode, e.g. night --levels grasp_easy catapult.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
MODE=${1:-night}
(( $# == 0 )) || shift
[[ $MODE == night || $MODE == smoke ]] || { echo "usage: bash night_learned.sh smoke|night [learn_budget.py args]"; exit 2; }
for arg in "$@"; do
  case "$arg" in --out|--out=*|--smoke|--preflight) echo "Use OUT=... and the shell mode instead of $arg"; exit 2;; esac
done
RTC="$HERE/third_party/real-time-chunking-kinetix"
[[ -x "$RTC/.venv/bin/python" && -d "$HERE/assets/bc" ]] || {
  echo "Kinetix environment/checkpoints missing. First run: cd $HERE && bash setup.sh"; exit 1;
}
for command in uv nvidia-smi flock setsid; do command -v "$command" >/dev/null || { echo "missing $command"; exit 1; }; done
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
unset JAX_PLATFORMS JAX_PLATFORM_NAME || true
OUT=$(realpath -m "${OUT:-$HERE/runs/learned_d2}")
[[ $MODE == smoke ]] && OUT="${OUT}_smoke"
mkdir -p "$OUT"
exec 9>"$OUT/night.lock"
flock -n 9 || { echo "Another run holds $OUT/night.lock"; exit 1; }
exec > >(tee -a "$OUT/night.log") 2>&1
worker=""
stop() {
  trap - INT TERM
  if [[ -n $worker ]]; then kill -TERM -- "-$worker" 2>/dev/null || true; wait "$worker" 2>/dev/null || true; fi
  echo "Stopped. Completed evaluations and the latest training iteration are retained."
  exit 130
}
trap stop INT TERM
trap 'echo "FAILED on line $LINENO. See $OUT/night.log; no failed rollout is counted as a policy failure."' ERR
run() {
  setsid "$@" & worker=$!
  local rc=0
  wait "$worker" || rc=$?
  worker=""
  return "$rc"
}
echo "$(date -Is) mode=$MODE GPU=$GPU OUT=$OUT"
git log -1 --oneline
echo "0 CPU tests (no GPU allocation)"
run env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" uv run --project "$RTC" --with pytest \
  python -m pytest -q "$HERE/tests"
echo "1 wait for GPU; require ${MIN_FREE_MB:-32000} MB free"
start=$SECONDS
while true; do
  free=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  [[ $free =~ ^[0-9]+$ ]] || { echo "Could not read free memory of GPU $GPU"; exit 1; }
  (( free < ${MIN_FREE_MB:-32000} )) || break
  (( SECONDS - start < ${WAIT_GPU_SECONDS:-7200} )) || { echo "GPU wait timeout"; exit 1; }
  echo "$(date -Is) GPU $GPU free ${free} MB; waiting"
  sleep 30
done
nvidia-smi --id="$GPU" --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo "2 real-model preflight: constant learned-controller path vs existing fixed RTC loop"
run bash "$HERE/run.sh" learn_budget.py --out "$OUT/preflight" --preflight "$@"
if [[ $MODE == smoke ]]; then
  echo "3 small end-to-end smoke: validation grid, one PPO iteration, held-out test"
  run bash "$HERE/run.sh" learn_budget.py --out "$OUT/experiment" --smoke "$@"
else
  echo "3 small end-to-end smoke before the full run (one level, four environments)"
  run bash "$HERE/run.sh" learn_budget.py --out "$OUT/smoke" --smoke "$@"
  read -ra seeds <<< "${MODEL_SEEDS:-0 1 2}"
  echo "4 full pipeline; training allocation ${TRAIN_HOURS:-4} hours total, evaluation/compilation additional"
  run bash "$HERE/run.sh" learn_budget.py --out "$OUT/experiment" \
    --train-hours "${TRAIN_HOURS:-4}" --iterations "${ITERATIONS:-40}" \
    --train-envs "${TRAIN_ENVS:-32}" --eval-envs "${EVAL_ENVS:-256}" --model-seeds "${seeds[@]}" "$@"
fi
echo "$(date -Is) DONE: $OUT/experiment/report.md"
cat "$OUT/experiment/report.md"
