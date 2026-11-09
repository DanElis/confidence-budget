#!/usr/bin/env bash
# Server only. CPU checks -> GPU wait -> smoke -> exploration -> offline gates -> validation -> test (budget_v2.py).
# A restart with the same OUT and arguments skips finished rollouts.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${OUT:-$HERE/runs/budget_v2_d7}
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
exec > >(tee -a "$OUT/night.log") 2>&1
trap 'echo "FAILED on line $LINENO; see $OUT/night.log. No infrastructure error is counted as a policy failure."' ERR
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTHONUNBUFFERED=1
echo "$(date -Is) OUT=$OUT GPU=${GPU:-0}"
git -C "$HERE" log -1 --oneline
bash "$HERE/patch_runner.sh"
bash "$HERE/patch_budget.sh"
cd "$HERE/../libero_pro"
echo "0 CPU tests (GPU hidden)"
CUDA_VISIBLE_DEVICES="" uv run --with websockets --with msgpack --with pytest python -m pytest -q "$HERE/tests"
echo "1 wait for GPU; require ${FREE_MB:-52000} MB free"
deadline=$((SECONDS + ${GPU_WAIT_SECONDS:-21600}))
until nvidia-smi --id="${GPU:-0}" --query-gpu=memory.free --format=csv,noheader,nounits \
    | awk -v need="${FREE_MB:-52000}" 'NR==1 {exit !($1>=need)}'; do
  (( SECONDS < deadline )) || { echo "GPU wait timed out"; exit 1; }
  sleep 30
done
nvidia-smi --id="${GPU:-0}" --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo "2 experiment"
uv run --with websockets --with msgpack --with pyyaml python "$HERE/budget_v2.py" --out "$OUT/experiment" "$@"
