#!/usr/bin/env bash
# AAC on openpi's pi05_libero, H = 10, with delay and naive switching, s = max(h* - d, d): our extension, the paper has no
# latency. The delay is simulated: the extra time of N = 20 samples is not added to d.
#
#   openpi_aac_d<d>          AAC, N = 20, sample 0
#   openpi_const_s<s>_d<d>   constant step of the same request rate (step_controls.py), one sample, the noise of sample 0
#
# H = 10 bounds the step: d <= s <= H - d. So d = 2 leaves s in 2...8, d = 4 only 4...6, and d = 5 no choice at all;
# d > 5 does not fit (LeRobot's d = 8 needs H = 50). The narrow range at d = 4 limits what any rule for s can change.
# Decided before the run. Main comparison at each d: AAC against that control, paired Δ over x0.1-x0.3 with a 95% bootstrap
# CI; two d: Bonferroni, so a gain counts only if it also holds at 97.5%.
#
# Usage (from experiments/libero_pro, after setup_openpi.sh; independent of openpi_aac.sh, may run beside it):
#   tmux new -s aac_async "bash scripts/stages/openpi_aac_async.sh"
# DELAYS="2 4" K=20 EPISODES_PER_TASK=20 SEED=1 BATCH_K=5 PORT=8780 by default; re-running skips finished suites.
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh
EPISODES=${EPISODES_PER_TASK:-20}
DELAYS=${DELAYS:-"2 4"}
K=${K:-20}
SEED=${SEED:-1}
BATCH_K=${BATCH_K:-5}
PORT=${PORT:-8780}
LEVELS=${LEVELS:-"pos:x0.1 pos:x0.2 pos:x0.3"}
SERVERS_FREE_MB=${SERVERS_FREE_MB:-45000}   # for all servers together, N = 20 per call
OPENPI=$PWD/third_party/openpi
SERVER_SCRIPT=$PWD/scripts/openpi_sample_server.py
OPENPI_PY="uv run --with websockets>=15 --with msgpack python"
REPORT=$RUNS/openpi_aac_async_report.md
SERVER_PIDS=()

start_server() {  # start_server PORT
  ( cd "$OPENPI" && exec env -u VIRTUAL_ENV XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES="${GPU:-0}" \
      .venv/bin/python "$SERVER_SCRIPT" --port "$1" ) > "$RUNS/openpi_aac_async_server_$1.log" 2>&1 &
  local pid=$!
  SERVER_PIDS+=("$pid")
  disown "$pid"  # outside lib.sh's job count and its final wait
  local i
  for (( i = 0; i < 120; i++ )); do
    curl -sf --noproxy '*' "http://127.0.0.1:$1/healthz" > /dev/null && { echo "$(date '+%F %T') server up on $1, pid $pid"; return 0; }
    kill -0 "$pid" 2>/dev/null || break
    sleep 5
  done
  echo "$(date '+%F %T') SERVER FAILED on $1"; tail -20 "$RUNS/openpi_aac_async_server_$1.log"
  return 1
}

stop_servers() { local pid; for pid in "${SERVER_PIDS[@]:-}"; do [[ -n $pid ]] && kill "$pid" 2>/dev/null; done; SERVER_PIDS=(); }

smoke() {
  local d log
  for d in $DELAYS; do
    log=$RUNS/openpi_aac_async_smoke_d$d.log
    $OPENPI_PY scripts/eval_openpi.py --suites pos:x0.2 --task-ids 0 --episodes-per-task 2 --batch-size 2 --signal aac --n-samples "$K" \
        --delay "$d" --seed "$SEED" --port "$PORT" --out "$RUNS/openpi_aac_async_smoke_d$d" > "$log" 2>&1 \
      || { echo "$(date '+%F %T') SMOKE FAILED: d=$d"; tail -20 "$log"; return 1; }
    echo "$(date '+%F %T') smoke d=$d: $(grep -o 'max diff [0-9.]*' "$log" | head -1), $(grep -o 'success [0-9.]*%' "$log" | tail -1)"
  done
}

per_level() {  # per_level NAME ARGS...: one process per level, each on its own server
  local name=$1 level port=$PORT; shift
  for level in $LEVELS; do
    PY=$OPENPI_PY EVAL_SCRIPT=scripts/eval_openpi.py MIN_FREE_MB=2000 run_eval "$name" "$level" --seed "$SEED" --port "$port" "$@"
    port=$(( port + 1 ))
  done
}

main() {
trap 'trap - INT TERM; stop_servers; kill 0' INT TERM
trap stop_servers EXIT
echo "$(date '+%F %T') openpi_aac_async: delays=$DELAYS K=$K EPISODES=$EPISODES seed=$SEED levels=$LEVELS"
git log -1 --oneline
[[ -x $OPENPI/.venv/bin/python ]] || { echo "no $OPENPI/.venv: run scripts/setup_openpi.sh"; exit 1; }
MIN_FREE_MB=$SERVERS_FREE_MB wait_for_gpu
local level port=$PORT d s steps
for level in $LEVELS; do start_server "$port" || exit 1; port=$(( port + 1 )); done
smoke || exit 1

for d in $DELAYS; do
  per_level "openpi_aac_d$d" --signal aac --n-samples "$K" --batch-size "$BATCH_K" --delay "$d"
  finish
done
for d in $DELAYS; do
  steps=$($PY scripts/step_controls.py "$RUNS/openpi_aac_d$d") || exit 1
  echo "$(date '+%F %T') d=$d: constant control s = $steps"
  for s in $steps; do per_level "openpi_const_s${s}_d$d" --stride "$s" --delay "$d"; done
  finish
done
stop_servers

: > "$REPORT"
for d in $DELAYS; do
  steps=$($PY scripts/step_controls.py "$RUNS/openpi_aac_d$d") || exit 1
  for s in $steps; do
    $PY scripts/compare.py --ref "$RUNS/openpi_const_s${s}_d$d" "$RUNS/openpi_const_s${s}_d$d" "$RUNS/openpi_aac_d$d" >> "$REPORT"
  done
done
echo "$(date '+%F %T') done: $REPORT"
}

main 2>&1 | tee -a "$RUNS/openpi_aac_async.log"
