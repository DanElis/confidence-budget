#!/usr/bin/env bash
# AAC [13] as in the authors' code (aac.py) on openpi's pi05_libero, H = 10, synchronous, the setting closest to the paper.
#
#   openpi_aac            AAC, N = 20, sample 0
#   openpi_const_s<s>     constant step of the same request rate (step_controls.py; the two neighbouring integers if none
#                         is within 5%), one sample per request, the same noise as sample 0 of AAC
#
# Decided before the run. Main comparison: AAC against that control, paired Δ over x0.1-x0.3 with a 95% bootstrap CI.
# Secondary, the paper's own comparison: AAC against openpi_s5 of openpi_base.sh (scenes shared, noise not).
# Paper for reference: AAC x0.2 57.4, x0.3 35.3; fixed s = 5: 53.2, 29.9.
# No delay: at H = 10 a large d leaves no room for the step.
#
# Usage (from experiments/libero_pro, after openpi_base.sh):  tmux new -s aac_openpi "bash scripts/stages/openpi_aac.sh"
# K=20 EPISODES_PER_TASK=20 SEED=1 BATCH_K=5 PORT=8770 by default; one server per level; re-running skips finished suites
# and resumes stopped ones after their last finished task.
SUITE_TIMEOUT=${SUITE_TIMEOUT:-12h}
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh
EPISODES=${EPISODES_PER_TASK:-20}
K=${K:-20}
SEED=${SEED:-1}
BATCH_K=${BATCH_K:-5}
PORT=${PORT:-8770}
LEVELS=${LEVELS:-"pos:x0.1 pos:x0.2 pos:x0.3"}
SERVERS_FREE_MB=${SERVERS_FREE_MB:-45000}   # for all servers together, N = 20 per call
OPENPI=$PWD/third_party/openpi
SERVER_SCRIPT=$PWD/scripts/openpi_sample_server.py
OPENPI_PY="uv run --with websockets>=15 --with msgpack python"
REPORT=$RUNS/openpi_aac_report.md
SERVER_PIDS=()
export JOBS=${JOBS:-4}

start_server() {  # start_server PORT
  ( cd "$OPENPI" && exec env -u VIRTUAL_ENV XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES="${GPU:-0}" \
      .venv/bin/python "$SERVER_SCRIPT" --port "$1" ) > "$RUNS/openpi_aac_server_$1.log" 2>&1 &
  local pid=$!
  SERVER_PIDS+=("$pid")
  disown "$pid"  # outside lib.sh's job count and its final wait
  local i
  for (( i = 0; i < 120; i++ )); do
    curl -sf --noproxy '*' "http://127.0.0.1:$1/healthz" > /dev/null && { echo "$(date '+%F %T') server up on $1, pid $pid"; return 0; }
    kill -0 "$pid" 2>/dev/null || break
    sleep 5
  done
  echo "$(date '+%F %T') SERVER FAILED on $1"; tail -20 "$RUNS/openpi_aac_server_$1.log"
  return 1
}

stop_servers() { local pid; for pid in "${SERVER_PIDS[@]:-}"; do [[ -n $pid ]] && kill "$pid" 2>/dev/null; done; SERVER_PIDS=(); }

checks() {  # the new server on stock libero_object must reach 50%, and AAC must run on it
  local out=$RUNS/openpi_aac_gate rate
  $OPENPI_PY scripts/eval_openpi.py --suites libero_object --episodes-per-task 2 --batch-size 2 --stride 5 --seed "$SEED" \
      --port "$PORT" --out "$out" > "$RUNS/openpi_aac_gate.log" 2>&1 \
    || { echo "$(date '+%F %T') GATE FAILED to run"; tail -20 "$RUNS/openpi_aac_gate.log"; return 1; }
  rate=$($OPENPI_PY -c "import json; print(round(100 * json.load(open('$out/libero_object.json'))['success_rate'], 1))" 2>/dev/null)
  echo "$(date '+%F %T') gate: stock libero_object $rate% on 20 episodes"
  $OPENPI_PY -c "raise SystemExit($rate < 50)" 2>/dev/null || { echo "gate: below 50%, stopping"; return 1; }
  $OPENPI_PY scripts/eval_openpi.py --suites pos:x0.2 --task-ids 0 --episodes-per-task 2 --batch-size 2 --signal aac --n-samples "$K" \
      --seed "$SEED" --port "$PORT" --out "$RUNS/openpi_aac_smoke" > "$RUNS/openpi_aac_smoke.log" 2>&1 \
    || { echo "$(date '+%F %T') SMOKE FAILED"; tail -20 "$RUNS/openpi_aac_smoke.log"; return 1; }
  echo "$(date '+%F %T') smoke: $(grep -o 'max diff [0-9.]*' "$RUNS/openpi_aac_smoke.log" | head -1), $(grep -o 'success [0-9.]*%' "$RUNS/openpi_aac_smoke.log" | tail -1)"
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
echo "$(date '+%F %T') openpi_aac: K=$K EPISODES=$EPISODES seed=$SEED levels=$LEVELS"
git log -1 --oneline
[[ -x $OPENPI/.venv/bin/python ]] || { echo "no $OPENPI/.venv: run scripts/setup_openpi.sh"; exit 1; }
ls "$RUNS"/openpi_s5/libero_object_pos_x0.2.json > /dev/null || { echo "openpi_s5 is missing: run openpi_base.sh first"; exit 1; }
MIN_FREE_MB=$SERVERS_FREE_MB wait_for_gpu
local level port=$PORT
for level in $LEVELS; do start_server "$port" || exit 1; port=$(( port + 1 )); done
checks || exit 1

per_level openpi_aac --signal aac --n-samples "$K" --batch-size "$BATCH_K"
finish
local s steps
steps=$($PY scripts/step_controls.py "$RUNS/openpi_aac") || exit 1
echo "$(date '+%F %T') constant control s = $steps"
for s in $steps; do per_level "openpi_const_s$s" --stride "$s"; done
finish
stop_servers

: > "$REPORT"
for s in $steps; do
  $PY scripts/compare.py --ref "$RUNS/openpi_const_s$s" "$RUNS/openpi_const_s$s" "$RUNS/openpi_aac" "$RUNS/openpi_s5" >> "$REPORT"
done
echo "$(date '+%F %T') done: $REPORT"
}

main 2>&1 | tee -a "$RUNS/openpi_aac.log"
