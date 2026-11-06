# openpi servers for the stage scripts, one per level. Source after lib.sh, from experiments/libero_pro:
#   source scripts/stages/lib.sh && source scripts/stages/openpi_lib.sh
# Needs PORT, LEVELS, SEED and SERVER_LOG (prefix of the server logs in $RUNS).
OPENPI=$PWD/third_party/openpi
SERVER_SCRIPT=$PWD/scripts/openpi_sample_server.py
OPENPI_PY="uv run --with websockets>=15 --with msgpack python"
SERVERS_FREE_MB=${SERVERS_FREE_MB:-45000}
SERVER_PIDS=()

start_server() {  # start_server PORT
  ( cd "$OPENPI" && exec env -u VIRTUAL_ENV XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES="${GPU:-0}" \
      .venv/bin/python "$SERVER_SCRIPT" --port "$1" ) > "$RUNS/${SERVER_LOG}_server_$1.log" 2>&1 &
  local pid=$!
  SERVER_PIDS+=("$pid")
  disown "$pid"  # outside lib.sh's job count and its final wait
  local i
  for (( i = 0; i < 120; i++ )); do
    curl -sf --noproxy '*' "http://127.0.0.1:$1/healthz" > /dev/null && { echo "$(date '+%F %T') server up on $1, pid $pid"; return 0; }
    kill -0 "$pid" 2>/dev/null || break
    sleep 5
  done
  echo "$(date '+%F %T') SERVER FAILED on $1"; tail -20 "$RUNS/${SERVER_LOG}_server_$1.log"
  return 1
}

start_servers() {  # one per level, PORT, PORT + 1, ...
  [[ -x $OPENPI/.venv/bin/python ]] || { echo "no $OPENPI/.venv: run scripts/setup_openpi.sh"; return 1; }
  MIN_FREE_MB=$SERVERS_FREE_MB wait_for_gpu
  local level port=$PORT
  for level in $LEVELS; do start_server "$port" || return 1; port=$(( port + 1 )); done
}

stop_servers() { local pid; for pid in "${SERVER_PIDS[@]:-}"; do [[ -n $pid ]] && kill "$pid" 2>/dev/null; done; SERVER_PIDS=(); }

per_level() {  # per_level NAME ARGS...: one process per level, each on its own server
  local name=$1 level port=$PORT; shift
  for level in $LEVELS; do
    PY=$OPENPI_PY EVAL_SCRIPT=scripts/eval_openpi.py MIN_FREE_MB=2000 run_eval "$name" "$level" --seed "$SEED" --port "$port" "$@"
    port=$(( port + 1 ))
  done
}

smoke_eval() {  # smoke_eval NAME ARGS...: 2 episodes of task 0 on the first level and server
  local log=$RUNS/$1_smoke.log name=$1; shift
  $OPENPI_PY scripts/eval_openpi.py --suites "${LEVELS%% *}" --task-ids 0 --episodes-per-task 2 --batch-size 2 --seed "$SEED" \
      --port "$PORT" --out "$RUNS/${name}_smoke" "$@" > "$log" 2>&1 \
    || { echo "$(date '+%F %T') SMOKE FAILED: $name"; tail -20 "$log"; return 1; }
  echo "$(date '+%F %T') smoke $name: $(grep -o 'RTC check.*' "$log" | head -1), $(grep -o 'success [0-9.]*%' "$log" | tail -1)"
}

percent() { awk -v q="$1" 'BEGIN { printf "%d", q * 100 + 0.5 }'; }  # 0.25 -> 25
