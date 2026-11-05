#!/usr/bin/env bash
# RTC of the paper (Black et al., 2025) on openpi's pi05_libero, H = 10, with delay, against naive switching at the same
# constant step and noise.
#
#   openpi_rtc_paper_s<s>_d<d>   Eq. 2 with the network's Jacobian, the paper's weights, beta = 5, mask to H - s
#   openpi_const_s<s>_d<d>       naive, from openpi_aac_async.sh; must be finished
#
# Differs from the paper: LIBERO-PRO with this checkpoint and H = 10, and 10 denoising steps, those of openpi and of the
# naive runs (the paper's real-world runs use 5). H = 10 leaves little to guide: at d = 4, s = 4 freezes 4 actions and
# softly attaches 2, s = 5 freezes 4 and attaches 1. The delay is simulated, and the extra time of the Jacobian is not
# added to it. The client checks the guidance on the real model before each suite (check_guidance).
# Decided before the run. Main comparison at each s: RTC against naive at that s, paired Δ over x0.1-x0.3 with a 95%
# bootstrap CI; two s: Bonferroni, so a gain counts only if it also holds at 97.5%. Smoothness (|Δa| at the switch, SPARC)
# is secondary.
#
# Usage (from experiments/libero_pro, after openpi_aac_async.sh has finished):
#   tmux new -d -s rtc_d4 "bash scripts/stages/openpi_rtc_async.sh"
# DELAY=4 STEPS="4 5" EPISODES_PER_TASK=20 SEED=1 PORT=8790 by default; re-running skips finished suites and resumes
# stopped ones after their last finished task.
SUITE_TIMEOUT=${SUITE_TIMEOUT:-12h}
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh
EPISODES=${EPISODES_PER_TASK:-20}
DELAY=${DELAY:-4}
STEPS=${STEPS:-"4 5"}
SEED=${SEED:-1}
PORT=${PORT:-8790}
LEVELS=${LEVELS:-"pos:x0.1 pos:x0.2 pos:x0.3"}
SERVERS_FREE_MB=${SERVERS_FREE_MB:-45000}
OPENPI=$PWD/third_party/openpi
SERVER_SCRIPT=$PWD/scripts/openpi_sample_server.py
OPENPI_PY="uv run --with websockets>=15 --with msgpack python"
REPORT=$RUNS/openpi_rtc_report.md
SERVER_PIDS=()

start_server() {  # start_server PORT
  ( cd "$OPENPI" && exec env -u VIRTUAL_ENV XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES="${GPU:-0}" \
      .venv/bin/python "$SERVER_SCRIPT" --port "$1" ) > "$RUNS/openpi_rtc_server_$1.log" 2>&1 &
  local pid=$!
  SERVER_PIDS+=("$pid")
  disown "$pid"  # outside lib.sh's job count and its final wait
  local i
  for (( i = 0; i < 120; i++ )); do
    curl -sf --noproxy '*' "http://127.0.0.1:$1/healthz" > /dev/null && { echo "$(date '+%F %T') server up on $1, pid $pid"; return 0; }
    kill -0 "$pid" 2>/dev/null || break
    sleep 5
  done
  echo "$(date '+%F %T') SERVER FAILED on $1"; tail -20 "$RUNS/openpi_rtc_server_$1.log"
  return 1
}

stop_servers() { local pid; for pid in "${SERVER_PIDS[@]:-}"; do [[ -n $pid ]] && kill "$pid" 2>/dev/null; done; SERVER_PIDS=(); }

smoke() {
  local s log
  for s in $STEPS; do
    log=$RUNS/openpi_rtc_smoke_s$s.log
    $OPENPI_PY scripts/eval_openpi.py --suites pos:x0.2 --task-ids 0 --episodes-per-task 2 --batch-size 2 --stride "$s" --delay "$DELAY" \
        --guidance paper --seed "$SEED" --port "$PORT" --out "$RUNS/openpi_rtc_smoke_s$s" > "$log" 2>&1 \
      || { echo "$(date '+%F %T') SMOKE FAILED: s=$s"; tail -20 "$log"; return 1; }
    echo "$(date '+%F %T') smoke s=$s: $(grep -o 'RTC check.*' "$log" | head -1), $(grep -o 'success [0-9.]*%' "$log" | tail -1)"
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
echo "$(date '+%F %T') openpi_rtc_async: d=$DELAY s=$STEPS EPISODES=$EPISODES seed=$SEED levels=$LEVELS"
git log -1 --oneline
[[ -x $OPENPI/.venv/bin/python ]] || { echo "no $OPENPI/.venv: run scripts/setup_openpi.sh"; exit 1; }
local level port=$PORT s
for s in $STEPS; do
  for level in $LEVELS; do
    [[ -f "$RUNS/openpi_const_s${s}_d$DELAY/$(suite_dir_name "$level").json" ]] \
      || { echo "naive s=$s d=$DELAY $level is missing: run openpi_aac_async.sh first"; exit 1; }
  done
done
MIN_FREE_MB=$SERVERS_FREE_MB wait_for_gpu
for level in $LEVELS; do start_server "$port" || exit 1; port=$(( port + 1 )); done
smoke || exit 1

for s in $STEPS; do per_level "openpi_rtc_paper_s${s}_d$DELAY" --stride "$s" --delay "$DELAY" --guidance paper; done
finish
stop_servers

: > "$REPORT"
for s in $STEPS; do
  $PY scripts/compare.py --ref "$RUNS/openpi_const_s${s}_d$DELAY" "$RUNS/openpi_const_s${s}_d$DELAY" "$RUNS/openpi_rtc_paper_s${s}_d$DELAY" >> "$REPORT"
done
echo "$(date '+%F %T') done: $REPORT"
}

main 2>&1 | tee -a "$RUNS/openpi_rtc_async.log"
