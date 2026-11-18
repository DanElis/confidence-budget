#!/usr/bin/env bash
# Why our fixed s = 5 lies below the AAC paper (x0.2 53.2, x0.3 29.9; ours aac_base_s5: 47.0, 16.5): the three differences
# between our run and the authors' client that can be measured without their server.
#
#   aac_base_s5       LeRobot lerobot/pi05_libero_finetuned, H = 50, render 360   (pure_aac.sh)
#   lerobot_s5_r256   the same, render 256: the effect of the image size alone
#   openpi_s5         openpi pi05_libero, H = 10, render 256: the effect of the model as well
#
# All synchronous, fixed s = 5, the same scenes as aac_base_s5 (compare.py pairs them; the noise is not shared with openpi).
# Decided before the run: openpi_s5 explains the gap if on both x0.2 and x0.3 the paper's value is within 1.96 standard
# errors of its success rate and its paired Δ to aac_base_s5 is above zero; then AAC is worth repeating on openpi
# (N samples need a server patch). The paper averages 50 init states per task and other seeds, so this is a coarse match.
# Validity gate: openpi on stock libero_object must reach 50% on 20 episodes, or the observation is wrong and the run stops.
#
# Usage (from experiments/libero_pro, after bash scripts/setup_openpi.sh):
#   tmux new -s openpi "bash scripts/stages/openpi_base.sh"
# SEED=1 EPISODES_PER_TASK=20 JOBS=4 PORT=8765 by default, as pure_aac.sh; re-running skips finished suites.
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh
EPISODES=${EPISODES_PER_TASK:-20}
SEED=${SEED:-1}
PORT=${PORT:-8765}
LEVELS=${LEVELS:-"pos:x0.1,pos:x0.2,pos:x0.3"}
OPENPI=$PWD/third_party/openpi
OPENPI_PY="uv run --with websockets>=15 --with msgpack python"
REPORT=$RUNS/openpi_base_report.md
SERVER=

start_server() {
  [[ -x $OPENPI/.venv/bin/python ]] || { echo "no $OPENPI/.venv: run scripts/setup_openpi.sh"; return 1; }
  ( cd "$OPENPI" && exec env -u VIRTUAL_ENV XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES="${GPU:-0}" \
      .venv/bin/python scripts/serve_policy.py --port "$PORT" --env LIBERO ) > "$RUNS/openpi_server.log" 2>&1 &
  SERVER=$!
  disown "$SERVER"  # outside lib.sh's job count and its final wait
  local i
  for (( i = 0; i < 120; i++ )); do
    curl -sf --noproxy '*' "http://127.0.0.1:$PORT/healthz" > /dev/null && { echo "$(date '+%F %T') openpi server up, pid $SERVER"; return 0; }
    kill -0 "$SERVER" 2>/dev/null || break
    sleep 5
  done
  echo "$(date '+%F %T') OPENPI SERVER FAILED, see $RUNS/openpi_server.log"; tail -20 "$RUNS/openpi_server.log"
  return 1
}

stop_server() { [[ -n $SERVER ]] && kill "$SERVER" 2>/dev/null; SERVER=; }

gate() {
  local out=$RUNS/openpi_gate rate
  $OPENPI_PY scripts/eval_openpi.py --suites libero_object --episodes-per-task 2 --batch-size 2 --stride 5 --seed "$SEED" \
      --port "$PORT" --out "$out" > "$RUNS/openpi_gate.log" 2>&1 \
    || { echo "$(date '+%F %T') GATE FAILED to run, see $RUNS/openpi_gate.log"; tail -20 "$RUNS/openpi_gate.log"; return 1; }
  rate=$($OPENPI_PY -c "import json; print(round(100 * json.load(open('$out/libero_object.json'))['success_rate'], 1))" 2>/dev/null)
  echo "$(date '+%F %T') gate: stock libero_object $rate% on 20 episodes"
  $OPENPI_PY -c "raise SystemExit($rate < 50)" 2>/dev/null || { echo "gate: below 50%, stopping"; return 1; }
}

main() {
trap 'trap - INT TERM; stop_server; kill 0' INT TERM
trap stop_server EXIT
echo "$(date '+%F %T') openpi_base: EPISODES=$EPISODES seed=$SEED levels=$LEVELS"
git log -1 --oneline
ls "$RUNS"/aac_base_s5/libero_object_pos_x0.2.json > /dev/null || { echo "aac_base_s5 is missing: run pure_aac.sh first"; exit 1; }
start_server || exit 1
gate || exit 1

PY=$OPENPI_PY EVAL_SCRIPT=scripts/eval_openpi.py MIN_FREE_MB=2000 run_eval openpi_s5 "$LEVELS" --stride 5 --seed "$SEED" --port "$PORT"
MIN_FREE_MB=18000 run_eval lerobot_s5_r256 "$LEVELS" --stride 5 --delay 0 --guidance none --resolution 256 --seed "$SEED"
finish
stop_server

{
  echo "Paper, fixed s = 5: x0.2 53.2, x0.3 29.9"
  echo
  $PY scripts/compare.py --ref "$RUNS/aac_base_s5" "$RUNS/aac_base_s5" "$RUNS/lerobot_s5_r256" "$RUNS/openpi_s5"
} > "$REPORT"
echo "$(date '+%F %T') done: $REPORT"
}

main 2>&1 | tee -a "$RUNS/openpi_base.log"
