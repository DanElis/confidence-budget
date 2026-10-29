#!/usr/bin/env bash
# One LIBERO-Dynamic configuration: our policy server(s) + the benchmark's own runner and controllers.
#
#   bash run_dynamic.sh NAME STRATEGY K LATENCY_S [server args...]
#     STRATEGY   sync | async | rtc   (scripts/run_pi05_{sync,async,rtc}_eval.py of LIBERO-Dynamic)
#     K          --replan-steps of the runner (the action horizon of the paper)
#     LATENCY_S  fixed simulated inference latency (paper, Pi0.5 PyTorch: 0.147 sync/async, 0.170 RTC)
#     server args are passed to policy_server.py (rtc adds --mode rtc)
#
# Environment: SPEEDS ("0.03 0.06 0.09"), LEVELS ("l1 l2 l3"), TRAJ ("0 1 2 3 4"), LIMIT (items per speed, empty =
# all 40 tasks x levels x trajectories), SERVERS (policy server processes, 4), WORKERS (simulator workers, 8),
# PORT (8124), RUNS (runs). Full configuration: 3 speeds x 600 = 1800 episodes. Finished speeds are skipped.
#
# RUNNER_ARGS: extra arguments for the benchmark runner (e.g. "--prefetch-threshold 4" for the adaptive step).
# KEEP_CHUNK=1: the client keeps the whole returned chunk instead of trimming it to K (patch_runner.sh).
#
# Adaptive step from plan agreement + confidence mask + coherent sample (RTC strategy of the runner; the server sets
# the step through the chunk length s + d, the runner requests when d actions are left; K = s_min + d):
#   KEEP_CHUNK=1 RUNNER_ARGS="--prefetch-threshold 4" bash run_dynamic.sh ada rtc 8 0.175 --n-samples 4 --select coherent \
#     --mask-mode rtc_x_conf --conf-sigma 1 --conf-ref 0.03716 --adaptive --eta 3 --s-max 20 --rtc-execution-horizon 16
#
# RTC with a transition whose length grows with the deviation of the old plan under the new frame (one sample;
# first the constant-length control, its median deviation "Δ med" from report_dynamic.py is --trans-ref):
#   bash run_dynamic.sh trans_c6 rtc 10 0.175 --mask-mode transition --trans-len 6 --trans-power 0
#   bash run_dynamic.sh trans_a6 rtc 10 0.175 --mask-mode transition --trans-len 6 --trans-power 1 --trans-ref REF
#
# Paper baselines (Pi0.5 PyTorch, best measured K):
#   bash run_dynamic.sh pt_sync_k10  sync  10 0.147
#   bash run_dynamic.sh pt_async_k6  async 6  0.147
#   bash run_dynamic.sh pt_rtc_k16   rtc   16 0.170
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME=$1 STRATEGY=$2 K=$3 LATENCY=$4; shift 4
SERVER_ARGS=("$@")
DYN="$HERE/third_party/LIBERO-Dynamic"
OPENPI="$HERE/third_party/openpi"
SPEEDS=${SPEEDS:-"0.03 0.06 0.09"}
LEVELS=${LEVELS:-"l1 l2 l3"}
TRAJ=${TRAJ:-"0 1 2 3 4"}
LIMIT=${LIMIT:-}
SERVERS=${SERVERS:-4}
WORKERS=${WORKERS:-8}
PORT=${PORT:-8124}
RUNS=${RUNS:-$HERE/runs}
OUT="$RUNS/$NAME"
mkdir -p "$OUT"
[[ $STRATEGY == rtc ]] && SERVER_ARGS=(--mode rtc "${SERVER_ARGS[@]}")
# the runner rejects chunks shorter than K; like the paper's servers, return max(10, K) actions (the RTC mask then
# also ends there, their default) unless --actions-per-chunk is given explicitly
[[ " ${SERVER_ARGS[*]} " == *" --actions-per-chunk "* ]] || SERVER_ARGS+=(--actions-per-chunk $(( K > 10 ? K : 10 )))
DPY="$DYN/.venv/bin/python"
export PYTHONPATH="$DYN:$DYN/scripts:$OPENPI/packages/openpi-client/src"
export LIBERO_CONFIG_PATH="$DYN/libero/libero/configs" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${GPU:-0}
export LIBERO_DYNAMIC_AUTO_FINALIZE=false
if [[ ${KEEP_CHUNK:-0} == 1 ]]; then
  grep -q LIBERO_DYNAMIC_KEEP_CHUNK "$DYN/scripts/eval_dynamic_libero_style.py" \
    || { echo "KEEP_CHUNK=1 needs the client patch: bash patch_runner.sh"; exit 1; }
  export LIBERO_DYNAMIC_KEEP_CHUNK=1
else
  unset LIBERO_DYNAMIC_KEEP_CHUNK
fi

pids=()
stop_servers() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; }
trap 'stop_servers; exit 130' INT TERM

speed_done() { [[ -f "$OUT/v$1/evaluation/summary.json" || -f "$OUT/v$1/summary.json" ]]; }
todo=()
for v in $SPEEDS; do speed_done "$v" || todo+=("$v"); done
if (( ${#todo[@]} == 0 )); then echo "$(date '+%F %T') $NAME: all speeds done"; exit 0; fi

echo "$(date '+%F %T') $NAME: strategy=$STRATEGY K=$K latency=$LATENCY speeds=${todo[*]} servers=$SERVERS workers=$WORKERS server_args=${SERVER_ARGS[*]}"
git -C "$HERE" log -1 --oneline
for (( i = 0; i < SERVERS; i++ )); do
  (cd "$HERE/../libero_pro" && exec uv run --with websockets --with msgpack python "$HERE/policy_server.py" \
     --port $((PORT + i)) --log-jsonl "$OUT/signals_$i.jsonl" "${SERVER_ARGS[@]}") > "$OUT/server_$i.log" 2>&1 &
  pids+=($!)
done
for (( i = 0; i < SERVERS; i++ )); do  # wait until every server answers its metadata
  until "$DPY" -c "from websockets.sync.client import connect; connect('ws://127.0.0.1:$((PORT + i))', open_timeout=1).close()" 2>/dev/null; do
    kill -0 "${pids[$i]}" 2>/dev/null || { echo "server $i died, see $OUT/server_$i.log"; tail -20 "$OUT/server_$i.log"; stop_servers; exit 1; }
    sleep 5
  done
done
echo "$(date '+%F %T') $SERVERS policy server(s) up"

rc=0
for v in "${todo[@]}"; do
  cfg="$OUT/configs_v$v"
  rm -rf "$cfg" && cp -r "$DYN/libero/libero/dynamic_configs" "$cfg"
  "$DPY" - "$cfg" "$v" <<'PY' || { rc=1; break; }
import sys
from pathlib import Path
import yaml
cfg, speed, n = Path(sys.argv[1]), float(sys.argv[2]), 0
def walk(x):
    global n
    if isinstance(x, dict):
        for k in x:
            if k == "alpha_mps":
                x[k] = speed; n += 1
            else:
                walk(x[k])
    elif isinstance(x, list):
        for c in x:
            walk(c)
for f in sorted(cfg.glob("libero_dynamic_l*.yaml")):
    d = yaml.safe_load(f.read_text()); walk(d); f.write_text(yaml.safe_dump(d, sort_keys=False))
assert n > 0, "no alpha_mps found"
print(f"speed {speed}: patched {n} alpha_mps entries")
PY
  extra=()
  [[ -n $LIMIT ]] && extra=(--start-index 0 --limit "$LIMIT")
  echo "$(date '+%F %T') start $NAME v=$v"
  (cd "$DYN" && "$DPY" "scripts/run_pi05_${STRATEGY}_eval.py" --config-dir "$cfg" --output-dir "$OUT/v$v" \
     --levels $LEVELS --traj-indices $TRAJ --num-workers "$WORKERS" --policy openpi_ws --openpi-host 127.0.0.1 \
     --openpi-port "$PORT" --policy-server-count "$SERVERS" --per-worker-policy-endpoint \
     --latency-source fixed --calibrated-latency-seconds "$LATENCY" --replan-steps "$K" --max-steps 600 \
     --stop-on-success --no-save-videos --no-save-model-input-videos --no-save-step-trace --camera-size 256 \
     "${extra[@]}" ${RUNNER_ARGS:-}) > "$OUT/eval_v$v.log" 2>&1 || { rc=1; echo "eval failed, see $OUT/eval_v$v.log"; tail -20 "$OUT/eval_v$v.log"; break; }
  "$DPY" - "$OUT/v$v" <<'PY'
import json, sys
from pathlib import Path
s = json.loads(next(Path(sys.argv[1]).rglob("summary.json")).read_text())
tot, ok = s.get("total"), s.get("success_count")
print(f"  success {ok}/{tot} = {100 * ok / max(tot, 1):.1f}%" if tot is not None else f"  summary keys: {list(s)}")
PY
done
stop_servers
exit $rc
