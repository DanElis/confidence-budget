#!/usr/bin/env bash
# Overnight pipeline of the correction head on LIBERO-Dynamic. Restartable: finished stages are skipped.
# From experiments/libero_dynamic (after setup.sh; the libero_pro venv must be synced):
#
#   tmux new -d -s corr "bash correction/night.sh 2>&1 | tee -a runs/correction_night.log"
#
# Stages (priority order):
#   0  checks: tests without a model, runner patch (corrected_rtc_queue registered)
#   1  data: download pi05_raw (1.1 GB), decode frames for the head (DATA/frames_128.u8, ~28 GB)
#   2  smoke replay (4 episodes) and the main replays of the async RTC process along the demonstrations:
#        rtc_s4   K 10, d 4, s 4, beta 2, window 6   (= rtc_k10_b2_w6 at 0.175 s)
#        rtc_var  K 16, s ~ U{4..12}, RTC scale ~ U[0.25, 1], random first tick (coverage for the adaptive variants)
#   3  head training on rtc_s4 + rtc_var, validation on rtc_s4 (held-out episodes); gate: val MSE corrected / base
#      <= 0.8 (offline_report.json). Without the gate closed loop is skipped (FORCE=1 runs it anyway).
#   4  closed loop, speed 0.06, 600 paired episodes, latency 0.175 (d = 4), K 10, beta 2, window 6:
#        corr_c1  rtc_queue, server with --return-latent (control: identical chunks until trajectories diverge)
#        corr_c2  corrected_rtc_queue, correction from the current frame (0 ticks)        <- main comparison
#        corr_c3  corrected_rtc_queue, correction from the previous frame (1 tick)
#   5  ablations offline: sync chunks (replay without RTC), tail of the previous chunk as an input
#
# Environment: DATA (data_correction), HEADS (heads), RUNS (runs), SPEEDS (0.06), STEPS (40000 training steps),
# BATCH (32 episodes per replay call), FORCE, plus run_dynamic.sh's SERVERS / WORKERS / PORT / GPU.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # experiments/libero_dynamic
cd "$HERE"
DATA=$(realpath -m "${DATA:-$HERE/data_correction}")
HEADS=$(realpath -m "${HEADS:-$HERE/heads}")
export RUNS=$(realpath -m "${RUNS:-$HERE/runs}")
export SPEEDS=${SPEEDS:-0.06}
STEPS=${STEPS:-40000} BATCH=${BATCH:-32} FORCE=${FORCE:-0}
mkdir -p "$DATA" "$HEADS" "$RUNS"
PY=(uv run --with websockets --with msgpack python)
lp() { (cd "$HERE/../libero_pro" && "${PY[@]}" "$@"); }  # python in the libero_pro venv
stamp() { echo "$(date '+%F %T') == $*"; }
die() { stamp "FAILED: $*"; exit 1; }

stamp "correction night: DATA=$DATA HEADS=$HEADS RUNS=$RUNS SPEEDS=$SPEEDS"
git log -1 --oneline
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv || true
df -h "$DATA" | tail -1

stamp "0 checks"
[[ -d third_party/LIBERO-Dynamic/.git ]] || bash setup.sh || die "setup.sh"
bash patch_runner.sh || die "patch_runner.sh"
lp -m pytest -q ../libero_dynamic/tests || die "tests"

stamp "1 data"
lp ../libero_dynamic/correction/prepare_data.py --data "$DATA" || die "prepare_data"

replay() {  # NAME ARGS...
  local name=$1; shift
  stamp "2 replay $name $*"
  lp ../libero_dynamic/correction/replay.py --data "$DATA" --name "$name" --batch "$BATCH" "$@" || die "replay $name"
}
replay smoke --limit 4 --K 10 --delay 4 --step 4
replay rtc_s4 --K 10 --delay 4 --step 4
replay rtc_var --K 16 --delay 4 --step-range 4 12 --scale-range 0.25 1 --max-offset 3 --seed 1

train() {  # OUT ARGS...
  local out=$1; shift
  stamp "3 train $out $*"
  lp ../libero_dynamic/correction/train_head.py --data "$DATA" --out "$HEADS/$out" --steps "$STEPS" "$@" \
    || die "train $out"
  python3 -c "import json,sys; r=json.load(open(sys.argv[1])); b=r['best']; print('offline', sys.argv[1], 'ratio', round(b['ratio'],3), 'arm', round(b['arm_ratio'],3), 'step', b['step'], 'pass', r['pass'])" \
    "$HEADS/$out/offline_report.json"
}
train main --replays rtc_s4 rtc_var --val-replays rtc_s4

gate=$(python3 -c "import json,sys; print(int(json.load(open(sys.argv[1]))['pass']))" "$HEADS/main/offline_report.json")
COMMON=(--max-guidance-weight 2 --rtc-execution-horizon 6 --return-latent)
if [[ $gate == 1 || $FORCE == 1 ]]; then
  stamp "4 closed loop (gate $gate, FORCE $FORCE)"
  bash run_dynamic.sh corr_c1 rtc 10 0.175 "${COMMON[@]}" || die "corr_c1"
  CORR_HEAD="$HEADS/main/best.pt" RUNNER_ARGS="--control-mode corrected_rtc_queue" \
    bash run_dynamic.sh corr_c2 rtc 10 0.175 "${COMMON[@]}" || die "corr_c2"
  python3 report_dynamic.py "$RUNS/corr_c1" "$RUNS/corr_c2" --ref "$RUNS/corr_c1" --speeds $SPEEDS \
    --out "$RUNS/correction_report.md"
  CORR_HEAD="$HEADS/main/best.pt" CORR_DELAY_TICKS=1 RUNNER_ARGS="--control-mode corrected_rtc_queue" \
    bash run_dynamic.sh corr_c3 rtc 10 0.175 "${COMMON[@]}" || die "corr_c3"
  python3 report_dynamic.py "$RUNS/corr_c1" "$RUNS/corr_c2" "$RUNS/corr_c3" --ref "$RUNS/corr_c1" \
    --speeds $SPEEDS --out "$RUNS/correction_report.md"
else
  stamp "4 closed loop skipped: offline gate failed (FORCE=1 to run anyway)"
fi

stamp "5 ablations (offline)"
replay sync_s4 --mode sync --K 10 --delay 4 --step 4 --seed 2
train sync --replays sync_s4 --val-replays rtc_s4
train tail --replays rtc_s4 rtc_var --val-replays rtc_s4 --tail
stamp "done"
