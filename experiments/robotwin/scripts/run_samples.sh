#!/usr/bin/env bash
# Dispersion across noise samples (as in AAC) as a reliability signal. From experiments/robotwin:
#
#   tmux new -s rt
#   bash scripts/run_samples.sh          # ~4 h: sync s=25 and RTC s=25 (d=6), K=8 samples, 25 episodes/task
#
# Same tasks, seeds and executed noise (sample 0) as runs/sync_s25 and runs/rtc_s25_d6, so the robot
# should do the same thing: the script reports how many episode outcomes are reproduced. Each chunk
# also logs the dispersion of K samples per action index, the entropy and the executed plan.
# At the end scripts/analyze_signals.py writes runs_samples/signals_report.md:
#   A - does dispersion in the first chunks separate failing episodes;
#   B - (sync only) is the plan revised more where the old chunk's dispersion was high.
set -u
cd "$(dirname "$0")/.."
HERE=$(pwd)
LIBERO_PRO=$HERE/../libero_pro
export PY=${PY:-"uv run --project $LIBERO_PRO python"}
export EVAL_SCRIPT=scripts/eval_robotwin.py
export RUNS=${RUNS:-runs_samples}
export JOBS=${JOBS:-1}
BATCH=${BATCH:-8}
K=${K:-8}
export MIN_FREE_MB=${MIN_FREE_MB:-$((22000 + BATCH * 5000))}   # K samples per env: larger policy batch
export RESERVE_S=${RESERVE_S:-240}
export SUITE_TIMEOUT=${SUITE_TIMEOUT:-5h}
EPISODES=${EPISODES:-25}
D=${D:-6}
TASKS=${TASKS:-"lift_pot,move_can_pot,place_a2b_left,place_bread_skillet,place_fan,place_mouse_pad,stamp_seal,turn_switch"}
source "$LIBERO_PRO/scripts/stages/lib.sh"

main() {
trap 'trap - INT TERM; kill 0' INT TERM
echo "$(date '+%F %T') samples: K=$K tasks=$TASKS episodes=$EPISODES batch=$BATCH"
S=(--batch-size "$BATCH" --n-samples "$K" --record-entropy --record-plans ${EXTRA:-})
EPISODES=$EPISODES run_eval "sync_s25_k$K"    "$TASKS" "${S[@]}" --stride 25 --delay 0 --guidance none
EPISODES=$EPISODES run_eval "rtc_s25_d${D}_k$K" "$TASKS" "${S[@]}" --stride 25 --delay "$D" --guidance lerobot
finish

python3 - "$RUNS" "$K" "$D" <<'PY'
import glob, json, os, sys
runs, k, d = sys.argv[1:]
for new, old in ((f"sync_s25_k{k}", "sync_s25"), (f"rtc_s25_d{d}_k{k}", f"rtc_s25_d{d}")):
    same = total = 0
    for f in glob.glob(f"{runs}/{new}/*.json"):
        if f.endswith(("_chunks.json", "config.json")):
            continue
        g = f"runs/{old}/{os.path.basename(f)}"
        if not os.path.exists(g):
            continue
        a = {e["init_state"]: e["success"] for e in json.load(open(f))["episodes"]}
        b = {e["init_state"]: e["success"] for e in json.load(open(g))["episodes"]}
        for i in a.keys() & b.keys():
            total += 1
            same += a[i] == b[i]
    if total:
        print(f"{new}: {same}/{total} episode outcomes as in runs/{old} (sample 0 = the same noise)")
PY
$PY scripts/analyze_signals.py "$RUNS/sync_s25_k$K" "$RUNS/rtc_s25_d${D}_k$K" --out "$RUNS/signals_report.md"
echo "$(date '+%F %T') done: $RUNS/signals_report.md"
}

mkdir -p "$RUNS"
main 2>&1 | tee -a "$RUNS/samples.log"
