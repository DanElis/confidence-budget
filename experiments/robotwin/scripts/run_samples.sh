#!/usr/bin/env bash
# Dispersion across noise samples (as in AAC) as a reliability signal. From experiments/robotwin:
#
#   tmux new -s rt
#   bash scripts/run_samples.sh          # ~8 h, 25 episodes/task, K=8 samples per inference
#
# Same tasks, seeds and executed noise (sample 0) as runs/sync_s25 and runs/rtc_s25_d6, so the robot
# should do the same thing: the script reports how many episode outcomes are reproduced. Each chunk
# also logs the dispersion of K samples per action index, the entropy and the executed plan.
# At the end scripts/analyze_signals.py writes runs_samples/signals_report.md:
#   A - does dispersion in the first chunks separate failing episodes;
#   B - (sync only) is the plan revised more where the old chunk's dispersion was high.
#
# Then adaptive chunking with h* = first index where the dispersion reaches eta (as in AAC). eta is set
# without looking at outcomes, so that the mean h* equals a constant that has been run already
# (scripts/calibrate_dispersion.py); the comparison isolates adaptivity at equal average horizon:
#   adaptive_disp25_k8          synchronous, s = h*;              vs runs/sync_s25 (s = 25)
#   budget_disp40_lam0.5_d6_k8  budget, d = 6, RTC, lambda = 0.5; vs runs/budget_const40_lam0.5_d6
# eta for the budget comes from the RTC run (guidance changes the dispersion profile).
# runs_samples/adaptive_report.md: paired comparison with the constant counterparts.
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
echo "$(date '+%F %T') signals: $RUNS/signals_report.md"

# adaptive chunking ---------------------------------------------------------------------------------
[[ -f $RUNS/hstar_disp25.json ]] || $PY scripts/calibrate_dispersion.py "$RUNS/sync_s25_k$K" --target 25 --out "$RUNS/hstar_disp25.json" || return 1
[[ -f $RUNS/hstar_disp40.json ]] || $PY scripts/calibrate_dispersion.py "$RUNS/rtc_s25_d${D}_k$K" --target 40 --out "$RUNS/hstar_disp40.json" || return 1
EPISODES=$EPISODES run_eval "adaptive_disp25_k$K" "$TASKS" "${S[@]}" --planner adaptive --signal dispersion \
  --hstar-config "$RUNS/hstar_disp25.json" --delay 0 --guidance none
EPISODES=$EPISODES run_eval "budget_disp40_lam0.5_d${D}_k$K" "$TASKS" "${S[@]}" --planner budget --signal dispersion \
  --hstar-config "$RUNS/hstar_disp40.json" --lam 0.5 --delay "$D" --guidance lerobot
finish
$PY "$LIBERO_PRO/scripts/compare.py" --ref runs/sync_s25 runs/sync_s25 "$RUNS/adaptive_disp25_k$K" \
  runs/budget_const40_lam0.5_d$D "$RUNS/budget_disp40_lam0.5_d${D}_k$K" runs/rtc_s25_d$D --suites "$TASKS" \
  --out "$RUNS/adaptive_report.md"
$PY "$LIBERO_PRO/scripts/compare.py" --ref runs/budget_const40_lam0.5_d$D runs/budget_const40_lam0.5_d$D \
  "$RUNS/budget_disp40_lam0.5_d${D}_k$K" --suites "$TASKS" >> "$RUNS/adaptive_report.md"
echo "$(date '+%F %T') done: $RUNS/signals_report.md, $RUNS/adaptive_report.md"
}

mkdir -p "$RUNS"
main 2>&1 | tee -a "$RUNS/samples.log"
