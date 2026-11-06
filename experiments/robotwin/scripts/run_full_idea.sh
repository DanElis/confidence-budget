#!/usr/bin/env bash
# The current idea on RoboTwin 2.0: adaptive replanning step from plan agreement + confidence-weighted RTC mask +
# coherent sample, K samples in one call with a shared prefix. From experiments/robotwin:
#
#   tmux new -s full "bash scripts/run_full_idea.sh"     # ~1.5 h with 25 episodes/task
#
# Same 8 tasks, seeds and executed noise of sample 0 as runs/ (sync_s25, rtc_*_d6, budget_*), so the result is paired
# with them. Fixed in advance, nothing tuned on these episodes: d = 6, K = 4, eta = 3 (agreement threshold),
# s in [d, H - d], mask to d + 5 (the best RTC window here), sigma = 1 * conf_ref.
# RULE=aac (default): on RoboTwin the agreement z is flat along the chunk (h* is 0 or H), the dispersion
# profile is not (smoke: h* 1..37), so h* = AAC Eq. 5 on the dispersion; RULE=agree uses z > ETA.
# conf_ref = median dispersion of the logged unguided samples of runs_samples/sync_s25_k8 (no outcomes used).
set -u
cd "$(dirname "$0")/.."
HERE=$(pwd)
LIBERO_PRO=$HERE/../libero_pro
export PY=${PY:-"uv run --project $LIBERO_PRO python"}
export EVAL_SCRIPT=scripts/eval_robotwin.py
export RUNS=${RUNS:-runs}
export JOBS=${JOBS:-1}
BATCH=${BATCH:-8}
K=${K:-4}
D=${D:-6}
ETA=${ETA:-3}
RULE=${RULE:-aac}   # aac: h* from the growth of the samples' dispersion; agree: plan agreement z > ETA
export MIN_FREE_MB=${MIN_FREE_MB:-40000}
export RESERVE_S=${RESERVE_S:-240}
export SUITE_TIMEOUT=${SUITE_TIMEOUT:-5h}
EPISODES=${EPISODES:-25}
TASKS=${TASKS:-"lift_pot,move_can_pot,place_a2b_left,place_bread_skillet,place_fan,place_mouse_pad,stamp_seal,turn_switch"}
source "$LIBERO_PRO/scripts/stages/lib.sh"

conf_ref() {
  python3 - <<'PY'
import glob, json, statistics
vals = [v for f in glob.glob("runs_samples/sync_s25_k8/*_chunks.json") for c in json.load(open(f))
        if "dispersion" in c for v in c["dispersion"]]
print(f"{statistics.median(vals):.5g}" if vals else "")
PY
}

main() {
trap 'trap - INT TERM; kill 0' INT TERM
REF=${CONF_REF:-$(conf_ref)}
if [[ -z $REF ]]; then echo "no runs_samples/sync_s25_k8 to take conf_ref from; set CONF_REF"; exit 1; fi
if [[ $RULE == aac ]]; then SIG=(--signal dispersion --hstar-rule "kind=aac,min_h=1"); NAME="full_aac_d${D}_k$K"
else SIG=(--signal agreement --hstar-rule "kind=above,eta=$ETA,k=1,min_h=1"); NAME="full_eta${ETA}_d${D}_k$K"; fi
echo "$(date '+%F %T') full idea: rule=$RULE d=$D K=$K eta=$ETA conf_ref=$REF tasks=$TASKS episodes=$EPISODES batch=$BATCH"
git log -1 --oneline
EPISODES=$EPISODES run_eval "$NAME" "$TASKS" --batch-size "$BATCH" --delay "$D" --guidance lerobot \
  --n-samples "$K" --shared-prefix --select coherent --mask-mode rtc_x_conf --conf-sigma 1 --conf-ref "$REF" \
  "${SIG[@]}" --planner adaptive --s-min "$D" --mask-r 5
finish
SUITES=$(echo "$TASKS")
REFS=()
for r in sync_s25 rtc_s25_m11_d6 rtc_s25_d6 naive_s25_d6 budget_const40_lam0.5_d6; do
  [[ -d $RUNS/$r ]] && REFS+=("$RUNS/$r")
done
$PY "$LIBERO_PRO/scripts/compare.py" --ref "$RUNS/sync_s25" "${REFS[@]}" "$RUNS/$NAME" \
  --suites "$SUITES" --out "$RUNS/${NAME}_report.md"
}

main 2>&1 | tee -a "$RUNS/full_idea.log"
