#!/usr/bin/env bash
# Overnight RoboTwin 2.0 run (from experiments/robotwin, after setup_robotwin.sh):
#
#   tmux new -s rt
#   bash scripts/run_night.sh                 # defaults below
#   EPISODES=25 bash scripts/run_night.sh     # half the episodes, ~half the time
#
# 0. sanity: synchronous s=50 (the checkpoint's own protocol) on beat_block_hammer, 10 episodes.
#    The README of SidneyXie/pi05_robotwin reports 93% there; the run stops if fewer than 5/10
#    succeed (wrong setup, no point running the grid).
# 1. grid on 8 mid-success tasks (lift_pot ... turn_switch, 41-60% in the README), $EPISODES each,
#    delay d=$D steps (25 Hz, so 6 = 240 ms), in priority order; every task of a configuration is its
#    own process, $JOBS processes at a time:
#      P1  sync s=50 | sync s=25 | naive s=25 | RTC s=25 (mask to H-s) | RTC s=25 short mask (d+5)
#          | RTC s=25 exact guidance
#      P2  naive s=10 | RTC s=10 short mask | budget with constant h* (30: lambda 0, 1; 40: lambda 0.5)
#    RTC s=25 and sync s=25 also record cross-attention entropy (for h* on RoboTwin later).
# 2. runs/report.md: success per task, paired difference to sync s=25 with 95% CI.
#
# Resumable: finished (configuration, task) pairs are skipped. Progress: runs/night.log
set -u
cd "$(dirname "$0")/.."
HERE=$(pwd)
LIBERO_PRO=$HERE/../libero_pro
export PY=${PY:-"uv run --project $LIBERO_PRO python"}
export EVAL_SCRIPT=scripts/eval_robotwin.py
export RUNS=${RUNS:-runs}
export JOBS=${JOBS:-3}
export MIN_FREE_MB=${MIN_FREE_MB:-24000}   # policy ~12 GB + simulators
export SUITE_TIMEOUT=${SUITE_TIMEOUT:-5h}
EPISODES=${EPISODES:-50}
D=${D:-6}
TASKS=${TASKS:-"lift_pot,move_can_pot,place_a2b_left,place_bread_skillet,place_fan,place_mouse_pad,stamp_seal,turn_switch"}
source "$LIBERO_PRO/scripts/stages/lib.sh"

main() {
trap 'trap - INT TERM; kill 0' INT TERM
echo "$(date '+%F %T') RoboTwin night: tasks=$TASKS episodes=$EPISODES d=$D jobs=$JOBS"

# 0. sanity ----------------------------------------------------------------------------
EPISODES=10 run_eval sanity_sync_s50 beat_block_hammer ${EXTRA:-} --stride 50 --delay 0 --guidance none
finish
ok=$(python3 -c "import json;r=json.load(open('$RUNS/sanity_sync_s50/beat_block_hammer.json'));print(sum(e['success'] for e in r['episodes']))" 2>/dev/null || echo 0)
echo "$(date '+%F %T') sanity: $ok/10 on beat_block_hammer (README: 93%)"
if (( ok < 5 )); then
  echo "STOP: the policy does not work in this setup; see $RUNS/sanity_sync_s50/beat_block_hammer.log"
  return 1
fi

# 1. grid --------------------------------------------------------------------------------
E=(--batch-size 8 ${EXTRA:-})   # EXTRA="--fake-sim --fake-policy" dry-runs the script without RoboTwin
EPISODES=$EPISODES run_eval sync_s50            "$TASKS" "${E[@]}" --stride 50 --delay 0 --guidance none
EPISODES=$EPISODES run_eval sync_s25            "$TASKS" "${E[@]}" --stride 25 --delay 0 --guidance none --record-entropy
EPISODES=$EPISODES run_eval naive_s25_d$D       "$TASKS" "${E[@]}" --stride 25 --delay "$D" --guidance none
EPISODES=$EPISODES run_eval rtc_s25_d$D         "$TASKS" "${E[@]}" --stride 25 --delay "$D" --guidance lerobot --record-entropy
EPISODES=$EPISODES run_eval rtc_s25_m$((D + 5))_d$D "$TASKS" "${E[@]}" --stride 25 --mask-end $((D + 5)) --delay "$D" --guidance lerobot
EPISODES=$EPISODES run_eval rtcexact_s25_d$D    "$TASKS" "${E[@]}" --stride 25 --delay "$D" --guidance exact
EPISODES=$EPISODES run_eval naive_s10_d$D       "$TASKS" "${E[@]}" --stride 10 --delay "$D" --guidance none
EPISODES=$EPISODES run_eval rtc_s10_m$((D + 5))_d$D "$TASKS" "${E[@]}" --stride 10 --mask-end $((D + 5)) --delay "$D" --guidance lerobot
for spec in "30 0" "30 1" "40 0.5"; do
  set -- $spec
  EPISODES=$EPISODES run_eval "budget_const$1_lam$2_d$D" "$TASKS" "${E[@]}" --planner budget \
    --hstar-rule "kind=const,c=$1" --lam "$2" --delay "$D" --guidance lerobot
done
finish

# 2. report --------------------------------------------------------------------------------
RUN_DIRS=$(ls -d "$RUNS"/*/ | grep -v -E "/(sanity|_smoke)" | tr '\n' ' ')
$PY "$LIBERO_PRO/scripts/compare.py" --ref "$RUNS/sync_s25" $RUN_DIRS --suites "$TASKS" --out "$RUNS/report.md"
echo "$(date '+%F %T') done: $RUNS/report.md"
}

mkdir -p "$RUNS"
main 2>&1 | tee -a "$RUNS/night.log"
