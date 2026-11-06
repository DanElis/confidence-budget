#!/usr/bin/env bash
# H1 under conflict between the previous chunk and the current observation.
#
# Two ways to make RTC guidance actually change the chunk:
#   shift  - the leftover is offset by sigma std of the normalised actions;
#   stale  - the previous chunk is planned from an observation L frames too old.
# Each level is run with both guidance variants. Already finished runs are
# skipped, so the script can be restarted after an interruption.
#
# Usage (from experiments/h1):
#   tmux new -s sweep
#   bash scripts/run_sweep.sh            # about 3-4 h on one A100
#   EPISODES=30 bash scripts/run_sweep.sh   # quicker, wider CIs

set -u
cd "$(dirname "$0")/.."

EPISODES=${EPISODES:-100}
NBOOT=${NBOOT:-1000}
GUIDANCES=${GUIDANCES:-"exact lerobot"}
SHIFTS=${SHIFTS:-"0.25 0.5 1.0 2.0"}
STALES=${STALES:-"25 50 100"}
RUNS=runs
mkdir -p "$RUNS"
LOG="$RUNS/sweep.log"

run() {  # name, extra collect args...
  local name=$1; shift
  if [[ -f "$RUNS/$name/summary.json" ]]; then
    echo "$(date '+%F %T') skip $name (done)" | tee -a "$LOG"
    return
  fi
  echo "$(date '+%F %T') start $name: $*" | tee -a "$LOG"
  if uv run python scripts/collect_h1.py --full-dataset --episodes "$EPISODES" --out "$RUNS/$name.npz" "$@" \
       > "$RUNS/$name.log" 2>&1 \
     && uv run python scripts/analyze_h1.py "$RUNS/$name.npz" --n-boot "$NBOOT" >> "$RUNS/$name.log" 2>&1; then
    echo "$(date '+%F %T') done  $name: $(grep -m1 '^H1' "$RUNS/$name.log")" | tee -a "$LOG"
  else
    echo "$(date '+%F %T') FAIL  $name, see $RUNS/$name.log" | tee -a "$LOG"
  fi
}

names=()
for g in $GUIDANCES; do
  run "h1_$g" --guidance "$g"; names+=("h1_$g")
  for s in $SHIFTS; do
    run "h1_shift${s}_$g" --guidance "$g" --prev-shift "$s"; names+=("h1_shift${s}_$g")
  done
  for l in $STALES; do
    run "h1_stale${l}_$g" --guidance "$g" --prev-source stale --stale-frames "$l"; names+=("h1_stale${l}_$g")
  done
done

uv run python scripts/sweep_table.py "${names[@]/#/$RUNS/}" --out "$RUNS/sweep.md" | tee -a "$LOG"
echo "$(date '+%F %T') sweep finished, table in $RUNS/sweep.md" | tee -a "$LOG"
