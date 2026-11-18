#!/usr/bin/env bash
# MimicGen Threading_D0 with pi0.5 vdb_pi05_threading_d0 (H = 50, 40% at s = 10 synchronous on 200 episodes of its own
# evaluation): checks, latency, and the constant step without and with delay. It fixes d and the step grids of the
# later stages (AAC, RTC gate, budgets: experiments/mimicgen/README.md). Descriptive: no gain is claimed here.
#
#   0 preflight  the client venv's imports, scene reproducibility (mimicgen_env.py --check), smoke: 2 episodes, s = 10
#   1 latency    median request time of one client, plain and with RTC of the paper -> d    $RUNS/latency.txt
#   2 sync       d = 0, s in SYNC_STEPS (s = 10 is the setting of the 40%)                 mg_sync_s<s>
#   3 naive      d = DELAY, s in STEPS                                                     mg_naive_s<s>_d<d>
#   4 rtc        RTC of the paper (VJP, beta = 5), d = DELAY, s in STEPS, mask to H - s    mg_rtc_s<s>_d<d>
#   report       compare.py: everything against sync s = 10; naive and RTC against sync and naive of the same s;
#                check_pairing.py: the same scene behind each item in every run      $RUNS/mimicgen_base_report.md
#
# Scenes: the tuning sets y0.1-y0.3 (labels, mimicgen_env.py), EPISODES_PER_TASK scenes in each of 10 blocks, seed 1.
# DELAY is not taken from step 1 automatically: the run prints d and goes on with DELAY (default 2: ~0.1 s at 20 Hz).
# If step 1 gives another d, stop and rerun with DELAY=<d>: finished suites are kept.
#
# Usage (from experiments/libero_pro):
#   tmux new -d -s mg_base "source ../mimicgen/profile.sh && bash scripts/stages/mimicgen_base.sh"
# DELAY=2 SYNC_STEPS="5 10 25" STEPS="5 10 25" EPISODES_PER_TASK=10 SEED=1 PORT=8900 JOBS=6 by default.
SUITE_TIMEOUT=${SUITE_TIMEOUT:-12h}
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh && source scripts/stages/openpi_lib.sh
[[ ${EVAL_ENV:-} == mimicgen ]] || { echo "source ../mimicgen/profile.sh first"; exit 1; }
JOBS=${JOBS:-6}   # two clients per server
EPISODES=${EPISODES_PER_TASK:-10}
DELAY=${DELAY:-2}
SYNC_STEPS=${SYNC_STEPS:-"5 10 25"}
STEPS=${STEPS:-"5 10 25"}
SEED=${SEED:-1}
PORT=${PORT:-8900}
LEVELS=${LEVELS:-"pos:y0.1 pos:y0.2 pos:y0.3"}
SERVER_LOG=mimicgen_base
REPORT=$RUNS/mimicgen_base_report.md
MG=../mimicgen

preflight() {
  $OPENPI_PY -c "import robosuite, mimicgen, h5py, torch, msgpack, websockets; print('robosuite', robosuite.__version__)" \
    || { echo "$(date '+%F %T') client venv lacks a package: see experiments/mimicgen/README.md"; return 1; }
  $OPENPI_PY $MG/mimicgen_env.py --check > "$RUNS/mimicgen_check.log" 2>&1 \
    || { echo "$(date '+%F %T') SCENE CHECK FAILED"; tail -20 "$RUNS/mimicgen_check.log"; return 1; }
  echo "$(date '+%F %T') $(tail -1 "$RUNS/mimicgen_check.log")"
}

main() {
trap 'trap - INT TERM; stop_servers; kill 0' INT TERM
trap stop_servers EXIT
echo "$(date '+%F %T') mimicgen_base: d=$DELAY sync=$SYNC_STEPS steps=$STEPS EPISODES=$EPISODES seed=$SEED levels=$LEVELS"
git log -1 --oneline
preflight || exit 1
start_servers || exit 1
smoke_eval mimicgen_base --stride 10 || exit 1
$OPENPI_PY $MG/latency.py --port "$PORT" --delay "$DELAY" --stride 10 --out "$RUNS/latency.txt" || exit 1
echo "$(date '+%F %T') latency (one client): $(tr '\n' ';' < "$RUNS/latency.txt"); running with DELAY=$DELAY"

local s runs=()
for s in $SYNC_STEPS; do per_level "mg_sync_s$s" --stride "$s"; runs+=("$RUNS/mg_sync_s$s"); done
for s in $STEPS; do per_level "mg_naive_s${s}_d$DELAY" --stride "$s" --delay "$DELAY"; runs+=("$RUNS/mg_naive_s${s}_d$DELAY"); done
for s in $STEPS; do
  per_level "mg_rtc_s${s}_d$DELAY" --stride "$s" --delay "$DELAY" --guidance paper
  runs+=("$RUNS/mg_rtc_s${s}_d$DELAY")
done
finish
stop_servers

{
  echo "# MimicGen base, d = $DELAY, seed $SEED, $LEVELS"
  echo; cat "$RUNS/latency.txt"; echo
  $OPENPI_PY $MG/check_pairing.py "${runs[@]}"; echo
  echo "## against sync s = 10"; $PY scripts/compare.py --ref "$RUNS/mg_sync_s10" "${runs[@]}"; echo
  for s in $STEPS; do
    echo "## s = $s: naive and RTC against sync of the same s (if run) and RTC against naive"
    [[ -d $RUNS/mg_sync_s$s ]] && $PY scripts/compare.py --ref "$RUNS/mg_sync_s$s" "$RUNS/mg_sync_s$s" \
        "$RUNS/mg_naive_s${s}_d$DELAY" "$RUNS/mg_rtc_s${s}_d$DELAY"
    $PY scripts/compare.py --ref "$RUNS/mg_naive_s${s}_d$DELAY" "$RUNS/mg_naive_s${s}_d$DELAY" "$RUNS/mg_rtc_s${s}_d$DELAY"
    echo
  done
} > "$REPORT" 2>&1
echo "$(date '+%F %T') done: $REPORT"
}

main 2>&1 | tee -a "$RUNS/mimicgen_base.log"
