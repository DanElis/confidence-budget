#!/usr/bin/env bash
# Night run: coherent sample selection, confidence-weighted RTC mask and agreement-based adaptive s on the
# LIBERO-Pro tuning levels (y0.1-y0.3, EPISODES per task), at delays 8 and 16 (20 Hz: 0.4 s and 0.8 s).
#
# 0. Smoke (1 task, 2 episodes) of every new code path; stops if any fails. Its dispersion gives conf_ref.
# 1. Core at d = 8, then d = 16 (queued in priority order, $JOBS processes):
#      sync s=10 (d = 0) | naive | RTC (mask d+5) | RTC with K samples (reference, sample 0) |
#      RTC + coherent sample | adaptive s from plan agreement (z = 1.5) + coherent
# 2. Extras: adaptive z = 3 | budget (agreement h*, lambda 0.5) | coherent + confidence mask (sigma = 1) |
#      naive + coherent | coherent with constant s = mean s of adaptive z = 1.5 (matched pair)
# Main comparison fixed in advance: coh and ada_z1.5 vs rtck at each d (same K samples, same noise).
# Tuning levels only: confirm whatever wins on x (report levels) and new seeds afterwards.
#
# Usage (from experiments/libero_pro):  tmux new -s night "bash scripts/stages/night_coherence.sh"
#   K=4 EPISODES=10 JOBS=4 by default; re-running skips finished suites.
#   Report at any time:  uv run python scripts/report_coherence.py --runs runs --out runs/coherence_report.md
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh
K=${K:-4}
EPISODES=${EPISODES:-10}
DELAYS=${DELAYS:-"8 16"}
MIN_FREE_K=${MIN_FREE_K:-20000}   # K samples per environment: batch BATCH_K * K sequences per policy call
BATCH_K=${BATCH_K:-5}
LEVELS=$Y_LEVELS
SMOKE=$RUNS/cs_smoke

stride_of() { echo $(( $1 > 10 ? $1 : 10 )); }   # s = max(d, 10)

smoke() {
  mkdir -p "$SMOKE"
  local common=(--suites pos:y0.1 --task-ids 0 --episodes-per-task 2 --batch-size 2 --delay 8 --n-samples "$K")
  local name args
  for name in rtck guided naive; do
    case $name in
      rtck)   args=(--stride 10 --mask-end 13 --guidance lerobot) ;;
      guided) args=(--planner adaptive --signal agreement --hstar-rule "kind=above,eta=1.5,k=1,min_h=1"
                    --s-min 8 --s-max 25 --mask-r 5 --select coherent --mask-mode rtc_x_conf --conf-sigma 0.3
                    --guidance lerobot) ;;
      naive)  args=(--planner budget --signal agreement --hstar-rule "kind=above,eta=1.5,k=1,min_h=1" --lam 0.5
                    --select coherent --guidance none) ;;
    esac
    [[ -f "$SMOKE/$name/libero_object_pos_y0.1.json" ]] && continue
    echo "$(date '+%F %T') smoke $name"
    local t0=$SECONDS
    if ! $PY scripts/eval_libero.py "${common[@]}" "${args[@]}" --out "$SMOKE/$name" > "$SMOKE/$name.log" 2>&1; then
      echo "$(date '+%F %T') SMOKE FAILED: $name, see $SMOKE/$name.log"; tail -20 "$SMOKE/$name.log"; return 1
    fi
    echo "$(date '+%F %T') smoke $name ok ($((SECONDS - t0)) s): $(grep -o 'success [0-9.]*%' "$SMOKE/$name.log" | tail -1)"
  done
}

conf_ref() {  # median dispersion of the smoke reference run
  $PY - "$SMOKE/rtck/libero_object_pos_y0.1_chunks.json" <<'PY'
import json, sys
import numpy as np
chunks = json.load(open(sys.argv[1]))
print(f"{np.median(np.concatenate([np.asarray(c['dispersion']) for c in chunks if 'dispersion' in c])):.5g}")
PY
}

mean_stride() {  # mean s over the episodes of a finished run, rounded
  $PY - "$RUNS/$1" <<'PY'
import glob, json, sys
import numpy as np
eps = [e for f in glob.glob(f"{sys.argv[1]}/libero_object_pos_y*.json") if not f.endswith("_chunks.json")
       for e in json.load(open(f))["episodes"]]
vals = [e["mean_stride"] for e in eps if e.get("mean_stride") is not None]
print(int(round(np.mean(vals))) if vals else "")
PY
}

ada_args() {  # ada_args d z
  local d=$1 s; s=$(stride_of "$1")
  echo --planner adaptive --signal agreement --hstar-rule "kind=above,eta=$2,k=1,min_h=1" \
       --s-min "$s" --s-max $((s + 15)) --mask-r 5 --select coherent --guidance lerobot --delay "$d" --n-samples "$K" --batch-size "$BATCH_K"
}

core() {  # core d
  local d=$1 s m; s=$(stride_of "$d"); m=$((d + 5))
  run_eval "cs_naive_s${s}_d$d" "$LEVELS" --stride "$s" --delay "$d" --guidance none
  run_eval "cs_rtc_s${s}_m${m}_d$d" "$LEVELS" --stride "$s" --mask-end "$m" --delay "$d" --guidance lerobot
  MIN_FREE_MB=$MIN_FREE_K run_eval "cs_rtck_s${s}_d$d" "$LEVELS" --stride "$s" --mask-end "$m" --delay "$d" \
    --guidance lerobot --n-samples "$K" --batch-size "$BATCH_K"
  MIN_FREE_MB=$MIN_FREE_K run_eval "cs_coh_s${s}_d$d" "$LEVELS" --stride "$s" --mask-end "$m" --delay "$d" \
    --guidance lerobot --n-samples "$K" --batch-size "$BATCH_K" --select coherent
  # shellcheck disable=SC2046
  MIN_FREE_MB=$MIN_FREE_K run_eval "cs_ada_z1.5_d$d" "$LEVELS" $(ada_args "$d" 1.5)
}

extras() {  # extras d conf_ref
  local d=$1 ref=$2 s m; s=$(stride_of "$d"); m=$((d + 5))
  # shellcheck disable=SC2046
  MIN_FREE_MB=$MIN_FREE_K run_eval "cs_ada_z3_d$d" "$LEVELS" $(ada_args "$d" 3)
  MIN_FREE_MB=$MIN_FREE_K run_eval "cs_budget_z1.5_lam0.5_d$d" "$LEVELS" --planner budget --signal agreement \
    --hstar-rule "kind=above,eta=1.5,k=1,min_h=1" --lam 0.5 --r-min 5 --select coherent --guidance lerobot \
    --delay "$d" --n-samples "$K" --batch-size "$BATCH_K"
  MIN_FREE_MB=$MIN_FREE_K run_eval "cs_cohconf1_d$d" "$LEVELS" --stride "$s" --mask-end "$m" --delay "$d" \
    --guidance lerobot --n-samples "$K" --batch-size "$BATCH_K" --select coherent --mask-mode rtc_x_conf --conf-sigma 1 --conf-ref "$ref"
  MIN_FREE_MB=$MIN_FREE_K run_eval "cs_naivecoh_s${s}_d$d" "$LEVELS" --stride "$s" --mask-end "$m" --delay "$d" \
    --guidance none --n-samples "$K" --batch-size "$BATCH_K" --select coherent
}

matched() {  # matched d: coherent with constant s = mean s of ada_z1.5 (needs that run finished)
  local d=$1 s; s=$(mean_stride "cs_ada_z1.5_d$d")
  if [[ -z $s ]]; then echo "$(date '+%F %T') no mean s for cs_ada_z1.5_d$d, skip matched"; return; fi
  (( s < d )) && s=$d
  MIN_FREE_MB=$MIN_FREE_K run_eval "cs_cohmatch_d$d" "$LEVELS" --stride "$s" --mask-end $((d + 5)) --delay "$d" \
    --guidance lerobot --n-samples "$K" --batch-size "$BATCH_K" --select coherent
}

main() {
trap 'trap - INT TERM; kill 0' INT TERM
echo "$(date '+%F %T') night_coherence: K=$K EPISODES=$EPISODES DELAYS=$DELAYS JOBS=$JOBS levels=$LEVELS"
git log -1 --oneline
smoke || exit 1
REF=$(conf_ref)
echo "$(date '+%F %T') conf_ref (median dispersion, smoke) = $REF"

run_eval cs_sync_s10 "$LEVELS" --stride 10 --delay 0 --guidance none
for d in $DELAYS; do core "$d"; done
for d in $DELAYS; do extras "$d" "$REF"; done
finish
for d in $DELAYS; do matched "$d"; done
finish
$PY scripts/report_coherence.py --runs "$RUNS" --delays $DELAYS --out "$RUNS/coherence_report.md"
}

main 2>&1 | tee -a "$RUNS/night_coherence.log"
