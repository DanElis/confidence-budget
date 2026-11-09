#!/usr/bin/env bash
# LIBERO-Dynamic with openpi's pi05_libero (H = 10, SERVER=openpi of run_dynamic.sh, openpi_dynamic_server.py), in the
# order of the plan: sync / async, then RTC and AAC; the step budgets follow (separate stage).
#
#   bash openpi_dynamic.sh calibrate   latency of one request on this GPU, without and with RTC (2 items x 2 runs)
#   bash openpi_dynamic.sh base        sync K = 10, async K in ASYNC_K, at the measured plain latency
#   bash openpi_dynamic.sh rtc         RTC of the paper (Jacobian, beta = 5), K = 10, at the measured RTC latency
#   bash openpi_dynamic.sh aac         AAC (run_pure_aac.sh): sync and with delay, each against a constant step of the
#                                      same request rate; N = 20 samples (their extra time is not added to the delay)
#   bash openpi_dynamic.sh all         all of the above, in this order
#
# Fixed before the run:
#   latency: the median server time of a request in the calibration runs (s), plain for sync / async / AAC, with RTC for
#            RTC; written to $RUNS/latency.txt and used by every later stage (a restart reuses it).
#   base:    the reference is sync K = 10; each async K against it, paired Δ over 600 items at v = 0.06, 95% CI
#            (Bonferroni over ASYNC_K for a claim). Underflow of the queue (queue_underflow_steps) makes a K unusable.
#   rtc:     against sync K = 10 and against async at the closest request rate, both secondary to base.
#   aac:     as run_pure_aac.sh: AAC against its constant control of the same rate, two comparisons, Bonferroni.
# From experiments/libero_dynamic, after libero_pro/scripts/setup_openpi.sh and setup.sh:
#   tmux new -d -s dyn_openpi "bash openpi_dynamic.sh all"
# SPEEDS=0.06 ASYNC_K="6 8 10" SERVERS=4 WORKERS=8 PORT=8124 RUNS=runs/openpi by default; finished speeds are skipped.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export SERVER=openpi
export RUNS=${RUNS:-$HERE/runs/openpi}
mkdir -p "$RUNS" && RUNS="$(cd "$RUNS" && pwd)" && export RUNS
export SPEEDS=${SPEEDS:-0.06}
ASYNC_K=${ASYNC_K:-"6 8 10"}
LAT=$RUNS/latency.txt
LOG=$RUNS/openpi_dynamic.log

median_ms() {  # median_ms RUN -> median server time per request from signals_*.jsonl, skipping the compiling calls
  python3 - "$1" <<'PY'
import json, statistics, sys
from pathlib import Path
rows = [json.loads(l) for f in sorted(Path(sys.argv[1]).glob("signals_*.jsonl")) for l in f.read_text().splitlines() if l.strip()]
ms = sorted(r["infer_ms"] for r in rows)[: max(1, int(0.95 * len(rows)))]  # the slowest 5% include JIT compilation
print(f"{statistics.median(ms):.1f}" if ms else "nan")
PY
}

ticks() { awk -v s="$1" 'BEGIN { t = s * 20; d = int(t); if (d < t) d++; print (d < 1 ? 1 : d) }'; }  # ceil(s * 20 Hz)

calibrate() {
  if [[ -f $LAT ]]; then echo "$(date '+%F %T') latency: $(cat "$LAT") (kept)"; return 0; fi
  local c=$RUNS/calibrate plain rtc
  LIMIT=2 RUNS=$c bash run_dynamic.sh plain sync 10 0.1 || return 1
  LIMIT=2 RUNS=$c bash run_dynamic.sh rtc rtc 10 0.1 || return 1
  plain=$(median_ms "$c/plain"); rtc=$(median_ms "$c/rtc")
  [[ $plain != nan && $rtc != nan ]] || { echo "no timing in $c/*/signals_*.jsonl"; return 1; }
  awk -v p="$plain" -v r="$rtc" 'BEGIN { printf "%.3f %.3f\n", p / 1000, r / 1000 }' > "$LAT"
  echo "$(date '+%F %T') latency: plain ${plain} ms, RTC ${rtc} ms -> $(cat "$LAT") s; d = $(ticks "$(awk '{print $1}' "$LAT")") / $(ticks "$(awk '{print $2}' "$LAT")") ticks"
}

latency() { [[ -f $LAT ]] || { echo "no $LAT: run calibrate first"; exit 1; }; awk -v i="$1" '{ print $i }' "$LAT"; }

base() {
  local lat k
  lat=$(latency 1)
  bash run_dynamic.sh sync_k10 sync 10 "$lat" || return 1
  for k in $ASYNC_K; do bash run_dynamic.sh "async_k$k" async "$k" "$lat" || return 1; done
  python3 report_dynamic.py "$RUNS/sync_k10" $(for k in $ASYNC_K; do echo "$RUNS/async_k$k"; done) \
    --ref "$RUNS/sync_k10" --speeds $SPEEDS --out "$RUNS/base_report.md"
}

rtc() {
  bash run_dynamic.sh rtc_k10 rtc 10 "$(latency 2)" || return 1
  python3 report_dynamic.py "$RUNS/sync_k10" "$RUNS/rtc_k10" $(for k in $ASYNC_K; do echo "$RUNS/async_k$k"; done) \
    --ref "$RUNS/sync_k10" --speeds $SPEEDS --out "$RUNS/rtc_report.md"
}

aac() {
  local lat
  lat=$(latency 1)
  LATENCY=$lat D=$(ticks "$lat") bash run_pure_aac.sh
}

main() {
  echo "$(date '+%F %T') openpi_dynamic $*: speeds=$SPEEDS async K=$ASYNC_K runs=$RUNS"
  git -C "$HERE" log -1 --oneline
  bash patch_runner.sh || exit 1
  local stage
  for stage in "$@"; do
    case $stage in
      calibrate) calibrate || exit 1 ;;
      base) base || exit 1 ;;
      rtc) rtc || exit 1 ;;
      aac) aac || exit 1 ;;
      all) calibrate && base && rtc && aac || exit 1 ;;
      *) echo "unknown stage $stage"; exit 1 ;;
    esac
  done
  echo "$(date '+%F %T') done: $*"
}

main "${@:-all}" 2>&1 | tee -a "$LOG"
