#!/usr/bin/env bash
# RTC transition mask on LIBERO-Dynamic as a pair: first the constant transition length L0 (control), then the length
# growing with the old plan's deviation, L = L0 * delta / REF with REF = the control's MEAN deviation (logged signals
# only, no outcomes), so that the mean length stays near L0 (the deviation is right-skewed: with the median the mean L
# came out ~1.4 * L0 in the smoke run); then the paired report. From experiments/libero_dynamic:
#
#   tmux new -d -s trans "SPEEDS=0.06 bash run_transition_pair.sh 2>&1 | tee runs/transition_pair.log"
#
# Environment: as run_dynamic.sh (SPEEDS, SERVERS, WORKERS, PORT, LIMIT, RUNS) plus L0 (6), BETA (max guidance weight,
# 2 as the weakened RTC runs), K (10: the runner replans every d steps, as rtc_k10_b2_w6), LATENCY (0.175: one extra
# expert pass over plain RTC's 0.170, both are d = 4 at 20 Hz), REF (given: skip the calibration).
# Finished speeds are skipped by run_dynamic.sh, so the script can be restarted.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS=${RUNS:-$HERE/runs}
mkdir -p "$RUNS" && RUNS="$(cd "$RUNS" && pwd)"  # absolute before the cd below (a relative RUNS is from the caller)
export RUNS
cd "$HERE"
L0=${L0:-6} BETA=${BETA:-2} K=${K:-10} LATENCY=${LATENCY:-0.175}
export SPEEDS=${SPEEDS:-"0.03 0.06 0.09"}
CTRL=trans_c${L0}_b$BETA
ADA=trans_a${L0}_b$BETA
COMMON=(--mask-mode transition --trans-len "$L0" --max-guidance-weight "$BETA")

bash run_dynamic.sh "$CTRL" rtc "$K" "$LATENCY" "${COMMON[@]}" --trans-power 0 || exit 1
REF=${REF:-$(python3 - "$RUNS/$CTRL" <<'PY'
import json
import statistics
import sys
from pathlib import Path

deltas = []
for f in Path(sys.argv[1]).glob("signals_*.jsonl"):
    for line in f.read_text().splitlines():
        if line.strip() and json.loads(line).get("delta") is not None:
            deltas.append(json.loads(line)["delta"])
print(f"{statistics.mean(deltas):.6g}" if deltas else "")
PY
)}
[[ -n $REF ]] || { echo "no deviations logged in $RUNS/$CTRL/signals_*.jsonl"; exit 1; }
echo "$(date '+%F %T') --trans-ref $REF (mean deviation of $CTRL)"
bash run_dynamic.sh "$ADA" rtc "$K" "$LATENCY" "${COMMON[@]}" --trans-power 1 --trans-ref "$REF" || exit 1
python3 report_dynamic.py "$RUNS/$CTRL" "$RUNS/$ADA" --ref "$RUNS/$CTRL" --speeds $SPEEDS
