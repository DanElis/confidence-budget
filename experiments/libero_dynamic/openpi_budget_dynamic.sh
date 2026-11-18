#!/usr/bin/env bash
# Option budget (budget_v2.py, BUDGET.md) on LIBERO-Dynamic with openpi's pi05_libero (H = 10). Two variants:
#
#   bash openpi_budget_dynamic.sh s    step budget, naive only: pairs (s, -1), s in {2, 4, 6, 8}, d = 2 at the measured
#                                      plain latency; the constant control is naive s = 5 at equal request rate
#   bash openpi_budget_dynamic.sh sr   step and RTC window: (3,-1) (3,2) (3,4) (5,-1) (5,2) (7,-1), d = 3 at the measured
#                                      RTC latency for every pair (naive pairs pay the RTC latency too: conservative
#                                      for them); the control is s = 5 with r in {-1, 2} chosen on validation
#
# r is the RTC window after the d frozen actions (mask end d + r), r = -1 no guidance. A pair is chosen before
# generation from deployable features and held for 40 ticks. Splits by trajectory of the same 120 scenes (40 tasks x
# 3 levels): train 0,1 / validation 2 / test 3,4 (240 paired test episodes), v = 0.06.
# Fixed before the run (as BUDGET.md): primary = learned against the constant control at equal rate (within 5%),
# 95% CI above zero; secondary = learned against marginal-random of the same frequencies. The offline gates (V AUROC,
# value of state dependence) are computed and reported in fit_1/fit.md; FORCE=1 (default here) continues past them,
# FORCE=0 stops before validation and test.
#
# Needs runs/openpi/latency.txt (openpi_dynamic.sh calibrate). From experiments/libero_dynamic, in tmux:
#   OUT=runs/openpi_budget_s bash openpi_budget_dynamic.sh s --smoke-only     # smoke, then the same OUT for the run
#   tmux new -d -s budget_dyn "bash openpi_budget_dynamic.sh s && bash openpi_budget_dynamic.sh sr"
# Extra arguments go to budget_v2.py. FREE_MB (40000), GPU, OUT (runs/openpi_budget_<variant>).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VARIANT=${1:?usage: openpi_budget_dynamic.sh s|sr [budget_v2 args]}; shift
LAT=$HERE/runs/openpi/latency.txt
[[ -f $LAT ]] || { echo "no $LAT: run openpi_dynamic.sh calibrate first"; exit 1; }
case $VARIANT in
  s)  latency=$(awk '{print $1}' "$LAT"); delay=2; pairs="2:-1 4:-1 6:-1 8:-1" ;;
  sr) latency=$(awk '{print $2}' "$LAT"); delay=3; pairs="3:-1 3:2 3:4 5:-1 5:2 7:-1" ;;
  *) echo "unknown variant $VARIANT (s|sr)"; exit 1 ;;
esac
OPENPI_PY="$HERE/../libero_pro/third_party/openpi/.venv/bin/python"
[[ -x $OPENPI_PY ]] || { echo "no $OPENPI_PY: run libero_pro/scripts/setup_openpi.sh"; exit 1; }
# the learned policy (budget_q.py) runs inside the openpi server: fail now, not after hours of exploration
env -u VIRTUAL_ENV -u PYTHONPATH "$OPENPI_PY" -c "import torch" \
  || { echo "torch is missing in the openpi venv: the learned budget cannot run inside openpi_dynamic_server.py"; exit 1; }
force=()
[[ ${FORCE:-1} == 1 ]] && force=(--force)
export OUT=${OUT:-$HERE/runs/openpi_budget_$VARIANT} FREE_MB=${FREE_MB:-40000}
echo "$(date '+%F %T') openpi budget $VARIANT: d=$delay latency=$latency pairs=[$pairs] OUT=$OUT"
exec bash "$HERE/night_budget_v2.sh" --server openpi --delay "$delay" --latency "$latency" --pairs "$pairs" \
  --reference-stride 5 "${force[@]}" "$@"
