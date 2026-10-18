# Shared helpers for the stage scripts. Source from experiments/libero_pro:  source scripts/stages/lib.sh
#
#   run_eval NAME SUITES EXTRA_ARGS...   evaluate one configuration into runs/NAME, one process per suite,
#                                         at most $JOBS processes at a time (the simulator is CPU-bound,
#                                         so several processes share one GPU well; each needs ~10 GB of GPU memory)
#   PY                                    python launcher (default "uv run python"; use ".venv/bin/python"
#                                         after "uv sync --no-sources")
set -u
PY=${PY:-"uv run python"}
JOBS=${JOBS:-4}
EPISODES=${EPISODES:-10}
RUNS=${RUNS:-runs}
mkdir -p "$RUNS"

Y_LEVELS="pos:y0.1,pos:y0.2,pos:y0.3"            # tuning
X_LEVELS="libero_object,pos:x0.1,pos:x0.2,pos:x0.3"   # final report

suite_dir_name() {  # pos:x0.2 -> libero_object_pos_x0.2
  case "$1" in pos:*) echo "libero_object_pos_${1#pos:}";; *) echo "$1";; esac
}

run_eval() {
  local name=$1 suites=$2; shift 2
  local out="$RUNS/$name"
  mkdir -p "$out"
  local s
  for s in ${suites//,/ }; do
    if [[ -f "$out/$(suite_dir_name "$s").json" ]]; then
      echo "$(date '+%F %T') skip $name/$s"
      continue
    fi
    while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
    echo "$(date '+%F %T') start $name/$s: $*"
    $PY scripts/eval_libero.py --suites "$s" --episodes-per-task "$EPISODES" --out "$out" "$@" \
      > "$out/$(suite_dir_name "$s").log" 2>&1 \
      && echo "$(date '+%F %T') done  $name/$s: $(grep -o 'success [0-9.]*%' "$out/$(suite_dir_name "$s").log" | tail -1)" \
      || echo "$(date '+%F %T') FAIL  $name/$s, see $out/$(suite_dir_name "$s").log" &
  done
}

finish() { wait; }

suites_csv() {  # pos:y0.1,pos:y0.2 -> libero_object_pos_y0.1,libero_object_pos_y0.2
  local out="" s
  for s in ${1//,/ }; do out="$out,$(suite_dir_name "$s")"; done
  echo "${out#,}"
}
