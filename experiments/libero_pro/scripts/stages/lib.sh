# Shared helpers for the stage scripts. Source from experiments/libero_pro:  source scripts/stages/lib.sh
#
#   run_eval NAME SUITES EXTRA_ARGS...   evaluate one configuration into runs/NAME, one process per suite,
#                                         at most $JOBS processes at a time (the simulator is CPU-bound,
#                                         so several processes share one GPU well; each needs ~14 GB of GPU memory;
#                                         a process starts only when MIN_FREE_MB is free, and is retried on CUDA OOM)
#   PY                                    python launcher (default "uv run python"; use ".venv/bin/python"
#                                         after "uv sync --no-sources")
set -u
PY=${PY:-"uv run python"}
JOBS=${JOBS:-4}
EPISODES=${EPISODES:-10}
RUNS=${RUNS:-runs}
MIN_FREE_MB=${MIN_FREE_MB:-18000}   # ~14 GB per process + margin
RESERVE_S=${RESERVE_S:-90}         # model load time before the next launch
OOM_RETRIES=${OOM_RETRIES:-3}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$RUNS"

Y_LEVELS="pos:y0.1,pos:y0.2,pos:y0.3"            # tuning
X_LEVELS="libero_object,pos:x0.1,pos:x0.2,pos:x0.3"   # final report

suite_dir_name() {  # pos:x0.2 -> libero_object_pos_x0.2
  case "$1" in pos:*) echo "libero_object_pos_${1#pos:}";; *) echo "$1";; esac
}

gpu_free_mb() {  # free memory of GPU 0 in MiB (huge number if nvidia-smi is unavailable)
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU:-0}" 2>/dev/null | head -1 || echo 999999
}

wait_for_gpu() {  # block until MIN_FREE_MB is free; RESERVE_S seconds between launches lets the model load
  local free
  while :; do
    free=$(gpu_free_mb); free=${free:-999999}
    (( free >= MIN_FREE_MB )) && break
    sleep 30
  done
}

eval_one() {  # eval_one OUT SUITE ARGS...: one suite, retried on CUDA OOM up to OOM_RETRIES times
  local out=$1 s=$2; shift 2
  local log="$out/$(suite_dir_name "$s").log" try
  for (( try = 0; try <= OOM_RETRIES; try++ )); do
    (( try > 0 )) && { echo "$(date '+%F %T') retry $try $(basename "$out")/$s after CUDA OOM"; sleep $((60 * try)); wait_for_gpu; }
    if $PY scripts/eval_libero.py --suites "$s" --episodes-per-task "$EPISODES" --out "$out" "$@" > "$log" 2>&1; then
      echo "$(date '+%F %T') done  $(basename "$out")/$s: $(grep -o 'success [0-9.]*%' "$log" | tail -1)"
      return 0
    fi
    grep -q "OutOfMemoryError\|CUDA out of memory" "$log" || break
  done
  echo "$(date '+%F %T') FAIL  $(basename "$out")/$s, see $log"
  return 1
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
    wait_for_gpu
    echo "$(date '+%F %T') start $name/$s: $*"
    eval_one "$out" "$s" "$@" &
    sleep "$RESERVE_S"  # let the new process take its GPU memory before the next free-memory check
  done
}

finish() { wait; }

suites_csv() {  # pos:y0.1,pos:y0.2 -> libero_object_pos_y0.1,libero_object_pos_y0.2
  local out="" s
  for s in ${1//,/ }; do out="$out,$(suite_dir_name "$s")"; done
  echo "${out#,}"
}
