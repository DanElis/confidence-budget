#!/usr/bin/env bash
# Run a script of this directory in the RTC venv: ./run.sh [script.py] args...  (default run_points.py)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
SCRIPT=run_points.py
if [[ "${1:-}" == *.py ]]; then SCRIPT="$1"; shift; fi
exec uv run --project "$HERE/third_party/real-time-chunking-kinetix" python "$HERE/$SCRIPT" "$@"
