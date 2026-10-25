#!/usr/bin/env bash
# Run run_points.py in the RTC venv. Arguments are passed through, see `./run.sh --help`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec uv run --project "$HERE/third_party/real-time-chunking-kinetix" python "$HERE/run_points.py" "$@"
