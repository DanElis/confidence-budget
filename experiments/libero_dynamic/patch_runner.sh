#!/usr/bin/env bash
# Minimal opt-in patch of the LIBERO-Dynamic client (scripts/eval_dynamic_libero_style.py @b66cb6b).
#
# The OpenPI client trims every returned chunk to --replan-steps, so the chunk length (and with it the replanning
# step under --prefetch-threshold) is fixed by the runner. Our adaptive step sets the length per chunk on the server
# (s + d), so with LIBERO_DYNAMIC_KEEP_CHUNK=1 the client keeps the whole returned chunk; --replan-steps then only
# acts as the minimum length. Without the variable the behaviour is unchanged (the paper's baselines).
# Idempotent; run by setup.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
F="$HERE/third_party/LIBERO-Dynamic/scripts/eval_dynamic_libero_style.py"
python3 - "$F" <<'PY'
import sys
from pathlib import Path

f = Path(sys.argv[1])
src = f.read_text()
old = "[: self.replan_steps"
new = '[: (None if __import__("os").environ.get("LIBERO_DYNAMIC_KEEP_CHUNK") == "1" else self.replan_steps)'
if new in src:
    print("patch_runner: already applied")
    sys.exit(0)
n = src.count(old)
if n != 2:
    sys.exit(f"patch_runner: expected 2 occurrences of {old!r} in {f}, found {n}; upstream changed?")
f.write_text(src.replace(old, new))
print(f"patch_runner: applied to {f} ({n} places)")
PY
