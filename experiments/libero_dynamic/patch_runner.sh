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

# Correction-head strategy (correction/corrected_rtc_queue.py): a symlink into the strategies package and the
# registration of --control-mode corrected_rtc_queue in control.py (runtime.py is not touched).
RT="$HERE/third_party/LIBERO-Dynamic/scripts/realtime_eval"
ln -sfn "$HERE/correction/corrected_rtc_queue.py" "$RT/strategies/corrected_rtc_queue.py"
python3 - "$RT/control.py" <<'PY'
import sys
from pathlib import Path

f = Path(sys.argv[1])
src = f.read_text()
if '"corrected_rtc_queue"' in src:
    print("patch_runner: corrected_rtc_queue already registered")
    sys.exit(0)
choices = '    "rtc_queue",\n    "vlash_future_state",\n'
branch = '    if canonical_mode == "vlash_future_state":\n'
if src.count(choices) != 1 or src.count(branch) != 1:
    sys.exit(f"patch_runner: control.py layout changed ({src.count(choices)}, {src.count(branch)}); upstream changed?")
src = src.replace(choices, '    "rtc_queue",\n    "corrected_rtc_queue",\n    "vlash_future_state",\n')
src = src.replace(branch, '''    if canonical_mode == "corrected_rtc_queue":
        from .strategies.corrected_rtc_queue import CorrectedRtcQueueController

        return CorrectedRtcQueueController(
            policy=policy,
            language=language,
            control_freq=control_freq,
            mode=mode,
            prefetch_threshold=prefetch_threshold,
            action_chunk_size=action_chunk_size,
            chunk_size_threshold=chunk_size_threshold,
            latency_sampler=latency_sampler,
            vlash_visual_delay_steps=vlash_visual_delay_steps,
        )
''' + branch)
f.write_text(src)
print("patch_runner: registered corrected_rtc_queue in control.py")
PY
