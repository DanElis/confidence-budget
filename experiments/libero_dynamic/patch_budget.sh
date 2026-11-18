#!/usr/bin/env bash
# Register an opt-in strategy; fail closed if the pinned upstream interface changes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RT="$HERE/third_party/LIBERO-Dynamic/scripts/realtime_eval"
ln -sfn "$HERE/learned_budget_queue.py" "$RT/strategies/learned_budget_queue.py"
python3 - "$RT/control.py" <<'PY'
import sys
from pathlib import Path
f = Path(sys.argv[1])
src = f.read_text()
if '"learned_budget_queue"' in src:
    print("patch_budget: already registered")
    sys.exit(0)
choice = '    "rtc_queue",\n'
branch = '    if canonical_mode == "vlash_future_state":\n'
if src.count(choice) != 1 or src.count(branch) != 1:
    sys.exit("patch_budget: unsupported upstream control.py; no file changed")
src = src.replace(choice, choice + '    "learned_budget_queue",\n')
src = src.replace(branch, '''    if canonical_mode == "learned_budget_queue":
        from .strategies.learned_budget_queue import LearnedBudgetQueueController

        return LearnedBudgetQueueController(
            policy=policy, language=language, control_freq=control_freq, mode=mode,
            prefetch_threshold=prefetch_threshold, action_chunk_size=action_chunk_size,
            chunk_size_threshold=chunk_size_threshold, latency_sampler=latency_sampler,
            vlash_visual_delay_steps=vlash_visual_delay_steps,
        )
''' + branch)
f.write_text(src)
print("patch_budget: registered learned_budget_queue")
PY
