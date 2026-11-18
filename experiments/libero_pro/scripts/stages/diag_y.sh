#!/usr/bin/env bash
# Diagnostics on the tuning levels (y), after run_all showed
#   (a) RTC with the paper's mask (guidance up to index H - s) far below naive async (13-21% vs 49-61%),
#   (b) the calibrated h* ~ 8 < 2d + r_min, so the budget always falls back to s = d and lambda does nothing.
#
# A. RTC vs length of the guided window: mask_end in {d+1, d+5, 15, 25, H-s} at s = 5 and 10, and
#    exact (Jacobian) guidance with the paper's mask. If short masks recover naive-level success,
#    the failure is over-anchoring to the old chunk, not a bug in the async loop.
# B. Does lambda matter once there is slack? Constant h* in {20, 30, 40} with lambda in {0, 0.5, 1}.
#
# ~45 suites of 100 episodes, ~4 h with JOBS=4.  Usage: bash scripts/stages/diag_y.sh
cd "$(dirname "$0")/../.." && source scripts/stages/lib.sh
D=${D:-4}
main() {
trap 'trap - INT TERM; kill 0' INT TERM
for s in 5 10; do
  for m in $((D + 1)) $((D + 5)) 15 25; do
    run_eval "y_rtc_s${s}_m${m}_d$D" "$Y_LEVELS" --stride "$s" --mask-end "$m" --delay "$D" --guidance lerobot
  done
  run_eval "y_rtcexact_s${s}_d$D" "$Y_LEVELS" --stride "$s" --delay "$D" --guidance exact
done
for c in 20 30 40; do
  for lam in 0 0.5 1; do
    run_eval "y_budget_const${c}_lam${lam}_d$D" "$Y_LEVELS" --planner budget --hstar-rule "kind=const,c=$c" \
      --lam "$lam" --delay "$D" --guidance lerobot
  done
done
finish
python3 - "$RUNS" "$D" <<'PY'
import glob, json, sys
import numpy as np
runs, d = sys.argv[1], sys.argv[2]
def row(name):
    eps = [e for f in glob.glob(f"{runs}/{name}/libero_object_pos_y*.json") if "_chunks" not in f
           for e in json.load(open(f))["episodes"]]
    if not eps:
        return f"| {name} | — | — | — |"
    sr = 100 * np.mean([e["success"] for e in eps])
    s = np.nanmean([e.get("mean_stride", np.nan) for e in eps])
    return f"| {name} | {sr:.1f} | {s:.1f} | {len(eps)} |"
names = ["y_sync_s5", "y_sync_s10", f"y_naive_s5_d{d}", f"y_naive_s10_d{d}"]
for s in (5, 10):
    names += [f"y_rtc_s{s}_m{m}_d{d}" for m in (int(d) + 1, int(d) + 5, 15, 25)] + [f"y_rtc_s{s}_d{d}", f"y_rtcexact_s{s}_d{d}"]
names += [f"y_budget_const{c}_lam{l}_d{d}" for c in (20, 30, 40) for l in ("0", "0.5", "1")] + [f"y_budget_lam0.5_d{d}"]
table = "| run | SR y, % | mean s | episodes |\n|---|---|---|---|\n" + "\n".join(row(n) for n in names) + "\n"
open(f"{runs}/diag_y.md", "w").write(table)
print(table)
PY
}
main 2>&1 | tee -a "$RUNS/diag.log"
