"""Compare fixed-controller rollouts without confusing float32 sums with timing.

No simulator or JAX imports. Outcomes and request timing must agree exactly.
Only first-episode actions matter. They must differ by <=1e-4 per coordinate
(absolute motor-command units). Accumulated GPU metrics are independently
checked against float64 recomputations from their OWN recorded actions.
"""

from __future__ import annotations

import numpy as np

ACTION_ATOL = 1e-4


def action_metrics(actions, due, elapsed, delay):
    actions = np.asarray(actions, np.float64)
    due = np.asarray(due, bool)
    elapsed = np.asarray(elapsed)
    jumps = np.linalg.norm(np.diff(actions, axis=0), axis=-1)
    valid = np.arange(1, len(actions))[:, None, None] < elapsed[None]
    switched = np.zeros_like(due)
    if delay == 0:
        switched[:] = due
    elif delay < len(due):
        switched[delay:] = due[:-delay]
    switches = switched[1:] & valid
    other = ~switched[1:] & valid
    return dict(action_change=np.where(valid, jumps, 0).sum(0),
                jump_switch=np.where(switches, jumps, 0).sum(0) / np.maximum(switches.sum(0), 1),
                jump_other=np.where(other, jumps, 0).sum(0) / np.maximum(other.sum(0), 1))


def compare_rollouts(actual, actual_logs, expected, expected_logs, *, delay):
    diagnostics = dict(passed=False, action_atol=ACTION_ATOL, failures=[], metrics={})
    failures = diagnostics["failures"]
    # These are structural checks, not floating-point approximations.
    for name in ("solved", "length", "unfinished", "elapsed", "n_infer", "mean_s", "mean_mask", "frac_sel"):
        x, y = np.asarray(actual[name]), np.asarray(expected[name])
        if not np.array_equal(x, y):
            failures.append(f"{name}: values or shape differ")
    due, reference_due = np.asarray(actual_logs["due"], bool), np.asarray(expected_logs["due"], bool)
    if not np.array_equal(due, reference_due):
        failures.append("request ticks differ")
    if due.shape != reference_due.shape:
        return diagnostics
    if not np.array_equal(np.asarray(actual_logs["end"])[due], np.asarray(expected_logs["end"])[due]):
        failures.append("RTC mask at a request differs")
    if not np.allclose(actual["ret"], expected["ret"], atol=1e-5, rtol=1e-5):
        failures.append("episode return differs")

    a, b = np.asarray(actual_logs["action"], np.float64), np.asarray(expected_logs["action"], np.float64)
    if a.shape != b.shape or a.shape[:-1] != due.shape:
        failures.append("action trace shapes differ")
        return diagnostics
    # The two lengths are required to match above. Use their union for diagnostics
    # if they do not: never hide the extra part of a mismatched first episode.
    elapsed = np.maximum(actual["elapsed"], expected["elapsed"])
    live = np.arange(len(a))[:, None, None] < np.asarray(elapsed)[None]
    errors = np.abs(a - b)[live]
    if not len(errors) or not np.all(np.isfinite(errors)):
        failures.append("empty or non-finite first-episode actions")
        return diagnostics
    diagnostics["max_abs_action_diff"] = float(errors.max())
    diagnostics["rms_action_diff"] = float(np.sqrt(np.mean(errors ** 2)))
    worst = np.unravel_index(np.argmax(np.where(live[..., None], np.abs(a - b), -1)), a.shape)
    diagnostics["worst_action"] = dict(tick=int(worst[0]), level=int(worst[1]), episode=int(worst[2]),
                                       coordinate=int(worst[3]), controlled=float(a[worst]), reference=float(b[worst]))
    if errors.max() > ACTION_ATOL:
        failures.append(f"first-episode action max error {errors.max():.6g} > {ACTION_ATOL}")

    recomputed = {}
    for label, result, logs, trace in (("controlled", actual, actual_logs, a),
                                        ("reference", expected, expected_logs, b)):
        host = action_metrics(trace, logs["due"], result["elapsed"], delay)
        recomputed[label] = host
        # Standard nonnegative float32 sum bound gamma_n, with allowance for
        # the action-vector norm operations and a factor two for their rounding.
        unit_roundoff = np.finfo(np.float32).eps / 2
        operations = np.asarray(result["elapsed"], np.float64) + trace.shape[-1] + 8
        gamma = operations * unit_roundoff / (1 - operations * unit_roundoff)
        for name, value in host.items():
            observed = np.asarray(result[name], np.float64)
            bound = 1e-6 + 2 * gamma * np.abs(value)
            error = np.abs(observed - value)
            diagnostics["metrics"][f"{label}.{name}"] = dict(
                max_abs_error=float(error.max()) if np.all(np.isfinite(error)) else None,
                max_rounding_bound=float(bound.max()) if np.all(np.isfinite(bound)) else None)
            if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(value)) or np.any(error > bound):
                failures.append(f"{label}.{name}: GPU metric inconsistent with its recorded actions")
    diagnostics["host_action_change_max_diff"] = float(np.max(np.abs(
        recomputed["controlled"]["action_change"] - recomputed["reference"]["action_change"])))
    diagnostics["passed"] = not failures
    return diagnostics
