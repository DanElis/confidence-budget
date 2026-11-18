"""GPU rounding is tolerated; changes to timing, outcomes, actions or metrics are not."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import preflight


def fixture():
    rng = np.random.default_rng(4)
    actions = rng.uniform(-1, 1, (500, 1, 2, 3)).astype(np.float32)
    due = np.zeros((500, 1, 2), bool)
    due[:400:2, 0, 0] = True
    due[:450:2, 0, 1] = True
    logs = dict(action=actions, due=due, end=np.where(due, 2, 0).astype(np.int32))
    elapsed = np.array([[400, 450]], np.int32)
    result = dict(elapsed=elapsed, length=elapsed.copy(), solved=np.array([[1, 0]], np.float32),
                  unfinished=np.zeros((1, 2), bool), n_infer=1 + due.sum(0),
                  mean_s=np.full((1, 2), 2.0, np.float32), mean_mask=np.full((1, 2), 2.0, np.float32),
                  frac_sel=np.zeros((1, 2), np.float32), ret=np.array([[1, 0]], np.float32),
                  **{k: v.astype(np.float32) for k, v in preflight.action_metrics(actions, due, elapsed, 2).items()})
    return result, logs


def test_accumulated_float32_rounding_passes_with_independent_trace_check():
    expected, logs = fixture()
    actual = deepcopy(expected)
    actual["action_change"] += np.float32(0.00125504)
    diagnostics = preflight.compare_rollouts(actual, logs, expected, logs, delay=2)
    assert diagnostics["passed"], diagnostics
    assert diagnostics["max_abs_action_diff"] == 0


def test_small_action_rounding_passes_but_requires_consistent_metrics():
    expected, logs = fixture()
    actual, perturbed = deepcopy(expected), deepcopy(logs)
    perturbed["action"] += np.float32(2e-6)
    actual.update({k: v.astype(np.float32) for k, v in preflight.action_metrics(
        perturbed["action"], perturbed["due"], actual["elapsed"], 2).items()})
    assert preflight.compare_rollouts(actual, perturbed, expected, logs, delay=2)["passed"]


@pytest.mark.parametrize("field", ["solved", "length", "elapsed", "n_infer", "mean_s", "mean_mask"])
def test_discrete_outcome_or_schedule_change_fails(field):
    expected, logs = fixture()
    actual = deepcopy(expected)
    actual[field][0, 0] += 1
    assert not preflight.compare_rollouts(actual, logs, expected, logs, delay=2)["passed"]


def test_shifted_request_fails_even_with_same_number_of_calls():
    expected, logs = fixture()
    shifted = deepcopy(logs)
    shifted["due"][10, 0, 0], shifted["due"][11, 0, 0] = False, True
    diagnostics = preflight.compare_rollouts(expected, shifted, expected, logs, delay=2)
    assert "request ticks differ" in diagnostics["failures"]


def test_large_action_difference_and_bad_accumulator_fail():
    expected, logs = fixture()
    changed = deepcopy(logs)
    changed["action"][10, 0, 0, 0] += 0.01
    assert not preflight.compare_rollouts(expected, changed, expected, logs, delay=2)["passed"]
    actual = deepcopy(expected)
    actual["action_change"] += 1.0
    assert not preflight.compare_rollouts(actual, logs, expected, logs, delay=2)["passed"]


def test_actions_after_first_episode_do_not_affect_comparison():
    expected, logs = fixture()
    changed = deepcopy(logs)
    changed["action"][400:, 0, 0] += 100
    changed["action"][450:, 0, 1] -= 100
    assert preflight.compare_rollouts(expected, changed, expected, logs, delay=2)["passed"]


def test_nonfinite_metric_fails_with_json_serializable_diagnostics():
    expected, logs = fixture()
    actual = deepcopy(expected)
    actual["action_change"][0, 0] = np.nan
    diagnostics = preflight.compare_rollouts(actual, logs, expected, logs, delay=2)
    assert not diagnostics["passed"]
    json.dumps(diagnostics, allow_nan=False)
