"""Consistency check between loaded rows and dataset metadata."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from collect_h1 import check_episodes  # noqa: E402

META = SimpleNamespace(episodes={
    0: {"length": 3, "dataset_from_index": 0},
    1: {"length": 2, "dataset_from_index": 3},
})


def _rows(episodes, frames):
    rows = {}
    for r, (e, f) in enumerate(zip(episodes, frames)):
        rows.setdefault(e, {})[f] = r
    return rows


def test_consistent_full_dataset():
    rows = _rows([0, 0, 0, 1, 1], [0, 1, 2, 0, 1])
    assert check_episodes(rows, np.arange(5), META, [0, 1], filtered=False) == []


def test_filtered_dataset_keeps_global_index():
    rows = _rows([1, 1], [0, 1])
    assert check_episodes(rows, np.array([3, 4]), META, [1], filtered=True) == []


def test_detects_missing_and_shifted_episodes():
    rows = _rows([0, 0, 0], [0, 1, 2])
    assert "no rows" in check_episodes(rows, np.arange(3), META, [1], filtered=True)[0]
    assert "global index" in check_episodes(rows, np.array([5, 6, 7]), META, [0], filtered=True)[0]
    shuffled = _rows([1, 1, 0, 0, 0], [0, 1, 0, 1, 2])
    problems = check_episodes(shuffled, np.array([3, 4, 0, 1, 2]), META, [0], filtered=False)
    assert "row order" in problems[0]
