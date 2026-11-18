"""Explicit item lists for budget splits: the same task/level scenes in every split, different trajectories.

Mirrors eval_dynamic_dummy_policy.build_items of LIBERO-Dynamic @b66cb6b (item_id format, enabled tasks, variants).
The runner's --item-ids-file rejects unknown ids and preserves the order; round-robin worker partitions then give
every configuration the same item -> worker slot -> env seed mapping for the same list and number of workers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def item_id(suite, task, level, traj):
    return f"{suite}_task{int(task):02d}__{level}__traj{int(traj):02d}"


def scenes_from_config(config, level):
    """Parsed libero_dynamic_<level>.yaml -> {(suite, task, level): number of trajectories}."""
    defaults = config.get("defaults") or {}
    scenes = {}
    for suite, tasks in (config.get("tasks") or {}).items():
        for task, raw in (tasks or {}).items():
            merged = {**defaults, **(raw or {})}
            if merged.get("enabled", False):
                scenes[str(suite), int(task), str(level)] = len(merged.get("activation_variants") or [])
    return scenes


def read_scenes(config_dir, levels):
    import yaml

    scenes = {}
    for level in levels:
        path = Path(config_dir) / f"libero_dynamic_{level}.yaml"
        if not path.exists():
            path = Path(config_dir) / f"libero_dynamic_manual_{level}.yaml"
        scenes.update(scenes_from_config(yaml.safe_load(path.read_text()), level))
    return scenes


def make_splits(scenes, splits, seed=0, scene_limit=None):
    """{name: trajectories} -> {name: [item ids]}. A scene enters only if it has every trajectory of every split,
    so the scene composition of the splits is identical by construction; scene_limit subsamples without outcomes."""
    trajectories = [t for values in splits.values() for t in values]
    if len(set(trajectories)) != len(trajectories):
        raise ValueError("trajectory splits must be disjoint")
    usable = sorted(k for k, n in scenes.items() if n > max(trajectories))
    if not usable:
        raise ValueError("no scene has all requested trajectories")
    order = np.random.default_rng(seed).permutation(len(usable))
    chosen = [usable[i] for i in order[:scene_limit]]
    return {name: [item_id(*scene, t) for scene in chosen for t in values] for name, values in splits.items()}


def write_splits(items, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, ids in items.items():
        paths[name] = directory / f"{name}.json"
        paths[name].write_text(json.dumps(ids, indent=0))
    return paths
