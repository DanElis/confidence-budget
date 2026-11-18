"""Register LIBERO-PRO position-perturbation suites on top of the stock LIBERO package.

LIBERO-PRO (Zhou et al., 2025, https://github.com/Zxy-MLlab/LIBERO-PRO) ships
the LIBERO-Object tasks with the target objects' placement regions shifted
along x or y (levels 0.1 ... 0.5; level 0.1 = 7 cm) as separate BDDL and
init-state folders. LeRobot builds the BDDL path as
``bddl_files/<problem_folder>/<task>.bddl`` and the init-state path as
``init_states/<problem_folder>/<task>.pruned_init``; an absolute
``problem_folder`` overrides both roots. So each level is materialised as one
folder holding both files (symlinks) and registered as a new suite, e.g.
``libero_object_pos_x0.2``. The installed ``libero`` package is not modified.
"""

from __future__ import annotations

import os
from pathlib import Path

POSITION_LEVELS = tuple(f"{axis}{level}" for axis in ("x", "y") for level in ("0.1", "0.2", "0.3", "0.4", "0.5"))
BASE_SUITE = "libero_object"
MAX_STEPS = 280  # LeRobot's limit for libero_object (longest demo: 254 steps)
LIBERO_PRO_COMMIT = "eafdb809426b13153aa1e4c42d6601844217dfec"


def import_libero_quietly():
    """Import ``libero.libero``; on first use it asks on stdin where to store datasets, answer "no"."""
    import builtins

    original = builtins.input
    builtins.input = lambda *args, **kwargs: "n"
    try:
        import libero.libero  # noqa: F401  (writes ~/.libero/config.yaml with default paths)
    finally:
        builtins.input = original


def suite_name(level: str) -> str:
    return f"{BASE_SUITE}_pos_{level}"


def materialise_level(pro_root: Path, level: str, out_root: Path, task_names: list[str]) -> Path:
    """Create ``out_root/<suite>/`` with symlinks to the level's .bddl and .pruned_init files."""
    src_bddl = pro_root / "libero" / "libero" / "bddl_files" / f"{BASE_SUITE}_temp_{level}"
    src_init = pro_root / "libero" / "libero" / "init_files" / f"{BASE_SUITE}_temp_{level}"
    folder = out_root / suite_name(level)
    folder.mkdir(parents=True, exist_ok=True)
    for task in task_names:
        for src in (src_bddl / f"{task}.bddl", src_init / f"{task}.pruned_init"):
            if not src.exists():
                raise FileNotFoundError(f"LIBERO-PRO file missing: {src} (is --libero-pro-root a LIBERO-PRO clone?)")
            dst = folder / src.name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink(src.resolve(), dst)
    return folder


def register_position_suites(pro_root: str | Path, levels=POSITION_LEVELS, out_root: str | Path | None = None) -> list[str]:
    """Register ``libero_object_pos_<level>`` suites in ``libero.libero.benchmark``. Returns their names."""
    import_libero_quietly()
    from libero.libero import benchmark as bm

    pro_root = Path(pro_root).expanduser().resolve()
    out_root = Path(out_root or pro_root / "_lerobot_suites").resolve()
    task_names = list(bm.libero_task_map[BASE_SUITE])
    names = []
    for level in levels:
        if level not in POSITION_LEVELS:
            raise ValueError(f"unknown level {level!r}, choose from {POSITION_LEVELS}")
        folder = materialise_level(pro_root, level, out_root, task_names)
        name = suite_name(level)
        bm.task_maps[name] = {
            task: bm.Task(
                name=task,
                language=bm.grab_language_from_filename(task + ".bddl"),
                problem="Libero",
                problem_folder=str(folder),
                bddl_file=f"{task}.bddl",
                init_states_file=f"{task}.pruned_init",
            )
            for task in task_names
        }

        def __init__(self, task_order_index=0, _name=name):
            bm.Benchmark.__init__(self, task_order_index=task_order_index)
            self.name = _name
            self._make_benchmark()

        cls = type(name.upper(), (bm.Benchmark,), {"__init__": __init__})
        bm.register_benchmark(cls)  # registered under the lower-cased class name, i.e. ``name``
        names.append(name)
    return names
