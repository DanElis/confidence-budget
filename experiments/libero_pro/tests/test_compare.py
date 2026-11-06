import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare import main  # noqa: E402


def write_run(d: Path, successes, suite="libero_object_pos_x0.2"):
    d.mkdir(parents=True)
    eps = [{"task_id": i // 10, "init_state": i % 10, "success": s, "n_inference": 20, "sparc": -2.0,
            "mean_stride": 10} for i, s in enumerate(successes)]
    (d / f"{suite}.json").write_text(json.dumps({"success_rate": sum(successes) / len(successes),
                                                 "episodes": eps, "jump_at_switch": 0.1}))


def test_paired_difference(tmp_path, capsys):
    write_run(tmp_path / "ref", [True] * 50 + [False] * 50)
    write_run(tmp_path / "better", [True] * 60 + [False] * 40)
    main([str(tmp_path / "ref"), str(tmp_path / "better"), "--n-boot", "500"])
    out = capsys.readouterr().out
    assert "| better | 60.0 | 60.0 | +10.0 [" in out
