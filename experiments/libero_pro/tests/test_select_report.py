import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from select_report import report  # noqa: E402


def write(runs: Path, name: str, rate: float, suites=("libero_object_pos_y0.1",), n=20):
    d = runs / name
    d.mkdir(parents=True, exist_ok=True)
    k = round(rate * n)
    for s in suites:
        eps = [{"task_id": i // 10, "init_state": i % 10, "success": i < k, "n_inference": 10, "sparc": -2.0,
                "mean_stride": 10} for i in range(n)]
        (d / f"{s}.json").write_text(json.dumps({"success_rate": k / n, "episodes": eps, "jump_at_switch": 0.1}))


def test_selection_uses_y_only(tmp_path):
    x = ("libero_object", "libero_object_pos_x0.2")
    for s, r in {"5": 0.5, "10": 0.6}.items():
        write(tmp_path, f"y_sync_s{s}", r)
        write(tmp_path, f"x_sync_s{s}", 0.9 - r, x)
    for s, r in {"5": 0.4, "10": 0.7, "15": 0.7, "20": 0.3}.items():  # tie 10/15 -> 10
        write(tmp_path, f"y_rtc_s{s}_d4", r)
        write(tmp_path, f"x_rtc_s{s}_d4", 1 - r, x)  # x would prefer s=20
        write(tmp_path, f"x_naive_s{s}_d4", 0.2, x)
    for v, r in {"0": 0.5, "0.25": 0.8, "0.5": 0.6, "0.75": 0.6, "1": 0.4}.items():
        write(tmp_path, f"y_budget_lam{v}_d4", r)
        write(tmp_path, f"x_budget_lam{v}_d4", 0.5, x)
    for v, r in {"1": 0.6, "2": 0.5, "3": 0.65}.items():
        write(tmp_path, f"y_budget_entropy_z{v}_d4", r)
        write(tmp_path, f"x_budget_entropy_z{v}_d4", 0.5, x)
    write(tmp_path, "abl_rmin3_d4", 0.4, x)
    text, chosen = report(tmp_path, ["4"], n_boot=100)
    assert chosen["4"] == {"s_sync": "10", "s": "10", "lam": "0.25", "z_max": "3"}
    assert "| RTC s=10 (reference) |" in text and "| budget lambda=0.25 |" in text
    assert "| abl_rmin3_d4 |" in text and "Full x grid" in text


def test_missing_runs_are_tolerated(tmp_path):
    write(tmp_path, "y_rtc_s10_d4", 0.5)
    text, chosen = report(tmp_path, ["4"], n_boot=100)
    assert chosen["4"]["s"] == "10" and chosen["4"]["lam"] is None
    assert "missing reference x_rtc_s10_d4" in text
