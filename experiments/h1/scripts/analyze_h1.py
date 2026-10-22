#!/usr/bin/env python
"""H1 verdict from the output of ``collect_h1.py``.

H1: the cross-attention entropy signal stays informative under RTC guidance.
Operationally: for RTC chunks, outside the frozen prefix, the Spearman
correlation between an action's attention entropy and its error against the
demonstration is at least ``--threshold`` and its 95% episode-bootstrap CI lies
above zero. By default the coefficient controls for the action index (see
``h1.stats.partial_spearman``); ``--criterion raw`` uses the plain coefficient.

Writes summary.json, summary.md and h1_report.png to ``--out-dir``.

Example:
    python scripts/analyze_h1.py runs/h1_d5_s25.npz --out-dir runs/h1_d5_s25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from h1.signal import ZONE_FROZEN, ZONE_NAMES, ZONE_RAMP, ZONE_TAIL  # noqa: E402
from h1.stats import (  # noqa: E402
    action_table,
    correlation_report,
    difference_report,
    frozen_deviation,
    h1_verdict,
    horizon_report,
    partial_by_step,
)

FREE_COLOR = "#2a78d6"
RTC_COLOR = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
CONDITIONS = ((False, "без guidance", FREE_COLOR), (True, "RTC", RTC_COLOR))
ZONE_LABELS = {
    "ramp+tail": "вне замороженной [d, H)",
    "frozen": "замороженная [0, d)",
    "ramp": "мягкая маска [d, конец маски)",
    "tail": "без привязки [конец маски, H)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path, help=".npz from collect_h1.py")
    p.add_argument("--out-dir", type=Path, default=None, help="default: next to the .npz, same stem")
    p.add_argument("--criterion", choices=["partial", "raw"], default="partial")
    p.add_argument("--threshold", type=float, default=0.15)
    p.add_argument("--denoise-step", default="-1", help="index of the denoising step, or 'mean'")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plateau-k", type=int, default=5)
    p.add_argument("--plateau-eta", type=float, default=0.95)
    p.add_argument("--plateau-tau", type=float, default=0.01)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def load(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as npz:
        data = {k: npz[k] for k in npz.files if k != "meta"}
        meta = json.loads(str(npz["meta"]))
    return data, meta


def analyze(data: dict, meta: dict, args: argparse.Namespace) -> tuple[dict, dict]:
    delay, mask_end = int(meta["delay"]), int(meta["mask_end"])
    action_std = np.asarray(meta["action_std"], dtype=np.float64)
    step = None if args.denoise_step == "mean" else int(args.denoise_step)
    plateau = {"k": args.plateau_k, "eta": args.plateau_eta, "tau": args.plateau_tau}
    table = action_table(data, action_std, delay, mask_end, denoise_step=step, plateau=plateau)
    outside = table["zone"] != ZONE_FROZEN

    correlations = {}
    for is_rtc, name, _ in CONDITIONS:
        cond = table["is_rtc"] == is_rtc
        key = "rtc" if is_rtc else "free"
        correlations[key] = {"ramp+tail": correlation_report(table, np.flatnonzero(cond & outside), args.n_boot, args.seed)}
        for zone in (ZONE_FROZEN, ZONE_RAMP, ZONE_TAIL):
            rows = np.flatnonzero(cond & (table["zone"] == zone))
            if rows.size >= 10:
                correlations[key][ZONE_NAMES[zone]] = correlation_report(table, rows, args.n_boot, args.seed)

    deviation = frozen_deviation(data, delay)
    frozen = {}
    for is_rtc, _, _ in CONDITIONS:
        dev = deviation[data["is_rtc"] == is_rtc]
        frozen["rtc" if is_rtc else "free"] = float(np.nanmedian(dev)) if np.isfinite(dev).any() else float("nan")

    entropy_norm = table["entropy_norm"]
    summary = {
        "input": str(args.data),
        "settings": {
            "delay": delay,
            "stride": int(meta["stride"]),
            "mask_end": mask_end,
            "horizon": int(meta["horizon"]),
            "denoise_step": args.denoise_step,
            "plateau": plateau,
            "n_boot": args.n_boot,
            "guidance": meta["args"].get("guidance"),
            "policy": meta["args"].get("policy"),
            "dataset": meta["args"].get("dataset"),
        },
        "n_chunks": int(data["episode"].shape[0]),
        "n_episodes": int(np.unique(data["episode"]).shape[0]),
        "verdict": h1_verdict(correlations["rtc"]["ramp+tail"], args.criterion, args.threshold),
        "correlations": correlations,
        "rtc_minus_free": difference_report(table, np.flatnonzero(outside), args.n_boot, args.seed),
        "horizon": {
            "free": horizon_report(table, is_rtc=False),
            "rtc": horizon_report(table, is_rtc=True),
        },
        "frozen_rmse_median": frozen,
        "entropy_norm_quantiles": {
            q: float(np.quantile(entropy_norm, float(q))) for q in ("0.05", "0.25", "0.5", "0.75", "0.95")
        },
        "partial_spearman_by_denoise_step": {
            "free": partial_by_step(data, action_std, delay, mask_end, is_rtc=False),
            "rtc": partial_by_step(data, action_std, delay, mask_end, is_rtc=True),
        },
    }
    summary["warnings"] = warnings(summary)
    return summary, table


def warnings(summary: dict) -> list[str]:
    out = []
    frozen = summary["frozen_rmse_median"]
    if np.isfinite(frozen["rtc"]) and np.isfinite(frozen["free"]) and frozen["rtc"] > 0.5 * frozen["free"]:
        out.append(
            "RTC почти не притягивает замороженный префикс к прошлому чанку "
            f"(медиана RMSE {frozen['rtc']:.3f} против {frozen['free']:.3f} без guidance): "
            "проверьте --guidance и --max-guidance-weight."
        )
    eta = summary["settings"]["plateau"]["eta"]
    for key, name in (("rtc", "RTC"), ("free", "без guidance")):
        share = summary["horizon"][key]["plateau_found_share"]
        if np.isfinite(share) and share < 0.1:
            out.append(
                f"{name}: плато найдено лишь в {share:.0%} чанков при eta = {eta}; "
                "h* почти всегда равен H. Сравните eta с квантилями нормированной энтропии."
            )
    for key, name in (("rtc", "RTC"), ("free", "без guidance")):
        if summary["correlations"][key]["ramp+tail"]["n_episodes"] < 30:
            out.append(f"{name}: меньше 30 эпизодов, доверительные интервалы ненадёжны.")
    return out


# ----------------------------------------------------------------------------- report


def fmt(value: float, ci: list[float] | None = None) -> str:
    if not np.isfinite(value):
        return "—"
    if ci is None:
        return f"{value:.3f}"
    return f"{value:.3f} [{ci[0]:.3f}; {ci[1]:.3f}]"


def markdown(summary: dict) -> str:
    v = summary["verdict"]
    s = summary["settings"]
    coef = "Частный коэффициент Спирмена (контроль индекса j)" if v["criterion"] == "partial" else "Коэффициент Спирмена"
    lines = [
        "# H1: сигнал энтропии cross-attention при генерации с RTC",
        "",
        f"**Вердикт: H1 {'подтверждена' if v['passed'] else 'не подтверждена'}.** "
        f"{coef} для RTC вне замороженной зоны: {fmt(v['value'])}, "
        f"нижняя граница 95% CI: {fmt(v['ci_low'])}. Критерий: не менее {v['threshold']} и нижняя граница больше 0.",
        "",
        f"Эпизодов: {summary['n_episodes']}, чанков: {summary['n_chunks']}; "
        f"d = {s['delay']}, s = {s['stride']}, конец маски = {s['mask_end']}, H = {s['horizon']}; "
        f"guidance: {s['guidance']}; шаг денойзинга: {s['denoise_step']}.",
        "",
        "## Корреляция энтропии и ошибки",
        "",
        "| Условие | Зона | Пар | Эпизодов | Спирмен [95% CI] | Частный Спирмен [95% CI] |",
        "|---|---|---|---|---|---|",
    ]
    for key, name in (("free", "без guidance"), ("rtc", "RTC")):
        for zone, rep in summary["correlations"][key].items():
            lines.append(
                f"| {name} | {ZONE_LABELS[zone]} | {rep['n_pairs']} | {rep['n_episodes']} | "
                f"{fmt(rep['spearman'], rep['spearman_ci'])} | {fmt(rep['partial_spearman'], rep['partial_spearman_ci'])} |"
            )
    diff = summary["rtc_minus_free"]
    lines += [
        "",
        f"Разность частного коэффициента RTC − без guidance (вне замороженной зоны): "
        f"{fmt(diff['partial_spearman_rtc_minus_free'], diff['ci'])}.",
        "",
        "## Горизонт h*",
        "",
        "| Условие | Доля чанков с плато | Медиана h* | Ошибка до h* | Ошибка после h* | После / до |",
        "|---|---|---|---|---|---|",
    ]
    for key, name in (("free", "без guidance"), ("rtc", "RTC")):
        h = summary["horizon"][key]
        lines.append(
            f"| {name} | {fmt(h['plateau_found_share'])} | {fmt(h['h_star_median'])} | "
            f"{fmt(h['error_before_h_star'])} | {fmt(h['error_after_h_star'])} | {fmt(h['error_ratio_after_before'])} |"
        )
    q = summary["entropy_norm_quantiles"]
    frozen = summary["frozen_rmse_median"]
    lines += [
        "",
        "## Контроль",
        "",
        f"- RMSE до прошлого чанка в замороженной зоне (медиана): без guidance {fmt(frozen['free'])}, RTC {fmt(frozen['rtc'])}.",
        f"- Квантили нормированной энтропии (5/25/50/75/95%): "
        + " / ".join(f"{q[k]:.3f}" for k in ("0.05", "0.25", "0.5", "0.75", "0.95"))
        + f"; порог плато eta = {s['plateau']['eta']}.",
        "- Частный Спирмен по шагам денойзинга, RTC: "
        + ", ".join(fmt(x) for x in summary["partial_spearman_by_denoise_step"]["rtc"]),
        "- Частный Спирмен по шагам денойзинга, без guidance: "
        + ", ".join(fmt(x) for x in summary["partial_spearman_by_denoise_step"]["free"]),
    ]
    if summary["warnings"]:
        lines += ["", "## Предупреждения", ""] + [f"- {w}" for w in summary["warnings"]]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- plots


def by_index_mean(values: np.ndarray, j: np.ndarray, horizon: int) -> np.ndarray:
    sums = np.bincount(j, values, minlength=horizon)
    counts = np.bincount(j, minlength=horizon)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / counts


def plot(summary: dict, table: dict, path: Path, n_bins: int = 10) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summary["settings"]
    horizon, delay, mask_end = s["horizon"], s["delay"], s["mask_end"]
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": INK_2,
            "axes.labelcolor": INK_2,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    j_axis = np.arange(horizon)

    def zone_lines(ax):
        for x, label in ((delay, "d"), (mask_end, "конец маски")):
            ax.axvline(x - 0.5, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
            ax.annotate(label, (x - 0.5, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -12),
                        textcoords="offset points", color=INK_2, fontsize=9)

    pending: dict = {}

    def end_label(ax, x, y, text, color):
        finite = np.flatnonzero(np.isfinite(y))
        if finite.size:
            i = finite[-1]
            ax.plot([x[i]], [y[i]], "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
            pending.setdefault(ax, []).append((float(x[i]), float(y[i]), text))

    def place_labels(min_gap=0.07):
        """Direct labels at line ends, pushed apart vertically so they never overlap."""
        for ax, items in pending.items():
            x0, x1 = ax.get_xlim()
            ax.set_xlim(x0, x1 + 0.2 * (x1 - x0))
            y0, y1 = ax.get_ylim()
            frac = sorted(((y - y0) / (y1 - y0), x, text) for x, y, text in items)
            placed = []
            for f, x, text in frac:
                if placed and f - placed[-1][0] < min_gap:
                    f = placed[-1][0] + min_gap
                placed.append((f, x, text))
            shift = max(0.0, placed[-1][0] - 0.97)
            for f, x, text in placed:
                ax.annotate(text, (x, y0 + (f - shift) * (y1 - y0)), xytext=(7, 0), textcoords="offset points",
                            color=INK, fontsize=9, va="center")
            ax.set_ylim(y0, y1)

    residual_curves = []
    for is_rtc, name, color in CONDITIONS:
        cond = table["is_rtc"] == is_rtc
        j = table["j"][cond]
        ent = by_index_mean(table["entropy_norm"][cond], j, horizon)
        err = by_index_mean(table["error"][cond], j, horizon)
        axes[0, 0].plot(j_axis, ent, color=color, linewidth=2, label=name)
        end_label(axes[0, 0], j_axis, ent, name, color)
        axes[0, 1].plot(j_axis, err, color=color, linewidth=2, label=name)
        end_label(axes[0, 1], j_axis, err, name, color)

        rows = cond & (table["zone"] != ZONE_FROZEN)
        jr = table["j"][rows]
        ent_mean = by_index_mean(table["entropy_norm"][rows], jr, horizon)
        err_mean = by_index_mean(table["error"][rows], jr, horizon)
        resid = table["entropy_norm"][rows] - ent_mean[jr]
        rel_err = table["error"][rows] / np.maximum(err_mean[jr], 1e-12)
        edges = np.quantile(resid, np.linspace(0, 1, n_bins + 1))
        bins = np.clip(np.searchsorted(edges, resid, side="right") - 1, 0, n_bins - 1)
        centers = np.array([resid[bins == b].mean() for b in range(n_bins)])
        rel = np.array([rel_err[bins == b].mean() for b in range(n_bins)])
        residual_curves.append((name, color, centers, rel))

        h_star = table["chunk_h_star"][np.unique(table["chunk"][cond])]
        axes[1, 1].hist(h_star, bins=np.arange(0.5, horizon + 1.5, 1), color=color, histtype="step",
                        linewidth=2, label=name)

    for name, color, centers, rel in residual_curves:
        axes[1, 0].plot(centers, rel, color=color, linewidth=2, marker="o", markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
        end_label(axes[1, 0], centers, rel, name, color)
    axes[1, 0].axhline(1.0, color=INK_2, linewidth=1)
    place_labels()

    for ax in (axes[0, 0], axes[0, 1]):
        zone_lines(ax)
        ax.set_xlabel("индекс действия j")
    axes[0, 0].set_ylabel("энтропия / log N")
    axes[0, 0].set_title("Нормированная энтропия по индексу", loc="left", color=INK)
    axes[0, 1].set_ylabel("MSE / std²")
    axes[0, 1].set_title("Ошибка по индексу", loc="left", color=INK)
    axes[1, 0].set_xlabel("энтропия минус среднее при том же j")
    axes[1, 0].set_ylabel("ошибка / средняя ошибка при том же j")
    axes[1, 0].set_title("Ошибка при отклонении энтропии (вне замороженной зоны)", loc="left", color=INK)
    axes[1, 1].set_xlabel("h*")
    axes[1, 1].set_ylabel("чанков")
    axes[1, 1].set_title("Распределение h*", loc="left", color=INK)
    for ax in axes.flat:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    v = summary["verdict"]
    fig.suptitle(
        f"H1 {'подтверждена' if v['passed'] else 'не подтверждена'}: "
        f"{'частный ' if v['criterion'] == 'partial' else ''}ρ Спирмена = {v['value']:.3f}, "
        f"нижняя граница 95% CI = {v['ci_low']:.3f}, порог {v['threshold']}",
        color=INK, fontsize=12, x=0.01, ha="left",
    )
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.data.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    data, meta = load(args.data)
    summary, table = analyze(data, meta, args)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out_dir / "summary.md").write_text(markdown(summary))
    if not args.no_plot:
        plot(summary, table, out_dir / "h1_report.png")
    v = summary["verdict"]
    print(
        f"H1 {'PASSED' if v['passed'] else 'NOT PASSED'}: {v['criterion']} spearman = {v['value']:.3f}, "
        f"CI low = {v['ci_low']:.3f}, threshold = {v['threshold']}"
    )
    for w in summary["warnings"]:
        print("warning:", w)
    print(f"report: {out_dir}")


if __name__ == "__main__":
    main()
