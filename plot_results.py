"""Plot experiment summary metrics as a compact Matplotlib dashboard."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_METRICS = (
    "coefficient",
    "sycophancy_answer_confidence",
    "sycophancy_apologize_rate",
    "sycophancy_truthfulness",
    "coherence_mean",
    "coherence_std",
    "coherence_failure_rate",
    "mean_output_tokens",
)

COLORS = {
    "truthfulness": "#0072B2",
    "confidence": "#D55E00",
    "apology": "#009E73",
    "coherence": "#6F4E9C",
    "tokens": "#CC79A7",
    "baseline": "#202124",
}


def load_summary(location: Path) -> list[dict[str, float]]:
    """Load and validate a ``summary.json`` file or results directory."""
    summary_path = location / "summary.json" if location.is_dir() else location
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Expected a non-empty list in {summary_path}")

    rows: list[dict[str, float]] = []
    for index, raw_row in enumerate(raw, start=1):
        if not isinstance(raw_row, dict):
            raise ValueError(f"Summary row {index} is not an object")
        missing = [metric for metric in REQUIRED_METRICS if metric not in raw_row]
        if missing:
            raise ValueError(
                f"Summary row {index} is missing metrics: {', '.join(missing)}"
            )
        row: dict[str, float] = {}
        for metric in REQUIRED_METRICS:
            value = raw_row[metric]
            if isinstance(value, bool):
                raise ValueError(f"Summary row {index} has non-numeric {metric}")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Summary row {index} has non-numeric {metric}: {value!r}"
                ) from exc
            if not math.isfinite(number):
                raise ValueError(
                    f"Summary row {index} has non-finite {metric}: {value!r}"
                )
            row[metric] = number
        rows.append(row)

    rows.sort(key=lambda row: row["coefficient"])
    coefficients = [row["coefficient"] for row in rows]
    if len(set(coefficients)) != len(coefficients):
        raise ValueError("Summary contains duplicate steering coefficients")
    return rows


def _style_axis(axis: Any) -> None:
    axis.set_facecolor("#FBFCFE")
    axis.grid(axis="y", color="#DDE2E8", linewidth=0.8, alpha=0.9)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#AAB2BD")
    axis.spines["bottom"].set_color("#AAB2BD")
    axis.tick_params(colors="#4B5563")
    axis.title.set_color("#202124")


def _mark_baseline(axis: Any, x: float, y: float) -> None:
    axis.scatter(
        [x],
        [y],
        marker="*",
        s=180,
        color=COLORS["baseline"],
        edgecolor="white",
        linewidth=0.9,
        zorder=6,
    )


def create_dashboard(
    rows: list[dict[str, float]], title: str
) -> Any:
    """Create and return the experiment dashboard figure."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, MultipleLocator, PercentFormatter

    coefficients = [row["coefficient"] for row in rows]
    truthfulness = [row["sycophancy_truthfulness"] for row in rows]
    confidence = [row["sycophancy_answer_confidence"] for row in rows]
    apology = [row["sycophancy_apologize_rate"] for row in rows]
    coherence = [row["coherence_mean"] for row in rows]
    coherence_std = [row["coherence_std"] for row in rows]
    failures = [row["coherence_failure_rate"] for row in rows]
    tokens = [row["mean_output_tokens"] for row in rows]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    figure.suptitle(title, fontsize=20, fontweight="bold", color="#202124")
    figure.get_layout_engine().set(
        h_pad=0.08,
        w_pad=0.08,
        hspace=0.06,
        wspace=0.06,
    )

    behavior_axis = axes[0, 0]
    behavior_series = (
        (truthfulness, "Truthfulness", COLORS["truthfulness"], "o"),
        (confidence, "Answer confidence", COLORS["confidence"], "s"),
        (apology, "Apology rate", COLORS["apology"], "^"),
    )
    for values, label, color, marker in behavior_series:
        behavior_axis.plot(
            coefficients,
            values,
            label=label,
            color=color,
            marker=marker,
            markersize=6.5,
            linewidth=2.2,
        )
    behavior_axis.set_title("Sycophancy behavior", loc="left")
    behavior_axis.set_ylabel("Rate")
    behavior_axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    rate_min = min(truthfulness + confidence + apology)
    rate_max = max(truthfulness + confidence + apology)
    behavior_axis.set_ylim(max(0.0, rate_min - 0.07), min(1.0, rate_max + 0.07))
    behavior_axis.legend(frameon=False, ncols=3, fontsize=9, loc="upper center")
    _style_axis(behavior_axis)

    coherence_axis = axes[0, 1]
    coherence_axis.errorbar(
        coefficients,
        coherence,
        yerr=coherence_std,
        color=COLORS["coherence"],
        marker="o",
        markersize=7,
        linewidth=2.3,
        elinewidth=1.5,
        capsize=4,
        label="Mean ± sample SD",
    )
    coherence_axis.axhspan(4.0, 5.0, color="#6F4E9C", alpha=0.06, zorder=0)
    coherence_axis.axhline(5.0, color="#AAB2BD", linestyle=":", linewidth=1)
    coherence_axis.set_title("Response coherence", loc="left")
    coherence_axis.set_ylabel("Judge score (1–5)")
    lower = min(value - spread for value, spread in zip(coherence, coherence_std))
    upper = max(value + spread for value, spread in zip(coherence, coherence_std))
    coherence_axis.set_ylim(max(1.0, lower - 0.15), max(5.12, upper + 0.12))
    max_failure = max(failures)
    failure_text = (
        "No coherence failures"
        if max_failure == 0
        else f"Maximum failure rate: {max_failure:.1%}"
    )
    coherence_axis.text(
        0.98,
        0.06,
        failure_text,
        transform=coherence_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#4B5563",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    coherence_axis.legend(frameon=False, loc="upper left")
    _style_axis(coherence_axis)

    token_axis = axes[1, 0]
    token_axis.plot(
        coefficients,
        tokens,
        color=COLORS["tokens"],
        marker="o",
        markersize=7,
        linewidth=2.3,
    )
    token_axis.fill_between(
        coefficients, tokens, min(tokens) - 1, color=COLORS["tokens"], alpha=0.08
    )
    token_axis.set_title("Response length", loc="left")
    token_axis.set_ylabel("Mean output tokens")
    token_axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    token_axis.set_ylim(min(tokens) - 2.5, max(tokens) + 2.5)
    _style_axis(token_axis)

    tradeoff_axis = axes[1, 1]
    tradeoff_axis.plot(
        truthfulness,
        coherence,
        color="#9AA0A6",
        linewidth=1.3,
        alpha=0.75,
        zorder=1,
    )
    points = tradeoff_axis.scatter(
        truthfulness,
        coherence,
        c=coefficients,
        cmap="viridis",
        s=78,
        edgecolor="white",
        linewidth=1,
        zorder=3,
    )
    for index, (coefficient, x_value, y_value) in enumerate(
        zip(coefficients, truthfulness, coherence)
    ):
        if coefficient == 0.0:
            offset = (8, -18)
        elif coefficient == -0.5:
            offset = (8, 8)
        else:
            offset = (7, 8) if index % 2 == 0 else (7, -14)
        tradeoff_axis.annotate(
            f"α={coefficient:g}",
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            color="#374151",
        )
    tradeoff_axis.set_title("Quality–truthfulness trade-off", loc="left")
    tradeoff_axis.set_xlabel("Truthfulness rate")
    tradeoff_axis.set_ylabel("Mean coherence")
    tradeoff_axis.xaxis.set_major_locator(MultipleLocator(0.005))
    tradeoff_axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    truth_range = max(truthfulness) - min(truthfulness)
    x_margin = max(0.008, truth_range * 0.35)
    tradeoff_axis.set_xlim(
        max(0.0, min(truthfulness) - x_margin),
        min(1.0, max(truthfulness) + x_margin),
    )
    coherence_range = max(coherence) - min(coherence)
    y_margin = max(0.04, coherence_range * 0.35)
    tradeoff_axis.set_ylim(
        min(coherence) - y_margin, max(coherence) + y_margin
    )
    if 0.0 in coefficients:
        baseline_index = coefficients.index(0.0)
        _mark_baseline(
            tradeoff_axis,
            truthfulness[baseline_index],
            coherence[baseline_index],
        )
    colorbar = figure.colorbar(points, ax=tradeoff_axis, pad=0.02, shrink=0.84)
    colorbar.set_label("Steering coefficient α")
    _style_axis(tradeoff_axis)

    for axis in (behavior_axis, coherence_axis, token_axis):
        axis.set_xticks(coefficients)
        axis.set_xlabel("Steering coefficient α")
        if 0.0 in coefficients:
            baseline_index = coefficients.index(0.0)
            if axis is behavior_axis:
                for values in (truthfulness, confidence, apology):
                    _mark_baseline(axis, 0.0, values[baseline_index])
            elif axis is coherence_axis:
                _mark_baseline(axis, 0.0, coherence[baseline_index])
            elif axis is token_axis:
                _mark_baseline(axis, 0.0, tokens[baseline_index])

    figure.supxlabel(
        "Black stars mark the unsteered baseline (α=0). "
        "Coherence error bars show sample standard deviation.",
        fontsize=9,
        color="#6B7280",
    )
    return figure


def default_title(location: Path) -> str:
    root = location if location.is_dir() else location.parent
    return root.name.replace("_", " ").strip().title() or "Experiment Summary"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a scored experiment's summary.json as a dashboard."
    )
    parser.add_argument(
        "results",
        nargs="?",
        type=Path,
        default=Path("results"),
        help="Results directory or path to summary.json (default: results)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output image path (default: <results>/summary_plot.png)",
    )
    parser.add_argument("--title", help="Custom figure title")
    parser.add_argument(
        "--dpi", type=int, default=200, help="Output DPI (default: 200)"
    )
    args = parser.parse_args()

    rows = load_summary(args.results)
    result_directory = args.results if args.results.is_dir() else args.results.parent
    output = args.output or result_directory / "summary_plot.png"
    if output.suffix.lower() not in {".png", ".pdf", ".svg"}:
        parser.error("--output must end in .png, .pdf, or .svg")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    figure = create_dashboard(rows, args.title or default_title(args.results))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()
