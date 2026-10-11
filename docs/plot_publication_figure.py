"""Build the compact README figure from the checked-in result table."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "probe_truthfulness.csv"
OUTPUT = HERE / "figures" / "probe_truthfulness.svg"


def main() -> None:
    with DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    coefficients = [float(row["coefficient"]) for row in rows]
    series = {
        "Layer 9": [float(row["layer_9"]) for row in rows],
        "Layer 18": [float(row["layer_18"]) for row in rows],
        "Layer 27": [float(row["layer_27"]) for row in rows],
    }

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["svg.hashsalt"] = "roles-and-sycophancy"
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    colors = ("#4477AA", "#EE6677", "#228833")
    for (label, values), color in zip(series.items(), colors, strict=True):
        axis.plot(
            coefficients,
            values,
            marker="o",
            linewidth=2,
            markersize=6,
            color=color,
            label=label,
        )

    axis.axvline(0, color="#666666", linewidth=1, linestyle="--", alpha=0.75)
    axis.set(
        title="Sycophancy truthfulness is nearly flat under Userness steering",
        xlabel="Steering coefficient (alpha)",
        ylabel="Truthfulness",
        xticks=coefficients,
        ylim=(0.79, 0.91),
    )
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.legend(frameon=False, ncol=3, loc="lower center")
    axis.text(
        0.01,
        0.02,
        "Same fixed 100 prompts at each layer and coefficient",
        transform=axis.transAxes,
        color="#555555",
        fontsize=9,
    )
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)


if __name__ == "__main__":
    main()
