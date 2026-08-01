from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from plot_results import create_dashboard, load_summary


def _summary_row(coefficient: float) -> dict[str, float]:
    return {
        "coefficient": coefficient,
        "sycophancy_answer_confidence": 0.6,
        "sycophancy_apologize_rate": 0.4,
        "sycophancy_truthfulness": 0.58,
        "coherence_mean": 4.8,
        "coherence_std": 0.2,
        "coherence_failure_rate": 0.0,
        "mean_output_tokens": 250.0,
    }


def test_load_summary_sorts_coefficients(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps([_summary_row(0.0), _summary_row(-1.0)]), encoding="utf-8"
    )

    rows = load_summary(tmp_path)

    assert [row["coefficient"] for row in rows] == [-1.0, 0.0]


def test_load_summary_rejects_missing_metrics(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text(
        json.dumps([{"coefficient": 0.0}]), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="missing metrics"):
        load_summary(tmp_path)


def test_create_dashboard_can_be_saved(tmp_path: Path) -> None:
    figure = create_dashboard(
        [_summary_row(-1.0), _summary_row(0.0)], "Test Experiment"
    )
    output = tmp_path / "plot.png"
    figure.savefig(output)
    plt.close(figure)

    assert output.stat().st_size > 0
