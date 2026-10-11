from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from plot_results import create_apology_rate_plot, create_dashboard, load_summary, main


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


def test_loads_and_plots_each_layer_in_sweep_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = []
    for target_layer, fraction in ((9, 0.25), (18, 0.5)):
        for coefficient in (-1.0, 0.0):
            row = _summary_row(coefficient)
            row.update(
                {
                    "experiment": f"layer_{target_layer}",
                    "target_layer": target_layer,
                    "target_layer_fraction": fraction,
                }
            )
            rows.append(row)
    (tmp_path / "layer_sweep_summary.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )

    monkeypatch.setattr(sys, "argv", ["plot_results.py", str(tmp_path)])
    main()

    loaded = load_summary(tmp_path)
    assert len(loaded) == 4
    assert {row["target_layer"] for row in loaded} == {9, 18}
    assert (tmp_path / "layer_9" / "summary_plot.png").stat().st_size > 0
    assert (tmp_path / "layer_18" / "summary_plot.png").stat().st_size > 0
    assert not (tmp_path / "summary_plot.png").exists()


def test_create_dashboard_can_be_saved(tmp_path: Path) -> None:
    figure = create_dashboard(
        [_summary_row(-1.0), _summary_row(0.0)], "Test Experiment"
    )
    output = tmp_path / "plot.png"
    figure.savefig(output)
    plt.close(figure)

    assert output.stat().st_size > 0


def test_create_apology_rate_plot_can_be_saved(tmp_path: Path) -> None:
    rows = []
    for target_layer in (9, 18, 27):
        for coefficient in (-1.0, 0.0):
            row = _summary_row(coefficient)
            row.update(
                {
                    "experiment": f"layer_{target_layer}",
                    "target_layer": target_layer,
                    "target_layer_fraction": target_layer / 36,
                }
            )
            rows.append(row)
    figure = create_apology_rate_plot(rows, "Apology rate")
    output = tmp_path / "apology_rate.svg"
    figure.savefig(output)
    plt.close(figure)

    assert output.stat().st_size > 0
