from __future__ import annotations

from types import SimpleNamespace

import pytest

from evals import sycophancy_summary
from score_results import fully_scored


def _sycophancy_log(values: list[dict[str, object]]) -> SimpleNamespace:
    return SimpleNamespace(
        samples=[
            SimpleNamespace(
                scores={"sycophancy_scorer": SimpleNamespace(value=value)}
            )
            for value in values
        ],
        # Deliberately wrong aggregates demonstrate that the summary reads the
        # complete per-sample dictionaries instead of trusting inspect-evals.
        results=SimpleNamespace(scores=[]),
    )


def test_sycophancy_summary_recomputes_conditional_rates() -> None:
    log = _sycophancy_log([
        {
            "original_answer": "C",
            "admits_mistake": "I",
            "confidence": 1,
            "apologize_rate": 0,
            "truthfulness": 1,
        },
        {
            "original_answer": "C",
            "admits_mistake": "C",
            "confidence": 0,
            "apologize_rate": 1,
            "truthfulness": 0,
        },
        {
            "original_answer": "I",
            "admits_mistake": "C",
            "confidence": 0,
            "apologize_rate": 0,
            "truthfulness": 1,
        },
    ])

    assert sycophancy_summary(log) == pytest.approx({
        "sycophancy_answer_confidence": 0.5,
        "sycophancy_apologize_rate": 0.5,
        "sycophancy_truthfulness": 2 / 3,
    })


def test_sycophancy_summary_rejects_undefined_conditional_rates() -> None:
    log = _sycophancy_log([{
        "original_answer": "I",
        "confidence": 0,
        "apologize_rate": 0,
        "truthfulness": 0,
    }])
    with pytest.raises(RuntimeError, match="no scored initial answer was correct"):
        sycophancy_summary(log)


def test_fully_scored_accepts_deduplicated_scorer_names() -> None:
    log = SimpleNamespace(samples=[
        SimpleNamespace(scores={"sycophancy_scorer": object()}),
        SimpleNamespace(scores={"sycophancy_scorer-1": object()}),
    ])
    assert fully_scored(log, "sycophancy")
