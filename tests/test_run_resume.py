from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import run


def sample(completion: str = "response") -> SimpleNamespace:
    return SimpleNamespace(
        error=None,
        output=SimpleNamespace(completion=completion),
    )


def log(samples: int, location: str) -> SimpleNamespace:
    return SimpleNamespace(
        status="started",
        samples=[sample() for _ in range(samples)],
        location=location,
    )


def test_generate_or_resume_task_reuses_complete_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete = log(100, "complete.eval")
    calls: list[str] = []
    monkeypatch.setattr(
        run,
        "existing_generation_logs",
        lambda *_args: [complete],
    )
    monkeypatch.setattr(
        run,
        "require_generated_eval",
        lambda *_args: calls.append("generated"),
    )

    selected = run.generate_or_resume_task(
        object(), {}, Path("logs"), "sycophancy", 100, -7.0
    )

    assert selected is complete
    assert calls == []


def test_generate_or_resume_task_retries_best_partial_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect_ai

    smaller = log(3, "smaller.eval")
    best = log(11, "best.eval")
    resumed = SimpleNamespace(
        status="success",
        samples=[sample() for _ in range(100)],
        location="resumed.eval",
    )
    retried: list[tuple[SimpleNamespace, dict[str, object]]] = []
    monkeypatch.setattr(
        run,
        "existing_generation_logs",
        lambda *_args: [smaller, best],
    )
    monkeypatch.setattr(
        inspect_ai,
        "eval_retry",
        lambda selected, **kwargs: (
            retried.append((selected, kwargs)) or [resumed]
        ),
    )

    selected = run.generate_or_resume_task(
        object(), {}, Path("logs"), "sycophancy", 100, -7.0
    )

    assert selected is resumed
    assert retried[0][0] is best
    assert retried[0][1]["score"] is False
    assert retried[0][1]["log_dir"] == "logs"


def test_cli_routes_explicit_resume_directories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[float, Path, Path]] = []
    monkeypatch.setattr(
        run,
        "resume_main",
        lambda fraction, artifacts, results: calls.append(
            (fraction, artifacts, results)
        ),
    )

    run.cli([
        "--resume-generation",
        "0.75",
        "--artifacts-dir",
        "artifacts_aug_2_v2",
        "--results-dir",
        "results_aug_2_v2",
    ])

    assert calls == [(
        0.75,
        Path("artifacts_aug_2_v2"),
        Path("results_aug_2_v2"),
    )]
