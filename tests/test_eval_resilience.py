from __future__ import annotations

from types import SimpleNamespace

import pytest

from run import require_usable_eval


def _log(
    *,
    status: str = "success",
    completed: int = 4,
    total: int = 4,
    error: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        error=error,
        results=SimpleNamespace(completed_samples=completed, total_samples=total),
    )


def test_complete_eval_is_accepted() -> None:
    log = _log()
    assert require_usable_eval([log], 4, "coherence") is log


def test_partial_eval_is_accepted(caplog: pytest.LogCaptureFixture) -> None:
    log = _log(status="error", completed=3)
    assert require_usable_eval([log], 4, "coherence") is log
    assert "completed 3/4 samples" in caplog.text


def test_eval_with_no_usable_samples_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="no usable samples"):
        require_usable_eval([_log(status="error", completed=0)], 4, "coherence")
