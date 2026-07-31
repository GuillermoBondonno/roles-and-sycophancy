from __future__ import annotations

from types import SimpleNamespace

import pytest

from run import require_generated_eval


def test_generation_without_results_is_accepted() -> None:
    samples = [
        SimpleNamespace(
            error=None,
            output=SimpleNamespace(completion="a generated response"),
        )
        for _ in range(4)
    ]
    log = SimpleNamespace(status="success", error=None, results=None, samples=samples)
    assert require_generated_eval([log], 4, "coherence") is log


def test_generation_with_no_responses_is_rejected() -> None:
    sample = SimpleNamespace(error=RuntimeError("failed"), output=None)
    log = SimpleNamespace(status="error", error=None, results=None, samples=[sample])
    with pytest.raises(RuntimeError, match="generated no usable responses"):
        require_generated_eval([log], 1, "coherence")
