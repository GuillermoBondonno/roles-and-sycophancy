from __future__ import annotations

import numpy as np
import pytest

from evals import parse_judge_json
from probe import (
    candidate_user_direction,
    content_token_indices,
    projection_scale,
    train_probe,
)


def test_content_span_excludes_native_template_tokens() -> None:
    rendered = "<role>alpha beta</role>"
    # Pretend a fast tokenizer emitted two template tokens and two content tokens.
    offsets = [(0, 6), (6, 11), (11, 16), (16, 23)]
    assert content_token_indices(rendered, "alpha beta", offsets) == [1, 2]


def test_content_span_rejects_boundary_merge() -> None:
    rendered = "<role>alpha beta</role>"
    # The first token improperly includes the final '>' from the role tag.
    offsets = [(0, 5), (5, 11), (11, 16), (16, 23)]
    with pytest.raises(ValueError, match="start"):
        content_token_indices(rendered, "alpha beta", offsets)


def test_content_span_excludes_suffix_merged_with_template() -> None:
    rendered = "<role>alpha beta\n</role>"
    offsets = [(0, 6), (6, 11), (11, 17), (17, 24)]
    # " beta\n" crosses the content/template boundary and is not a pure
    # content-token activation, so only "alpha" is retained.
    assert content_token_indices(rendered, "alpha beta", offsets) == [1]


def test_content_span_accepts_overlapping_byte_token_offsets() -> None:
    rendered = "<role>a 辨 b</role>"
    # Byte-level tokenizers can split one Unicode character into tokens whose
    # character offsets overlap. Their union still covers the content exactly.
    offsets = [(0, 6), (6, 8), (7, 8), (7, 8), (8, 10), (10, 17)]
    assert content_token_indices(rendered, "a 辨 b", offsets) == [1, 2, 3, 4]


def test_content_span_rejects_actual_uncovered_gap() -> None:
    rendered = "<role>alpha beta</role>"
    offsets = [(0, 6), (6, 11), (12, 16), (16, 23)]
    with pytest.raises(ValueError, match="gap"):
        content_token_indices(rendered, "alpha beta", offsets)


def test_candidate_direction_is_centered_and_scaled_on_holdout() -> None:
    roles = ("user", "assistant", "tool")
    weights = np.array([[3.0, 1.0], [0.0, 1.0], [1.0, -1.0]])
    raw, unit = candidate_user_direction(weights, roles)
    expected = weights[0] - weights[1:].mean(axis=0)
    np.testing.assert_allclose(raw, expected)
    np.testing.assert_allclose(np.linalg.norm(unit), 1.0)
    held_out = np.array([[0.0, 0.0], [1.0, 2.0], [-1.0, 1.0], [2.0, -2.0]])
    expected_scale = np.std(held_out @ unit, ddof=1)
    assert projection_scale(held_out, unit) == pytest.approx(expected_scale)


def test_judge_parser_validates_all_dimensions() -> None:
    parsed = parse_judge_json(
        '```json\n{"grammar":5,"local_consistency":4,"global_consistency":4,'
        '"relevance":5,"non_degeneracy":5,"reason":"clear"}\n```'
    )
    assert parsed["grammar"] == 5
    assert parsed["reason"] == "clear"


def test_probe_monotonicity_is_independent_of_coefficient_order() -> None:
    rng = np.random.default_rng(42)
    centers = np.array([[3.0, 0.0], [-3.0, 0.0], [0.0, 3.0]])
    x = np.concatenate([center + rng.normal(0, 0.2, (12, 2)) for center in centers])
    y = np.repeat(np.arange(3), 12)
    train = np.tile([True] * 8 + [False] * 4, 3)
    _, diagnostics = train_probe(
        x[train],
        y[train],
        x[~train],
        y[~train],
        ("user", "assistant", "system"),
        (1.0,),
        (1.0, -1.0, 0.0),
        42,
    )
    assert diagnostics["userness_diagnostic_monotonic"] is True
    assert [
        row["coefficient"] for row in diagnostics["userness_steering_diagnostic"]
    ] == [1.0, -1.0, 0.0]
