from __future__ import annotations

import pytest

from model_adapter import (
    decoder_layer_count,
    resolve_layer_fractions,
    resolve_steering_site,
)


class _Hookable:
    def register_forward_pre_hook(self, _hook):  # pragma: no cover - capability marker
        raise NotImplementedError


class _Block:
    def __init__(self, feed_forward_name: str) -> None:
        setattr(self, feed_forward_name, _Hookable())


class _Object:
    pass


def _nested_model() -> _Object:
    model = _Object()
    model.transformer = _Object()
    model.transformer.h = [_Block("feed_forward"), _Block("feed_forward")]
    return model


def test_discovers_an_alternative_decoder_layout_without_model_imports() -> None:
    site = resolve_steering_site(_nested_model(), target_layer=1)
    assert site.module_path == "transformer.h.1.feed_forward"
    assert site.activation_site == "transformer.h.1.feed_forward input"


def test_explicit_path_supports_custom_architectures() -> None:
    model = _nested_model()
    site = resolve_steering_site(
        model, target_layer=0, module_path="transformer.h[{layer}].feed_forward"
    )
    assert site.module is model.transformer.h[0].feed_forward


def test_discovery_error_explains_the_override() -> None:
    with pytest.raises(ValueError, match="STEERING_MODULE_PATH"):
        resolve_steering_site(_Object(), target_layer=0)


def test_negative_layers_are_rejected() -> None:
    with pytest.raises(IndexError, match="non-negative"):
        resolve_steering_site(_nested_model(), target_layer=-1)


def test_decoder_layer_count_uses_discovered_stack() -> None:
    assert decoder_layer_count(_nested_model()) == 2


def test_layer_fractions_resolve_against_full_depth() -> None:
    assert resolve_layer_fractions((0.0, 0.25, 0.5, 0.75, 1.0), 28) == [
        (0.0, 0),
        (0.25, 7),
        (0.5, 14),
        (0.75, 21),
        (1.0, 27),
    ]


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan")])
def test_invalid_layer_fractions_are_rejected(fraction: float) -> None:
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        resolve_layer_fractions((fraction,), 28)


def test_fractions_that_collapse_to_one_layer_are_rejected() -> None:
    with pytest.raises(ValueError, match="both map to layer"):
        resolve_layer_fractions((0.0, 0.1), 2)
