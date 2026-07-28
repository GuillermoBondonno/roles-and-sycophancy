from __future__ import annotations

import pytest

from model_adapter import resolve_steering_site


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
