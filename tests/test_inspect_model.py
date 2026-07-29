from __future__ import annotations

from types import SimpleNamespace

from inspect_ai.model._model import ModelName

from steered_model import make_inspect_model


def test_local_steered_api_has_inspect_registry_info() -> None:
    wrapper = SimpleNamespace(tokenizer=None)

    model = make_inspect_model(
        wrapper,
        model_name="example/model",
        max_new_tokens=16,
        temperature=0.0,
        top_p=1.0,
        seed=42,
    )

    assert str(ModelName(model)) == "local-steered/example/model"
