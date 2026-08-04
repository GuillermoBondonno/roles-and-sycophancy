from __future__ import annotations

import asyncio
from types import SimpleNamespace

import torch
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, get_model
from inspect_ai.model._model import ModelName

from steered_model import (
    STEERING_SCOPE_ALWAYS,
    STEERING_SCOPE_SECOND_ASSISTANT,
    make_inspect_model,
)


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

    reconstructed = get_model(str(ModelName(model)), memoize=False)

    assert str(ModelName(reconstructed)) == "local-steered/example/model"


class _FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(
        self, *_args: object, **_kwargs: object
    ) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def decode(self, _ids: torch.Tensor, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return "response"


class _FakeModel:
    def parameters(self):  # type: ignore[no-untyped-def]
        yield torch.zeros(1)


def _recording_wrapper(coefficient: float) -> tuple[SimpleNamespace, list[float]]:
    applied: list[float] = []
    wrapper = SimpleNamespace(
        tokenizer=_FakeTokenizer(),
        model=_FakeModel(),
        coefficient=coefficient,
        target_layer=7,
    )

    def generate(**kwargs: torch.Tensor) -> torch.Tensor:
        applied.append(float(wrapper.coefficient))
        return torch.cat((kwargs["input_ids"], torch.tensor([[3]])), dim=1)

    wrapper.generate = generate
    return wrapper, applied


def test_second_assistant_scope_leaves_initial_answer_unsteered() -> None:
    wrapper, applied = _recording_wrapper(-3.0)
    model = make_inspect_model(
        wrapper,
        model_name="example/model",
        max_new_tokens=16,
        temperature=0.0,
        top_p=1.0,
        seed=42,
        api_name="test-second-assistant-only",
        steering_scope=STEERING_SCOPE_SECOND_ASSISTANT,
    )

    async def generate(messages):  # type: ignore[no-untyped-def]
        return await model.api.generate(messages, [], None, model.config)

    first = asyncio.run(generate([ChatMessageUser(content="Question")]))
    second = asyncio.run(
        generate([
            ChatMessageUser(content="Question"),
            ChatMessageAssistant(content="Initial answer"),
            ChatMessageUser(content="Are you sure?"),
        ])
    )

    assert applied == [0.0, -3.0]
    assert wrapper.coefficient == -3.0
    assert first.metadata["applied_steering_coefficient"] == 0.0
    assert first.metadata["assistant_response_number"] == 1
    assert second.metadata["applied_steering_coefficient"] == -3.0
    assert second.metadata["assistant_response_number"] == 2


def test_always_scope_steers_single_turn_coherence_response() -> None:
    wrapper, applied = _recording_wrapper(-3.0)
    model = make_inspect_model(
        wrapper,
        model_name="example/model",
        max_new_tokens=16,
        temperature=0.0,
        top_p=1.0,
        seed=42,
        api_name="test-all-assistant-responses",
        steering_scope=STEERING_SCOPE_ALWAYS,
    )

    asyncio.run(
        model.api.generate(
            [ChatMessageUser(content="Coherence prompt")], [], None, model.config
        )
    )

    assert applied == [-3.0]
    assert wrapper.coefficient == -3.0
