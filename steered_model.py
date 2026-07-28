"""Candidate-direction activation steering and the in-process Inspect model."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch


class SteeredModel:
    """A loaded causal LM with a mutable activation-steering coefficient."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        steering_module: Any,
        steering_module_path: str,
        target_layer: int,
        unit_direction: Any,
        projection_scale: float,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.target_layer = target_layer
        self.steering_module = steering_module
        self.steering_module_path = steering_module_path
        self.activation_site = f"{steering_module_path} input"
        self.coefficient = 0.0
        self.projection_scale = float(projection_scale)
        direction = torch.as_tensor(unit_direction, dtype=torch.float32)
        if direction.ndim != 1 or direction.numel() == 0:
            raise ValueError(
                f"Steering direction must be a non-empty vector, got {direction.shape}"
            )
        if not torch.isclose(direction.norm(), torch.tensor(1.0), atol=1e-5):
            raise ValueError("Steering direction must be unit normalized")
        self.unit_direction = direction

    @contextmanager
    def steering_hook(self) -> Iterator[None]:
        """Steer only the final prefill position or current cached token."""
        def add_direction(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            if not inputs:
                raise AssertionError("Steering module pre-hook received no inputs")
            states = inputs[0]
            if not isinstance(states, torch.Tensor):
                raise AssertionError("Steering module must receive tensor states first")
            if states.ndim != 3:
                raise AssertionError(f"Expected [batch, sequence, hidden], got {states.shape}")
            if states.shape[-1] != self.unit_direction.numel():
                raise AssertionError(
                    "Steering direction width does not match the resolved module input: "
                    f"{self.unit_direction.numel()} != {states.shape[-1]}"
                )
            direction = self.unit_direction.to(device=states.device, dtype=states.dtype)
            shift = direction * (self.coefficient * self.projection_scale)
            position_mask = torch.zeros(
                states.shape[:-1] + (1,), device=states.device, dtype=states.dtype
            )
            position_mask[:, -1, :] = 1
            steered = states + position_mask * shift
            if steered.shape != states.shape or steered.dtype != states.dtype:
                raise AssertionError("Steering changed activation shape or dtype")
            return (steered,) + inputs[1:]

        handle = self.steering_module.register_forward_pre_hook(add_direction)
        try:
            yield
        finally:
            handle.remove()

    @torch.inference_mode()
    def generate(self, **kwargs: Any) -> Any:
        with self.steering_hook():
            return self.model.generate(use_cache=True, **kwargs)

    @torch.inference_mode()
    def logits(self, **kwargs: Any) -> torch.Tensor:
        """Forward logits through the identical hook path (used by the zero check)."""
        with self.steering_hook():
            return self.model(**kwargs, use_cache=False, return_dict=True).logits


def assert_zero_equivalence(wrapper: SteeredModel, input_ids: torch.Tensor,
                            attention_mask: torch.Tensor) -> None:
    """Confirm alpha=0 is numerically identical to a hook-free forward pass."""
    old = wrapper.coefficient
    try:
        wrapper.coefficient = 0.0
        with torch.inference_mode():
            baseline = wrapper.model(
                input_ids=input_ids, attention_mask=attention_mask,
                use_cache=False, return_dict=True,
            ).logits
            steered = wrapper.logits(input_ids=input_ids, attention_mask=attention_mask)
        torch.testing.assert_close(steered, baseline, rtol=0.0, atol=0.0)
    finally:
        wrapper.coefficient = old


def equal_norm_random_direction(reference_unit: Any, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random = torch.randn(len(reference_unit), generator=generator, dtype=torch.float32)
    return random / random.norm()


def make_inspect_model(
    wrapper: SteeredModel,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Expose the existing local model instance directly to current Inspect AI."""
    from inspect_ai.model import (
        ChatMessage,
        GenerateConfig,
        Model,
        ModelAPI,
        ModelOutput,
        ModelUsage,
    )
    from inspect_ai.tool import ToolChoice, ToolInfo

    class LocalSteeredAPI(ModelAPI):
        def __init__(self) -> None:
            super().__init__(
                model_name=model_name,
                config=GenerateConfig(
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    max_connections=1,
                    adaptive_connections=False,
                    cache=False,
                ),
            )

        async def count_text_tokens(self, text: str) -> int:
            return len(wrapper.tokenizer(text, add_special_tokens=False)["input_ids"])

        async def generate(
            self,
            input: list[ChatMessage],
            tools: list[ToolInfo],
            tool_choice: ToolChoice,
            config: GenerateConfig,
        ) -> ModelOutput:
            if tools:
                raise NotImplementedError("The local evaluation model does not emit tool calls")
            messages = [{"role": message.role, "content": message.text} for message in input]
            encoded = wrapper.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                **(chat_template_kwargs or {}),
            )
            if "attention_mask" not in encoded:
                encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
            device = next(wrapper.model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            max_tokens = config.max_tokens or max_new_tokens
            effective_temperature = temperature if config.temperature is None else config.temperature
            do_sample = effective_temperature is not None and effective_temperature > 0
            generation: dict[str, Any] = {
                **encoded,
                "max_new_tokens": max_tokens,
                "do_sample": do_sample,
                "pad_token_id": wrapper.tokenizer.pad_token_id,
            }
            if do_sample:
                generation["temperature"] = effective_temperature
                generation["top_p"] = top_p if config.top_p is None else config.top_p
            if config.stop_seqs:
                generation["stop_strings"] = config.stop_seqs
                generation["tokenizer"] = wrapper.tokenizer
            generated = wrapper.generate(**generation)
            prompt_tokens = encoded["input_ids"].shape[1]
            new_ids = generated[0, prompt_tokens:]
            completion = wrapper.tokenizer.decode(new_ids, skip_special_tokens=True)
            stop_reason = "max_tokens" if len(new_ids) >= max_tokens else "stop"
            output = ModelOutput.from_content(model_name, completion, stop_reason=stop_reason)
            output.usage = ModelUsage(
                input_tokens=int(prompt_tokens),
                output_tokens=int(len(new_ids)),
                total_tokens=int(prompt_tokens + len(new_ids)),
            )
            output.metadata = {
                "steering_coefficient": wrapper.coefficient,
                "target_layer": wrapper.target_layer,
            }
            return output

    config = GenerateConfig(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_connections=1,
        adaptive_connections=False,
        cache=False,
    )
    return Model(LocalSteeredAPI(), config)
