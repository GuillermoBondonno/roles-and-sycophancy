from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch is a platform-specific dependency")

from model_adapter import resolve_steering_site  # noqa: E402
from steered_model import SteeredModel, assert_zero_equivalence  # noqa: E402


class _ToyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.Identity()

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.mlp(states)


class _ToyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_ToyLayer()])


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.model = _ToyBackbone()

    def forward(self, input_ids: torch.Tensor, attention_mask=None,
                use_cache=False, return_dict=True):
        del attention_mask, use_cache, return_dict
        return SimpleNamespace(logits=self.model.layers[0](input_ids.float()))


def _wrapper() -> SteeredModel:
    model = _ToyModel()
    site = resolve_steering_site(model, target_layer=0)
    return SteeredModel(
        model, tokenizer=None, steering_module=site.module,
        steering_module_path=site.module_path, target_layer=0,
        unit_direction=np.array([1.0, 0.0, 0.0]), projection_scale=2.0,
    )


def test_hook_shape_site_and_sign() -> None:
    wrapper = _wrapper()
    states = torch.zeros((2, 3, 3))
    wrapper.coefficient = 1.5
    positive = wrapper.logits(input_ids=states)
    wrapper.coefficient = -1.5
    negative = wrapper.logits(input_ids=states)
    assert positive.shape == states.shape
    torch.testing.assert_close(positive[:, :-1], states[:, :-1])
    torch.testing.assert_close(positive[:, -1, 0], torch.full((2,), 3.0))
    torch.testing.assert_close(negative[:, -1, 0], torch.full((2,), -3.0))
    torch.testing.assert_close(positive[:, -1, 1:], torch.zeros((2, 2)))


def test_coefficient_zero_exactly_matches_unhooked_logits() -> None:
    wrapper = _wrapper()
    states = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
    assert_zero_equivalence(wrapper, states, torch.ones((2, 3), dtype=torch.long))

