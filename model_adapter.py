"""Locate an activation-steering site in decoder-only Transformers models."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# Ordered from the most common Hugging Face layout to less common layouts.
_LAYER_STACK_PATHS = (
    "model.layers",
    "model.decoder.layers",
    "transformer.h",
    "transformer.layers",
    "gpt_neox.layers",
    "model.transformer.h",
    "model.transformer.blocks",
    "transformer.blocks",
    "backbone.layers",
    "decoder.layers",
)
_FEED_FORWARD_NAMES = (
    "mlp",
    "feed_forward",
    "ffn",
    "ff",
    "block_sparse_moe",
    "moe",
    "fc1",
    "dense_h_to_4h",
)


@dataclass(frozen=True)
class SteeringSite:
    """Resolved module whose input is both probed and steered."""

    module: Any
    module_path: str
    target_layer: int

    @property
    def activation_site(self) -> str:
        return f"{self.module_path} input"


def _path_parts(path: str) -> list[str]:
    # Accept both ``layers.3`` and the more familiar ``layers[3]`` notation.
    normalized = re.sub(r"\[([^]]+)\]", r".\1", path).strip(".")
    return [part for part in normalized.split(".") if part]


def resolve_module_path(root: Any, path: str) -> Any:
    """Resolve attributes and integer indexes in a dotted module path."""
    current = root
    traversed: list[str] = []
    for part in _path_parts(path):
        traversed.append(part)
        try:
            current = current[int(part)] if part.isdigit() else getattr(current, part)
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            location = ".".join(traversed)
            raise ValueError(
                f"Cannot resolve module path {path!r}; failed at {location!r}"
            ) from exc
    return current


def _direct_feed_forward(layer: Any) -> tuple[Any, str] | None:
    for name in _FEED_FORWARD_NAMES:
        module = getattr(layer, name, None)
        if module is not None and hasattr(module, "register_forward_pre_hook"):
            return module, name
    return None


def _candidate_layer_stacks(model: Any) -> list[tuple[str, Any]]:
    candidates: list[tuple[str, Any]] = []
    seen: set[int] = set()
    for path in _LAYER_STACK_PATHS:
        try:
            stack = resolve_module_path(model, path)
        except ValueError:
            continue
        if id(stack) not in seen:
            candidates.append((path, stack))
            seen.add(id(stack))

    # This fallback also supports custom architectures that register their block
    # stack under a name not known above.
    if hasattr(model, "named_modules"):
        for path, module in model.named_modules():
            if not path or id(module) in seen:
                continue
            if module.__class__.__name__ not in {"ModuleList", "Sequential"}:
                continue
            candidates.append((path, module))
            seen.add(id(module))
    return candidates


def resolve_steering_site(
    model: Any,
    target_layer: int,
    module_path: str | None = None,
) -> SteeringSite:
    """Resolve the feed-forward input at ``target_layer``.

    ``module_path`` can override discovery for any custom architecture. It may
    contain ``{layer}``, for example ``transformer.blocks.{layer}.ffn``.
    """
    if target_layer < 0:
        raise IndexError(f"TARGET_LAYER must be non-negative, got {target_layer}")

    if module_path is not None:
        rendered_path = module_path.format(layer=target_layer)
        module = resolve_module_path(model, rendered_path)
        if not hasattr(module, "register_forward_pre_hook"):
            raise TypeError(f"Resolved object at {rendered_path!r} is not a torch module")
        return SteeringSite(module, rendered_path, target_layer)

    inspected: list[str] = []
    for stack_path, layers in _candidate_layer_stacks(model):
        try:
            layer_count = len(layers)
        except TypeError:
            continue
        inspected.append(f"{stack_path} ({layer_count} layers)")
        if target_layer >= layer_count:
            continue
        found = _direct_feed_forward(layers[target_layer])
        if found is not None:
            module, child_name = found
            return SteeringSite(
                module=module,
                module_path=f"{stack_path}.{target_layer}.{child_name}",
                target_layer=target_layer,
            )

    detail = ", ".join(inspected) if inspected else "no decoder layer stacks"
    raise ValueError(
        "Could not discover a feed-forward steering module; inspected "
        f"{detail}. Set STEERING_MODULE_PATH to a dotted path containing "
        "{layer}, pointing at the module whose input should be probed and steered."
    )
