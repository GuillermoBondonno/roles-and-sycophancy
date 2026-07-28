"""Controlled role-probe data, feed-forward activation extraction, and fitting."""

from __future__ import annotations

import hashlib
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:  # Pure direction/span tests can run without installing the large torch wheel.
    import torch
except ImportError:  # pragma: no cover - exercised only in dependency-light checks
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class BaseSequence:
    base_id: str
    text: str
    source: str


@dataclass(frozen=True)
class RoleVariant:
    base_id: str
    source: str
    role: str
    text: str
    rendered: str
    input_ids: list[int]
    content_token_indices: list[int]
    offsets: list[tuple[int, int]]


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _looks_neutral(text: str) -> bool:
    """Conservative filter for non-dialogue, non-instructional web prose."""
    lowered = f" {text.lower()} "
    instructional = (
        "?",
        " user:",
        " assistant:",
        "system:",
        " instructions:",
        " please ",
        " click ",
        " sign up ",
        " log in ",
        " you should ",
        " your task ",
    )
    return len(text.split()) >= 80 and not any(marker in lowered for marker in instructional)


def load_neutral_base_sequences(
    tokenizer: Any,
    n_sequences: int,
    sequence_length: int,
    seed: int,
) -> list[BaseSequence]:
    """Stream distinct neutral C4 validation passages and truncate by tokens."""
    from datasets import load_dataset

    stream = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=50_000)
    result: list[BaseSequence] = []
    seen: set[str] = set()
    for row in stream:
        raw = " ".join(str(row["text"]).split())
        if not _looks_neutral(raw):
            continue
        ids = tokenizer(raw, add_special_tokens=False, truncation=True,
                        max_length=sequence_length)["input_ids"]
        if len(ids) < min(64, sequence_length):
            continue
        text = tokenizer.decode(ids, skip_special_tokens=False).strip()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        if digest in seen or not _looks_neutral(text):
            continue
        seen.add(digest)
        result.append(BaseSequence(base_id=f"c4-{digest}", text=text, source="c4"))
        if len(result) == n_sequences:
            break
    if len(result) != n_sequences:
        raise RuntimeError(f"Found only {len(result)} suitable C4 sequences")
    return result


def content_token_indices(
    rendered: str,
    content: str,
    offsets: Iterable[tuple[int, int]],
) -> list[int]:
    """Map a unique content character span to tokens wholly inside that span.

    Tokens crossing a template/content boundary are deliberately excluded.
    The exact coverage assertions make boundary-token loss visible rather than
    silently treating a template fragment as content.
    """
    start = rendered.find(content)
    if start < 0 or rendered.find(content, start + 1) >= 0:
        raise ValueError("Content must occur exactly once in the rendered template")
    end = start + len(content)
    offsets = list(offsets)
    selected = [
        i for i, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_start >= start and token_end <= end
    ]
    if not selected:
        raise ValueError("No content tokens were identified")
    spans = [offsets[i] for i in selected]
    if spans[0][0] != start:
        raise ValueError("Tokenizer merged the start of content with a template boundary")
    if any(left[1] != right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("Content token offsets contain a gap")
    if spans[-1][1] != end:
        # Fast tokenizers can merge the last content character with a
        # template-inserted suffix. Exclude the
        # mixed token and verify explicitly that it is the only missing span.
        boundary_tokens = [
            (token_start, token_end) for token_start, token_end in offsets
            if token_start == spans[-1][1] and token_start < end < token_end
        ]
        if len(boundary_tokens) != 1:
            raise ValueError("Content suffix is not explained by one boundary-crossing token")
    return selected


def render_role_variant(
    tokenizer: Any,
    base: BaseSequence,
    role: str,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> RoleVariant:
    """Render one native chat role and verify the content-token mapping."""
    rendered = tokenizer.apply_chat_template(
        [{"role": role, "content": base.text}],
        tokenize=False,
        add_generation_prompt=False,
        **(chat_template_kwargs or {}),
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    indices = content_token_indices(rendered, base.text, offsets)
    ids = list(encoded["input_ids"])
    covered_end = offsets[indices[-1]][1]
    covered_text = rendered[offsets[indices[0]][0]:covered_end]
    if tokenizer.decode([ids[i] for i in indices], skip_special_tokens=False) != covered_text:
        raise ValueError("Mapped tokens do not decode back to their exact content substring")
    if not base.text.startswith(covered_text):
        raise ValueError("Mapped token span is not a prefix of the base content")
    return RoleVariant(
        base_id=base.base_id,
        source=base.source,
        role=role,
        text=base.text,
        rendered=rendered,
        input_ids=ids,
        content_token_indices=indices,
        offsets=offsets,
    )


def split_base_ids(
    bases: list[BaseSequence], train_fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    ids = np.array([base.base_id for base in bases], dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    cut = int(round(len(ids) * train_fraction))
    cut = min(max(cut, 1), len(ids) - 1)
    train_ids, validation_ids = set(ids[:cut]), set(ids[cut:])
    assert train_ids.isdisjoint(validation_ids)
    return train_ids, validation_ids


def build_role_variants(
    tokenizer: Any,
    bases: list[BaseSequence],
    roles: tuple[str, ...],
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[RoleVariant]:
    if len(roles) < 2 or len(set(roles)) != len(roles) or "user" not in roles:
        raise ValueError("ROLES must contain user and at least one distinct comparison role")
    variants = [
        render_role_variant(tokenizer, base, role, chat_template_kwargs)
        for base in bases
        for role in roles
    ]
    aligned: list[RoleVariant] = []
    for base in bases:
        group = [variant for variant in variants if variant.base_id == base.base_id]
        content_ids = [
            [variant.input_ids[i] for i in variant.content_token_indices]
            for variant in group
        ]
        common_length = min(map(len, content_ids))
        if common_length == 0 or any(
            ids[:common_length] != content_ids[0][:common_length]
            for ids in content_ids[1:]
        ):
            raise ValueError(f"Content tokenization differs across roles for {base.base_id}")
        # Restrict every role copy to exactly the same content-token prefix.
        aligned.extend(
            replace(
                variant,
                content_token_indices=variant.content_token_indices[:common_length],
            )
            for variant in group
        )
    return aligned


def extract_activations(
    model: Any,
    variants: list[RoleVariant],
    steering_module: Any,
    roles: tuple[str, ...],
    max_tokens_per_base_role: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Extract content-token states at the exact module input used for steering."""
    if torch is None:
        raise RuntimeError("PyTorch is required for activation extraction")
    role_to_label = {role: index for index, role in enumerate(roles)}
    activations: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    device = next(model.parameters()).device

    with torch.inference_mode():
        for variant in variants:
            captured: list[Any] = []

            def capture(_module: Any, inputs: tuple[Any, ...]) -> None:
                if not inputs or not isinstance(inputs[0], torch.Tensor):
                    raise AssertionError("Steering module did not receive tensor states first")
                states = inputs[0]
                if states.ndim != 3 or states.shape[0] != 1:
                    raise AssertionError(
                        f"Unexpected steering-site shape: {tuple(states.shape)}"
                    )
                if not states.dtype.is_floating_point:
                    raise AssertionError(f"Unexpected activation dtype: {states.dtype}")
                captured.append(states.detach())

            handle = steering_module.register_forward_pre_hook(capture)
            try:
                input_ids = torch.tensor([variant.input_ids], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids)
                model(input_ids=input_ids, attention_mask=attention_mask,
                      use_cache=False, return_dict=True)
            finally:
                handle.remove()
            if len(captured) != 1:
                raise AssertionError(f"Expected one activation capture, got {len(captured)}")
            states = captured[0][0]
            indices = variant.content_token_indices
            keep = min(len(indices), max_tokens_per_base_role)
            # Same deterministic ordinal positions are retained for all role copies.
            ordinals = np.linspace(0, len(indices) - 1, keep, dtype=int)
            chosen = [indices[int(i)] for i in ordinals]
            selected = states[chosen].to(dtype=torch.float32, device="cpu").numpy()
            if selected.ndim != 2 or selected.shape[0] != keep:
                raise AssertionError(f"Bad selected activation shape: {selected.shape}")
            activations.append(selected)
            labels.extend([role_to_label[variant.role]] * keep)
            for ordinal, token_index in zip(ordinals.tolist(), chosen):
                metadata.append({
                    "base_id": variant.base_id,
                    "source": variant.source,
                    "role": variant.role,
                    "content_token_position": ordinal,
                    "rendered_token_index": token_index,
                    "token_id": variant.input_ids[token_index],
                    "char_start": variant.offsets[token_index][0],
                    "char_end": variant.offsets[token_index][1],
                })
    x = np.concatenate(activations, axis=0).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or len(x) != len(y) or len(metadata) != len(y):
        raise AssertionError("Activation, label, and metadata shapes disagree")
    counts = np.bincount(y, minlength=len(roles))
    if len(set(counts.tolist())) != 1:
        raise AssertionError(f"Role classes are not balanced: {counts.tolist()}")
    return x, y, metadata


def candidate_user_direction(weights: np.ndarray, roles: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return centered raw and unit candidate Userness directions."""
    roles = list(roles)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape[0] != len(roles) or "user" not in roles:
        raise ValueError("Weights must have one row per role including user")
    user_index = roles.index("user")
    other = np.delete(weights, user_index, axis=0)
    raw = weights[user_index] - other.mean(axis=0)
    norm = np.linalg.norm(raw)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Candidate direction has zero or non-finite norm")
    return raw, raw / norm


def projection_scale(held_out_activations: np.ndarray, unit_direction: np.ndarray) -> float:
    projections = np.asarray(held_out_activations) @ np.asarray(unit_direction)
    scale = float(np.std(projections, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Held-out projection standard deviation is not positive")
    return scale


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def train_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    roles: tuple[str, ...],
    regularization_grid: Iterable[float],
    coefficients: Iterable[float],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit and select an unstandardized multinomial logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, log_loss

    candidates: list[tuple[float, float, float, Any]] = []
    for regularization in regularization_grid:
        if regularization <= 0:
            raise ValueError("Regularization strengths must be positive")
        clf = LogisticRegression(
            C=1.0 / regularization,
            penalty="l2",
            solver="lbfgs",
            fit_intercept=True,
            max_iter=5_000,
            random_state=seed,
        )
        clf.fit(x_train, y_train)
        probabilities = clf.predict_proba(x_validation)
        accuracy = float(accuracy_score(y_validation, probabilities.argmax(axis=1)))
        nll = float(log_loss(y_validation, probabilities, labels=np.arange(len(roles))))
        candidates.append((accuracy, -nll, regularization, clf))
    accuracy, neg_nll, selected_regularization, clf = max(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    if list(clf.classes_) != list(range(len(roles))):
        raise AssertionError(f"Unexpected sklearn class order: {clf.classes_}")

    raw, unit = candidate_user_direction(clf.coef_, roles)
    scale = projection_scale(x_validation, unit)
    validation_probabilities = clf.predict_proba(x_validation)
    validation_predictions = validation_probabilities.argmax(axis=1)
    user_index = roles.index("user")
    diagnostic_rows = []
    for alpha in coefficients:
        shifted = x_validation + float(alpha) * scale * unit
        probs = _softmax(shifted @ clf.coef_.T + clf.intercept_)
        row: dict[str, Any] = {
            "coefficient": float(alpha),
            "mean_userness": float(probs[:, user_index].mean()),
        }
        for role_index, role in enumerate(roles):
            row[f"mean_userness_true_{role}"] = float(
                probs[y_validation == role_index, user_index].mean()
            )
        diagnostic_rows.append(row)
    means = [row["mean_userness"] for row in diagnostic_rows]
    monotonic = all(right >= left - 1e-10 for left, right in zip(means, means[1:]))

    per_role_accuracy = {
        role: float(accuracy_score(y_validation == i, validation_predictions == i))
        for i, role in enumerate(roles)
    }
    # The conventional per-class recall is more interpretable than one-vs-rest
    # accuracy; save it under an unambiguous key.
    per_role_recall = {
        role: float((validation_predictions[y_validation == i] == i).mean())
        for i, role in enumerate(roles)
    }
    userness_distributions = {
        role: {
            "mean": float(validation_probabilities[y_validation == i, user_index].mean()),
            "std": float(validation_probabilities[y_validation == i, user_index].std()),
            "quantiles": np.quantile(
                validation_probabilities[y_validation == i, user_index],
                [0.05, 0.25, 0.5, 0.75, 0.95],
            ).tolist(),
        }
        for i, role in enumerate(roles)
    }
    mean_direction = (
        x_train[y_train == user_index].mean(axis=0)
        - x_train[y_train != user_index].mean(axis=0)
    )
    mean_direction_norm = float(np.linalg.norm(mean_direction))
    if mean_direction_norm <= 0 or not np.isfinite(mean_direction_norm):
        raise ValueError("Mean-difference direction has zero or non-finite norm")
    mean_direction_unit = mean_direction / mean_direction_norm
    mean_direction_scale = projection_scale(x_validation, mean_direction_unit)
    artifact = {
        "coefficients": clf.coef_.astype(np.float32),
        "intercept": clf.intercept_.astype(np.float32),
        "direction_raw": raw.astype(np.float32),
        "direction_unit": unit.astype(np.float32),
        "projection_scale": np.asarray(scale, dtype=np.float32),
        "mean_difference_direction": mean_direction.astype(np.float32),
        "mean_difference_direction_unit": mean_direction_unit.astype(np.float32),
        "mean_difference_projection_scale": np.asarray(
            mean_direction_scale, dtype=np.float32
        ),
    }
    diagnostics = {
        "selected_regularization_lambda": selected_regularization,
        "selected_sklearn_C": 1.0 / selected_regularization,
        "validation_accuracy": accuracy,
        "validation_log_loss": -neg_nll,
        "per_role_one_vs_rest_accuracy": per_role_accuracy,
        "per_role_recall": per_role_recall,
        "confusion_matrix": confusion_matrix(
            y_validation, validation_predictions, labels=np.arange(len(roles))
        ).tolist(),
        "validation_userness_distributions": userness_distributions,
        "userness_steering_diagnostic": diagnostic_rows,
        "userness_diagnostic_monotonic": monotonic,
        "regularization_search": [
            {"lambda": reg, "C": 1.0 / reg, "validation_accuracy": acc,
             "validation_log_loss": -nll}
            for acc, nll, reg, _ in candidates
        ],
    }
    return artifact, diagnostics


def save_probe(
    directory: Path,
    artifact: dict[str, Any],
    diagnostics: dict[str, Any],
    metadata: dict[str, Any],
    activation_metadata: list[dict[str, Any]],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "probe.npz", **artifact)
    (directory / "probe_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / "probe_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    diagnostic_rows = diagnostics["userness_steering_diagnostic"]
    (directory / "userness_steering_diagnostic.json").write_text(
        json.dumps(diagnostic_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (directory / "userness_steering_diagnostic.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostic_rows[0]))
        writer.writeheader()
        writer.writerows(diagnostic_rows)
    with (directory / "activation_metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in activation_metadata:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
