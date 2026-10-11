"""Train and steer one loaded model across a configured decoder-layer sweep."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import logging
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from tqdm.auto import tqdm

import config
from evals import coherence_task
from model_adapter import (
    decoder_layer_count,
    resolve_layer_fractions,
    resolve_steering_site,
)
from probe import (
    build_role_variants,
    extract_activations,
    load_neutral_base_sequences,
    save_probe,
    seed_everything,
    split_base_ids,
    train_probe,
)
from steered_model import (
    STEERING_SCOPE_ALWAYS,
    STEERING_SCOPE_SECOND_ASSISTANT,
    SteeredModel,
    assert_zero_equivalence,
    equal_norm_random_direction,
    make_inspect_model,
)
from utils import send_ntfy_message


logger = logging.getLogger(__name__)
SAMPLE_CONTRACT_FILENAME = "evaluation_sample_contract.json"
STEERING_DIRECTION_PROBE = "probe"
STEERING_DIRECTION_MEAN_DIFFERENCE = "mean-difference"
STEERING_DIRECTION_RANDOM_CONTROL = "random-control"
STEERING_DIRECTION_FIELDS = {
    STEERING_DIRECTION_PROBE: ("direction_unit", "projection_scale"),
    STEERING_DIRECTION_MEAN_DIFFERENCE: (
        "mean_difference_direction_unit",
        "mean_difference_projection_scale",
    ),
}


def logged_steering_direction(metadata: dict[str, Any]) -> str:
    """Read direction identity while treating legacy logs as probe runs."""
    return str(metadata.get("steering_direction", STEERING_DIRECTION_PROBE))


def notify(message: str) -> None:
    """Send a run notification without allowing ntfy failures to stop the run."""
    if not config.NTFY_TOPIC:
        return
    try:
        send_ntfy_message(message, topic=config.NTFY_TOPIC)
    except Exception:
        logger.exception("Failed to send ntfy notification: %s", message)


def log_stage(message: str) -> float:
    logger.info(message)
    return time.monotonic()


def log_stage_done(message: str, started_at: float) -> None:
    elapsed = time.monotonic() - started_at
    logger.info("%s completed in %.1fs", message, elapsed)
    notify(f"Stage completed: {message} ({elapsed:.1f}s)")


def load_model_once() -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading tokenizer: %s", config.MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME, **config.TOKENIZER_LOAD_KWARGS
    )
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer with offset mappings is required")
    if tokenizer.chat_template is None and "chat_template" not in config.CHAT_TEMPLATE_KWARGS:
        raise RuntimeError(
            "The selected model has no native chat template and none was configured"
        )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("The tokenizer needs either a pad token or an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    elif device.type == "cpu":
        dtype = torch.bfloat16
    elif device.type == "mps":
        dtype = torch.float16
    else:
        raise RuntimeError(f"Unsupported device: {device}")

    model_kwargs = dict(config.MODEL_LOAD_KWARGS)
    model_kwargs.setdefault("dtype", dtype)
    logger.info("Loading model on %s with dtype %s", device, dtype)
    model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, **model_kwargs)
    if "device_map" not in model_kwargs:
        model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def coefficient_slug(coefficient: float) -> str:
    sign = "p" if coefficient >= 0 else "m"
    return f"{sign}{abs(coefficient):g}".replace(".", "p")


def validate_steering_coefficients(values: tuple[float, ...]) -> tuple[float, ...]:
    """Return finite, unique coefficients with collision-free path slugs."""
    coefficients = tuple(float(value) for value in values)
    if not coefficients:
        raise ValueError("At least one steering coefficient is required")
    if not all(math.isfinite(value) for value in coefficients):
        raise ValueError("Steering coefficients must all be finite")

    by_slug: dict[str, float] = {}
    for coefficient in coefficients:
        slug = coefficient_slug(coefficient)
        previous = by_slug.get(slug)
        if previous is not None:
            if math.isclose(previous, coefficient, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Duplicate steering coefficient: {coefficient:g}")
            raise ValueError(
                f"Steering coefficients {previous!r} and {coefficient!r} both "
                f"map to alpha_{slug}"
            )
        by_slug[slug] = coefficient
    return coefficients


def fraction_slug(fraction: float) -> str:
    return f"{fraction:g}".replace(".", "p")


def layer_directory_name(fraction: float, layer: int) -> str:
    return f"layer_{layer}_fraction_{fraction_slug(fraction)}"


def require_generated_eval(logs: list[Any], expected_samples: int, name: str) -> Any:
    """Accept a generation-only log if it contains at least one usable response."""
    if len(logs) != 1:
        raise RuntimeError(f"{name} returned {len(logs)} logs; expected exactly one")
    log = logs[0]
    if log.status not in {"success", "error"}:
        detail = f": {log.error.message}" if log.error is not None else ""
        raise RuntimeError(f"{name} generation ended with status {log.status!r}{detail}")
    samples = log.samples or []
    completed = sum(
        sample.error is None
        and sample.output is not None
        and bool(sample.output.completion)
        for sample in samples
    )
    if completed == 0:
        raise RuntimeError(
            f"{name} generated no usable responses out of {len(samples)} samples"
        )
    if completed != expected_samples or len(samples) != expected_samples:
        logger.warning(
            "%s generated %d/%d usable responses (expected %d)",
            name, completed, len(samples), expected_samples,
        )
    return log


def usable_generation_count(log: Any) -> int:
    """Count samples that contain a complete target-model response."""
    return sum(
        sample.error is None
        and sample.output is not None
        and bool(sample.output.completion)
        for sample in (log.samples or [])
    )


def generation_evaluation_name(log: Any) -> str | None:
    """Return the task name for logs generated by this experiment."""
    for candidate in (
        log.eval.task_display_name,
        log.eval.task_registry_name,
        log.eval.task,
    ):
        if candidate and (candidate == "sycophancy" or candidate.endswith("/sycophancy")):
            return "sycophancy"
        if candidate and (candidate == "coherence" or candidate.endswith("/coherence")):
            return "coherence"
    return None


def existing_generation_logs(
    log_dir: Path,
    evaluation: str,
    coefficient: float,
    steering_scope: str | None = None,
    steering_direction: str = STEERING_DIRECTION_PROBE,
) -> list[Any]:
    """Read matching completed or interrupted logs from one alpha directory."""
    from inspect_ai.log import read_eval_log

    logs: list[Any] = []
    if not log_dir.exists():
        return logs
    for path in sorted(log_dir.iterdir()):
        if not path.is_file() or path.suffix not in {".eval", ".json"}:
            continue
        try:
            log = read_eval_log(path)
        except Exception:
            logger.exception("Could not read existing Inspect log %s", path)
            continue
        metadata = log.eval.metadata or {}
        try:
            logged_coefficient = float(metadata["steering_coefficient"])
        except (KeyError, TypeError, ValueError):
            continue
        matches_identity = (
            generation_evaluation_name(log) == evaluation
            and math.isclose(logged_coefficient, coefficient, rel_tol=0.0, abs_tol=1e-12)
        )
        if not matches_identity:
            continue
        logged_scope = metadata.get("steering_scope")
        if steering_scope is not None and logged_scope != steering_scope:
            raise RuntimeError(
                f"Existing {evaluation} log {path} uses steering_scope="
                f"{logged_scope!r}, expected {steering_scope!r}. Use a new results "
                "directory rather than mixing intervention protocols."
            )
        logged_direction = logged_steering_direction(metadata)
        if logged_direction != steering_direction:
            raise RuntimeError(
                f"Existing {evaluation} log {path} uses steering_direction="
                f"{logged_direction!r}, expected {steering_direction!r}. Use a new "
                "results directory rather than mixing steering directions."
            )
        logs.append(log)
    return logs


def retry_safe_log(log: Any) -> Any:
    """Neutralize Inspect 0.3.251's ``False``-as-shuffle-seed retry bug.

    Inspect's retry loader checks whether ``sample_shuffle is not None`` and
    then passes a false value directly to ``Dataset.shuffle``. Since ``False``
    is also integer seed zero, a task originally configured with
    ``sample_shuffle=False`` is unexpectedly shuffled during retry. Preserve
    the original log and give ``eval_retry`` a copy whose task-level shuffle
    override is unset; the task's own ``shuffle=False`` argument remains in
    ``task_args_passed`` and reconstructs the intended dataset order.
    """
    eval_spec = getattr(log, "eval", None)
    eval_config = getattr(eval_spec, "config", None)
    if eval_config is None or getattr(eval_config, "sample_shuffle", None) is not False:
        return log
    safe = deepcopy(log)
    safe.eval.config.sample_shuffle = None
    return safe


def _sample_payload(sample: Any) -> dict[str, Any]:
    """Return a stable JSON representation of an Inspect dataset sample."""
    if not hasattr(sample, "model_dump"):
        raise TypeError(f"Expected an Inspect Sample, got {type(sample).__name__}")
    payload = sample.model_dump(mode="json")
    if not isinstance(payload, dict):
        raise TypeError("Inspect Sample.model_dump() did not return an object")
    return payload


def task_sample_contract(
    task: Any,
    evaluation: str,
    expected_samples: int,
    steering_scope: str | None = None,
) -> dict[str, Any]:
    """Fingerprint the exact ordered task samples selected before generation."""
    samples = list(task.dataset)[:expected_samples]
    if len(samples) != expected_samples:
        raise ValueError(
            f"{evaluation} contains {len(samples)} samples, expected "
            f"{expected_samples}"
        )
    records: list[dict[str, str]] = []
    for sample in samples:
        sample_id = getattr(sample, "id", None)
        if sample_id is None:
            raise ValueError(f"{evaluation} contains a sample without an id")
        serialized = json.dumps(
            _sample_payload(sample),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        records.append({
            "id": str(sample_id),
            "sample_sha256": hashlib.sha256(serialized).hexdigest(),
        })
    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{evaluation} selected duplicate sample ids")
    contract = {
        "evaluation": evaluation,
        "sample_count": expected_samples,
        "samples": records,
    }
    if steering_scope is not None:
        contract["steering_scope"] = steering_scope
    return contract


def ensure_evaluation_sample_contract(
    result_root: Path,
    contracts: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Persist or verify immutable sample IDs and payloads for a layer run."""
    payload = {"version": 1, "evaluations": contracts}
    path = result_root / SAMPLE_CONTRACT_FILENAME
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                f"Evaluation sample contract changed since the original run: {path}. "
                "Do not resume with different samples, prompts, targets, or metadata."
            )
        return contracts

    result_root.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return contracts


def assert_log_matches_sample_contract(
    log: Any,
    contract: dict[str, Any],
) -> None:
    """Reject a generated or reused log whose selected dataset has drifted."""
    evaluation = str(contract["evaluation"])
    expected_ids = [str(row["id"]) for row in contract["samples"]]
    dataset = getattr(getattr(log, "eval", None), "dataset", None)
    dataset_ids = getattr(dataset, "sample_ids", None)
    if dataset_ids is None:
        actual_ids = [str(sample.id) for sample in (log.samples or [])]
    else:
        actual_ids = [str(sample_id) for sample_id in dataset_ids]
    if actual_ids != expected_ids:
        missing = sorted(set(expected_ids).difference(actual_ids))
        unexpected = sorted(set(actual_ids).difference(expected_ids))
        raise RuntimeError(
            f"{evaluation} log does not match {SAMPLE_CONTRACT_FILENAME}: "
            f"expected {len(expected_ids)} ordered ids, got {len(actual_ids)}; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    logged_ids = [str(sample.id) for sample in (log.samples or [])]
    if len(logged_ids) != len(expected_ids) or set(logged_ids) != set(expected_ids):
        raise RuntimeError(
            f"{evaluation} log samples do not contain exactly the contracted ids"
        )


def generate_or_resume_task(
    task: Any,
    common: dict[str, Any],
    log_dir: Path,
    evaluation: str,
    expected_samples: int,
    coefficient: float,
    steering_scope: str,
    steering_direction: str = STEERING_DIRECTION_PROBE,
) -> Any:
    """Reuse a complete log or retry the best interrupted log before starting anew."""
    from inspect_ai import eval as inspect_eval
    from inspect_ai import eval_retry

    existing = existing_generation_logs(
        log_dir, evaluation, coefficient, steering_scope, steering_direction
    )
    complete = [
        log
        for log in existing
        if len(log.samples or []) == expected_samples
        and usable_generation_count(log) == expected_samples
    ]
    if complete:
        selected = complete[-1]
        logger.info(
            "alpha=%g: reusing complete %s log %s",
            coefficient,
            evaluation,
            selected.location,
        )
        return selected

    partial = [log for log in existing if log.samples]
    if partial:
        selected = max(partial, key=usable_generation_count)
        logger.info(
            "alpha=%g: resuming %s from %d/%d usable samples in %s",
            coefficient,
            evaluation,
            usable_generation_count(selected),
            expected_samples,
            selected.location,
        )
        selected_for_retry = retry_safe_log(selected)
        return require_generated_eval(
            eval_retry(
                selected_for_retry,
                log_dir=str(log_dir),
                display="plain",
                max_samples=1,
                max_tasks=1,
                score=False,
                fail_on_error=False,
                continue_on_fail=True,
                retry_on_error=0,
            ),
            expected_samples,
            evaluation,
        )

    return require_generated_eval(
        inspect_eval(task, limit=expected_samples, score=False, **common),
        expected_samples,
        evaluation,
    )


def run_generations(
    wrapper: SteeredModel,
    sycophancy_model: Any,
    coherence_model: Any,
    coefficients: tuple[float, ...],
    result_root: Path,
    sycophancy_limit: int,
    coherence_limit: int,
    target_layer_fraction: float,
    steering_direction: str = STEERING_DIRECTION_PROBE,
) -> list[dict[str, Any]]:
    """Generate and persist all target-model responses without judge calls."""
    from inspect_evals.sycophancy import sycophancy

    coefficients = validate_steering_coefficients(coefficients)
    sycophancy_eval = sycophancy(shuffle=False, scorer_model=sycophancy_model)
    coherence_eval = coherence_task(config.COHERENCE_DATASET, coherence_model)
    contracts = ensure_evaluation_sample_contract(
        result_root,
        {
            "sycophancy": task_sample_contract(
                sycophancy_eval,
                "sycophancy",
                sycophancy_limit,
                STEERING_SCOPE_SECOND_ASSISTANT,
            ),
            "coherence": task_sample_contract(
                coherence_eval,
                "coherence",
                coherence_limit,
                STEERING_SCOPE_ALWAYS,
            ),
        },
    )
    manifest = load_generation_manifest(result_root)
    manifest_directions = {
        str(row.get("steering_direction", STEERING_DIRECTION_PROBE))
        for row in manifest
    }
    if manifest_directions and manifest_directions != {steering_direction}:
        raise RuntimeError(
            f"Existing generation manifest in {result_root} uses steering "
            f"direction(s) {sorted(manifest_directions)!r}, expected "
            f"{steering_direction!r}. Use a new results directory rather than "
            "mixing steering directions."
        )
    existing_coefficients = tuple(
        float(row["coefficient"])
        for row in manifest
        if "coefficient" in row
    )
    validate_steering_coefficients(
        tuple(dict.fromkeys((*existing_coefficients, *coefficients)))
    )
    for coefficient in tqdm(
        coefficients, desc="Generating steering coefficients", unit="coefficient"
    ):
        wrapper.coefficient = float(coefficient)
        slug = coefficient_slug(coefficient)
        log_dir = result_root / "inspect_logs" / f"alpha_{slug}"
        common = {
            "log_dir": str(log_dir),
            "log_format": "eval",
            "log_samples": True,
            "display": "plain",
            "max_samples": 1,
            "max_tasks": 1,
            "temperature": config.TEMPERATURE,
            "top_p": config.TOP_P,
            "max_tokens": config.MAX_NEW_TOKENS,
            "seed": config.SEED,
            "cache": False,
            "retry_on_error": 0,
            "task_retry_attempts": 0,
            "fail_on_error": False,
            "continue_on_fail": True,
        }
        base_metadata = {
            "steering_coefficient": coefficient,
            "steering_direction": steering_direction,
            "target_layer": wrapper.target_layer,
            "target_layer_fraction": target_layer_fraction,
            "activation_site": wrapper.activation_site,
        }
        logger.info(
            "alpha=%g: generating sycophancy responses (%d samples)",
            coefficient,
            sycophancy_limit,
        )
        syc_log = generate_or_resume_task(
            # Inspect constructs task scorers even when score=False. Give it
            # the already loaded local model so GPU generation requires no
            # hosted-judge credentials; local rescoring replaces the scorer.
            sycophancy_eval,
            {
                **common,
                "model": sycophancy_model,
                "metadata": {
                    **base_metadata,
                    "steering_scope": STEERING_SCOPE_SECOND_ASSISTANT,
                },
            },
            log_dir,
            "sycophancy",
            sycophancy_limit,
            coefficient,
            STEERING_SCOPE_SECOND_ASSISTANT,
            steering_direction,
        )
        assert_log_matches_sample_contract(syc_log, contracts["sycophancy"])
        logger.info(
            "alpha=%g: generating coherence responses (%d samples)",
            coefficient,
            coherence_limit,
        )
        coherence_log = generate_or_resume_task(
            coherence_eval,
            {
                **common,
                "model": coherence_model,
                "metadata": {
                    **base_metadata,
                    "steering_scope": STEERING_SCOPE_ALWAYS,
                },
            },
            log_dir,
            "coherence",
            coherence_limit,
            coefficient,
            STEERING_SCOPE_ALWAYS,
            steering_direction,
        )
        assert_log_matches_sample_contract(coherence_log, contracts["coherence"])
        generated_rows: list[dict[str, Any]] = []
        for evaluation, log in (
            ("sycophancy", syc_log),
            ("coherence", coherence_log),
        ):
            location = Path(log.location) if log.location is not None else None
            try:
                portable_location = str(
                    location.resolve().relative_to(result_root.resolve())
                )
            except (AttributeError, ValueError):
                portable_location = str(location) if location is not None else None
            generated_rows.append(
                {
                    "coefficient": float(coefficient),
                    "steering_direction": steering_direction,
                    "evaluation": evaluation,
                    "log": portable_location,
                    "samples": len(log.samples or []),
                    "target_layer": wrapper.target_layer,
                    "target_layer_fraction": target_layer_fraction,
                    "steering_scope": (
                        STEERING_SCOPE_SECOND_ASSISTANT
                        if evaluation == "sycophancy"
                        else STEERING_SCOPE_ALWAYS
                    ),
                }
            )
        manifest = merge_generation_manifest(result_root, generated_rows)
    return manifest


def load_generation_manifest(result_root: Path) -> list[dict[str, Any]]:
    """Load a layer manifest, preserving prior runs when generation is extended."""
    path = result_root / "generation_manifest.json"
    if not path.is_file():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a list of objects in {path}")
    return rows


def _generation_manifest_key(row: dict[str, Any]) -> tuple[float, str]:
    try:
        coefficient = float(row["coefficient"])
        evaluation = str(row["evaluation"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Generation manifest rows require coefficient and evaluation"
        ) from exc
    return coefficient, evaluation


def merge_generation_manifest(
    result_root: Path,
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Upsert completed tasks without dropping entries from earlier runs."""
    manifest = load_generation_manifest(result_root)
    positions: dict[tuple[float, str], int] = {}
    for index, row in enumerate(manifest):
        key = _generation_manifest_key(row)
        if key in positions:
            raise ValueError(f"Duplicate generation manifest entry: {key!r}")
        positions[key] = index
    for row in new_rows:
        key = _generation_manifest_key(row)
        if key in positions:
            manifest[positions[key]] = row
        else:
            positions[key] = len(manifest)
            manifest.append(row)

    result_root.mkdir(parents=True, exist_ok=True)
    path = result_root / "generation_manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return manifest


def write_sweep_manifest(
    rows: list[dict[str, Any]],
    artifact_root: Path | None = None,
    result_root: Path | None = None,
) -> None:
    """Checkpoint the layer mapping and per-layer generation status."""
    payload = json.dumps(rows, indent=2, sort_keys=True) + "\n"
    for root in (
        artifact_root or config.ARTIFACTS_DIR,
        result_root or config.RESULTS_DIR,
    ):
        root.mkdir(parents=True, exist_ok=True)
        (root / "layer_sweep_manifest.json").write_text(payload, encoding="utf-8")


def load_saved_probe(
    artifact_directory: Path,
    steering_direction: str = STEERING_DIRECTION_PROBE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the small saved state needed to generate from an existing probe."""
    try:
        direction_field, scale_field = STEERING_DIRECTION_FIELDS[steering_direction]
    except KeyError as exc:
        raise ValueError(
            f"Unknown steering direction {steering_direction!r}; expected one of "
            f"{sorted(STEERING_DIRECTION_FIELDS)}"
        ) from exc
    metadata_path = artifact_directory / "probe_metadata.json"
    probe_path = artifact_directory / "probe.npz"
    if not metadata_path.is_file() or not probe_path.is_file():
        raise FileNotFoundError(
            f"Resume requires both {metadata_path} and {probe_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(probe_path, allow_pickle=False) as saved:
        required = {direction_field, scale_field}
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(
                f"Saved probe {probe_path} is missing arrays: {sorted(missing)}"
            )
        artifact = {
            "direction_unit": saved[direction_field].copy(),
            "projection_scale": saved[scale_field].copy(),
        }
    return artifact, metadata


def set_sweep_entry_status(
    artifact_root: Path,
    result_root: Path,
    fraction: float,
    layer: int,
    status: str,
    error: str | None = None,
) -> None:
    """Update only the resumed entry while preserving the downloaded sweep state."""
    manifest_paths = (
        artifact_root / "layer_sweep_manifest.json",
        result_root / "layer_sweep_manifest.json",
    )
    source = next((path for path in manifest_paths if path.is_file()), None)
    if source is None:
        rows = [{
            "target_layer_fraction": fraction,
            "target_layer": layer,
            "decoder_layer_count": None,
            "artifacts": layer_directory_name(fraction, layer),
            "results": layer_directory_name(fraction, layer),
            "status": status,
        }]
    else:
        rows = json.loads(source.read_text(encoding="utf-8"))
        matches = [
            row for row in rows
            if int(row["target_layer"]) == layer
            and math.isclose(
                float(row["target_layer_fraction"]),
                fraction,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one sweep manifest entry for fraction={fraction:g}, "
                f"layer={layer}; found {len(matches)}"
            )
        matches[0]["status"] = status
        if error is None:
            matches[0].pop("error", None)
        else:
            matches[0]["error"] = error
    write_sweep_manifest(rows, artifact_root, result_root)


def resume_layer_generation(
    model: Any,
    tokenizer: Any,
    target_layer_fraction: float,
    artifact_root: Path,
    result_root: Path,
    steering_coefficients: tuple[float, ...] | None = None,
    steering_direction: str = STEERING_DIRECTION_PROBE,
) -> None:
    """Generate one layer from its saved probe, reusing interrupted Inspect logs."""
    import torch

    layer_count = decoder_layer_count(model)
    fraction, target_layer = resolve_layer_fractions(
        (target_layer_fraction,), layer_count
    )[0]
    directory_name = layer_directory_name(fraction, target_layer)
    artifact_directory = artifact_root / directory_name
    result_directory = result_root / directory_name
    artifact, metadata = load_saved_probe(artifact_directory, steering_direction)

    expected = {
        "model_id": config.MODEL_NAME,
        "target_layer": target_layer,
        "decoder_layer_count": layer_count,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Saved probe {key} is {metadata.get(key)!r}, expected {value!r}"
            )
    if not math.isclose(
        float(metadata.get("target_layer_fraction", float("nan"))),
        fraction,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Saved probe target_layer_fraction does not match the requested fraction"
        )
    saved_scale = float(artifact["projection_scale"])
    _, scale_field = STEERING_DIRECTION_FIELDS[steering_direction]
    metadata_scale = metadata.get(scale_field)
    if metadata_scale is not None and not math.isclose(
        float(metadata_scale), saved_scale, rel_tol=1e-6, abs_tol=1e-8
    ):
        raise ValueError(
            f"probe.npz and probe_metadata.json disagree on {scale_field}"
        )

    steering_site = resolve_steering_site(
        model,
        target_layer,
        metadata.get("steering_module_path") or config.STEERING_MODULE_PATH,
    )
    if metadata.get("activation_site") != steering_site.activation_site:
        raise ValueError(
            f"Resolved activation site {steering_site.activation_site!r} does not "
            f"match saved probe site {metadata.get('activation_site')!r}"
        )
    wrapper = SteeredModel(
        model,
        tokenizer,
        steering_site.module,
        steering_site.module_path,
        target_layer,
        artifact["direction_unit"],
        saved_scale,
    )
    check = tokenizer.apply_chat_template(
        [{"role": "user", "content": "A short deterministic check."}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        **config.CHAT_TEMPLATE_KWARGS,
    )
    device = next(model.parameters()).device
    if "attention_mask" not in check:
        check["attention_mask"] = torch.ones_like(check["input_ids"])
    assert_zero_equivalence(
        wrapper, check["input_ids"].to(device), check["attention_mask"].to(device)
    )
    direction_api_suffix = (
        "" if steering_direction == STEERING_DIRECTION_PROBE
        else f"-{steering_direction}"
    )
    sycophancy_model = make_inspect_model(
        wrapper,
        config.MODEL_NAME,
        config.MAX_NEW_TOKENS,
        config.TEMPERATURE,
        config.TOP_P,
        config.SEED,
        config.CHAT_TEMPLATE_KWARGS,
        api_name=f"local-second-response{direction_api_suffix}-layer-{target_layer}",
        steering_scope=STEERING_SCOPE_SECOND_ASSISTANT,
    )
    coherence_model = make_inspect_model(
        wrapper,
        config.MODEL_NAME,
        config.MAX_NEW_TOKENS,
        config.TEMPERATURE,
        config.TOP_P,
        config.SEED,
        config.CHAT_TEMPLATE_KWARGS,
        api_name=f"local-all-responses{direction_api_suffix}-layer-{target_layer}",
        steering_scope=STEERING_SCOPE_ALWAYS,
    )
    if steering_coefficients is None:
        coefficients = validate_steering_coefficients(
            tuple(float(value) for value in metadata["steering_coefficients"])
        )
    else:
        coefficients = validate_steering_coefficients(steering_coefficients)
    sycophancy_limit = int(metadata["sycophancy_sample_limit"])
    coherence_limit = int(metadata["coherence_sample_limit"])
    logger.info(
        "Resuming fraction=%g (layer=%d, direction=%s) for coefficients %s "
        "from %s into %s",
        fraction,
        target_layer,
        steering_direction,
        ", ".join(f"{value:g}" for value in coefficients),
        artifact_directory,
        result_directory,
    )
    set_sweep_entry_status(
        artifact_root, result_root, fraction, target_layer, "running"
    )
    try:
        started = log_stage(f"Layer {target_layer}: resuming behavioral generation")
        run_generations(
            wrapper,
            sycophancy_model,
            coherence_model,
            coefficients,
            result_directory,
            sycophancy_limit,
            coherence_limit,
            fraction,
            steering_direction,
        )
        log_stage_done(f"Layer {target_layer}: behavioral generation", started)
    except Exception as exc:
        set_sweep_entry_status(
            artifact_root,
            result_root,
            fraction,
            target_layer,
            "failed",
            str(exc),
        )
        raise
    set_sweep_entry_status(
        artifact_root, result_root, fraction, target_layer, "generated"
    )


def run_layer_experiment(
    model: Any,
    tokenizer: Any,
    variants: list[Any],
    train_ids: set[str],
    validation_ids: set[str],
    target_layer_fraction: float,
    target_layer: int,
    layer_count: int,
) -> str:
    """Fit, validate, and behaviorally evaluate one resolved sweep layer."""
    import torch

    directory_name = layer_directory_name(target_layer_fraction, target_layer)
    artifact_directory = config.ARTIFACTS_DIR / directory_name
    result_directory = config.RESULTS_DIR / directory_name
    logger.info(
        "Layer fraction %g resolved to decoder layer %d/%d",
        target_layer_fraction,
        target_layer,
        layer_count - 1,
    )
    steering_site = resolve_steering_site(
        model, target_layer, config.STEERING_MODULE_PATH
    )
    logger.info("Using steering site: %s", steering_site.activation_site)

    started = log_stage(f"Layer {target_layer}: extracting model activations")
    x, y, activation_metadata = extract_activations(
        model,
        variants,
        steering_site.module,
        config.ROLES,
        config.MAX_PROBE_TOKENS_PER_BASE_ROLE,
    )
    log_stage_done(f"Layer {target_layer}: activation extraction", started)
    train_mask = np.asarray([
        row["base_id"] in train_ids for row in activation_metadata
    ])
    validation_mask = np.asarray([
        row["base_id"] in validation_ids for row in activation_metadata
    ])
    if np.any(train_mask & validation_mask) or not np.all(train_mask | validation_mask):
        raise AssertionError("Invalid base-level probe split")
    for row, in_train in zip(activation_metadata, train_mask.tolist()):
        row["split"] = "train" if in_train else "validation"

    started = log_stage(f"Layer {target_layer}: fitting and validating role probe")
    artifact, diagnostics = train_probe(
        x[train_mask],
        y[train_mask],
        x[validation_mask],
        y[validation_mask],
        config.ROLES,
        config.REGULARIZATION_GRID,
        config.STEERING_COEFFICIENTS,
        config.SEED,
    )
    log_stage_done(f"Layer {target_layer}: probe fit", started)
    metadata = {
        "model_id": config.MODEL_NAME,
        "target_layer": target_layer,
        "target_layer_fraction": target_layer_fraction,
        "decoder_layer_count": layer_count,
        "activation_site": steering_site.activation_site,
        "steering_module_path": steering_site.module_path,
        "roles": list(config.ROLES),
        "role_order": list(config.ROLES),
        "n_base_sequences": config.N_BASE_SEQUENCES,
        "sequence_length": config.SEQUENCE_LENGTH,
        "train_fraction": config.TRAIN_FRACTION,
        "max_probe_tokens_per_base_role": config.MAX_PROBE_TOKENS_PER_BASE_ROLE,
        "direction_formula": "w_user - mean(w_non_user)",
        "direction_normalization": "L2 unit norm",
        "direction_scale": "sample std (ddof=1) of held-out projections",
        "projection_scale": float(artifact["projection_scale"]),
        "mean_difference_projection_scale": float(
            artifact["mean_difference_projection_scale"]
        ),
        "seed": config.SEED,
        "steering_coefficients": list(config.STEERING_COEFFICIENTS),
        "judge_model": config.JUDGE_MODEL,
        "sycophancy_steering_scope": STEERING_SCOPE_SECOND_ASSISTANT,
        "coherence_steering_scope": STEERING_SCOPE_ALWAYS,
        "sycophancy_sample_limit": config.SYCOPHANCY_SAMPLE_LIMIT,
        "coherence_sample_limit": config.COHERENCE_SAMPLE_LIMIT,
    }
    save_probe(
        artifact_directory, artifact, diagnostics, metadata, activation_metadata
    )
    # The activation matrix can be large and is not needed during behavioral
    # generation. Release it before moving on to generation or the next layer.
    del x, y, train_mask, validation_mask
    if not diagnostics["userness_diagnostic_monotonic"]:
        logger.warning(
            "Layer %d held-out mean Userness is not monotonic; diagnostics were "
            "saved to %s and behavioral generation for this layer is skipped",
            target_layer,
            artifact_directory,
        )
        return "diagnostics_only_non_monotonic"

    wrapper = SteeredModel(
        model,
        tokenizer,
        steering_site.module,
        steering_site.module_path,
        target_layer,
        artifact["direction_unit"],
        float(artifact["projection_scale"]),
    )
    check = tokenizer.apply_chat_template(
        [{"role": "user", "content": "A short deterministic check."}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        **config.CHAT_TEMPLATE_KWARGS,
    )
    device = next(model.parameters()).device
    if "attention_mask" not in check:
        check["attention_mask"] = torch.ones_like(check["input_ids"])
    assert_zero_equivalence(
        wrapper, check["input_ids"].to(device), check["attention_mask"].to(device)
    )
    sycophancy_model = make_inspect_model(
        wrapper,
        config.MODEL_NAME,
        config.MAX_NEW_TOKENS,
        config.TEMPERATURE,
        config.TOP_P,
        config.SEED,
        config.CHAT_TEMPLATE_KWARGS,
        api_name=f"local-second-response-layer-{target_layer}",
        steering_scope=STEERING_SCOPE_SECOND_ASSISTANT,
    )
    coherence_model = make_inspect_model(
        wrapper,
        config.MODEL_NAME,
        config.MAX_NEW_TOKENS,
        config.TEMPERATURE,
        config.TOP_P,
        config.SEED,
        config.CHAT_TEMPLATE_KWARGS,
        api_name=f"local-all-responses-layer-{target_layer}",
        steering_scope=STEERING_SCOPE_ALWAYS,
    )
    started = log_stage(f"Layer {target_layer}: generating behavioral responses")
    run_generations(
        wrapper,
        sycophancy_model,
        coherence_model,
        config.STEERING_COEFFICIENTS,
        result_directory,
        config.SYCOPHANCY_SAMPLE_LIMIT,
        config.COHERENCE_SAMPLE_LIMIT,
        target_layer_fraction,
    )
    log_stage_done(f"Layer {target_layer}: behavioral generation", started)

    if config.RUN_RANDOM_DIRECTION_CONTROL:
        primary = wrapper.unit_direction.clone()
        try:
            wrapper.unit_direction = equal_norm_random_direction(primary, config.SEED)
            run_generations(
                wrapper,
                sycophancy_model,
                coherence_model,
                config.STEERING_COEFFICIENTS,
                result_directory / "random_direction_control",
                min(config.SYCOPHANCY_SAMPLE_LIMIT, config.RANDOM_CONTROL_SAMPLE_LIMIT),
                min(config.COHERENCE_SAMPLE_LIMIT, config.RANDOM_CONTROL_SAMPLE_LIMIT),
                target_layer_fraction,
                STEERING_DIRECTION_RANDOM_CONTROL,
            )
        finally:
            wrapper.unit_direction = primary
    return "generated"


def _run() -> None:
    import torch

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    seed_everything(config.SEED)
    if config.JUDGE_MODEL == config.MODEL_NAME:
        raise ValueError("JUDGE_MODEL must be independent of the steered target model")
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = log_stage("Loading tokenizer and model")
    model, tokenizer = load_model_once()
    log_stage_done("Model load", started)
    layer_count = decoder_layer_count(model)
    layers = resolve_layer_fractions(config.TARGET_LAYER_FRACTIONS, layer_count)
    logger.info(
        "Resolved layer sweep: %s",
        ", ".join(f"{fraction:g} -> {layer}" for fraction, layer in layers),
    )

    started = log_stage("Selecting neutral C4 base passages")
    bases = load_neutral_base_sequences(
        tokenizer, config.N_BASE_SEQUENCES, config.SEQUENCE_LENGTH, config.SEED
    )
    log_stage_done("Base-passage selection", started)
    # This split intentionally occurs before any role copy is constructed and
    # is shared by every layer in the sweep.
    train_ids, validation_ids = split_base_ids(bases, config.TRAIN_FRACTION, config.SEED)
    started = log_stage("Rendering role variants")
    variants = build_role_variants(
        tokenizer, bases, config.ROLES, config.CHAT_TEMPLATE_KWARGS
    )
    log_stage_done("Role-variant rendering", started)
    write_jsonl(config.ARTIFACTS_DIR / "base_sequences.jsonl", [
        {
            "base_id": base.base_id,
            "source": base.source,
            "text": base.text,
            "split": "train" if base.base_id in train_ids else "validation",
        }
        for base in bases
    ])

    sweep_rows: list[dict[str, Any]] = [
        {
            "target_layer_fraction": fraction,
            "target_layer": layer,
            "decoder_layer_count": layer_count,
            "artifacts": layer_directory_name(fraction, layer),
            "results": layer_directory_name(fraction, layer),
            "status": "pending",
        }
        for fraction, layer in layers
    ]
    write_sweep_manifest(sweep_rows)
    for index, ((fraction, layer), row) in enumerate(zip(layers, sweep_rows), start=1):
        logger.info(
            "Starting layer sweep entry %d/%d (fraction=%g, layer=%d)",
            index,
            len(layers),
            fraction,
            layer,
        )
        row["status"] = "running"
        write_sweep_manifest(sweep_rows)
        try:
            row["status"] = run_layer_experiment(
                model,
                tokenizer,
                variants,
                train_ids,
                validation_ids,
                fraction,
                layer,
                layer_count,
            )
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            write_sweep_manifest(sweep_rows)
            raise
        write_sweep_manifest(sweep_rows)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(
        "GPU layer sweep complete. Download %s and run `python score_results.py` locally",
        config.RESULTS_DIR,
    )
    notify(f"run.py completed successfully. Results: {config.RESULTS_DIR}")


def saved_layer_fractions(artifact_root: Path) -> tuple[float, ...]:
    """Read saved layer fractions in sweep order from an artifact root."""
    manifest_path = artifact_root / "layer_sweep_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Resume-all requires the saved sweep manifest {manifest_path}"
        )
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty list in {manifest_path}")
    try:
        fractions = tuple(float(row["target_layer_fraction"]) for row in rows)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Every row in {manifest_path} needs target_layer_fraction"
        ) from exc
    if not all(math.isfinite(fraction) for fraction in fractions):
        raise ValueError(f"Saved layer fractions in {manifest_path} must be finite")
    if len(set(fractions)) != len(fractions):
        raise ValueError(f"Saved layer fractions in {manifest_path} must be unique")
    return fractions


def _resume_generation(
    fraction: float | None,
    artifact_root: Path,
    result_root: Path,
    steering_coefficients: tuple[float, ...] | None = None,
    steering_direction: str = STEERING_DIRECTION_PROBE,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    seed_everything(config.SEED)
    artifact_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    started = log_stage("Loading tokenizer and model for generation resume")
    model, tokenizer = load_model_once()
    log_stage_done("Model load", started)
    fractions = (
        saved_layer_fractions(artifact_root) if fraction is None else (fraction,)
    )
    for saved_fraction in fractions:
        resume_layer_generation(
            model,
            tokenizer,
            saved_fraction,
            artifact_root,
            result_root,
            steering_coefficients,
            steering_direction,
        )
    logger.info(
        "Generation resume complete for %d layer(s). Download %s and run "
        "`python score_results.py %s` locally",
        len(fractions),
        result_root,
        result_root,
    )
    notify(
        f"run.py generation resume completed successfully for {len(fractions)} "
        "layer(s). "
        f"Results: {result_root}"
    )


def main() -> None:
    """Run the experiment and report any uncaught error before propagating it."""
    try:
        _run()
    except Exception as exc:
        notify(f"run.py failed with {type(exc).__name__}: {exc}")
        raise


def resume_main(
    fraction: float | None,
    artifact_root: Path,
    result_root: Path,
    steering_coefficients: tuple[float, ...] | None = None,
    steering_direction: str = STEERING_DIRECTION_PROBE,
) -> None:
    """Resume saved layers and report any uncaught error before propagating it."""
    try:
        if steering_coefficients is not None:
            steering_coefficients = validate_steering_coefficients(
                steering_coefficients
            )
        _resume_generation(
            fraction,
            artifact_root,
            result_root,
            steering_coefficients,
            steering_direction,
        )
    except Exception as exc:
        notify(f"run.py resume failed with {type(exc).__name__}: {exc}")
        raise


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train a layer sweep or resume generation from a saved layer probe."
    )
    resume_mode = parser.add_mutually_exclusive_group()
    resume_mode.add_argument(
        "--resume-generation",
        type=float,
        metavar="FRACTION",
        help="Skip probe training and generate only this saved layer fraction.",
    )
    resume_mode.add_argument(
        "--resume-all",
        action="store_true",
        help="Generate from every layer listed in the saved artifact manifest.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=config.ARTIFACTS_DIR,
        help="Artifact root containing the saved layer probe.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=config.RESULTS_DIR,
        help="Result root containing any interrupted Inspect logs.",
    )
    parser.add_argument(
        "--steering-coefficients",
        type=float,
        nargs="+",
        metavar="ALPHA",
        help=(
            "Coefficients to generate during resume. Existing logs and manifest "
            "entries are preserved and reused."
        ),
    )
    parser.add_argument(
        "--steering-direction",
        choices=tuple(STEERING_DIRECTION_FIELDS),
        default=STEERING_DIRECTION_PROBE,
        help=(
            "Saved activation direction to use during resume generation "
            f"(default: {STEERING_DIRECTION_PROBE})."
        ),
    )
    args = parser.parse_args(argv)
    is_resume = args.resume_generation is not None or args.resume_all
    if not is_resume:
        if (
            args.artifacts_dir != config.ARTIFACTS_DIR
            or args.results_dir != config.RESULTS_DIR
        ):
            parser.error(
                "--artifacts-dir and --results-dir are only valid with "
                "--resume-generation or --resume-all"
            )
        if args.steering_coefficients is not None:
            parser.error("--steering-coefficients is only valid during resume")
        if args.steering_direction != STEERING_DIRECTION_PROBE:
            parser.error("--steering-direction is only valid during resume")
        main()
    else:
        try:
            coefficients = (
                validate_steering_coefficients(tuple(args.steering_coefficients))
                if args.steering_coefficients is not None
                else None
            )
        except ValueError as exc:
            parser.error(str(exc))
        resume_main(
            None if args.resume_all else args.resume_generation,
            args.artifacts_dir,
            args.results_dir,
            coefficients,
            args.steering_direction,
        )


if __name__ == "__main__":
    cli()
