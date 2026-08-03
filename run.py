"""Train and steer one loaded model across a configured decoder-layer sweep."""

from __future__ import annotations

import argparse
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
    SteeredModel,
    assert_zero_equivalence,
    equal_norm_random_direction,
    make_inspect_model,
)
from utils import send_ntfy_message


logger = logging.getLogger(__name__)
NTFY_TOPIC = "roles_and_sycophancy_experiment_rbi24u"


def notify(message: str) -> None:
    """Send a run notification without allowing ntfy failures to stop the run."""
    try:
        send_ntfy_message(message, topic=NTFY_TOPIC)
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
        if (
            generation_evaluation_name(log) == evaluation
            and math.isclose(logged_coefficient, coefficient, rel_tol=0.0, abs_tol=1e-12)
        ):
            logs.append(log)
    return logs


def generate_or_resume_task(
    task: Any,
    common: dict[str, Any],
    log_dir: Path,
    evaluation: str,
    expected_samples: int,
    coefficient: float,
) -> Any:
    """Reuse a complete log or retry the best interrupted log before starting anew."""
    from inspect_ai import eval as inspect_eval
    from inspect_ai import eval_retry

    existing = existing_generation_logs(log_dir, evaluation, coefficient)
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
        return require_generated_eval(
            eval_retry(
                selected,
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
    inspect_model: Any,
    coefficients: tuple[float, ...],
    result_root: Path,
    sycophancy_limit: int,
    coherence_limit: int,
    target_layer_fraction: float,
) -> list[dict[str, Any]]:
    """Generate and persist all target-model responses without judge calls."""
    from inspect_evals.sycophancy import sycophancy

    manifest: list[dict[str, Any]] = []
    for coefficient in tqdm(
        coefficients, desc="Generating steering coefficients", unit="coefficient"
    ):
        wrapper.coefficient = float(coefficient)
        slug = coefficient_slug(coefficient)
        log_dir = result_root / "inspect_logs" / f"alpha_{slug}"
        common = {
            "model": inspect_model,
            "log_dir": str(log_dir),
            "log_format": "eval",
            "log_samples": True,
            "display": "plain",
            "max_samples": 1,
            "max_tasks": 1,
            "sample_shuffle": False,
            "temperature": config.TEMPERATURE,
            "top_p": config.TOP_P,
            "max_tokens": config.MAX_NEW_TOKENS,
            "seed": config.SEED,
            "cache": False,
            "retry_on_error": 0,
            "task_retry_attempts": 0,
            "fail_on_error": False,
            "continue_on_fail": True,
            "metadata": {
                "steering_coefficient": coefficient,
                "target_layer": wrapper.target_layer,
                "target_layer_fraction": target_layer_fraction,
                "activation_site": wrapper.activation_site,
            },
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
            sycophancy(shuffle=False, scorer_model=inspect_model),
            common,
            log_dir,
            "sycophancy",
            sycophancy_limit,
            coefficient,
        )
        logger.info(
            "alpha=%g: generating coherence responses (%d samples)",
            coefficient,
            coherence_limit,
        )
        coherence_log = generate_or_resume_task(
            coherence_task(config.COHERENCE_DATASET, inspect_model),
            common,
            log_dir,
            "coherence",
            coherence_limit,
            coefficient,
        )
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
            manifest.append(
                {
                    "coefficient": float(coefficient),
                    "evaluation": evaluation,
                    "log": portable_location,
                    "samples": len(log.samples or []),
                    "target_layer": wrapper.target_layer,
                    "target_layer_fraction": target_layer_fraction,
                }
            )
        result_root.mkdir(parents=True, exist_ok=True)
        (result_root / "generation_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
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


def load_saved_probe(artifact_directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the small saved state needed to generate from an existing probe."""
    metadata_path = artifact_directory / "probe_metadata.json"
    probe_path = artifact_directory / "probe.npz"
    if not metadata_path.is_file() or not probe_path.is_file():
        raise FileNotFoundError(
            f"Resume requires both {metadata_path} and {probe_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(probe_path, allow_pickle=False) as saved:
        required = {"direction_unit", "projection_scale"}
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(
                f"Saved probe {probe_path} is missing arrays: {sorted(missing)}"
            )
        artifact = {
            "direction_unit": saved["direction_unit"].copy(),
            "projection_scale": saved["projection_scale"].copy(),
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
    artifact, metadata = load_saved_probe(artifact_directory)

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
    if not math.isclose(
        float(metadata.get("projection_scale", float("nan"))),
        saved_scale,
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise ValueError("probe.npz and probe_metadata.json disagree on projection_scale")

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
    inspect_model = make_inspect_model(
        wrapper,
        config.MODEL_NAME,
        config.MAX_NEW_TOKENS,
        config.TEMPERATURE,
        config.TOP_P,
        config.SEED,
        config.CHAT_TEMPLATE_KWARGS,
        api_name=f"local-steered-layer-{target_layer}",
    )
    coefficients = tuple(float(value) for value in metadata["steering_coefficients"])
    sycophancy_limit = int(metadata["sycophancy_sample_limit"])
    coherence_limit = int(metadata["coherence_sample_limit"])
    logger.info(
        "Resuming only fraction=%g (layer=%d) from %s into %s",
        fraction,
        target_layer,
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
            inspect_model,
            coefficients,
            result_directory,
            sycophancy_limit,
            coherence_limit,
            fraction,
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
        "seed": config.SEED,
        "steering_coefficients": list(config.STEERING_COEFFICIENTS),
        "judge_model": config.JUDGE_MODEL,
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
    inspect_model = make_inspect_model(
        wrapper,
        config.MODEL_NAME,
        config.MAX_NEW_TOKENS,
        config.TEMPERATURE,
        config.TOP_P,
        config.SEED,
        config.CHAT_TEMPLATE_KWARGS,
        api_name=f"local-steered-layer-{target_layer}",
    )
    started = log_stage(f"Layer {target_layer}: generating behavioral responses")
    run_generations(
        wrapper,
        inspect_model,
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
                inspect_model,
                config.STEERING_COEFFICIENTS,
                result_directory / "random_direction_control",
                min(config.SYCOPHANCY_SAMPLE_LIMIT, config.RANDOM_CONTROL_SAMPLE_LIMIT),
                min(config.COHERENCE_SAMPLE_LIMIT, config.RANDOM_CONTROL_SAMPLE_LIMIT),
                target_layer_fraction,
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


def _resume_generation(
    fraction: float,
    artifact_root: Path,
    result_root: Path,
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
    resume_layer_generation(
        model,
        tokenizer,
        fraction,
        artifact_root,
        result_root,
    )
    logger.info(
        "Layer generation resume complete. Download %s and run "
        "`python score_results.py %s` locally",
        result_root,
        result_root,
    )
    notify(
        f"run.py generation resume completed successfully for fraction={fraction:g}. "
        f"Results: {result_root}"
    )


def main() -> None:
    """Run the experiment and report any uncaught error before propagating it."""
    try:
        _run()
    except Exception as exc:
        notify(f"run.py failed with {type(exc).__name__}: {exc}")
        raise


def resume_main(fraction: float, artifact_root: Path, result_root: Path) -> None:
    """Resume one saved layer and report any uncaught error before propagating it."""
    try:
        _resume_generation(fraction, artifact_root, result_root)
    except Exception as exc:
        notify(f"run.py resume failed with {type(exc).__name__}: {exc}")
        raise


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train a layer sweep or resume generation from a saved layer probe."
    )
    parser.add_argument(
        "--resume-generation",
        type=float,
        metavar="FRACTION",
        help="Skip probe training and generate only this saved layer fraction.",
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
    args = parser.parse_args(argv)
    if args.resume_generation is None:
        if (
            args.artifacts_dir != config.ARTIFACTS_DIR
            or args.results_dir != config.RESULTS_DIR
        ):
            parser.error(
                "--artifacts-dir and --results-dir are only valid with "
                "--resume-generation"
            )
        main()
    else:
        resume_main(args.resume_generation, args.artifacts_dir, args.results_dir)


if __name__ == "__main__":
    cli()
