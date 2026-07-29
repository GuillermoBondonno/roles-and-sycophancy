"""Train once, steer one loaded model, and run identical Inspect evals."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
from tqdm.auto import tqdm

import config
from evals import coherence_summary, coherence_task, save_per_sample, sycophancy_summary
from model_adapter import resolve_steering_site
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


logger = logging.getLogger(__name__)


def log_stage(message: str) -> float:
    logger.info(message)
    return time.monotonic()


def log_stage_done(message: str, started_at: float) -> None:
    logger.info("%s completed in %.1fs", message, time.monotonic() - started_at)


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


def run_evaluations(
    wrapper: SteeredModel,
    inspect_model: Any,
    coefficients: tuple[float, ...],
    result_root: Path,
    sycophancy_limit: int,
    coherence_limit: int,
) -> list[dict[str, float]]:
    from inspect_ai import eval as inspect_eval
    from inspect_evals.sycophancy import sycophancy

    rows: list[dict[str, float]] = []
    for coefficient in tqdm(
        coefficients, desc="Evaluating steering coefficients", unit="coefficient"
    ):
        wrapper.coefficient = float(coefficient)
        slug = coefficient_slug(coefficient)
        log_dir = result_root / "inspect_logs" / f"alpha_{slug}"
        common = {
            "model": inspect_model,
            "log_dir": str(log_dir),
            "log_format": "json",
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
            "metadata": {
                "steering_coefficient": coefficient,
                "target_layer": config.TARGET_LAYER,
                "activation_site": wrapper.activation_site,
            },
        }
        logger.info("alpha=%g: running sycophancy evaluation (%d samples)", coefficient,
                    sycophancy_limit)
        syc_log = inspect_eval(
            sycophancy(shuffle=False, scorer_model=config.JUDGE_MODEL),
            limit=sycophancy_limit,
            **common,
        )[0]
        logger.info("alpha=%g: running coherence evaluation (%d samples)", coefficient,
                    coherence_limit)
        coherence_log = inspect_eval(
            coherence_task(config.COHERENCE_DATASET, config.JUDGE_MODEL),
            limit=coherence_limit,
            **common,
        )[0]
        save_per_sample(
            syc_log, result_root / "per_sample" / f"sycophancy_alpha_{slug}.jsonl",
            coefficient, "sycophancy",
        )
        save_per_sample(
            coherence_log, result_root / "per_sample" / f"coherence_alpha_{slug}.jsonl",
            coefficient, "coherence",
        )
        rows.append({
            "coefficient": float(coefficient),
            **sycophancy_summary(syc_log),
            **coherence_summary(coherence_log),
        })
    return rows


def save_summary(rows: list[dict[str, float]], directory: Path, stem: str = "summary") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if rows:
        with (directory / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
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
    started = log_stage("Stage 1/7: loading tokenizer and model")
    model, tokenizer = load_model_once()
    log_stage_done("Stage 1/7", started)
    started = log_stage("Stage 2/7: resolving the steering module")
    steering_site = resolve_steering_site(
        model, config.TARGET_LAYER, config.STEERING_MODULE_PATH
    )
    logger.info("Using steering site: %s", steering_site.activation_site)
    log_stage_done("Stage 2/7", started)

    started = log_stage("Stage 3/7: selecting neutral C4 base passages")
    bases = load_neutral_base_sequences(
        tokenizer, config.N_BASE_SEQUENCES, config.SEQUENCE_LENGTH, config.SEED
    )
    log_stage_done("Stage 3/7", started)
    # This split intentionally occurs before any role copy is constructed.
    train_ids, validation_ids = split_base_ids(bases, config.TRAIN_FRACTION, config.SEED)
    started = log_stage("Stage 4/7: rendering role variants")
    variants = build_role_variants(
        tokenizer, bases, config.ROLES, config.CHAT_TEMPLATE_KWARGS
    )
    log_stage_done("Stage 4/7", started)
    started = log_stage("Stage 5/7: extracting model activations")
    x, y, activation_metadata = extract_activations(
        model, variants, steering_site.module, config.ROLES,
        config.MAX_PROBE_TOKENS_PER_BASE_ROLE,
    )
    log_stage_done("Stage 5/7", started)
    train_mask = np.asarray([row["base_id"] in train_ids for row in activation_metadata])
    validation_mask = np.asarray([
        row["base_id"] in validation_ids for row in activation_metadata
    ])
    if np.any(train_mask & validation_mask) or not np.all(train_mask | validation_mask):
        raise AssertionError("Invalid base-level probe split")
    for row, in_train in zip(activation_metadata, train_mask.tolist()):
        row["split"] = "train" if in_train else "validation"

    started = log_stage("Stage 6/7: fitting and validating the role probe")
    artifact, diagnostics = train_probe(
        x[train_mask], y[train_mask], x[validation_mask], y[validation_mask],
        config.ROLES, config.REGULARIZATION_GRID,
        config.STEERING_COEFFICIENTS, config.SEED,
    )
    log_stage_done("Stage 6/7", started)
    metadata = {
        "model_id": config.MODEL_NAME,
        "target_layer": config.TARGET_LAYER,
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
        config.ARTIFACTS_DIR, artifact, diagnostics, metadata, activation_metadata
    )
    if not diagnostics["userness_diagnostic_monotonic"]:
        raise RuntimeError(
            "Held-out mean Userness was not monotonic across coefficients; "
            "diagnostics were saved, and behavioral evaluation was stopped"
        )
    write_jsonl(config.ARTIFACTS_DIR / "base_sequences.jsonl", [
        {
            "base_id": base.base_id,
            "source": base.source,
            "text": base.text,
            "split": "train" if base.base_id in train_ids else "validation",
        }
        for base in bases
    ])

    wrapper = SteeredModel(
        model, tokenizer, steering_site.module, steering_site.module_path,
        config.TARGET_LAYER,
        artifact["direction_unit"], float(artifact["projection_scale"]),
    )
    check = tokenizer.apply_chat_template(
        [{"role": "user", "content": "A short deterministic check."}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
        return_dict=True, **config.CHAT_TEMPLATE_KWARGS,
    )
    device = next(model.parameters()).device
    if "attention_mask" not in check:
        check["attention_mask"] = torch.ones_like(check["input_ids"])
    assert_zero_equivalence(
        wrapper, check["input_ids"].to(device), check["attention_mask"].to(device)
    )
    inspect_model = make_inspect_model(
        wrapper, config.MODEL_NAME, config.MAX_NEW_TOKENS,
        config.TEMPERATURE, config.TOP_P, config.SEED,
        config.CHAT_TEMPLATE_KWARGS,
    )
    started = log_stage("Stage 7/7: running behavioral evaluations")
    primary_rows = run_evaluations(
        wrapper, inspect_model, config.STEERING_COEFFICIENTS,
        config.RESULTS_DIR, config.SYCOPHANCY_SAMPLE_LIMIT,
        config.COHERENCE_SAMPLE_LIMIT,
    )
    save_summary(primary_rows, config.RESULTS_DIR)
    log_stage_done("Stage 7/7", started)

    if config.RUN_RANDOM_DIRECTION_CONTROL:
        primary = wrapper.unit_direction.clone()
        try:
            wrapper.unit_direction = equal_norm_random_direction(primary, config.SEED)
            random_rows = run_evaluations(
                wrapper, inspect_model, config.STEERING_COEFFICIENTS,
                config.RESULTS_DIR / "random_direction_control",
                min(config.SYCOPHANCY_SAMPLE_LIMIT, config.RANDOM_CONTROL_SAMPLE_LIMIT),
                min(config.COHERENCE_SAMPLE_LIMIT, config.RANDOM_CONTROL_SAMPLE_LIMIT),
            )
            save_summary(
                random_rows, config.RESULTS_DIR / "random_direction_control",
                stem="random_direction_summary",
            )
        finally:
            wrapper.unit_direction = primary

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Experiment complete. Results are in %s", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
