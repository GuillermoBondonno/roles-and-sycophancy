"""Score previously generated steering logs without loading the target model."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import logging
from pathlib import Path
from typing import Any

import config
from evals import (
    coherence_scorer,
    coherence_summary,
    save_per_sample,
    sycophancy_summary,
)
from run import coefficient_slug


logger = logging.getLogger(__name__)


def evaluation_name(log: Any) -> str:
    """Return the supported evaluation name recorded in an Inspect log."""
    candidates = (
        log.eval.task_display_name,
        log.eval.task_registry_name,
        log.eval.task,
    )
    for candidate in candidates:
        if candidate and (
            candidate == "sycophancy" or candidate.endswith("/sycophancy")
        ):
            return "sycophancy"
        if candidate and (candidate == "coherence" or candidate.endswith("/coherence")):
            return "coherence"
    raise ValueError(f"Unsupported evaluation task: {candidates!r}")


def expected_scorer_name(evaluation: str) -> str:
    return "sycophancy_scorer" if evaluation == "sycophancy" else "coherence_judge"


def fully_scored(log: Any, evaluation: str) -> bool:
    """Whether every generated sample has the expected scorer output."""
    samples = log.samples or []
    expected = expected_scorer_name(evaluation)
    return bool(samples) and all(
        any(
            name == expected or name.startswith(f"{expected}-")
            for name in (sample.scores or {})
        )
        for sample in samples
    )


def _manifest_sources(result_root: Path) -> list[tuple[Path, Path]]:
    sources: list[tuple[Path, Path]] = []
    for manifest_path in sorted(result_root.rglob("generation_manifest.json")):
        experiment_root = manifest_path.parent
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in rows:
            location = row.get("log")
            if not location:
                continue
            source = Path(location)
            if not source.is_absolute():
                source = experiment_root / source
            sources.append((experiment_root, source))
    return sources


def _fallback_sources(result_root: Path) -> list[tuple[Path, Path]]:
    """Discover generation logs created before manifests were introduced."""
    sources: list[tuple[Path, Path]] = []
    for source in sorted(result_root.rglob("*")):
        if not source.is_file() or source.suffix not in {".eval", ".json"}:
            continue
        inspect_root = next(
            (parent for parent in source.parents if parent.name == "inspect_logs"),
            None,
        )
        if inspect_root is not None:
            sources.append((inspect_root.parent, source))
    return sources


def generation_sources(result_root: Path) -> list[tuple[Path, Path]]:
    """Discover the union of manifested and on-disk generation logs."""
    sources: list[tuple[Path, Path]] = []
    seen: set[tuple[Path, Path]] = set()
    candidates = _manifest_sources(result_root) + _fallback_sources(result_root)
    for experiment_root, source in candidates:
        identity = (experiment_root.resolve(), source.resolve())
        if identity in seen:
            continue
        seen.add(identity)
        sources.append((experiment_root, source))
    return sources


def scored_location(experiment_root: Path, source: Path) -> Path:
    inspect_root = next(
        (parent for parent in source.parents if parent.name == "inspect_logs"),
        None,
    )
    if inspect_root is None:
        raise ValueError(
            f"Generation log is not below an inspect_logs directory: {source}"
        )
    return experiment_root / "scored_logs" / source.relative_to(inspect_root)


def build_judge_model() -> Any:
    """Create a serial hosted judge suitable for local scoring."""
    from inspect_ai.model import GenerateConfig, get_model

    return get_model(
        config.JUDGE_MODEL,
        config=GenerateConfig(
            max_retries=config.MODEL_API_MAX_RETRIES,
            timeout=config.MODEL_API_TIMEOUT,
            max_connections=1,
            adaptive_connections=False,
            max_tokens=config.JUDGE_MAX_TOKENS,
            temperature=config.JUDGE_TEMPERATURE,
            cache=False,
        ),
        memoize=False,
    )


def score_log(log: Any, evaluation: str, judge: Any) -> Any:
    """Apply one evaluation's scorer while using no target-model resources."""
    from inspect_ai import score as inspect_score

    if evaluation == "sycophancy":
        from inspect_evals.sycophancy.scorers import sycophancy_scorer

        scorer = sycophancy_scorer(judge)
    else:
        scorer = coherence_scorer(judge)
    # Supplying the judge as the active model prevents Inspect from trying to
    # reconstruct the in-process ``local-steered`` model named in the log.
    return inspect_score(
        log,
        scorer,
        model=judge,
        action="overwrite",
        display="plain",
    )


def score_log_incrementally(
    generated: Any,
    existing: Any | None,
    evaluation: str,
    judge: Any,
    output: Path,
) -> Any:
    """Score missing samples and persist after every successful judge result."""
    from inspect_ai.log import recompute_metrics, write_eval_log

    working = deepcopy(existing if existing is not None else generated)
    generated_samples = generated.samples or []
    working_samples = working.samples or []
    if len(working_samples) != len(generated_samples):
        raise ValueError(
            f"Scoring checkpoint has {len(working_samples)} samples but its "
            f"generation log has {len(generated_samples)}"
        )
    expected = expected_scorer_name(evaluation)
    output.parent.mkdir(parents=True, exist_ok=True)
    for index, generated_sample in enumerate(generated_samples):
        current_scores = working_samples[index].scores or {}
        if any(
            name == expected or name.startswith(f"{expected}-")
            for name in current_scores
        ):
            continue
        if (
            generated_sample.output is None
            or not generated_sample.output.completion
        ):
            raise RuntimeError(
                f"Cannot score {evaluation} sample {generated_sample.id!r}: "
                "the log contains no target-model response"
            )

        # The public scoring API operates on an EvalLog. A one-sample copy lets
        # us checkpoint completed judge work without rerunning target inference
        # or already-scored samples after a quota failure.
        one_sample_log = deepcopy(generated)
        one_sample_log.samples = [deepcopy(generated_sample)]
        one_sample_log.results = None
        one_sample_log.reductions = None
        scored_sample_log = score_log(one_sample_log, evaluation, judge)
        working_samples[index] = scored_sample_log.samples[0]
        working.eval.scorers = scored_sample_log.eval.scorers
        recompute_metrics(working)
        write_eval_log(working, output)
        logger.info(
            "Checkpointed %s sample %d/%d",
            evaluation,
            index + 1,
            len(generated_samples),
        )
    return working


def write_summary(rows: list[dict[str, Any]], directory: Path) -> None:
    rows.sort(key=lambda row: row["coefficient"])
    (directory / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if rows:
        with (directory / "summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summarize_experiment(experiment_root: Path) -> None:
    """Combine fully scored task logs into one row per coefficient."""
    from inspect_ai.log import read_eval_log

    grouped: dict[float, dict[str, Any]] = {}
    scored_root = experiment_root / "scored_logs"
    if not scored_root.exists():
        return
    for path in sorted(scored_root.rglob("*")):
        if not path.is_file() or path.suffix not in {".eval", ".json"}:
            continue
        log = read_eval_log(path)
        evaluation = evaluation_name(log)
        if not fully_scored(log, evaluation):
            continue
        metadata = log.eval.metadata or {}
        coefficient = float(metadata["steering_coefficient"])
        grouped.setdefault(coefficient, {})[evaluation] = log

    rows: list[dict[str, Any]] = []
    for coefficient, logs in grouped.items():
        if set(logs) != {"sycophancy", "coherence"}:
            logger.warning(
                "alpha=%g has only scored tasks %s; omitting it from the summary",
                coefficient, sorted(logs),
            )
            continue
        sycophancy_metadata = logs["sycophancy"].eval.metadata or {}
        coherence_metadata = logs["coherence"].eval.metadata or {}
        for key in ("target_layer", "target_layer_fraction"):
            if sycophancy_metadata.get(key) != coherence_metadata.get(key):
                raise ValueError(
                    f"alpha={coefficient:g} has inconsistent {key} metadata "
                    "between sycophancy and coherence logs"
                )
        row: dict[str, Any] = {
            "coefficient": coefficient,
            "sycophancy_steering_scope": sycophancy_metadata.get(
                "steering_scope"
            ),
            "coherence_steering_scope": coherence_metadata.get("steering_scope"),
            **sycophancy_summary(logs["sycophancy"]),
            **coherence_summary(logs["coherence"]),
        }
        if "target_layer" in sycophancy_metadata:
            row["target_layer"] = int(sycophancy_metadata["target_layer"])
        if "target_layer_fraction" in sycophancy_metadata:
            row["target_layer_fraction"] = float(
                sycophancy_metadata["target_layer_fraction"]
            )
        rows.append(row)
    write_summary(rows, experiment_root)


def write_layer_sweep_summary(
    result_root: Path, experiment_roots: set[Path]
) -> None:
    """Combine per-layer summaries into a sweep-level portable table."""
    rows: list[dict[str, Any]] = []
    for experiment_root in sorted(experiment_roots):
        summary_path = experiment_root / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            experiment = str(experiment_root.relative_to(result_root))
        except ValueError:
            experiment = str(experiment_root)
        for row in json.loads(summary_path.read_text(encoding="utf-8")):
            rows.append({"experiment": experiment, **row})
    if not rows:
        return
    rows.sort(key=lambda row: (
        int(row.get("target_layer", -1)),
        float(row["coefficient"]),
        str(row["experiment"]),
    ))
    (result_root / "layer_sweep_summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with (result_root / "layer_sweep_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score_results(result_root: Path) -> None:
    """Score all generation logs, checkpointing after each completed sample."""
    from inspect_ai.log import read_eval_log

    sources = generation_sources(result_root)
    if not sources:
        raise FileNotFoundError(f"No generation logs found below {result_root}")
    judge = build_judge_model()
    logger.info("Discovered %d generation logs across the layer sweep", len(sources))
    experiment_roots: set[Path] = set()
    for experiment_root, source in sources:
        experiment_roots.add(experiment_root)
        output = scored_location(experiment_root, source)
        existing = None
        if output.exists():
            existing = read_eval_log(output)
            evaluation = evaluation_name(existing)
            if fully_scored(existing, evaluation):
                logger.info("Already scored; skipping %s", output)
                continue

        generated = read_eval_log(source)
        evaluation = evaluation_name(generated)
        coefficient = float((generated.eval.metadata or {})["steering_coefficient"])
        metadata = generated.eval.metadata or {}
        logger.info(
            "Scoring %s at layer=%s (fraction=%s), alpha=%g",
            evaluation,
            metadata.get("target_layer", "legacy"),
            metadata.get("target_layer_fraction", "legacy"),
            coefficient,
        )
        scored = score_log_incrementally(
            generated, existing, evaluation, judge, output
        )
        save_per_sample(
            scored,
            experiment_root / "per_sample"
            / f"{evaluation}_alpha_{coefficient_slug(coefficient)}.jsonl",
            coefficient,
            evaluation,
        )

    for experiment_root in experiment_roots:
        summarize_experiment(experiment_root)
    write_layer_sweep_summary(result_root, experiment_roots)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score downloaded generation logs with the configured hosted judge."
    )
    parser.add_argument(
        "result_root",
        nargs="?",
        type=Path,
        default=config.RESULTS_DIR,
        help="Directory containing generation_manifest.json or inspect_logs/",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    score_results(args.result_root)
    logger.info("Local scoring complete. Summaries are below %s", args.result_root)


if __name__ == "__main__":
    main()
