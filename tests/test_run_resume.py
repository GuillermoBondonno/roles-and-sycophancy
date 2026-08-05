from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import run


def sample(completion: str = "response") -> SimpleNamespace:
    return SimpleNamespace(
        error=None,
        output=SimpleNamespace(completion=completion),
    )


def log(samples: int, location: str) -> SimpleNamespace:
    return SimpleNamespace(
        status="started",
        samples=[sample() for _ in range(samples)],
        location=location,
    )


def test_generate_or_resume_task_reuses_complete_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete = log(100, "complete.eval")
    calls: list[str] = []
    monkeypatch.setattr(
        run,
        "existing_generation_logs",
        lambda *_args: [complete],
    )
    monkeypatch.setattr(
        run,
        "require_generated_eval",
        lambda *_args: calls.append("generated"),
    )

    selected = run.generate_or_resume_task(
        object(), {}, Path("logs"), "sycophancy", 100, -7.0,
        run.STEERING_SCOPE_SECOND_ASSISTANT,
    )

    assert selected is complete
    assert calls == []


def test_generate_or_resume_task_retries_best_partial_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect_ai

    smaller = log(3, "smaller.eval")
    best = log(11, "best.eval")
    resumed = SimpleNamespace(
        status="success",
        samples=[sample() for _ in range(100)],
        location="resumed.eval",
    )
    retried: list[tuple[SimpleNamespace, dict[str, object]]] = []
    monkeypatch.setattr(
        run,
        "existing_generation_logs",
        lambda *_args: [smaller, best],
    )
    monkeypatch.setattr(
        inspect_ai,
        "eval_retry",
        lambda selected, **kwargs: (
            retried.append((selected, kwargs)) or [resumed]
        ),
    )

    selected = run.generate_or_resume_task(
        object(), {}, Path("logs"), "sycophancy", 100, -7.0,
        run.STEERING_SCOPE_SECOND_ASSISTANT,
    )

    assert selected is resumed
    assert retried[0][0] is best
    assert retried[0][1]["score"] is False
    assert retried[0][1]["log_dir"] == "logs"


def test_generate_or_resume_task_disables_task_shuffle_override_on_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect_ai

    partial = log(11, "partial.eval")
    partial.eval = SimpleNamespace(
        config=SimpleNamespace(sample_shuffle=False),
    )
    resumed = SimpleNamespace(
        status="success",
        samples=[sample() for _ in range(100)],
        location="resumed.eval",
    )
    retried: list[SimpleNamespace] = []
    monkeypatch.setattr(
        run,
        "existing_generation_logs",
        lambda *_args: [partial],
    )
    monkeypatch.setattr(
        inspect_ai,
        "eval_retry",
        lambda selected, **_kwargs: retried.append(selected) or [resumed],
    )

    run.generate_or_resume_task(
        object(), {}, Path("logs"), "sycophancy", 100, -7.0,
        run.STEERING_SCOPE_SECOND_ASSISTANT,
    )

    assert retried[0] is not partial
    assert retried[0].eval.config.sample_shuffle is None
    assert partial.eval.config.sample_shuffle is False


def test_legacy_generation_log_is_probe_direction_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect_ai.log

    path = tmp_path / "legacy.eval"
    path.touch()
    legacy = SimpleNamespace(
        eval=SimpleNamespace(
            task_display_name="sycophancy",
            task_registry_name=None,
            task=None,
            metadata={
                "steering_coefficient": -3.0,
                "steering_scope": run.STEERING_SCOPE_SECOND_ASSISTANT,
            },
        )
    )
    monkeypatch.setattr(inspect_ai.log, "read_eval_log", lambda _path: legacy)

    assert run.existing_generation_logs(
        tmp_path,
        "sycophancy",
        -3.0,
        run.STEERING_SCOPE_SECOND_ASSISTANT,
        run.STEERING_DIRECTION_PROBE,
    ) == [legacy]
    with pytest.raises(RuntimeError, match="mixing steering directions"):
        run.existing_generation_logs(
            tmp_path,
            "sycophancy",
            -3.0,
            run.STEERING_SCOPE_SECOND_ASSISTANT,
            run.STEERING_DIRECTION_MEAN_DIFFERENCE,
        )


def test_evaluation_sample_contract_rejects_dataset_drift(tmp_path: Path) -> None:
    from inspect_ai.dataset import MemoryDataset, Sample

    task = SimpleNamespace(dataset=MemoryDataset([
        Sample(id="one", input="First prompt", target="A"),
        Sample(id="two", input="Second prompt", target="B"),
    ]))
    contract = run.task_sample_contract(task, "sycophancy", 2)
    run.ensure_evaluation_sample_contract(
        tmp_path, {"sycophancy": contract}
    )

    changed_task = SimpleNamespace(dataset=MemoryDataset([
        Sample(id="one", input="Changed prompt", target="A"),
        Sample(id="two", input="Second prompt", target="B"),
    ]))
    changed = run.task_sample_contract(changed_task, "sycophancy", 2)

    with pytest.raises(ValueError, match="sample contract changed"):
        run.ensure_evaluation_sample_contract(
            tmp_path, {"sycophancy": changed}
        )


def test_log_sample_contract_rejects_retry_shuffle() -> None:
    contract = {
        "evaluation": "sycophancy",
        "sample_count": 2,
        "samples": [
            {"id": "one", "sample_sha256": "a"},
            {"id": "two", "sample_sha256": "b"},
        ],
    }
    generated = SimpleNamespace(
        eval=SimpleNamespace(
            dataset=SimpleNamespace(sample_ids=["two", "one"]),
        ),
        samples=[
            SimpleNamespace(id="one"),
            SimpleNamespace(id="two"),
        ],
    )

    with pytest.raises(RuntimeError, match="does not match"):
        run.assert_log_matches_sample_contract(generated, contract)


def test_cli_routes_explicit_resume_directories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        run,
        "resume_main",
        lambda fraction, artifacts, results, coefficients, direction: calls.append(
            (fraction, artifacts, results, coefficients, direction)
        ),
    )

    run.cli([
        "--resume-generation",
        "0.75",
        "--artifacts-dir",
        "artifacts_aug_2_v2",
        "--results-dir",
        "results_aug_2_v2",
    ])

    assert calls == [(
        0.75,
        Path("artifacts_aug_2_v2"),
        Path("results_aug_2_v2"),
        None,
        run.STEERING_DIRECTION_PROBE,
    )]


def test_cli_routes_all_layers_and_explicit_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        run,
        "resume_main",
        lambda fraction, artifacts, results, coefficients, direction: calls.append(
            (fraction, artifacts, results, coefficients, direction)
        ),
    )

    run.cli(
        [
            "--resume-all",
            "--artifacts-dir",
            "artifacts_aug_2_v2_combined",
            "--results-dir",
            "results_aug_2_v2_combined",
            "--steering-coefficients",
            "-8",
            "-6",
            "-4",
            "-2",
        ]
    )

    assert calls == [
        (
            None,
            Path("artifacts_aug_2_v2_combined"),
            Path("results_aug_2_v2_combined"),
            (-8.0, -6.0, -4.0, -2.0),
            run.STEERING_DIRECTION_PROBE,
        )
    ]


def test_cli_routes_mean_difference_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        run,
        "resume_main",
        lambda fraction, artifacts, results, coefficients, direction: calls.append(
            (fraction, artifacts, results, coefficients, direction)
        ),
    )

    run.cli([
        "--resume-all",
        "--artifacts-dir",
        "artifacts",
        "--results-dir",
        "results_mean_difference",
        "--steering-direction",
        "mean-difference",
        "--steering-coefficients",
        "-3",
        "0",
        "3",
    ])

    assert calls == [(
        None,
        Path("artifacts"),
        Path("results_mean_difference"),
        (-3.0, 0.0, 3.0),
        run.STEERING_DIRECTION_MEAN_DIFFERENCE,
    )]


def test_load_saved_probe_selects_mean_difference_arrays(tmp_path: Path) -> None:
    (tmp_path / "probe_metadata.json").write_text("{}", encoding="utf-8")
    np.savez(
        tmp_path / "probe.npz",
        direction_unit=np.array([1.0, 0.0], dtype=np.float32),
        projection_scale=np.array(0.25, dtype=np.float32),
        mean_difference_direction_unit=np.array([0.0, 1.0], dtype=np.float32),
        mean_difference_projection_scale=np.array(0.75, dtype=np.float32),
    )

    artifact, _ = run.load_saved_probe(
        tmp_path, run.STEERING_DIRECTION_MEAN_DIFFERENCE
    )

    np.testing.assert_array_equal(
        artifact["direction_unit"], np.array([0.0, 1.0], dtype=np.float32)
    )
    assert float(artifact["projection_scale"]) == pytest.approx(0.75)


def test_resume_all_dispatches_every_saved_layer_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    result_root = tmp_path / "results"
    artifact_root.mkdir()
    (artifact_root / "layer_sweep_manifest.json").write_text(
        json.dumps(
            [
                {"target_layer_fraction": 0.25},
                {"target_layer_fraction": 0.5},
                {"target_layer_fraction": 0.75},
            ]
        ),
        encoding="utf-8",
    )
    calls: list[tuple[float, tuple[float, ...] | None, str]] = []
    model = object()
    tokenizer = object()
    monkeypatch.setattr(run, "seed_everything", lambda _seed: None)
    monkeypatch.setattr(run, "load_model_once", lambda: (model, tokenizer))
    monkeypatch.setattr(run, "log_stage", lambda _message: 0.0)
    monkeypatch.setattr(run, "log_stage_done", lambda *_args: None)
    monkeypatch.setattr(run, "notify", lambda _message: None)

    def record_layer(
        loaded_model: object,
        loaded_tokenizer: object,
        fraction: float,
        artifacts: Path,
        results: Path,
        coefficients: tuple[float, ...] | None,
        direction: str,
    ) -> None:
        assert loaded_model is model
        assert loaded_tokenizer is tokenizer
        assert artifacts == artifact_root
        assert results == result_root
        calls.append((fraction, coefficients, direction))

    monkeypatch.setattr(
        run,
        "resume_layer_generation",
        record_layer,
    )

    run._resume_generation(
        None,
        artifact_root,
        result_root,
        (-8.0, -6.0),
    )

    assert calls == [
        (0.25, (-8.0, -6.0), run.STEERING_DIRECTION_PROBE),
        (0.5, (-8.0, -6.0), run.STEERING_DIRECTION_PROBE),
        (0.75, (-8.0, -6.0), run.STEERING_DIRECTION_PROBE),
    ]


def test_merge_generation_manifest_preserves_and_replaces_entries(
    tmp_path: Path,
) -> None:
    original = [
        {"coefficient": -7.0, "evaluation": "sycophancy", "log": "old.eval"},
        {"coefficient": -7.0, "evaluation": "coherence", "log": "coh.eval"},
    ]
    (tmp_path / "generation_manifest.json").write_text(
        json.dumps(original), encoding="utf-8"
    )

    merged = run.merge_generation_manifest(
        tmp_path,
        [
            {"coefficient": -7.0, "evaluation": "sycophancy", "log": "new.eval"},
            {"coefficient": -6.0, "evaluation": "sycophancy", "log": "added.eval"},
        ],
    )

    assert merged == [
        {"coefficient": -7.0, "evaluation": "sycophancy", "log": "new.eval"},
        {"coefficient": -7.0, "evaluation": "coherence", "log": "coh.eval"},
        {"coefficient": -6.0, "evaluation": "sycophancy", "log": "added.eval"},
    ]
    assert json.loads(
        (tmp_path / "generation_manifest.json").read_text(encoding="utf-8")
    ) == merged


def test_rejects_coefficients_with_colliding_directory_slugs() -> None:
    with pytest.raises(ValueError, match="both map"):
        run.validate_steering_coefficients((1.0000001, 1.0000002))
