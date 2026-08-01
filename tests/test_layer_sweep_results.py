from __future__ import annotations

import json
from pathlib import Path

from score_results import generation_sources, write_layer_sweep_summary


def test_generation_sources_discovers_every_layer_manifest(tmp_path: Path) -> None:
    expected: set[Path] = set()
    for layer in (7, 14, 21):
        experiment_root = tmp_path / f"layer_{layer}_fraction_0p5"
        log = experiment_root / "inspect_logs" / "alpha_p0" / "run.eval"
        log.parent.mkdir(parents=True)
        log.touch()
        (experiment_root / "generation_manifest.json").write_text(
            json.dumps([{"log": str(log.relative_to(experiment_root))}]),
            encoding="utf-8",
        )
        expected.add(log)

    sources = generation_sources(tmp_path)

    assert {source for _, source in sources} == expected
    assert {root.name for root, _ in sources} == {
        "layer_7_fraction_0p5",
        "layer_14_fraction_0p5",
        "layer_21_fraction_0p5",
    }


def test_writes_combined_layer_sweep_summary(tmp_path: Path) -> None:
    roots: set[Path] = set()
    for layer, fraction in ((7, 0.25), (14, 0.5)):
        root = tmp_path / f"layer_{layer}"
        root.mkdir()
        roots.add(root)
        (root / "summary.json").write_text(
            json.dumps([{
                "coefficient": 0.0,
                "target_layer": layer,
                "target_layer_fraction": fraction,
            }]),
            encoding="utf-8",
        )

    write_layer_sweep_summary(tmp_path, roots)

    combined = json.loads(
        (tmp_path / "layer_sweep_summary.json").read_text(encoding="utf-8")
    )
    assert [row["target_layer"] for row in combined] == [7, 14]
    assert (tmp_path / "layer_sweep_summary.csv").is_file()
