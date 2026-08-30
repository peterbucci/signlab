from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import onnx
import pytest

from signlab.contracts.canonical import canonical_json_bytes
from signlab.contracts.pipeline import (
    assert_model_compatible,
    validate_dataset_manifest_v1,
    validate_model_manifest,
    validate_preprocessing_plan,
    validate_resolved_configuration,
    validate_run_record,
    validate_split_manifest,
)
from signlab.contracts.taxonomy import EXPECTED_CLASS_IDS
from signlab.experiments import reference_experiment, tracking
from signlab.experiments.reference_experiment import (
    ReferenceExperimentError,
    run_reference_experiment,
)
from signlab.experiments.tracking import ReferenceRunInput, ReferenceRunReceipt
from signlab.features.transforms import derive_feature_source

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/experiments/signlab-reference-experiment-v1.json"


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_bytes()))


def _receipt() -> ReferenceRunReceipt:
    return ReferenceRunReceipt(
        run_id="e" * 32,
        experiment_id="1",
        artifact_sha256={
            "configuration.json": "sha256:" + "1" * 64,
            "report.json": "sha256:" + "2" * 64,
            "confusion-matrix.json": "sha256:" + "3" * 64,
            "predictions.csv": "sha256:" + "4" * 64,
        },
    )


def test_checked_recipe_is_compact_canonical_and_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = CONFIG.read_bytes()
    payload = _json(CONFIG)

    assert len(raw) <= 32 * 1024
    assert raw == canonical_json_bytes(payload) + b"\n"
    assert payload["labels"] == list(EXPECTED_CLASS_IDS)
    assert payload["classifier_samples"] == 36
    assert payload["signer_groups"] == 6
    assert payload["sessions_per_signer"] == 2
    assert payload["source_kind"] == "project_authored_synthetic"
    assert payload["license_spdx"] == "MIT"
    assert payload["contains_person_data"] is False

    payload["unreviewed_option"] = True
    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(canonical_json_bytes(payload) + b"\n")
    with pytest.raises(ReferenceExperimentError, match="recipe is unavailable or invalid"):
        run_reference_experiment(invalid, output_root=tmp_path / "output")

    def dirty_checkout(_path: Path) -> tuple[Path, str]:
        raise ReferenceExperimentError("reference experiment requires a clean committed checkout")

    monkeypatch.setattr(reference_experiment, "_git_identity", dirty_checkout)
    with pytest.raises(ReferenceExperimentError, match="clean committed checkout"):
        run_reference_experiment(CONFIG, output_root=tmp_path / "output")


@pytest.mark.integration
def test_one_reference_run_proves_the_complete_synthetic_mechanics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt()
    tracking_uri = f"sqlite:///{(tmp_path / 'tracking.sqlite').as_posix()}"
    tracking_events: list[str] = []
    tracked_runs: list[ReferenceRunInput] = []
    feature_shapes: list[tuple[int, int]] = []

    def observe_feature(*args: Any, **kwargs: Any) -> Any:
        sequence = derive_feature_source(*args, **kwargs)
        feature_shapes.append((len(sequence.values_q), len(sequence.values_q[0])))
        return sequence

    def log_run(
        run: ReferenceRunInput,
        *,
        tracking_uri: str | None = None,
    ) -> ReferenceRunReceipt:
        tracking_events.append(f"log:{tracking_uri}")
        tracked_runs.append(run)
        paths = (
            run.configuration_path,
            run.report_path,
            run.confusion_matrix_path,
            run.predictions_path,
        )
        assert [path.name for path in paths] == [
            "configuration.json",
            "evaluation.json",
            "confusion-matrix.json",
            "predictions.csv",
        ]
        assert all(path.is_file() for path in paths)
        return receipt

    def verify_run(
        run_id: str,
        *,
        tracking_uri: str | None = None,
    ) -> ReferenceRunReceipt:
        tracking_events.append(f"verify:{run_id}:{tracking_uri}")
        return receipt

    monkeypatch.setattr(reference_experiment, "derive_feature_source", observe_feature)
    monkeypatch.setattr(reference_experiment, "_git_identity", lambda _path: (ROOT, "d" * 40))
    monkeypatch.setattr(tracking, "log_reference_run", log_run)
    monkeypatch.setattr(tracking, "verify_reference_run", verify_run)
    output = tmp_path / "pack"

    started = perf_counter()
    result = run_reference_experiment(
        CONFIG,
        output_root=output,
        tracking_uri=tracking_uri,
    )
    elapsed = perf_counter() - started

    assert elapsed < 30
    assert result.summary_path == output.resolve() / "summary.json"
    assert result.sample_count == 36
    assert result.label_count == 6
    assert result.parity_maximum_absolute_difference <= 1e-5
    assert result.tracking == receipt
    assert feature_shapes == [(64, 126)] * 38
    assert tracking_events == [f"log:{tracking_uri}", f"verify:{receipt.run_id}:{tracking_uri}"]
    tracked = tracked_runs[0]
    assert tracked.parameters == {
        "classifier_samples": 36,
        "model": "nearest_centroid",
        "threads": 1,
    }
    assert set(tracked.metrics) == {
        "test.balanced_accuracy",
        "test.macro_f1",
        "validation.balanced_accuracy",
        "validation.macro_f1",
    }

    assert (output / "native-parameters.json").is_file()
    assert (output / "recipe.json").read_bytes() == CONFIG.read_bytes()

    dataset = validate_dataset_manifest_v1((output / "dataset.json").read_bytes())
    split = validate_split_manifest((output / "split.json").read_bytes())
    preprocessing = validate_preprocessing_plan((output / "preprocessing.json").read_bytes())
    configuration = validate_resolved_configuration((output / "configuration.json").read_bytes())
    run = validate_run_record((output / "run.json").read_bytes())
    model = validate_model_manifest((output / "model.json").read_bytes())
    assert_model_compatible(dataset, split, preprocessing, configuration, run, model)

    samples = {sample.sample_id: sample for sample in dataset.content.samples}
    assert {sample.label_id for sample in samples.values()} == set(EXPECTED_CLASS_IDS)
    assert len({sample.participant_id for sample in samples.values()}) == 6
    assert len({sample.session_id for sample in samples.values()}) == 12
    for field in ("sample_ids", "participant_ids", "session_ids", "source_recording_ids"):
        memberships = [set(getattr(partition, field)) for partition in split.partitions]
        assert all(left.isdisjoint(right) for left, right in pairwise(memberships))
        assert memberships[0].isdisjoint(memberships[2])
    for partition in split.partitions:
        assert len(partition.sample_ids) == 12
        assert len(partition.participant_ids) == 2
        assert len(partition.session_ids) == 4
        assert {samples[sample_id].label_id for sample_id in partition.sample_ids} == set(
            EXPECTED_CLASS_IDS
        )

    step_parameters = {item.name: item.value for item in preprocessing.steps[0].parameters}
    assert step_parameters == {"frames": 64, "width": 126}
    assert configuration.model.implementation_id == "nearest_centroid_linear_scores"
    assert {item.name: item.value for item in configuration.optimizer.parameters} == {"passes": 1}
    assert {item.name: item.value for item in configuration.trainer.parameters} == {"threads": 1}

    onnx_model = onnx.load_model_from_string((output / "model.onnx").read_bytes())
    onnx.checker.check_model(onnx_model)
    shapes = [
        [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        for value in (*onnx_model.graph.input, *onnx_model.graph.output)
    ]
    assert shapes == [[1, 64, 126], [1, 6]]
    parity = _json(output / "parity.json")
    assert parity["status"] == "pass"
    assert parity["providers"] == ["CPUExecutionProvider"]
    assert parity["cases"] == 36
    assert parity["class_mismatches"] == 0
    assert (parity["absolute_tolerance"], parity["relative_tolerance"]) == (1e-5, 1e-5)
    assert parity["maximum_absolute_difference"] <= 1e-5

    evaluation = _json(output / "evaluation.json")
    assert evaluation["claim_scope"] == "synthetic_mechanics_only"
    assert evaluation["data"]["learned_other"] == {"kind": "oov_gesture", "samples": 6}
    assert evaluation["data"]["inactive"] == {
        "cases": 2,
        "handled_as_inactive": 2,
        "included_in_classifier": False,
    }
    assert set(evaluation["partitions"]) == {"validation", "test"}

    summary = _json(result.summary_path)
    assert summary["status"] == "pass"
    assert summary["structure"] == _json(
        ROOT / "configs/experiments/signlab-reference-experiment-expected-v1.json"
    )
    expected_metrics = summary["structure"]["metric_expectations"]
    assert all(
        abs(tracked.metrics[name] - expected) <= expected_metrics["absolute_tolerance"]
        for name, expected in expected_metrics["values"].items()
    )
    assert summary["tracking"] == {"run_id": receipt.run_id, "verified": True}
    assert summary["budgets"]["pack_bytes"] <= 5 * 1024 * 1024
    assert sum(path.stat().st_size for path in output.iterdir()) <= 5 * 1024 * 1024
    assert (output / "dataset-card.md").stat().st_size < 512
    assert (output / "model-card.md").stat().st_size < 512
    textual = (
        b"\n".join(
            path.read_bytes()
            for path in output.iterdir()
            if path.suffix in {".json", ".csv", ".md"}
        )
        .decode(errors="replace")
        .casefold()
    )
    assert str(ROOT).casefold() not in textual
    assert str(tmp_path).casefold() not in textual
    assert "browser-model-bundle/1" not in textual
    assert "quality claim" in textual
    assert "\nruns/\n" in (ROOT / ".gitignore").read_text(encoding="utf-8")
