from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from typer.testing import CliRunner

from feature_fixtures import EXTRACTION_CONFIG_SHA256, make_feature_fixture
from signlab import cli
from signlab.datasets.public_split import PublicCorpusSample
from signlab.experiments import legacy_gru
from signlab.experiments.tracking import ReferenceRunInput, ReferenceRunReceipt
from signlab.features.resources import load_packaged_default_feature_plan
from signlab.features.transforms import derive_feature_sequence

_CONFIG_PATH = Path("configs/experiments/popsign-legacy-gru-compatibility-v1.json")


def _config() -> legacy_gru.LegacyGruCompatibilityConfig:
    return legacy_gru.load_legacy_gru_compatibility_config(_CONFIG_PATH)[0]


def test_checked_in_config_is_canonical_frozen_and_vectorizes_one_sequence() -> None:
    config, raw = legacy_gru.load_legacy_gru_compatibility_config(_CONFIG_PATH)
    fixture = make_feature_fixture()
    feature = derive_feature_sequence(
        fixture.table,
        fixture.sequence,
        fixture.quality,
        load_packaged_default_feature_plan("combined"),
        extraction_config_sha256=EXTRACTION_CONFIG_SHA256,
    )
    sample = PublicCorpusSample(
        partition="train",
        target_label_id="hello",
        sample_id="sample_" + "1" * 32,
        source_recording_id=feature.source_recording_id,
        source_signer_id="participant_" + "2" * 32,
        quality_disposition="pass",
        feature=feature,
    )

    matrix, labels = legacy_gru._vectorize((sample,), config)

    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\r\n")
    assert config.legacy_run_id == "20251222_154233_gru_phase_3_run_001"
    assert config.legacy_model_sha256 == (
        "sha256:f69c07838a477df0853a6cdb71b1acb9a933e0de1491359da0cae5462584e46c"
    )
    assert config.labels == legacy_gru.TARGET_LABELS
    assert (config.input_frames, config.input_width) == (64, 134)
    assert (config.gru_layers, config.gru_units, config.parameter_count) == (2, 128, 201_222)
    assert (config.batch_size, config.maximum_epochs, config.early_stopping_patience) == (
        32,
        30,
        8,
    )
    assert (config.onnx_opset, config.parity_absolute_tolerance) == (18, 1e-5)
    promoted = json.loads(Path("docs/legacy/export/v1/promoted-artifacts.json").read_bytes())
    legacy = next(row for row in promoted["artifacts"] if row["run_id"] == config.legacy_run_id)
    assert "sha256:" + legacy["model"]["sha256"] == config.legacy_model_sha256
    assert "sha256:" + legacy["label_map"]["sha256"] == config.legacy_label_map_sha256
    assert legacy["model_key"] == "causal_gru"
    assert legacy["validity"]["data_role"] == "development-only"
    assert matrix.shape == (1, 64, 134)
    assert matrix.dtype == np.float32
    assert matrix.flags.c_contiguous
    assert labels.dtype == np.int64
    assert labels.tolist() == [0]
    np.testing.assert_allclose(
        matrix[0],
        np.asarray(feature.values_q, dtype=np.float32) / np.float32(1_000_000),
    )


def test_frozen_config_rejects_a_second_model_choice(tmp_path: Path) -> None:
    payload = json.loads(_CONFIG_PATH.read_bytes())
    payload["gru_units"] = 64
    drifted = tmp_path / "drifted.json"
    drifted.write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(
        legacy_gru.LegacyGruCompatibilityError,
        match="configuration is invalid",
    ):
        legacy_gru.load_legacy_gru_compatibility_config(drifted)


def test_real_model_has_the_recovered_topology_and_repeatable_initialization() -> None:
    pytest.importorskip("keras")
    pytest.importorskip("tensorflow")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    config = _config()
    runtime = legacy_gru._runtime()

    legacy_gru._seed_runtime(runtime, config.seed)
    first = legacy_gru._build_model(config, runtime)
    first_weights = tuple(np.asarray(weight).copy() for weight in first.get_weights())
    legacy_gru._seed_runtime(runtime, config.seed)
    second = legacy_gru._build_model(config, runtime)

    assert first.input_shape == (None, 64, 134)
    assert first.output_shape == (None, 5)
    assert first.count_params() == 201_222
    assert [layer.name for layer in first.layers] == [
        "input",
        "causal_gru_0",
        "causal_gru_1",
        "attn_scores",
        "attn_softmax",
        "attn_reduce",
        "features",
        "probabilities",
    ]
    gru_layers = [layer for layer in first.layers if layer.name.startswith("causal_gru_")]
    assert [layer.get_config()["units"] for layer in gru_layers] == [128, 128]
    assert all(layer.get_config()["return_sequences"] for layer in gru_layers)
    assert all(layer.use_cudnn is False for layer in gru_layers)
    assert all(
        np.array_equal(expected, actual)
        for expected, actual in zip(first_weights, second.get_weights(), strict=True)
    )


def test_real_model_completes_one_bounded_fit_and_saves(tmp_path: Path) -> None:
    pytest.importorskip("keras")
    pytest.importorskip("tensorflow")
    config = _config().model_copy(update={"maximum_epochs": 1, "early_stopping_patience": 1})
    runtime = legacy_gru._runtime()
    legacy_gru._seed_runtime(runtime, config.seed)
    model = legacy_gru._build_model(config, runtime)
    train_x = np.zeros((10, 64, 134), dtype=np.float32)
    train_y = np.repeat(np.arange(5, dtype=np.int64), 2)
    validation_x = np.zeros((5, 64, 134), dtype=np.float32)
    validation_y = np.arange(5, dtype=np.int64)

    training = legacy_gru._train_model(
        model,
        train_x,
        train_y,
        validation_x,
        validation_y,
        config,
        runtime,
    )
    artifact = legacy_gru._save_model(model, tmp_path / "model.keras")

    assert training.epochs_completed == 1
    assert training.best_epoch == 1
    assert 0.0 <= training.best_validation_accuracy <= 1.0
    assert training.best_validation_loss >= 0.0
    assert len(artifact) > 0


def test_real_fixed_shape_onnx_export_passes_checker_and_cpu_parity(tmp_path: Path) -> None:
    pytest.importorskip("keras")
    pytest.importorskip("tensorflow")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    config = _config()
    runtime = legacy_gru._runtime()
    legacy_gru._seed_runtime(runtime, config.seed)
    model = legacy_gru._build_model(config, runtime)
    validation = np.zeros((2, 64, 134), dtype=np.float32)
    expected = legacy_gru._predict_probabilities(model, validation)

    payload, parity = legacy_gru._export_and_compare(
        model,
        validation,
        expected,
        tmp_path / "model.onnx",
        config,
        runtime,
    )

    assert len(payload) > 0
    assert parity["compared_examples"] == 2
    assert parity["all_outputs_within_tolerance"] is True
    assert parity["identical_predicted_labels"] is True
    assert parity["maximum_absolute_difference"] <= 1e-5


def test_onnx_comparison_fails_closed_when_outputs_diverge(tmp_path: Path) -> None:
    config = _config()
    exported = tmp_path / "model.onnx"
    captured_providers: list[list[str]] = []

    class Model:
        def export(self, path: Path, **_kwargs: object) -> None:
            path.write_bytes(b"fixture-onnx")

    class Session:
        def get_inputs(self) -> list[Any]:
            return [SimpleNamespace(name="input", shape=[1, 64, 134])]

        def run(self, _outputs: object, _inputs: object) -> list[NDArray[np.float32]]:
            return [np.asarray([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)]

    def capture_session(_path: str, *, providers: list[str]) -> Session:
        captured_providers.append(list(providers))
        return Session()

    runtime = legacy_gru._Runtime(
        keras=SimpleNamespace(InputSpec=lambda **kwargs: kwargs),
        tensorflow=SimpleNamespace(),
        onnx=SimpleNamespace(checker=SimpleNamespace(check_model=lambda *_args, **_kwargs: None)),
        onnxruntime=SimpleNamespace(InferenceSession=capture_session),
    )
    validation = np.zeros((1, 64, 134), dtype=np.float32)
    keras_probabilities = np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    with pytest.raises(
        legacy_gru.LegacyGruCompatibilityError,
        match="Keras and ONNX validation outputs do not match",
    ):
        legacy_gru._export_and_compare(
            Model(),
            validation,
            keras_probabilities,
            exported,
            config,
            runtime,
        )

    assert captured_providers == [["CPUExecutionProvider"]]


def test_optional_runtime_boundaries_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KERAS_BACKEND", "jax")
    with pytest.raises(legacy_gru.LegacyGruCompatibilityError, match="TensorFlow backend"):
        legacy_gru._runtime()

    class InvalidPredictor:
        def predict(self, *_args: object, **_kwargs: object) -> NDArray[np.float32]:
            return np.zeros((1, 4), dtype=np.float32)

    with pytest.raises(legacy_gru.LegacyGruCompatibilityError, match="output is invalid"):
        legacy_gru._predict_probabilities(
            InvalidPredictor(), np.zeros((1, 64, 134), dtype=np.float32)
        )

    class UnsavableModel:
        def save(self, _path: Path) -> None:
            raise RuntimeError("private runtime detail")

    with pytest.raises(
        legacy_gru.LegacyGruCompatibilityError,
        match="artifact could not be saved",
    ) as failure:
        legacy_gru._save_model(UnsavableModel(), tmp_path / "model.keras")
    assert "private runtime detail" not in str(failure.value)

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(legacy_gru.LegacyGruCompatibilityError, match="must not already exist"):
        legacy_gru._new_path(existing, label="compatibility output")


def _samples(partition: str, per_class: int) -> tuple[Any, ...]:
    return tuple(
        SimpleNamespace(
            partition=partition,
            target_label_id=label,
            sample_id=f"private_sample_{label}_{position}",
            source_recording_id=f"private_recording_{label}_{position}",
            source_signer_id=f"private_signer_{position % 2}",
            quality_disposition="pass",
        )
        for label in legacy_gru.TARGET_LABELS
        for position in range(per_class)
    )


def test_one_run_never_opens_test_and_hands_sanitized_evidence_to_tracker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    captured: list[ReferenceRunInput] = []
    requested_partitions: list[tuple[str, ...]] = []
    development = (*_samples("train", 10), *_samples("validation", 3))
    receipt = ReferenceRunReceipt(
        run_id="e" * 32,
        experiment_id="1",
        artifact_sha256={
            "configuration.json": "sha256:" + "1" * 64,
            "report.json": "sha256:" + "2" * 64,
            "confusion-matrix.json": "sha256:" + "3" * 64,
            "predictions.csv": "sha256:" + "4" * 64,
        },
    )

    monkeypatch.setattr(legacy_gru, "_git_identity", lambda _path: (Path.cwd(), "d" * 40, False))
    monkeypatch.setattr(legacy_gru, "_load_inputs", lambda *_args: (b"split", b"manifest"))

    def load_samples(
        _split: bytes,
        _manifest: bytes,
        _root: Path,
        partitions: tuple[str, ...],
    ) -> tuple[Any, ...]:
        events.append("load:" + ",".join(partitions))
        requested_partitions.append(partitions)
        if "test" in partitions:
            raise AssertionError("Story #26 must not open the test partition")
        return development

    def vectorize(
        samples: Sequence[Any],
        _config_value: legacy_gru.LegacyGruCompatibilityConfig,
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        events.append("vectorize:" + samples[0].partition)
        labels = np.asarray(
            [legacy_gru.TARGET_LABELS.index(sample.target_label_id) for sample in samples],
            dtype=np.int64,
        )
        matrix = np.zeros((len(samples), 64, 134), dtype=np.float32)
        matrix[:, 0, 0] = labels
        return matrix, labels

    def predict(
        _model: object,
        matrix: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        events.append("predict")
        return np.eye(5, dtype=np.float32)[matrix[:, 0, 0].astype(np.int64)]

    def runtime() -> object:
        events.append("runtime")
        return object()

    def build_model(*_args: object) -> object:
        events.append("build")
        return model

    def train_model(*_args: object) -> legacy_gru._Training:
        events.append("fit")
        return legacy_gru._Training(
            epochs_completed=4,
            best_epoch=3,
            best_validation_accuracy=1.0,
            best_validation_loss=0.25,
        )

    def save_model(*_args: object) -> bytes:
        events.append("save")
        return b"keras-fixture"

    def export_and_compare(*_args: object) -> tuple[bytes, dict[str, bool | float | int]]:
        events.append("export")
        return b"onnx-fixture", {
            "all_outputs_within_tolerance": True,
            "compared_examples": 15,
            "identical_predicted_labels": True,
            "maximum_absolute_difference": 0.0,
        }

    monkeypatch.setattr(legacy_gru, "_load_samples", load_samples)
    monkeypatch.setattr(legacy_gru, "_vectorize", vectorize)
    monkeypatch.setattr(legacy_gru, "_runtime", runtime)
    monkeypatch.setattr(
        legacy_gru,
        "_seed_runtime",
        lambda *_args: events.append("seed"),
    )
    model = object()
    monkeypatch.setattr(legacy_gru, "_build_model", build_model)
    monkeypatch.setattr(legacy_gru, "_train_model", train_model)
    monkeypatch.setattr(legacy_gru, "_predict_probabilities", predict)
    monkeypatch.setattr(legacy_gru, "_save_model", save_model)
    monkeypatch.setattr(legacy_gru, "_export_and_compare", export_and_compare)
    monkeypatch.setattr(legacy_gru, "version", lambda _name: "fixture")

    def log_run(run: ReferenceRunInput, **_kwargs: object) -> ReferenceRunReceipt:
        events.append("track")
        captured.append(run)
        return receipt

    def verify_run(_run_id: str, **_kwargs: object) -> ReferenceRunReceipt:
        events.append("verify")
        return receipt

    monkeypatch.setattr(legacy_gru, "log_reference_run", log_run)
    monkeypatch.setattr(legacy_gru, "verify_reference_run", verify_run)
    output = tmp_path / "compatibility-run"
    public_report = tmp_path / "sanitized-report.md"

    result = legacy_gru.run_legacy_gru_compatibility(
        _CONFIG_PATH,
        corpus_root=tmp_path / "private-corpus",
        external_manifest_path=tmp_path / "private-manifest.json",
        output_root=output,
        public_report_path=public_report,
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.sqlite').as_posix()}",
    )

    assert requested_partitions == [("train", "validation")]
    assert events == [
        "load:train,validation",
        "vectorize:train",
        "vectorize:validation",
        "runtime",
        "seed",
        "build",
        "fit",
        "predict",
        "save",
        "export",
        "track",
        "verify",
    ]
    assert result.validation_macro_f1 == 1.0
    assert result.epochs_completed == 4
    assert result.tracking == receipt
    assert len(captured) == 1
    tracked = captured[0]
    assert tracked.parameters["fit_calls"] == 1
    assert tracked.parameters["test_status"] == "sealed_not_opened"
    assert tracked.metrics["validation.onnx_maximum_absolute_difference"] == 0.0
    assert tracked.metrics["validation.loss"] == 0.25
    assert all(path.is_file() for path in tracked.model_dump().values() if isinstance(path, Path))
    report = json.loads((output / "report.json").read_bytes())
    assert report["claim_scope"] == "compatibility_and_export_smoke_only"
    assert report["data"] == {
        "labels": list(legacy_gru.TARGET_LABELS),
        "opened_partition_counts": {"train": 50, "validation": 15},
        "test_status": "sealed_not_opened",
    }
    assert report["training"]["fit_calls"] == 1
    assert report["onnx"]["fixed_input_shape"] == [1, 64, 134]
    assert (output / "model.keras").read_bytes() == b"keras-fixture"
    assert (output / "model.onnx").read_bytes() == b"onnx-fixture"
    assert public_report.is_file()
    assert not (output / "public-report.md").exists()
    public_evidence = b"\n".join(
        path.read_bytes() for path in (*output.glob("*.json"), *output.glob("*.csv"), public_report)
    )
    assert b"private" not in public_evidence.lower()


def test_cli_reports_the_single_verified_run_without_echoing_private_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        legacy_gru,
        "run_legacy_gru_compatibility",
        lambda *_args, **_kwargs: SimpleNamespace(
            validation_macro_f1=0.625,
            tracking=SimpleNamespace(run_id="e" * 32),
        ),
    )

    result = CliRunner(env={"NO_COLOR": "1"}).invoke(
        cli.app,
        [
            "train",
            "legacy-gru-compatibility",
            str(_CONFIG_PATH),
            "--corpus-root",
            str(tmp_path / "private-corpus"),
            "--external-manifest",
            str(tmp_path / "private-manifest.json"),
            "--output-root",
            str(tmp_path / "private-output"),
        ],
    )

    assert result.exit_code == 0
    assert result.output.strip() == (
        f"Legacy GRU compatibility verified: validation macro-F1 0.625; ledger run {'e' * 32}."
    )
    assert "private" not in result.output.casefold()


def test_cli_reports_a_stable_compatibility_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise legacy_gru.LegacyGruCompatibilityError("compatibility partition quotas drifted")

    monkeypatch.setattr(legacy_gru, "run_legacy_gru_compatibility", fail)

    result = CliRunner(env={"NO_COLOR": "1"}).invoke(
        cli.app,
        [
            "train",
            "legacy-gru-compatibility",
            str(_CONFIG_PATH),
            "--corpus-root",
            "corpus",
            "--external-manifest",
            "external.json",
            "--output-root",
            "run",
        ],
    )

    assert result.exit_code == 1
    assert result.output.strip() == (
        "Legacy GRU compatibility run failed: compatibility partition quotas drifted"
    )
