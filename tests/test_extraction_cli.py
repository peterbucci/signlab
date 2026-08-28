from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from signlab import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"NO_COLOR": "1"})


def _manifest() -> object:
    return SimpleNamespace(
        manifest_sha256="sha256:" + "a" * 64,
        config_sha256="sha256:" + "b" * 64,
        raw_data_sha256="sha256:" + "c" * 64,
        participant_id="participant_ffffffffffffffffffffffffffffffff",
    )


def _validation() -> object:
    return SimpleNamespace(
        raw_bundle_integrity="verified",
        consent_boundary="synthetic_fixture_only",
        manifest_integrity="verified",
        registered_model_identity="verified",
        parquet_byte_integrity="verified",
        semantic_integrity="verified",
        exact_inventory="verified",
        sequence_count=2,
        frame_count=61,
        invalid_frame_count=1,
    )


def _expected_summary(status: str) -> list[str]:
    return [
        f"Landmark extraction: {status}.",
        "Extraction manifest SHA-256: sha256:" + "a" * 64,
        "Extraction configuration SHA-256: sha256:" + "b" * 64,
        "Raw data SHA-256: sha256:" + "c" * 64,
        "Extracted sequences: 2",
        "Landmark frames: 61",
        "Invalid frames: 1",
        "Extraction bundle integrity: verified",
    ]


def test_extract_landmarks_reads_bytes_and_delegates_only_the_stable_batch_inputs(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from signlab.extraction import batch, resources

    raw_manifest = tmp_path / "participant-private-raw-manifest.json"
    raw_bundle_root = tmp_path / "participant-private-raw-bundle"
    model_root = tmp_path / "participant-private-models"
    output = tmp_path / "participant-private-landmarks"
    raw_manifest.write_bytes(b"synthetic-raw-manifest")
    config = object()
    calls: list[tuple[bytes, Path, Path, object, Path]] = []

    monkeypatch.setattr(resources, "load_packaged_default_extraction_config", lambda: config)

    def extract(
        document: bytes,
        *,
        raw_bundle_root: Path,
        model_root: Path,
        config: object,
        destination: Path,
    ) -> object:
        calls.append((document, raw_bundle_root, model_root, config, destination))
        return SimpleNamespace(
            status="published",
            manifest=_manifest(),
            validation=_validation(),
        )

    monkeypatch.setattr(batch, "extract_raw_dataset", extract)

    result = runner.invoke(
        cli.app,
        [
            "data",
            "extract-landmarks",
            str(raw_manifest),
            "--raw-bundle-root",
            str(raw_bundle_root),
            "--model-root",
            str(model_root),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        (
            b"synthetic-raw-manifest",
            raw_bundle_root,
            model_root,
            config,
            output,
        )
    ]
    assert result.output.splitlines() == _expected_summary("published")
    assert "participant" not in result.output.casefold()
    assert str(tmp_path) not in result.output


def test_validate_extraction_reads_both_manifests_and_delegates_exact_roots(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from signlab.extraction import batch

    extraction_manifest = tmp_path / "participant-private-extraction.json"
    raw_manifest = tmp_path / "participant-private-raw.json"
    workspace_root = tmp_path / "participant-private-landmarks"
    raw_bundle_root = tmp_path / "participant-private-raw-bundle"
    extraction_manifest.write_bytes(b"synthetic-extraction-manifest")
    raw_manifest.write_bytes(b"synthetic-raw-manifest")
    calls: list[tuple[bytes, Path, bytes, Path]] = []

    def validate(
        document: bytes,
        workspace_root: Path,
        *,
        raw_manifest: bytes,
        raw_bundle_root: Path,
    ) -> object:
        calls.append((document, workspace_root, raw_manifest, raw_bundle_root))
        return SimpleNamespace(manifest=_manifest(), validation=_validation())

    monkeypatch.setattr(batch, "validate_landmark_extraction_bundle", validate)

    result = runner.invoke(
        cli.app,
        [
            "data",
            "validate-extraction",
            str(extraction_manifest),
            "--workspace-root",
            str(workspace_root),
            "--raw-manifest",
            str(raw_manifest),
            "--raw-bundle-root",
            str(raw_bundle_root),
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        (
            b"synthetic-extraction-manifest",
            workspace_root,
            b"synthetic-raw-manifest",
            raw_bundle_root,
        )
    ]
    assert result.output.splitlines() == _expected_summary("verified")
    assert "participant" not in result.output.casefold()
    assert str(tmp_path) not in result.output


@pytest.mark.parametrize(
    "error",
    [
        OSError("participant-private-model-path"),
        ValueError("participant-private-vendor-error"),
    ],
)
def test_extract_landmarks_redacts_os_and_value_errors(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    from signlab.extraction import batch, resources

    raw_manifest = tmp_path / "participant-private-raw.json"
    raw_manifest.write_bytes(b"participant-private-content")
    monkeypatch.setattr(resources, "load_packaged_default_extraction_config", lambda: object())

    def fail(*_args: object, **_kwargs: object) -> object:
        raise error

    monkeypatch.setattr(batch, "extract_raw_dataset", fail)

    result = runner.invoke(
        cli.app,
        [
            "data",
            "extract-landmarks",
            str(raw_manifest),
            "--raw-bundle-root",
            str(tmp_path / "participant-private-raw-root"),
            "--model-root",
            str(tmp_path / "participant-private-model-root"),
            "--output",
            str(tmp_path / "participant-private-output"),
        ],
    )

    assert result.exit_code == 1
    assert result.output.strip() == "Landmark extraction failed."
    assert "participant-private" not in result.output
    assert str(tmp_path) not in result.output
    assert "Traceback" not in result.output


def test_validate_extraction_redacts_batch_errors(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from signlab.extraction import batch

    extraction_manifest = tmp_path / "participant-private-extraction.json"
    raw_manifest = tmp_path / "participant-private-raw.json"
    extraction_manifest.write_bytes(b"participant-private-extraction-content")
    raw_manifest.write_bytes(b"participant-private-raw-content")

    def fail(*_args: object, **_kwargs: object) -> object:
        raise batch.ExtractionBatchError("bundle.invalid")

    monkeypatch.setattr(batch, "validate_landmark_extraction_bundle", fail)

    result = runner.invoke(
        cli.app,
        [
            "data",
            "validate-extraction",
            str(extraction_manifest),
            "--workspace-root",
            str(tmp_path / "participant-private-workspace"),
            "--raw-manifest",
            str(raw_manifest),
            "--raw-bundle-root",
            str(tmp_path / "participant-private-raw-root"),
        ],
    )

    assert result.exit_code == 1
    assert result.output.strip() == "Landmark extraction validation failed."
    assert "participant-private" not in result.output
    assert str(tmp_path) not in result.output
    assert "Traceback" not in result.output


def test_dataset_resource_validation_includes_extraction_and_feature_resources(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from signlab.datasets import external_resources, ingest_resources
    from signlab.datasets import resources as dataset_resources
    from signlab.extraction import resources as extraction_resources
    from signlab.features import resources as feature_resources
    from signlab.quality import resources as quality_resources

    calls: list[str] = []
    monkeypatch.setattr(
        dataset_resources,
        "validate_packaged_dataset_resources",
        lambda: calls.append("dataset"),
    )
    monkeypatch.setattr(
        external_resources,
        "validate_packaged_external_dataset_resources",
        lambda: calls.append("external"),
    )
    monkeypatch.setattr(
        ingest_resources,
        "validate_packaged_ingest_resources",
        lambda: calls.append("ingest"),
    )
    monkeypatch.setattr(
        extraction_resources,
        "validate_packaged_extraction_resources",
        lambda: calls.append("extraction"),
    )
    monkeypatch.setattr(
        quality_resources,
        "validate_packaged_quality_resources",
        lambda: calls.append("quality"),
    )
    monkeypatch.setattr(
        feature_resources,
        "validate_packaged_feature_resources",
        lambda: calls.append("feature"),
    )

    result = runner.invoke(cli.app, ["data", "validate-resources"])

    assert result.exit_code == 0
    assert calls == ["dataset", "external", "ingest", "extraction", "quality", "feature"]
    assert result.output.strip() == (
        "Packaged dataset, external, ingest, extraction, quality, and feature resources are valid."
    )


def test_extraction_help_does_not_import_optional_native_or_batch_modules() -> None:
    probe = (
        "import sys; from typer.testing import CliRunner; import signlab.cli as cli; "
        "runner = CliRunner(env={'NO_COLOR': '1'}); "
        "results = [runner.invoke(cli.app, ['data', '--help']), "
        "runner.invoke(cli.app, ['data', 'extract-landmarks', '--help']), "
        "runner.invoke(cli.app, ['data', 'validate-extraction', '--help'])]; "
        "blocked = {'av', 'mediapipe', 'pyarrow', 'signlab.extraction.batch', "
        "'signlab.extraction.resources'} & set(sys.modules); "
        "failed = any(result.exit_code for result in results); "
        "raise SystemExit(','.join(sorted(blocked)) if blocked else (1 if failed else 0))"
    )

    subprocess.run([sys.executable, "-c", probe], check=True)
