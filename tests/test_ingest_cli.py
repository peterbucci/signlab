from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from signlab import cli
from signlab.contracts.ingest import validate_capture_identifier_set
from signlab.datasets import importer, raw_bundle


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"NO_COLOR": "1"})


def test_allocate_capture_ids_persists_opaque_ids_without_printing_them(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    output = tmp_path / "private-person-capture-ids.json"

    first = runner.invoke(cli.app, ["data", "allocate-capture-ids", str(output)])
    second = runner.invoke(cli.app, ["data", "allocate-capture-ids", str(output)])

    assert first.exit_code == second.exit_code == 0
    assert first.output.splitlines()[0] == "Capture identifiers: created."
    assert second.output.splitlines()[0] == "Capture identifiers: unchanged."
    assert first.output.splitlines()[1] == second.output.splitlines()[1]
    checked = validate_capture_identifier_set(output.read_bytes())
    assert checked.recording_id not in first.output
    assert output.name not in first.output
    assert str(tmp_path) not in first.output


def test_validate_capture_reports_only_counts_and_contract_identity(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar = tmp_path / "private-sidecar.json"
    sidecar.write_bytes(b"synthetic")
    checked = SimpleNamespace(
        state="complete",
        occurrences=(
            SimpleNamespace(
                state="accepted",
                attempts=(SimpleNamespace(outcome="accepted"),),
            ),
            SimpleNamespace(
                state="skipped",
                attempts=(SimpleNamespace(outcome="quarantined"),),
            ),
        ),
        annotations=(object(),),
    )
    monkeypatch.setattr("signlab.contracts.ingest.validate_collection_sidecar", lambda _: checked)
    monkeypatch.setattr(
        "signlab.contracts.ingest.collection_sidecar_digest",
        lambda _: "sha256:" + "a" * 64,
    )

    result = runner.invoke(cli.app, ["data", "validate-capture", str(sidecar)])

    assert result.exit_code == 0
    assert result.output.splitlines() == [
        "Collection sidecar SHA-256: sha256:" + "a" * 64,
        "Collection state: complete",
        "Capture outcomes: 1 accepted, 0 retry, 1 quarantined, 1 skipped.",
        "Annotation histories: 1",
    ]
    assert sidecar.name not in result.output
    assert str(tmp_path) not in result.output


def test_import_capture_delegates_without_exporting_source_paths(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar = tmp_path / "private-sidecar.json"
    source_map = tmp_path / "private-source-map.json"
    source_root = tmp_path / "private-source-root"
    output = tmp_path / "private-output"
    sidecar.write_bytes(b"synthetic-sidecar")
    source_map.write_text('{"source_00000000000000000000000000000001":"opaque.webm"}')
    calls: list[tuple[bytes, Path, dict[str, str], Path]] = []

    def import_capture(
        document: bytes,
        *,
        source_root: Path,
        source_map: dict[str, str],
        destination: Path,
    ) -> object:
        calls.append((document, source_root, source_map, destination))
        return SimpleNamespace(
            status="published",
            accepted_recordings=1,
            retry_attempts=0,
            quarantined_attempts=0,
            skipped_occurrences=0,
            manifest=SimpleNamespace(raw_data_sha256="sha256:" + "b" * 64),
        )

    monkeypatch.setattr(importer, "import_collection_sidecar", import_capture)

    result = runner.invoke(
        cli.app,
        [
            "data",
            "import-capture",
            str(sidecar),
            "--source-map",
            str(source_map),
            "--source-root",
            str(source_root),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        (
            b"synthetic-sidecar",
            source_root,
            {"source_00000000000000000000000000000001": "opaque.webm"},
            output,
        )
    ]
    assert result.output.splitlines() == [
        "Capture import: published.",
        "Imported outcomes: 1 accepted, 0 retry, 0 quarantined, 0 skipped.",
        "Raw data SHA-256: sha256:" + "b" * 64,
        "Raw bundle integrity: verified.",
    ]
    assert "private" not in result.output.casefold()
    assert str(tmp_path) not in result.output


def test_import_capture_redacts_invalid_source_map(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "participant-private-sidecar.json"
    source_map = tmp_path / "participant-private-map.json"
    source_root = tmp_path / "participant-private-source"
    output = tmp_path / "participant-private-output"
    sidecar.write_bytes(b"participant-private")
    source_map.write_text('{"source_key": ["participant-private-path"]}')

    result = runner.invoke(
        cli.app,
        [
            "data",
            "import-capture",
            str(sidecar),
            "--source-map",
            str(source_map),
            "--source-root",
            str(source_root),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 1
    assert result.output.strip() == "Capture import failed."
    assert "participant-private" not in result.output
    assert str(tmp_path) not in result.output
    assert "Traceback" not in result.output


def test_validate_raw_dataset_reports_every_boundary_without_paths(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "private-raw-manifest.json"
    root = tmp_path / "private-raw-root"
    manifest.write_bytes(b"synthetic-manifest")
    calls: list[tuple[bytes, Path]] = []

    def validate(document: bytes, workspace_root: Path) -> object:
        calls.append((document, workspace_root))
        return SimpleNamespace(
            validation=SimpleNamespace(
                raw_data_sha256="sha256:" + "c" * 64,
                parquet_table_bytes="verified",
                semantic_integrity="verified",
                artifact_byte_integrity="verified",
                collection_sidecar_integrity="verified",
                lineage_inventory_integrity="verified",
                quarantine_inventory_integrity="verified",
                consent_authorization="not_checked",
            )
        )

    monkeypatch.setattr(raw_bundle, "validate_raw_dataset_bundle", validate)

    result = runner.invoke(
        cli.app,
        [
            "data",
            "validate-raw-dataset",
            str(manifest),
            "--workspace-root",
            str(root),
        ],
    )

    assert result.exit_code == 0
    assert calls == [(b"synthetic-manifest", root)]
    assert result.output.splitlines() == [
        "Raw data SHA-256: sha256:" + "c" * 64,
        "Parquet table bytes: verified",
        "Raw dataset semantics: verified",
        "Referenced artifact bytes: verified",
        "Collection sidecar: verified",
        "Lineage inventory: verified",
        "Quarantine inventory: verified",
        "Current consent authorization: not checked",
    ]
    assert "private" not in result.output.casefold()
    assert str(tmp_path) not in result.output
