from __future__ import annotations

from pathlib import Path

import pytest

from signlab.contracts.dataset import DatasetManifestV2, DatasetTable
from signlab.datasets import bundle
from signlab.datasets.parquet import ParquetTableResult, write_dataset_table
from signlab.datasets.resources import DatasetResourceBundle, build_example_dataset_bundle
from signlab.datasets.storage import DatasetStorageError


@pytest.fixture
def example_bundle() -> DatasetResourceBundle:
    return build_example_dataset_bundle()


def _assert_no_staging_directories(parent: Path, destination_name: str) -> None:
    assert not tuple(parent.glob(f".{destination_name}.staging-*"))


def test_complete_bundle_atomically_replaces_an_empty_destination(
    tmp_path: Path,
    example_bundle: DatasetResourceBundle,
) -> None:
    destination = tmp_path / "dataset"
    destination.mkdir()

    written = bundle.write_dataset_bundle(
        example_bundle.manifest,
        example_bundle.tables,
        destination,
    )

    assert written.manifest_path == destination / "dataset-manifest.json"
    assert written.manifest_path.is_file()
    assert written.validation.parquet_table_bytes == "verified"
    assert written.validation.semantic_integrity == "verified"
    assert len(tuple((destination / "tables").glob("*.parquet"))) == 6
    _assert_no_staging_directories(tmp_path, destination.name)


def test_writer_reports_a_malformed_template_as_manifest_invalid(
    tmp_path: Path,
    example_bundle: DatasetResourceBundle,
) -> None:
    with pytest.raises(bundle.DatasetBundleError) as caught:
        bundle.write_dataset_bundle(
            b"{}",
            example_bundle.tables,
            tmp_path / "dataset",
        )

    assert caught.value.category == "manifest.invalid"


def test_table_write_failure_never_publishes_partial_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    example_bundle: DatasetResourceBundle,
) -> None:
    destination = tmp_path / "dataset"
    write_count = 0

    def fail_during_second_table(
        document: DatasetTable,
        output: str | Path,
    ) -> ParquetTableResult:
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise OSError("seeded table failure")
        return write_dataset_table(document, output)

    monkeypatch.setattr(bundle, "write_dataset_table", fail_during_second_table)

    with pytest.raises(bundle.DatasetBundleError) as caught:
        bundle.write_dataset_bundle(
            example_bundle.manifest,
            example_bundle.tables,
            destination,
        )

    assert caught.value.category == "table_bytes.invalid"
    assert not destination.exists()
    _assert_no_staging_directories(tmp_path, destination.name)


def test_manifest_write_failure_never_publishes_completed_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    example_bundle: DatasetResourceBundle,
) -> None:
    destination = tmp_path / "dataset"

    def fail_manifest_write(path: Path, _manifest: object) -> None:
        assert len(tuple((path.parent / "tables").glob("*.parquet"))) == 6
        assert not destination.exists()
        raise OSError("seeded manifest failure")

    monkeypatch.setattr(bundle, "_write_manifest_durably", fail_manifest_write)

    with pytest.raises(bundle.DatasetBundleError) as caught:
        bundle.write_dataset_bundle(
            example_bundle.manifest,
            example_bundle.tables,
            destination,
        )

    assert caught.value.category == "table_bytes.invalid"
    assert not destination.exists()
    _assert_no_staging_directories(tmp_path, destination.name)


def test_each_parquet_file_is_flushed_before_the_manifest_is_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    example_bundle: DatasetResourceBundle,
) -> None:
    destination = tmp_path / "dataset"
    flushed: list[Path] = []
    real_manifest_write = bundle._write_manifest_durably

    monkeypatch.setattr(bundle, "_fsync_file", flushed.append)

    def assert_tables_flushed(path: Path, manifest: DatasetManifestV2) -> None:
        assert {item.name for item in flushed} == {
            "annotations.parquet",
            "clips.parquet",
            "derived_artifacts.parquet",
            "participants.parquet",
            "recordings.parquet",
            "sessions.parquet",
        }
        real_manifest_write(path, manifest)

    monkeypatch.setattr(bundle, "_write_manifest_durably", assert_tables_flushed)

    bundle.write_dataset_bundle(
        example_bundle.manifest,
        example_bundle.tables,
        destination,
    )

    assert (destination / "dataset-manifest.json").is_file()


def test_post_write_validation_failure_leaves_existing_destination_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    example_bundle: DatasetResourceBundle,
) -> None:
    destination = tmp_path / "dataset"
    destination.mkdir()

    def fail_validation(_manifest: object, workspace_root: Path) -> object:
        assert (workspace_root / "dataset-manifest.json").is_file()
        assert len(tuple((workspace_root / "tables").glob("*.parquet"))) == 6
        assert not tuple(destination.iterdir())
        raise bundle.DatasetBundleError("semantics.invalid")

    monkeypatch.setattr(bundle, "validate_dataset_bundle", fail_validation)

    with pytest.raises(bundle.DatasetBundleError) as caught:
        bundle.write_dataset_bundle(
            example_bundle.manifest,
            example_bundle.tables,
            destination,
        )

    assert caught.value.category == "semantics.invalid"
    assert destination.is_dir()
    assert not tuple(destination.iterdir())
    _assert_no_staging_directories(tmp_path, destination.name)


def test_row_artifact_verification_is_explicit_and_propagates_positive_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    example_bundle: DatasetResourceBundle,
) -> None:
    destination = tmp_path / "dataset"
    written = bundle.write_dataset_bundle(
        example_bundle.manifest,
        example_bundle.tables,
        destination,
    )
    calls = 0

    def verify(_tables: object, workspace_root: str | Path) -> object:
        nonlocal calls
        calls += 1
        assert Path(workspace_root) == destination
        return object()

    monkeypatch.setattr(bundle, "verify_dataset_row_artifacts", verify)

    unchecked = bundle.validate_dataset_bundle(written.manifest, destination)
    checked = bundle.validate_dataset_bundle(
        written.manifest,
        destination,
        verify_row_artifacts=True,
    )

    assert unchecked.artifact_byte_integrity == "not_checked"
    assert checked.artifact_byte_integrity == "verified"
    assert calls == 1


def test_row_artifact_verification_failure_has_a_distinct_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    example_bundle: DatasetResourceBundle,
) -> None:
    destination = tmp_path / "dataset"
    written = bundle.write_dataset_bundle(
        example_bundle.manifest,
        example_bundle.tables,
        destination,
    )
    monkeypatch.setattr(
        bundle,
        "verify_dataset_row_artifacts",
        lambda *_args: (_ for _ in ()).throw(DatasetStorageError()),
    )

    with pytest.raises(bundle.DatasetBundleError) as raised:
        bundle.validate_dataset_bundle(
            written.manifest,
            destination,
            verify_row_artifacts=True,
        )

    assert raised.value.category == "row_artifact_bytes.invalid"
