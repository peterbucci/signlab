"""Application service for validating a complete table-backed dataset bundle."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from signlab.contracts.core import ArtifactRefV1, WorkspaceRelativeLocatorV1
from signlab.contracts.dataset import (
    DATASET_TABLE_SCHEMA_VERSIONS,
    DatasetContentV2,
    DatasetManifestV2,
    DatasetTable,
    DatasetTableRefV1,
    DatasetTableSetV1,
    TableName,
    dataset_content_digest,
)
from signlab.contracts.governance import (
    ConsentAuthorizationVerifier,
    ScopePermission,
)
from signlab.contracts.pipeline import SplitManifestV1, validate_dataset_manifest_v2
from signlab.datasets.parquet import read_dataset_table, write_dataset_table
from signlab.datasets.validation import (
    ConsentEvidenceLookup,
    DatasetValidationError,
    validate_dataset_manifest_tables,
)

type DatasetManifestV2Input = DatasetManifestV2 | str | bytes | bytearray | Mapping[str, object]
type DatasetBundleErrorCategory = Literal[
    "manifest.invalid",
    "table_bytes.invalid",
    "semantics.invalid",
]

_ERROR_MESSAGES: dict[DatasetBundleErrorCategory, str] = {
    "manifest.invalid": "dataset bundle manifest is invalid",
    "table_bytes.invalid": "dataset bundle table bytes could not be verified",
    "semantics.invalid": "dataset bundle semantic relationships are invalid",
}


@dataclass(frozen=True, slots=True)
class DatasetBundleValidationResult:
    """Positive bundle checks while keeping row-artifact verification distinct."""

    data_sha256: str
    parquet_table_bytes: Literal["verified"]
    semantic_integrity: Literal["verified"]
    artifact_byte_integrity: Literal["verified", "not_checked"]
    split_compatibility: Literal["verified", "not_checked"]
    consent_authorization: Literal["verified", "not_checked"]


@dataclass(frozen=True, slots=True)
class WrittenDatasetBundle:
    """A fully verified manifest written last after its six Parquet tables."""

    manifest: DatasetManifestV2
    manifest_path: Path
    validation: DatasetBundleValidationResult


class DatasetBundleError(ValueError):
    """A stable, sanitized failure at the dataset-bundle boundary."""

    def __init__(self, category: DatasetBundleErrorCategory) -> None:
        self.category = category
        self.code = f"dataset.bundle.{category}"
        super().__init__(_ERROR_MESSAGES[category])


def _require_publishable_destination(destination: Path) -> None:
    """Reject destinations that could hide or overwrite existing content."""

    try:
        if destination.is_symlink():
            raise DatasetBundleError("table_bytes.invalid")
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise DatasetBundleError("table_bytes.invalid")
    except DatasetBundleError:
        raise
    except OSError as error:
        raise DatasetBundleError("table_bytes.invalid") from error


def _write_manifest_durably(path: Path, manifest: DatasetManifestV2) -> None:
    """Write the completion marker once and flush it before publication."""

    payload = (
        json.dumps(
            manifest.model_dump(mode="json", round_trip=True),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_file(path: Path) -> None:
    """Flush a closed staged file before its completion marker can be published."""

    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    """Persist staged directory entries where directory fsync is supported."""

    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_staged_directory(staging: Path, destination: Path) -> None:
    """Atomically expose a complete sibling directory at ``destination``."""

    removed_empty_destination = False
    try:
        _require_publishable_destination(destination)
        if destination.exists():
            destination.rmdir()
            removed_empty_destination = True
        staging.rename(destination)
        with suppress(OSError):
            _fsync_directory(destination.parent)
    except DatasetBundleError:
        raise
    except OSError as error:
        if removed_empty_destination and not destination.exists():
            with suppress(OSError):
                destination.mkdir()
        raise DatasetBundleError("table_bytes.invalid") from error


def _load_tables(
    manifest: DatasetManifestV2,
    workspace_root: str | Path,
) -> dict[TableName, DatasetTable]:
    tables: dict[TableName, DatasetTable] = {}
    try:
        for table_name in DATASET_TABLE_SCHEMA_VERSIONS:
            reference = getattr(manifest.content.tables, table_name)
            tables[table_name] = read_dataset_table(reference, workspace_root)
    except (OSError, TypeError, ValueError) as error:
        raise DatasetBundleError("table_bytes.invalid") from error
    return tables


def validate_dataset_bundle(
    manifest: DatasetManifestV2Input,
    workspace_root: str | Path,
    *,
    split: SplitManifestV1 | None = None,
    consent_evidence_lookup: ConsentEvidenceLookup | None = None,
    consent_authorization_verifier: ConsentAuthorizationVerifier | None = None,
    authorization_permission: ScopePermission | None = None,
    authorization_at: str | None = None,
) -> DatasetBundleValidationResult:
    """Verify six exact Parquet files and their semantic dataset relationships.

    Current consent authorization remains explicitly unchecked unless every
    authenticated dependency is supplied. Media referenced by table rows is not
    silently treated as verified; this boundary verifies the manifest-bound table
    artifacts themselves.
    """

    try:
        checked_manifest = validate_dataset_manifest_v2(manifest)
    except (TypeError, ValueError) as error:
        raise DatasetBundleError("manifest.invalid") from error
    tables = _load_tables(checked_manifest, workspace_root)
    try:
        result = validate_dataset_manifest_tables(
            checked_manifest,
            cast(Mapping[str, DatasetTable], tables),
            split=split,
            consent_evidence_lookup=consent_evidence_lookup,
            consent_authorization_verifier=consent_authorization_verifier,
            authorization_permission=authorization_permission,
            authorization_at=authorization_at,
        )
    except DatasetValidationError as error:
        raise DatasetBundleError("semantics.invalid") from error
    return DatasetBundleValidationResult(
        data_sha256=checked_manifest.data_sha256,
        parquet_table_bytes="verified",
        semantic_integrity=result.semantic_integrity,
        artifact_byte_integrity=result.artifact_byte_integrity,
        split_compatibility=result.split_compatibility,
        consent_authorization=result.consent_authorization,
    )


def write_dataset_bundle(
    manifest: DatasetManifestV2Input,
    tables: Mapping[TableName, DatasetTable],
    destination: str | Path,
) -> WrittenDatasetBundle:
    """Materialize a validated logical bundle into a new or empty directory.

    The supplied manifest is a semantic template. Its table byte references are
    replaced with hashes, sizes, and workspace-relative locators for the bytes
    written here; `data_sha256` must remain unchanged.
    """

    try:
        template = validate_dataset_manifest_v2(manifest)
    except (TypeError, ValueError) as error:
        raise DatasetBundleError("manifest.invalid") from error
    try:
        validate_dataset_manifest_tables(
            template,
            cast(Mapping[str, DatasetTable], tables),
        )
    except (TypeError, ValueError) as error:
        raise DatasetBundleError("semantics.invalid") from error

    root = Path(destination)
    _require_publishable_destination(root)
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise DatasetBundleError("table_bytes.invalid") from error

    try:
        with tempfile.TemporaryDirectory(
            dir=root.parent,
            prefix=f".{root.name}.staging-",
        ) as temporary_directory:
            staging = Path(temporary_directory)
            (staging / "tables").mkdir()

            references: dict[TableName, DatasetTableRefV1] = {}
            for table_name in DATASET_TABLE_SCHEMA_VERSIONS:
                table = tables[table_name]
                relative_path = f"tables/{table_name}.parquet"
                staged_table_path = staging.joinpath(*relative_path.split("/"))
                written = write_dataset_table(
                    table,
                    staged_table_path,
                )
                _fsync_file(staged_table_path)
                template_reference = getattr(template.content.tables, table_name)
                references[table_name] = DatasetTableRefV1(
                    schema_version="dataset-table-reference/1",
                    table_name=table_name,
                    table_schema_version=written.table_schema_version,
                    row_count=written.row_count,
                    content_sha256=written.content_sha256,
                    artifact=ArtifactRefV1(
                        schema_version="artifact-reference/1",
                        artifact_id=template_reference.artifact.artifact_id,
                        role="dataset_table",
                        media_type="application/vnd.apache.parquet",
                        sha256=written.sha256,
                        size_bytes=written.size_bytes,
                        locator=WorkspaceRelativeLocatorV1(
                            kind="workspace_relative",
                            path=relative_path,
                        ),
                    ),
                )

            table_set = DatasetTableSetV1(
                schema_version="dataset-table-set/1",
                **references,
            )
            content = DatasetContentV2(
                schema_version="dataset-content/2",
                taxonomy=template.content.taxonomy,
                governance_policy=template.content.governance_policy,
                lineage_inventory_sha256=template.content.lineage_inventory_sha256,
                sample_schema_version=template.content.sample_schema_version,
                tables=table_set,
                samples=template.content.samples,
            )
            checked_manifest = DatasetManifestV2(
                schema_version="dataset-manifest/2",
                dataset_id=template.dataset_id,
                version=template.version,
                content=content,
                data_sha256=dataset_content_digest(content),
            )
            if checked_manifest.data_sha256 != template.data_sha256:
                raise DatasetBundleError("semantics.invalid")

            staged_manifest_path = staging / "dataset-manifest.json"
            _write_manifest_durably(staged_manifest_path, checked_manifest)
            validation = validate_dataset_bundle(staged_manifest_path.read_bytes(), staging)
            _fsync_directory(staging / "tables")
            _fsync_directory(staging)
            _publish_staged_directory(staging, root)
    except DatasetBundleError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise DatasetBundleError("table_bytes.invalid") from error

    manifest_path = root / "dataset-manifest.json"
    return WrittenDatasetBundle(
        manifest=checked_manifest,
        manifest_path=manifest_path,
        validation=validation,
    )


__all__ = [
    "DatasetBundleError",
    "DatasetBundleErrorCategory",
    "DatasetBundleValidationResult",
    "DatasetManifestV2Input",
    "WrittenDatasetBundle",
    "validate_dataset_bundle",
    "write_dataset_bundle",
]
