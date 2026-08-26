"""Synthetic-gated, atomic publication of deterministic quality reports."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from signlab.contracts.canonical import CanonicalizationError, canonical_json_bytes
from signlab.contracts.core import WorkspaceRelativeLocatorV1
from signlab.contracts.dataset import RecordingRowV1, RecordingsTableV1
from signlab.contracts.extraction import LandmarkExtractionManifestV1
from signlab.contracts.ingest import RawDatasetManifestV1, validate_raw_dataset_manifest
from signlab.contracts.quality import (
    DatasetQualityReportV1,
    LandmarkQualityManifestV1,
    LandmarkQualityPolicyV1,
    QualityContractError,
    SequenceQualityReportV1,
    assert_landmark_quality_bound_to_extraction,
    assert_sequence_quality_report_matches_table,
    landmark_quality_manifest_digest,
    landmark_quality_policy_digest,
    validate_landmark_quality_manifest,
    validate_landmark_quality_policy,
)
from signlab.datasets.parquet import DatasetParquetError, read_dataset_table
from signlab.datasets.raw_bundle import RawDatasetManifestInput
from signlab.extraction.batch import (
    ExtractionBatchError,
    ValidatedLandmarkExtractionBundle,
    validate_landmark_extraction_bundle,
)
from signlab.quality.policy import (
    QualityPolicyError,
    aggregate_quality_reports,
    assess_landmark_sequence,
)

LANDMARK_QUALITY_MANIFEST_FILENAME: Final = "landmark-quality-manifest.json"
LANDMARK_QUALITY_ID: Final = "landmark_quality_assessment"
LANDMARK_QUALITY_VERSION: Final = "1.0.0"
_MAX_MANIFEST_BYTES: Final = 64 * 1024 * 1024

type QualityBatchStatus = Literal["published", "unchanged"]
type ExtractionManifestInput = (
    LandmarkExtractionManifestV1 | str | bytes | bytearray | Mapping[str, object]
)
type QualityManifestInput = (
    LandmarkQualityManifestV1 | str | bytes | bytearray | Mapping[str, object]
)
type QualityPolicyInput = LandmarkQualityPolicyV1 | str | bytes | bytearray | Mapping[str, object]
type QualityBatchErrorCategory = Literal[
    "bundle.invalid",
    "consent.unauthorized",
    "destination.conflict",
    "destination.invalid",
    "execution.failed",
    "policy.invalid",
    "publication.failed",
    "source.invalid",
]

_ERROR_MESSAGES: Final[dict[QualityBatchErrorCategory, str]] = {
    "bundle.invalid": "landmark quality bundle is invalid",
    "consent.unauthorized": "landmark quality assessment is not authorized",
    "destination.conflict": "landmark quality destination conflicts with this run",
    "destination.invalid": "landmark quality destination is invalid",
    "execution.failed": "landmark quality assessment could not be completed",
    "policy.invalid": "landmark quality policy is invalid",
    "publication.failed": "landmark quality bundle could not be published",
    "source.invalid": "landmark quality source bundles are invalid",
}


class QualityBatchError(ValueError):
    """A stable, path-free quality publication or validation failure."""

    def __init__(self, category: QualityBatchErrorCategory) -> None:
        self.category = category
        self.code = f"quality.batch.{category}"
        super().__init__(_ERROR_MESSAGES[category])


@dataclass(frozen=True, slots=True)
class QualityBundleValidationResult:
    """Positive evidence returned only after full source and report reconciliation."""

    raw_bundle_integrity: Literal["verified"]
    consent_boundary: Literal["synthetic_fixture_only"]
    extraction_bundle_integrity: Literal["verified"]
    manifest_integrity: Literal["verified"]
    report_recomputation: Literal["verified"]
    exact_inventory: Literal["verified"]
    sequence_count: int
    pass_count: int
    warning_count: int
    quarantine_count: int
    reject_count: int


@dataclass(frozen=True, slots=True)
class ValidatedLandmarkQualityBundle:
    """One canonical report manifest reconciled to immutable source bundles."""

    manifest: LandmarkQualityManifestV1
    validation: QualityBundleValidationResult


@dataclass(frozen=True, slots=True)
class QualityBatchResult:
    """A newly published or already-identical landmark quality bundle."""

    status: QualityBatchStatus
    manifest: LandmarkQualityManifestV1
    manifest_path: Path
    validation: QualityBundleValidationResult


@dataclass(frozen=True, slots=True)
class _ValidatedSources:
    extraction: ValidatedLandmarkExtractionBundle
    raw_manifest: RawDatasetManifestV1
    recordings: tuple[RecordingRowV1, ...]


def _is_linklike(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _is_same_or_descendant(path: Path, ancestor: Path) -> bool:
    try:
        path.relative_to(ancestor)
    except ValueError:
        return False
    return True


def _resolved_source_root(root_input: str | Path) -> Path:
    try:
        candidate = Path(root_input)
        if _is_linklike(candidate):
            raise QualityBatchError("source.invalid")
        root = candidate.resolve(strict=True)
        if not root.is_dir():
            raise QualityBatchError("source.invalid")
        return root
    except QualityBatchError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise QualityBatchError("source.invalid") from None


def _resolved_destination(destination: str | Path) -> Path:
    try:
        candidate = Path(destination)
        if _is_linklike(candidate):
            raise QualityBatchError("destination.invalid")
        return candidate.resolve(strict=False)
    except QualityBatchError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise QualityBatchError("destination.invalid") from None


def _require_nonoverlapping(
    quality_root: Path,
    extraction_root: Path,
    raw_root: Path,
    *,
    category: Literal["bundle.invalid", "destination.invalid"],
) -> None:
    if any(
        _is_same_or_descendant(quality_root, source) or _is_same_or_descendant(source, quality_root)
        for source in (extraction_root, raw_root)
    ):
        raise QualityBatchError(category)


def _validated_policy(policy: QualityPolicyInput) -> LandmarkQualityPolicyV1:
    try:
        return validate_landmark_quality_policy(policy)
    except (QualityContractError, RuntimeError, TypeError, ValueError):
        raise QualityBatchError("policy.invalid") from None


def _recordings_from_verified_raw(
    raw_manifest: RawDatasetManifestInput,
    raw_root: Path,
) -> tuple[RawDatasetManifestV1, tuple[RecordingRowV1, ...]]:
    """Read only the hash-bound recordings table after full raw verification."""

    try:
        checked = validate_raw_dataset_manifest(raw_manifest)
        reference = checked.content.tables.recordings
        locator = reference.artifact.locator
        if not isinstance(locator, WorkspaceRelativeLocatorV1):
            raise QualityBatchError("source.invalid")
        candidate = raw_root.joinpath(*locator.path.split("/"))
        if _is_linklike(candidate):
            raise QualityBatchError("source.invalid")
        table = read_dataset_table(reference, raw_root)
        if not isinstance(table, RecordingsTableV1):
            raise QualityBatchError("source.invalid")
        recordings = table.rows
        if not recordings or tuple(row.recording_id for row in recordings) != tuple(
            sorted({row.recording_id for row in recordings})
        ):
            raise QualityBatchError("source.invalid")
        return checked, recordings
    except QualityBatchError:
        raise
    except (DatasetParquetError, OSError, RuntimeError, TypeError, ValueError):
        raise QualityBatchError("source.invalid") from None


def _validated_sources(
    extraction_manifest: ExtractionManifestInput,
    extraction_root: Path,
    raw_manifest: RawDatasetManifestInput,
    raw_root: Path,
) -> _ValidatedSources:
    try:
        checked_raw = validate_raw_dataset_manifest(raw_manifest)
        extraction = validate_landmark_extraction_bundle(
            extraction_manifest,
            extraction_root,
            raw_manifest=checked_raw,
            raw_bundle_root=raw_root,
        )
    except ExtractionBatchError as error:
        if error.category == "consent.unauthorized":
            raise QualityBatchError("consent.unauthorized") from None
        raise QualityBatchError("source.invalid") from None
    except (OSError, RuntimeError, TypeError, ValueError):
        raise QualityBatchError("source.invalid") from None
    checked_raw, recordings = _recordings_from_verified_raw(checked_raw, raw_root)
    expected_ids = tuple(
        sequence.lineage.source_recording_id for sequence in extraction.manifest.sequences
    )
    if expected_ids != tuple(recording.recording_id for recording in recordings):
        raise QualityBatchError("source.invalid")
    return _ValidatedSources(
        extraction=extraction,
        raw_manifest=checked_raw,
        recordings=recordings,
    )


def _assess_reports(
    sources: _ValidatedSources,
    policy: LandmarkQualityPolicyV1,
) -> tuple[tuple[SequenceQualityReportV1, ...], DatasetQualityReportV1]:
    reports: list[SequenceQualityReportV1] = []
    recordings = {recording.recording_id: recording for recording in sources.recordings}
    for sequence in sources.extraction.manifest.sequences:
        recording_id = sequence.lineage.source_recording_id
        table = sources.extraction.tables.get(recording_id)
        recording = recordings.get(recording_id)
        if table is None or recording is None:
            raise QualityPolicyError("quality source sequence is incomplete")
        report = assess_landmark_sequence(sequence, table, recording, policy)
        assert_sequence_quality_report_matches_table(report, table)
        reports.append(report)
    checked_reports = tuple(reports)
    return checked_reports, aggregate_quality_reports(checked_reports, policy)


def _quality_manifest(
    extraction: LandmarkExtractionManifestV1,
    policy: LandmarkQualityPolicyV1,
    reports: tuple[SequenceQualityReportV1, ...],
    dataset_report: DatasetQualityReportV1,
) -> LandmarkQualityManifestV1:
    payload: dict[str, object] = {
        "schema_version": "landmark-quality-manifest/1",
        "quality_id": LANDMARK_QUALITY_ID,
        "version": LANDMARK_QUALITY_VERSION,
        "raw_dataset_id": extraction.raw_dataset_id,
        "raw_dataset_version": extraction.raw_dataset_version,
        "raw_data_sha256": extraction.raw_data_sha256,
        "raw_dataset_manifest_sha256": extraction.raw_dataset_manifest_sha256,
        "extraction_id": extraction.extraction_id,
        "extraction_version": extraction.version,
        "extraction_manifest_sha256": extraction.manifest_sha256,
        "extraction_config_sha256": extraction.config_sha256,
        "policy": policy.model_dump(mode="json", round_trip=True),
        "policy_sha256": landmark_quality_policy_digest(policy),
        "sequence_reports": [report.model_dump(mode="json", round_trip=True) for report in reports],
        "dataset_report": dataset_report.model_dump(mode="json", round_trip=True),
    }
    payload["manifest_sha256"] = landmark_quality_manifest_digest(payload)
    return validate_landmark_quality_manifest(payload)


def _resolved_quality_root(root_input: str | Path) -> Path:
    try:
        candidate = Path(root_input)
        if _is_linklike(candidate):
            raise QualityBatchError("bundle.invalid")
        root = candidate.resolve(strict=True)
        if not root.is_dir():
            raise QualityBatchError("bundle.invalid")
        return root
    except QualityBatchError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise QualityBatchError("bundle.invalid") from None


def _manifest_bytes(root: Path) -> bytes:
    path = root / LANDMARK_QUALITY_MANIFEST_FILENAME
    try:
        if _is_linklike(path):
            raise QualityBatchError("bundle.invalid")
        resolved = path.resolve(strict=True)
        if (
            not resolved.is_relative_to(root)
            or not resolved.is_file()
            or resolved.stat().st_size > _MAX_MANIFEST_BYTES
        ):
            raise QualityBatchError("bundle.invalid")
        return resolved.read_bytes()
    except QualityBatchError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise QualityBatchError("bundle.invalid") from None


def _verify_exact_inventory(root: Path) -> None:
    try:
        entries = tuple(root.iterdir())
        if len(entries) != 1:
            raise QualityBatchError("bundle.invalid")
        entry = entries[0]
        if (
            entry.name != LANDMARK_QUALITY_MANIFEST_FILENAME
            or _is_linklike(entry)
            or not entry.is_file()
        ):
            raise QualityBatchError("bundle.invalid")
    except QualityBatchError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise QualityBatchError("bundle.invalid") from None


def _validation_result(manifest: LandmarkQualityManifestV1) -> QualityBundleValidationResult:
    report = manifest.dataset_report
    return QualityBundleValidationResult(
        raw_bundle_integrity="verified",
        consent_boundary="synthetic_fixture_only",
        extraction_bundle_integrity="verified",
        manifest_integrity="verified",
        report_recomputation="verified",
        exact_inventory="verified",
        sequence_count=report.sequence_count,
        pass_count=report.pass_count,
        warning_count=report.warning_count,
        quarantine_count=report.quarantine_count,
        reject_count=report.reject_count,
    )


def _validate_quality_bundle_against_sources(
    manifest: QualityManifestInput,
    root: Path,
    sources: _ValidatedSources,
) -> ValidatedLandmarkQualityBundle:
    try:
        checked = validate_landmark_quality_manifest(manifest)
        disk_bytes = _manifest_bytes(root)
        disk_manifest = validate_landmark_quality_manifest(disk_bytes)
        if disk_manifest != checked or disk_bytes != canonical_json_bytes(disk_manifest) + b"\n":
            raise QualityBatchError("bundle.invalid")
        _verify_exact_inventory(root)
        assert_landmark_quality_bound_to_extraction(checked, sources.extraction.manifest)
        reports, dataset_report = _assess_reports(sources, checked.policy)
        if reports != checked.sequence_reports or dataset_report != checked.dataset_report:
            raise QualityBatchError("bundle.invalid")
        return ValidatedLandmarkQualityBundle(
            manifest=checked,
            validation=_validation_result(checked),
        )
    except QualityBatchError:
        raise
    except (
        CanonicalizationError,
        QualityContractError,
        QualityPolicyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        raise QualityBatchError("bundle.invalid") from None


def validate_landmark_quality_bundle(
    manifest: QualityManifestInput,
    workspace_root: str | Path,
    *,
    extraction_manifest: ExtractionManifestInput,
    extraction_root: str | Path,
    raw_manifest: RawDatasetManifestInput,
    raw_bundle_root: str | Path,
) -> ValidatedLandmarkQualityBundle:
    """Verify exact sources, canonical inventory, and every recomputed report."""

    checked_extraction_root = _resolved_source_root(extraction_root)
    checked_raw_root = _resolved_source_root(raw_bundle_root)
    sources = _validated_sources(
        extraction_manifest,
        checked_extraction_root,
        raw_manifest,
        checked_raw_root,
    )
    quality_root = _resolved_quality_root(workspace_root)
    _require_nonoverlapping(
        quality_root,
        checked_extraction_root,
        checked_raw_root,
        category="bundle.invalid",
    )
    return _validate_quality_bundle_against_sources(manifest, quality_root, sources)


def _destination_state(destination: Path) -> Literal["absent", "empty", "occupied"]:
    try:
        if _is_linklike(destination) or (destination.exists() and not destination.is_dir()):
            raise QualityBatchError("destination.invalid")
        if not destination.exists():
            return "absent"
        return "occupied" if any(destination.iterdir()) else "empty"
    except QualityBatchError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise QualityBatchError("destination.invalid") from None


def _existing_result(
    destination: Path,
    policy: LandmarkQualityPolicyV1,
    sources: _ValidatedSources,
) -> QualityBatchResult:
    try:
        validated = _validate_quality_bundle_against_sources(
            _manifest_bytes(_resolved_quality_root(destination)),
            destination,
            sources,
        )
    except QualityBatchError:
        raise QualityBatchError("destination.conflict") from None
    if (
        validated.manifest.policy != policy
        or validated.manifest.quality_id != LANDMARK_QUALITY_ID
        or validated.manifest.version != LANDMARK_QUALITY_VERSION
    ):
        raise QualityBatchError("destination.conflict")
    return QualityBatchResult(
        status="unchanged",
        manifest=validated.manifest,
        manifest_path=destination / LANDMARK_QUALITY_MANIFEST_FILENAME,
        validation=validated.validation,
    )


def _write_manifest_durably(path: Path, manifest: LandmarkQualityManifestV1) -> None:
    try:
        captured = canonical_json_bytes(manifest) + b"\n"
        with path.open("xb") as stream:
            stream.write(captured)
            stream.flush()
            os.fsync(stream.fileno())
    except (CanonicalizationError, OSError, RuntimeError, TypeError, ValueError):
        raise QualityBatchError("publication.failed") from None


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_or_reconcile(
    staging: Path,
    destination: Path,
    manifest: LandmarkQualityManifestV1,
    policy: LandmarkQualityPolicyV1,
    sources: _ValidatedSources,
) -> QualityBatchResult:
    state = _destination_state(destination)
    if state == "occupied":
        existing = _existing_result(destination, policy, sources)
        if existing.manifest != manifest:
            raise QualityBatchError("destination.conflict")
        return existing
    removed_empty = False
    try:
        if state == "empty":
            destination.rmdir()
            removed_empty = True
        staging.rename(destination)
        with suppress(OSError):
            _fsync_directory(destination.parent)
        validated = _validate_quality_bundle_against_sources(manifest, destination, sources)
        return QualityBatchResult(
            status="published",
            manifest=validated.manifest,
            manifest_path=destination / LANDMARK_QUALITY_MANIFEST_FILENAME,
            validation=validated.validation,
        )
    except QualityBatchError:
        if removed_empty and not destination.exists():
            with suppress(OSError):
                destination.mkdir()
        if destination.exists():
            existing = _existing_result(destination, policy, sources)
            if existing.manifest == manifest:
                return existing
        raise
    except (OSError, RuntimeError, ValueError):
        if removed_empty and not destination.exists():
            with suppress(OSError):
                destination.mkdir()
        if destination.exists():
            try:
                existing = _existing_result(destination, policy, sources)
                if existing.manifest == manifest:
                    return existing
            except QualityBatchError:
                pass
        raise QualityBatchError("publication.failed") from None


def assess_landmark_quality(
    extraction_manifest: ExtractionManifestInput,
    *,
    extraction_root: str | Path,
    raw_manifest: RawDatasetManifestInput,
    raw_bundle_root: str | Path,
    policy: QualityPolicyInput,
    destination: str | Path,
) -> QualityBatchResult:
    """Assess and atomically publish one synthetic extraction report bundle."""

    checked_extraction_root = _resolved_source_root(extraction_root)
    checked_raw_root = _resolved_source_root(raw_bundle_root)
    sources = _validated_sources(
        extraction_manifest,
        checked_extraction_root,
        raw_manifest,
        checked_raw_root,
    )
    checked_policy = _validated_policy(policy)
    output = _resolved_destination(destination)
    _require_nonoverlapping(
        output,
        checked_extraction_root,
        checked_raw_root,
        category="destination.invalid",
    )
    if _destination_state(output) == "occupied":
        return _existing_result(output, checked_policy, sources)
    try:
        reports, dataset_report = _assess_reports(sources, checked_policy)
        manifest = _quality_manifest(
            sources.extraction.manifest,
            checked_policy,
            reports,
            dataset_report,
        )
    except (QualityContractError, QualityPolicyError, RuntimeError, TypeError, ValueError):
        raise QualityBatchError("execution.failed") from None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=output.parent,
            prefix=f".{output.name}.staging-",
        ) as temporary_directory:
            staging = Path(temporary_directory)
            _write_manifest_durably(
                staging / LANDMARK_QUALITY_MANIFEST_FILENAME,
                manifest,
            )
            _fsync_directory(staging)
            _validate_quality_bundle_against_sources(manifest, staging, sources)
            return _publish_or_reconcile(
                staging,
                output,
                manifest,
                checked_policy,
                sources,
            )
    except QualityBatchError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise QualityBatchError("publication.failed") from None


__all__ = [
    "LANDMARK_QUALITY_ID",
    "LANDMARK_QUALITY_MANIFEST_FILENAME",
    "LANDMARK_QUALITY_VERSION",
    "QualityBatchError",
    "QualityBatchErrorCategory",
    "QualityBatchResult",
    "QualityBatchStatus",
    "QualityBundleValidationResult",
    "ValidatedLandmarkQualityBundle",
    "assess_landmark_quality",
    "validate_landmark_quality_bundle",
]
