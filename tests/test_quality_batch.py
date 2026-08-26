from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, cast

import pytest

import signlab.extraction.batch as extraction_batch
import signlab.quality.batch as batch
from signlab.contracts.canonical import CanonicalizationError, canonical_json_bytes
from signlab.contracts.extraction import (
    LandmarkExtractionManifestV1,
    raw_dataset_manifest_digest,
)
from signlab.contracts.ingest import RawDatasetManifestV1
from signlab.contracts.quality import (
    LandmarkQualityManifestV1,
    LandmarkQualityPolicyV1,
    landmark_quality_manifest_digest,
    sequence_quality_report_digest,
    validate_landmark_quality_manifest,
    validate_landmark_quality_policy,
)
from signlab.datasets.raw_bundle import (
    RawDatasetManifestInput,
    ValidatedRawDatasetBundle,
    validate_raw_dataset_bundle,
)
from signlab.quality.batch import (
    QualityBatchError,
    QualityBatchResult,
    assess_landmark_quality,
    validate_landmark_quality_bundle,
)
from signlab.quality.policy import QualityPolicyError
from signlab.quality.resources import build_default_quality_policy
from test_extraction_batch import _extract, _tree_snapshot


def _assess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    destination_name: str = "quality",
) -> tuple[
    QualityBatchResult,
    Path,
    RawDatasetManifestV1,
    Path,
    LandmarkExtractionManifestV1,
]:
    extraction, raw_root, raw_manifest, _runtimes, _trackers = _extract(
        monkeypatch,
        tmp_path,
    )
    extraction_root = tmp_path / "landmarks"
    result = assess_landmark_quality(
        extraction.manifest,
        extraction_root=extraction_root,
        raw_manifest=raw_manifest,
        raw_bundle_root=raw_root,
        policy=build_default_quality_policy(),
        destination=tmp_path / destination_name,
    )
    return result, raw_root, raw_manifest, extraction_root, extraction.manifest


def _validate(
    result: QualityBatchResult,
    *,
    quality_root: Path,
    extraction_manifest: LandmarkExtractionManifestV1,
    extraction_root: Path,
    raw_manifest: RawDatasetManifestV1,
    raw_root: Path,
) -> LandmarkQualityManifestV1:
    validated = validate_landmark_quality_bundle(
        result.manifest,
        quality_root,
        extraction_manifest=extraction_manifest,
        extraction_root=extraction_root,
        raw_manifest=raw_manifest,
        raw_bundle_root=raw_root,
    )
    assert validated.validation.report_recomputation == "verified"
    return validated.manifest


def _different_valid_policy() -> LandmarkQualityPolicyV1:
    payload = build_default_quality_policy().model_dump(mode="json", round_trip=True)
    rules = cast(list[dict[str, Any]], payload["threshold_rules"])
    rules[0]["warning"] = 949_999
    return validate_landmark_quality_policy(payload)


def _rewrite_manifest(root: Path, payload: dict[str, Any]) -> LandmarkQualityManifestV1:
    payload["manifest_sha256"] = landmark_quality_manifest_digest(payload)
    checked = validate_landmark_quality_manifest(payload)
    (root / batch.LANDMARK_QUALITY_MANIFEST_FILENAME).write_bytes(
        canonical_json_bytes(checked) + b"\n"
    )
    return checked


def test_assessment_publishes_one_canonical_report_and_preserves_source_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    extraction, raw_root, raw_manifest, _runtimes, _trackers = _extract(
        monkeypatch,
        tmp_path,
    )
    extraction_root = tmp_path / "landmarks"
    raw_before = _tree_snapshot(raw_root)
    extraction_before = _tree_snapshot(extraction_root)
    destination = tmp_path / "quality"

    result = assess_landmark_quality(
        extraction.manifest,
        extraction_root=extraction_root,
        raw_manifest=raw_manifest,
        raw_bundle_root=raw_root,
        policy=build_default_quality_policy(),
        destination=destination,
    )

    assert result.status == "published"
    assert tuple(path.name for path in destination.iterdir()) == (
        batch.LANDMARK_QUALITY_MANIFEST_FILENAME,
    )
    assert result.manifest_path.read_bytes() == canonical_json_bytes(result.manifest) + b"\n"
    assert result.manifest.extraction_manifest_sha256 == extraction.manifest.manifest_sha256
    assert result.manifest.raw_dataset_manifest_sha256 == raw_dataset_manifest_digest(raw_manifest)
    assert result.validation.sequence_count == len(extraction.manifest.sequences)
    assert (
        result.validation.pass_count
        + result.validation.warning_count
        + result.validation.quarantine_count
        + result.validation.reject_count
        == result.validation.sequence_count
    )
    assert _tree_snapshot(raw_root) == raw_before
    assert _tree_snapshot(extraction_root) == extraction_before
    assert (
        _validate(
            result,
            quality_root=destination,
            extraction_manifest=extraction.manifest,
            extraction_root=extraction_root,
            raw_manifest=raw_manifest,
            raw_root=raw_root,
        )
        == result.manifest
    )


def test_assessment_performs_one_public_private_media_validation_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    extraction, raw_root, raw_manifest, _runtimes, _trackers = _extract(
        monkeypatch,
        tmp_path,
    )
    calls = 0
    original = validate_raw_dataset_bundle

    def counted(
        manifest: RawDatasetManifestInput,
        workspace_root: str | Path,
    ) -> ValidatedRawDatasetBundle:
        nonlocal calls
        calls += 1
        return original(manifest, workspace_root)

    monkeypatch.setattr(extraction_batch, "validate_raw_dataset_bundle", counted)
    assess_landmark_quality(
        extraction.manifest,
        extraction_root=tmp_path / "landmarks",
        raw_manifest=raw_manifest,
        raw_bundle_root=raw_root,
        policy=build_default_quality_policy(),
        destination=tmp_path / "quality",
    )

    assert calls == 1


def test_identical_replay_is_unchanged_and_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first, raw_root, raw_manifest, extraction_root, extraction_manifest = _assess(
        monkeypatch,
        tmp_path,
    )
    before = _tree_snapshot(tmp_path / "quality")

    second = assess_landmark_quality(
        extraction_manifest,
        extraction_root=extraction_root,
        raw_manifest=raw_manifest,
        raw_bundle_root=raw_root,
        policy=build_default_quality_policy(),
        destination=tmp_path / "quality",
    )

    assert second.status == "unchanged"
    assert second.manifest == first.manifest
    assert _tree_snapshot(tmp_path / "quality") == before


def test_occupied_destination_fails_closed_for_policy_conflict_or_extra_inventory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _result, raw_root, raw_manifest, extraction_root, extraction_manifest = _assess(
        monkeypatch,
        tmp_path,
    )

    with pytest.raises(QualityBatchError) as policy_error:
        assess_landmark_quality(
            extraction_manifest,
            extraction_root=extraction_root,
            raw_manifest=raw_manifest,
            raw_bundle_root=raw_root,
            policy=_different_valid_policy(),
            destination=tmp_path / "quality",
        )
    assert policy_error.value.category == "destination.conflict"

    (tmp_path / "quality" / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(QualityBatchError) as inventory_error:
        assess_landmark_quality(
            extraction_manifest,
            extraction_root=extraction_root,
            raw_manifest=raw_manifest,
            raw_bundle_root=raw_root,
            policy=build_default_quality_policy(),
            destination=tmp_path / "quality",
        )
    assert inventory_error.value.category == "destination.conflict"


def test_validation_recomputes_reports_instead_of_trusting_self_digests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, raw_root, raw_manifest, extraction_root, extraction_manifest = _assess(
        monkeypatch,
        tmp_path,
    )
    payload = result.manifest.model_dump(mode="json", round_trip=True)
    reports = cast(list[dict[str, Any]], payload["sequence_reports"])
    gaps = cast(list[dict[str, Any]], reports[0]["gaps"])
    assert gaps
    gaps[0]["gap_id"] = "gap_tampered"
    reports[0]["report_sha256"] = sequence_quality_report_digest(reports[0])
    tampered = _rewrite_manifest(tmp_path / "quality", payload)

    with pytest.raises(QualityBatchError) as captured:
        validate_landmark_quality_bundle(
            tampered,
            tmp_path / "quality",
            extraction_manifest=extraction_manifest,
            extraction_root=extraction_root,
            raw_manifest=raw_manifest,
            raw_bundle_root=raw_root,
        )

    assert captured.value.category == "bundle.invalid"


def test_validation_rejects_source_identity_substitution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, raw_root, raw_manifest, extraction_root, extraction_manifest = _assess(
        monkeypatch,
        tmp_path,
    )
    payload = result.manifest.model_dump(mode="json", round_trip=True)
    reports = cast(list[dict[str, Any]], payload["sequence_reports"])
    reports[0]["source_landmark_parquet_sha256"] = "sha256:" + "0" * 64
    reports[0]["report_sha256"] = sequence_quality_report_digest(reports[0])
    substituted = _rewrite_manifest(tmp_path / "quality", payload)

    with pytest.raises(QualityBatchError) as captured:
        validate_landmark_quality_bundle(
            substituted,
            tmp_path / "quality",
            extraction_manifest=extraction_manifest,
            extraction_root=extraction_root,
            raw_manifest=raw_manifest,
            raw_bundle_root=raw_root,
        )

    assert captured.value.category == "bundle.invalid"


@pytest.mark.parametrize("mutation", ["missing", "extra", "symlink"])
def test_validation_requires_exact_regular_file_inventory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    result, raw_root, raw_manifest, extraction_root, extraction_manifest = _assess(
        monkeypatch,
        tmp_path,
    )
    quality_root = tmp_path / "quality"
    manifest_path = quality_root / batch.LANDMARK_QUALITY_MANIFEST_FILENAME
    if mutation == "missing":
        manifest_path.unlink()
    elif mutation == "extra":
        (quality_root / "unexpected.json").write_text("{}", encoding="utf-8")
    else:
        external = tmp_path / "external-manifest.json"
        external.write_bytes(manifest_path.read_bytes())
        manifest_path.unlink()
        try:
            manifest_path.symlink_to(external)
        except OSError:
            pytest.skip("symbolic links are unavailable in this environment")

    with pytest.raises(QualityBatchError) as captured:
        validate_landmark_quality_bundle(
            result.manifest,
            quality_root,
            extraction_manifest=extraction_manifest,
            extraction_root=extraction_root,
            raw_manifest=raw_manifest,
            raw_bundle_root=raw_root,
        )

    assert captured.value.category == "bundle.invalid"


@pytest.mark.parametrize("placement", ["inside_raw", "inside_extraction", "ancestor"])
def test_destination_must_not_overlap_source_bundles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    placement: str,
) -> None:
    extraction, raw_root, raw_manifest, _runtimes, _trackers = _extract(
        monkeypatch,
        tmp_path,
    )
    extraction_root = tmp_path / "landmarks"
    before_raw = _tree_snapshot(raw_root)
    before_extraction = _tree_snapshot(extraction_root)
    destination = {
        "inside_raw": raw_root / "quality",
        "inside_extraction": extraction_root / "quality",
        "ancestor": tmp_path,
    }[placement]

    with pytest.raises(QualityBatchError) as captured:
        assess_landmark_quality(
            extraction.manifest,
            extraction_root=extraction_root,
            raw_manifest=raw_manifest,
            raw_bundle_root=raw_root,
            policy=build_default_quality_policy(),
            destination=destination,
        )

    assert captured.value.category == "destination.invalid"
    assert _tree_snapshot(raw_root) == before_raw
    assert _tree_snapshot(extraction_root) == before_extraction


def test_empty_destination_is_atomically_replaced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    extraction, raw_root, raw_manifest, _runtimes, _trackers = _extract(
        monkeypatch,
        tmp_path,
    )
    destination = tmp_path / "quality"
    destination.mkdir()

    result = assess_landmark_quality(
        extraction.manifest,
        extraction_root=tmp_path / "landmarks",
        raw_manifest=raw_manifest,
        raw_bundle_root=raw_root,
        policy=build_default_quality_policy(),
        destination=destination,
    )

    assert result.status == "published"
    assert tuple(path.name for path in destination.iterdir()) == (
        batch.LANDMARK_QUALITY_MANIFEST_FILENAME,
    )


def test_publication_failure_leaves_no_partial_bundle_or_staging_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    extraction, raw_root, raw_manifest, _runtimes, _trackers = _extract(
        monkeypatch,
        tmp_path,
    )

    def fail_canonicalization(_value: object) -> bytes:
        raise CanonicalizationError("private path")

    monkeypatch.setattr(batch, "canonical_json_bytes", fail_canonicalization)
    with pytest.raises(QualityBatchError) as captured:
        assess_landmark_quality(
            extraction.manifest,
            extraction_root=tmp_path / "landmarks",
            raw_manifest=raw_manifest,
            raw_bundle_root=raw_root,
            policy=build_default_quality_policy(),
            destination=tmp_path / "quality",
        )

    assert captured.value.category == "publication.failed"
    assert "private" not in str(captured.value)
    assert not (tmp_path / "quality").exists()
    assert not tuple(tmp_path.glob(".quality.staging-*"))


def test_policy_engine_failures_are_sanitized_and_do_not_publish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    extraction, raw_root, raw_manifest, _runtimes, _trackers = _extract(
        monkeypatch,
        tmp_path,
    )

    def fail_assessment(*_args: object, **_kwargs: object) -> object:
        raise QualityPolicyError("participant-private absolute path")

    monkeypatch.setattr(batch, "assess_landmark_sequence", fail_assessment)
    with pytest.raises(QualityBatchError) as captured:
        assess_landmark_quality(
            extraction.manifest,
            extraction_root=tmp_path / "landmarks",
            raw_manifest=raw_manifest,
            raw_bundle_root=raw_root,
            policy=build_default_quality_policy(),
            destination=tmp_path / "quality",
        )

    assert captured.value.category == "execution.failed"
    assert "private" not in str(captured.value)
    assert not (tmp_path / "quality").exists()


def test_consent_denial_from_public_extraction_validation_cannot_be_bypassed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    extraction, raw_root, raw_manifest, _runtimes, _trackers = _extract(
        monkeypatch,
        tmp_path,
    )

    def deny_consent(*_args: object, **_kwargs: object) -> object:
        raise extraction_batch.ExtractionBatchError("consent.unauthorized")

    def forbidden_assessment(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("quality policy must not run after consent denial")

    monkeypatch.setattr(batch, "validate_landmark_extraction_bundle", deny_consent)
    monkeypatch.setattr(batch, "assess_landmark_sequence", forbidden_assessment)
    with pytest.raises(QualityBatchError) as captured:
        assess_landmark_quality(
            extraction.manifest,
            extraction_root=tmp_path / "landmarks",
            raw_manifest=raw_manifest,
            raw_bundle_root=raw_root,
            policy=build_default_quality_policy(),
            destination=tmp_path / "quality",
        )

    assert captured.value.category == "consent.unauthorized"
    assert not (tmp_path / "quality").exists()


@pytest.mark.parametrize("race", ["identical", "conflict", "empty_cleanup"])
def test_atomic_rename_races_reconcile_or_fail_closed_without_staging_leaks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    race: str,
) -> None:
    winner, raw_root, raw_manifest, extraction_root, extraction_manifest = _assess(
        monkeypatch,
        tmp_path,
        destination_name="winner",
    )
    destination = tmp_path / "quality"
    if race == "empty_cleanup":
        destination.mkdir()
    original_rename = Path.rename

    def raced_rename(source: Path, target: str | Path) -> Path:
        target_path = Path(target)
        if target_path == destination and ".staging-" in source.name:
            if race == "identical":
                shutil.copytree(tmp_path / "winner", destination)
            elif race == "conflict":
                destination.mkdir()
                (destination / "conflict").write_text("occupied", encoding="utf-8")
            raise OSError("private race path")
        return original_rename(source, target)

    monkeypatch.setattr(Path, "rename", raced_rename)
    if race == "identical":
        reconciled = assess_landmark_quality(
            extraction_manifest,
            extraction_root=extraction_root,
            raw_manifest=raw_manifest,
            raw_bundle_root=raw_root,
            policy=build_default_quality_policy(),
            destination=destination,
        )
        assert reconciled.status == "unchanged"
        assert reconciled.manifest == winner.manifest
    else:
        with pytest.raises(QualityBatchError) as captured:
            assess_landmark_quality(
                extraction_manifest,
                extraction_root=extraction_root,
                raw_manifest=raw_manifest,
                raw_bundle_root=raw_root,
                policy=build_default_quality_policy(),
                destination=destination,
            )
        assert captured.value.category == "publication.failed"
        assert "private" not in str(captured.value)
        if race == "empty_cleanup":
            assert destination.is_dir()
            assert not any(destination.iterdir())
    assert not tuple(tmp_path.glob(".quality.staging-*"))
