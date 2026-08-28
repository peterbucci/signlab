"""Content-addressed storage for immutable portable landmark features."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Final

from signlab.contracts.canonical import CanonicalizationError, canonical_json_bytes
from signlab.contracts.extraction import LandmarkSequenceRefV1
from signlab.contracts.features import (
    FeatureCacheKeyV1,
    FeatureStatisticsV1,
    LandmarkFeaturePlanV1,
    PortableFeatureSequenceV1,
    feature_cache_key_digest,
    landmark_feature_plan_digest,
    validate_portable_feature_sequence,
)
from signlab.contracts.quality import SequenceQualityReportV1

FEATURE_CACHE_FILENAME: Final = "feature.json"
_MAX_FEATURE_BYTES: Final = 256 * 1024 * 1024


class FeatureCacheError(ValueError):
    """Raised when a feature cache key, object, or filesystem boundary is invalid."""


def _validated_sources(
    sequence_reference: LandmarkSequenceRefV1,
    quality_report: SequenceQualityReportV1,
) -> tuple[LandmarkSequenceRefV1, SequenceQualityReportV1]:
    if not isinstance(sequence_reference, LandmarkSequenceRefV1) or not isinstance(
        quality_report, SequenceQualityReportV1
    ):
        raise FeatureCacheError("feature cache sources must be validated contracts")
    try:
        return (
            LandmarkSequenceRefV1.model_validate_json(
                canonical_json_bytes(sequence_reference), strict=True
            ),
            SequenceQualityReportV1.model_validate_json(
                canonical_json_bytes(quality_report), strict=True
            ),
        )
    except (CanonicalizationError, TypeError, ValueError):
        raise FeatureCacheError("feature cache sources must be validated contracts") from None


def _validated_preprocessing(
    plan: LandmarkFeaturePlanV1,
    statistics: FeatureStatisticsV1 | None,
) -> tuple[LandmarkFeaturePlanV1, FeatureStatisticsV1 | None]:
    if not isinstance(plan, LandmarkFeaturePlanV1) or (
        statistics is not None and not isinstance(statistics, FeatureStatisticsV1)
    ):
        raise FeatureCacheError("feature cache preprocessing must be validated")
    try:
        checked_plan = LandmarkFeaturePlanV1.model_validate_json(
            canonical_json_bytes(plan), strict=True
        )
        checked_statistics = (
            None
            if statistics is None
            else FeatureStatisticsV1.model_validate_json(
                canonical_json_bytes(statistics), strict=True
            )
        )
        return checked_plan, checked_statistics
    except (CanonicalizationError, TypeError, ValueError):
        raise FeatureCacheError("feature cache preprocessing must be validated") from None


def _validated_sequence(sequence: PortableFeatureSequenceV1) -> PortableFeatureSequenceV1:
    if not isinstance(sequence, PortableFeatureSequenceV1):
        raise FeatureCacheError("feature cache content must be a validated sequence")
    try:
        return PortableFeatureSequenceV1.model_validate_json(
            canonical_json_bytes(sequence), strict=True
        )
    except (CanonicalizationError, TypeError, ValueError):
        raise FeatureCacheError("feature cache content must be a validated sequence") from None


def _cache_key(payload: dict[str, object]) -> FeatureCacheKeyV1:
    payload["cache_key_sha256"] = feature_cache_key_digest(payload)
    try:
        return FeatureCacheKeyV1.model_validate_json(
            canonical_json_bytes(payload),
            strict=True,
        )
    except (CanonicalizationError, TypeError, ValueError):
        raise FeatureCacheError("feature cache inputs are not portable") from None


def _assert_statistics_match_plan(
    plan: LandmarkFeaturePlanV1,
    plan_sha256: str,
    statistics: FeatureStatisticsV1 | None,
) -> None:
    if statistics is None:
        return
    if plan.learned_statistics.mode == "none":
        raise FeatureCacheError("a statistics-free feature plan cannot use fitted statistics")
    if (
        statistics.feature_plan_sha256 != plan_sha256
        or statistics.feature_names != plan.feature_order
        or statistics.quantization_scale != plan.quantization_scale
    ):
        raise FeatureCacheError("feature statistics do not match the preprocessing plan")


def build_feature_cache_key(
    sequence_reference: LandmarkSequenceRefV1,
    quality_report: SequenceQualityReportV1,
    plan: LandmarkFeaturePlanV1,
    *,
    extraction_config_sha256: str,
    statistics: FeatureStatisticsV1 | None = None,
) -> FeatureCacheKeyV1:
    """Bind exact extraction, quality, preprocessing, and optional fit identities."""

    checked_reference, checked_quality = _validated_sources(sequence_reference, quality_report)
    checked_plan, checked_statistics = _validated_preprocessing(plan, statistics)

    recording_id = checked_reference.lineage.source_recording_id
    expected_quality_source = (
        recording_id,
        checked_reference.content_sha256,
        checked_reference.lineage.artifact.sha256,
    )
    actual_quality_source = (
        checked_quality.source_recording_id,
        checked_quality.source_sequence_content_sha256,
        checked_quality.source_landmark_parquet_sha256,
    )
    if actual_quality_source != expected_quality_source:
        raise FeatureCacheError("quality evidence does not match the landmark sequence")

    plan_sha256 = landmark_feature_plan_digest(checked_plan)
    _assert_statistics_match_plan(checked_plan, plan_sha256, checked_statistics)
    return _cache_key(
        {
            "schema_version": "feature-cache-key/1",
            "source_recording_id": recording_id,
            "source_media_sha256": checked_reference.source_media_sha256,
            "source_landmarks_sha256": checked_reference.content_sha256,
            "extraction_config_sha256": extraction_config_sha256,
            "quality_policy_sha256": checked_quality.policy_sha256,
            "quality_report_sha256": checked_quality.report_sha256,
            "feature_plan_sha256": plan_sha256,
            "statistics_sha256": (
                None if checked_statistics is None else checked_statistics.statistics_sha256
            ),
        }
    )


def _cache_key_from_validated_sequence(
    sequence: PortableFeatureSequenceV1,
) -> FeatureCacheKeyV1:
    return _cache_key(
        {
            "schema_version": "feature-cache-key/1",
            "source_recording_id": sequence.source_recording_id,
            "source_media_sha256": sequence.source_media_sha256,
            "source_landmarks_sha256": sequence.source_landmarks_sha256,
            "extraction_config_sha256": sequence.extraction_config_sha256,
            "quality_policy_sha256": sequence.quality_policy_sha256,
            "quality_report_sha256": sequence.quality_report_sha256,
            "feature_plan_sha256": sequence.feature_plan_sha256,
            "statistics_sha256": sequence.statistics_sha256,
        }
    )


def feature_cache_key_from_sequence(
    sequence: PortableFeatureSequenceV1,
) -> FeatureCacheKeyV1:
    """Reconstruct the cache key using only a feature sequence's bound provenance."""

    return _cache_key_from_validated_sequence(_validated_sequence(sequence))


def _validated_key(key: FeatureCacheKeyV1) -> FeatureCacheKeyV1:
    if not isinstance(key, FeatureCacheKeyV1):
        raise FeatureCacheError("feature cache key must be a validated contract")
    try:
        return FeatureCacheKeyV1.model_validate_json(canonical_json_bytes(key), strict=True)
    except (CanonicalizationError, TypeError, ValueError):
        raise FeatureCacheError("feature cache key is invalid") from None


def _digest_parts(key: FeatureCacheKeyV1) -> tuple[str, str]:
    digest = key.cache_key_sha256.removeprefix("sha256:")
    return f"p-{digest[:2]}", f"sha256-{digest}"


def feature_cache_path(cache_root: str | Path, key: FeatureCacheKeyV1) -> Path:
    """Return the sole cache location, derived only from the cache-key digest."""

    checked = _validated_key(key)
    prefix, object_name = _digest_parts(checked)
    return Path(cache_root) / "objects" / "sha256" / prefix / object_name / FEATURE_CACHE_FILENAME


def _is_linklike(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _resolved_cache_root(cache_root: str | Path, *, create: bool) -> Path:
    try:
        candidate = Path(cache_root)
        if _is_linklike(candidate):
            raise FeatureCacheError("feature cache root cannot be a link")
        if not candidate.exists():
            if not create:
                raise FeatureCacheError("feature cache root does not exist")
            parent = candidate.parent.resolve(strict=True)
            if _is_linklike(candidate.parent) or not parent.is_dir():
                raise FeatureCacheError("feature cache parent is invalid")
            candidate = parent / candidate.name
            try:
                candidate.mkdir()
            except FileExistsError:
                # Another writer may have published the shared root after our
                # existence check. Reconcile only an ordinary directory; links
                # and non-directories remain fail-closed below.
                if _is_linklike(candidate) or not candidate.is_dir():
                    raise FeatureCacheError("feature cache root is invalid") from None
        root = candidate.resolve(strict=True)
        if _is_linklike(candidate) or not root.is_dir():
            raise FeatureCacheError("feature cache root is invalid")
        return root
    except FeatureCacheError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise FeatureCacheError("feature cache root is invalid") from None


def _require_within(path: Path, root: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise FeatureCacheError("feature cache path escapes its root")
        return resolved
    except FeatureCacheError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise FeatureCacheError("feature cache path is invalid") from None


def _ensure_prefix(root: Path, key: FeatureCacheKeyV1) -> tuple[Path, Path]:
    prefix_name, object_name = _digest_parts(key)
    current = root
    for part in ("objects", "sha256", prefix_name):
        candidate = current / part
        try:
            if _is_linklike(candidate):
                raise FeatureCacheError("feature cache hierarchy cannot contain links")
            try:
                candidate.mkdir()
            except FileExistsError:
                # Shared cache writers commonly race while establishing a new
                # digest prefix. Accept only the directory another writer won.
                if _is_linklike(candidate) or not candidate.is_dir():
                    raise FeatureCacheError("feature cache hierarchy is invalid") from None
            current = _require_within(candidate, root)
        except FeatureCacheError:
            raise
        except (OSError, RuntimeError, ValueError):
            raise FeatureCacheError("feature cache hierarchy is invalid") from None
    return current, current / object_name


def _object_path(root: Path, key: FeatureCacheKeyV1) -> Path:
    prefix_name, object_name = _digest_parts(key)
    return root / "objects" / "sha256" / prefix_name / object_name


def _load_from_root(root: Path, key: FeatureCacheKeyV1) -> PortableFeatureSequenceV1:
    object_root = _object_path(root, key)
    feature_path = object_root / FEATURE_CACHE_FILENAME
    try:
        for directory in (root / "objects", root / "objects" / "sha256", object_root.parent):
            if _is_linklike(directory) or not directory.is_dir():
                raise FeatureCacheError("feature cache hierarchy is invalid")
            _require_within(directory, root)
        if _is_linklike(object_root) or not object_root.is_dir():
            raise FeatureCacheError("feature cache object is missing or invalid")
        _require_within(object_root, root)
        inventory = tuple(sorted(path.name for path in object_root.iterdir()))
        if inventory != (FEATURE_CACHE_FILENAME,):
            raise FeatureCacheError("feature cache object inventory is invalid")
        if _is_linklike(feature_path) or not feature_path.is_file():
            raise FeatureCacheError("feature cache content is invalid")
        resolved_feature = _require_within(feature_path, root)
        size = resolved_feature.stat().st_size
        if size <= 0 or size > _MAX_FEATURE_BYTES:
            raise FeatureCacheError("feature cache content size is invalid")
        captured = resolved_feature.read_bytes()
        sequence = validate_portable_feature_sequence(captured)
        if captured != canonical_json_bytes(sequence) + b"\n":
            raise FeatureCacheError("feature cache content is not canonical")
        if _cache_key_from_validated_sequence(sequence) != key:
            raise FeatureCacheError("feature cache content does not match its key")
        return sequence
    except FeatureCacheError:
        raise
    except (CanonicalizationError, OSError, RuntimeError, TypeError, ValueError):
        raise FeatureCacheError("feature cache content is invalid") from None


def load_cached_feature(
    cache_root: str | Path,
    key: FeatureCacheKeyV1,
) -> PortableFeatureSequenceV1:
    """Load and fully verify one canonical feature cache object."""

    checked = _validated_key(key)
    root = _resolved_cache_root(cache_root, create=False)
    return _load_from_root(root, checked)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_feature(path: Path, sequence: PortableFeatureSequenceV1) -> None:
    try:
        captured = canonical_json_bytes(sequence) + b"\n"
        with path.open("xb") as stream:
            stream.write(captured)
            stream.flush()
            os.fsync(stream.fileno())
    except (CanonicalizationError, OSError, RuntimeError, TypeError, ValueError):
        raise FeatureCacheError("feature cache content could not be written") from None


def store_cached_feature(
    cache_root: str | Path,
    key: FeatureCacheKeyV1,
    sequence: PortableFeatureSequenceV1,
) -> Path:
    """Atomically publish or reconcile one immutable cache object."""

    checked = _validated_key(key)
    checked_sequence = _validated_sequence(sequence)
    if _cache_key_from_validated_sequence(checked_sequence) != checked:
        raise FeatureCacheError("feature cache content does not match its key")

    root = _resolved_cache_root(cache_root, create=True)
    prefix, object_root = _ensure_prefix(root, checked)
    final_path = object_root / FEATURE_CACHE_FILENAME
    if object_root.exists() or _is_linklike(object_root):
        existing = _load_from_root(root, checked)
        if existing != checked_sequence:
            raise FeatureCacheError("feature cache key already has different content")
        return final_path

    try:
        with tempfile.TemporaryDirectory(
            dir=prefix,
            prefix=f".{object_root.name}.staging-",
        ) as temporary_directory:
            staging = Path(temporary_directory)
            _write_feature(staging / FEATURE_CACHE_FILENAME, checked_sequence)
            _fsync_directory(staging)
            try:
                staging.rename(object_root)
                _fsync_directory(prefix)
            except OSError:
                if not object_root.exists() or _is_linklike(object_root):
                    raise FeatureCacheError("feature cache object could not be published") from None
        stored = _load_from_root(root, checked)
        if stored != checked_sequence:
            raise FeatureCacheError("feature cache key already has different content")
        return final_path
    except FeatureCacheError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise FeatureCacheError("feature cache object could not be published") from None


__all__ = [
    "FEATURE_CACHE_FILENAME",
    "FeatureCacheError",
    "build_feature_cache_key",
    "feature_cache_key_from_sequence",
    "feature_cache_path",
    "load_cached_feature",
    "store_cached_feature",
]
