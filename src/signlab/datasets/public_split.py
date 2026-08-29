"""Freeze the exact PopSign smoke split from retained landmark attempts."""

from __future__ import annotations

import hashlib
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self, cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pydantic import Field, ValidationError, model_validator

from signlab.contracts.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    canonical_sha256,
    parse_json_object,
)
from signlab.contracts.core import StrictContractModel, WorkspaceRelativeLocatorV1
from signlab.contracts.external_dataset import (
    ExternalDatasetManifestV1,
    ExternalMediaRecordV1,
    ExternalSplit,
    ExternalTargetLabel,
    OpaqueParticipantId,
    OpaqueRecordingId,
    OpaqueSampleId,
    validate_external_dataset_manifest,
)
from signlab.contracts.extraction import (
    landmark_frames_table_digest,
    mediapipe_extraction_config_digest,
)
from signlab.contracts.features import (
    LandmarkFeaturePlanV1,
    PortableFeatureSequenceV1,
    landmark_feature_plan_digest,
    validate_portable_feature_sequence,
)
from signlab.contracts.quality import (
    LandmarkQualityPolicyV1,
    landmark_quality_policy_digest,
)
from signlab.contracts.taxonomy import Sha256Digest
from signlab.datasets.popsign_window import (
    POPSIGN_WINDOW_RULE_ID,
    PopSignWindow,
    materialize_popsign_window,
    select_popsign_window,
)
from signlab.datasets.public_corpus import (
    _SOURCE_MIRROR_STATE,
    _content_path,
    _duration_us,
    _fresh_output_root,
    _write_bytes,
)
from signlab.extraction.parquet import read_landmark_frames, write_landmark_frames
from signlab.extraction.resources import load_packaged_default_extraction_config
from signlab.features.resources import load_packaged_default_feature_plan
from signlab.features.transforms import derive_feature_source
from signlab.quality.policy import assess_landmark_source
from signlab.quality.resources import load_packaged_default_quality_policy

type PublicPartition = Literal["train", "validation", "test"]
type PublicSplitInput = str | bytes | bytearray | Mapping[str, object]
type ExternalManifestInput = ExternalDatasetManifestV1 | PublicSplitInput

_SOURCE_CORPUS_SHA256 = "sha256:bd2e552d28792346b7c8e345f8387ebcc52938692cde7f0a316763aa09bdceb9"
_ATTEMPT_INVENTORY_SHA256 = (
    "sha256:fa2028b212c342fe273b0afd9101114c157b4822f3cc95860535848d993f9f44"
)
_EXPECTED_REPLAY = {"no_window": 11, "pass": 582, "quarantine": 46, "warning": 111}
_ATTEMPT_COUNT = 750
_TARGETS: tuple[ExternalTargetLabel, ...] = ("hello", "no", "please", "thank_you", "yes")
_QUOTAS: dict[ExternalSplit, int] = {"train": 10, "val": 3, "test": 3}
_PARTITION_BY_SOURCE: dict[ExternalSplit, PublicPartition] = {
    "train": "train",
    "val": "validation",
    "test": "test",
}
_PARTITION_ORDER = {"train": 0, "validation": 1, "test": 2}
_SELECTION_RULE = "first_usable_distinct_signer_by_stable_sample_id_per_split_target/1"
_FORMAT = "signlab-public-corpus-split/1"
_INVENTORY_DOMAIN = "signlab.popsign.quality-diagnostic.landmark-inventory/1"


class PublicCorpusSplitError(ValueError):
    """Raised when the frozen public split cannot be trusted."""


class PublicReplayCountsV1(StrictContractModel):
    """Exact post-window outcomes for the immutable 750 attempts."""

    pass_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    quarantine_count: int = Field(ge=0)
    no_window_count: int = Field(ge=0)
    usable_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _require_frozen_replay(self) -> Self:
        observed = {
            "no_window": self.no_window_count,
            "pass": self.pass_count,
            "quarantine": self.quarantine_count,
            "warning": self.warning_count,
        }
        if (
            observed != _EXPECTED_REPLAY
            or self.usable_count != self.pass_count + self.warning_count
        ):
            raise ValueError("public split replay counts changed from frozen evidence")
        return self


class PublicSplitAssignmentV1(StrictContractModel):
    """One selected sample and the exact artifacts used by experiments."""

    sample_id: OpaqueSampleId
    source_recording_id: OpaqueRecordingId
    source_signer_id: OpaqueParticipantId
    source_split: ExternalSplit
    partition: PublicPartition
    target_label_id: ExternalTargetLabel
    quality_disposition: Literal["pass", "warning"]
    feature_sequence_sha256: Sha256Digest
    feature_path: str

    @model_validator(mode="after")
    def _require_partition_and_path(self) -> Self:
        if self.partition != _PARTITION_BY_SOURCE[self.source_split]:
            raise ValueError("public partition does not preserve the official source split")
        value = self.feature_sequence_sha256.removeprefix("sha256:")
        if self.feature_path != f"features/sha256/{value[:2]}/{value}.json":
            raise ValueError("public split feature path is not content-addressed")
        return self


class PublicCorpusSplitV1(StrictContractModel):
    """One immutable 80-sample public split; no participant records are implied."""

    format: Literal["signlab-public-corpus-split/1"]
    split_id: Literal["popsign-five-isolated-smoke-v1"]
    version: Literal["1.0.0"]
    source_corpus_sha256: Sha256Digest
    external_dataset_sha256: Sha256Digest
    attempt_inventory_sha256: Sha256Digest
    extraction_config_sha256: Sha256Digest
    quality_policy_sha256: Sha256Digest
    feature_plan_id: Literal["combined_64_frames"]
    feature_plan_sha256: Sha256Digest
    active_window_rule_id: Literal["popsign_longest_detected_hand_episode/1"]
    selection_rule: Literal["first_usable_distinct_signer_by_stable_sample_id_per_split_target/1"]
    replay: PublicReplayCountsV1
    assignments: Annotated[tuple[PublicSplitAssignmentV1, ...], Field(min_length=80, max_length=80)]
    split_sha256: Sha256Digest

    @model_validator(mode="after")
    def _require_exact_leakage_free_membership(self) -> Self:
        keys = tuple(
            (_PARTITION_ORDER[row.partition], row.target_label_id, row.sample_id)
            for row in self.assignments
        )
        if keys != tuple(sorted(keys)):
            raise ValueError("public split assignments are not in canonical order")
        counts = Counter((row.source_split, row.target_label_id) for row in self.assignments)
        expected = {
            (source_split, target): quota
            for source_split, quota in _QUOTAS.items()
            for target in _TARGETS
        }
        if counts != expected:
            raise ValueError("public split does not meet exact split-target quotas")
        for label, values in (
            ("sample", tuple(row.sample_id for row in self.assignments)),
            ("recording", tuple(row.source_recording_id for row in self.assignments)),
            ("feature", tuple(row.feature_sequence_sha256 for row in self.assignments)),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"public split repeats a {label} identity")
        signer_partitions: dict[str, PublicPartition] = {}
        group_signers: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in self.assignments:
            prior = signer_partitions.setdefault(row.source_signer_id, row.partition)
            if prior != row.partition:
                raise ValueError("public split signer crosses partitions")
            group = (row.source_split, row.target_label_id)
            if row.source_signer_id in group_signers[group]:
                raise ValueError("public split repeats a signer within one split-target group")
            group_signers[group].add(row.source_signer_id)
        if self.split_sha256 != public_corpus_split_digest(self):
            raise ValueError("public split digest does not match canonical content")
        return self


@dataclass(frozen=True, slots=True)
class PublicSelectionCandidate:
    """Fields used by the fixed selection rule; disposition is never a rank."""

    sample_id: str
    source_recording_id: str
    source_signer_id: str
    source_split: ExternalSplit
    target_label_id: ExternalTargetLabel
    quality_disposition: Literal["pass", "warning"]


@dataclass(frozen=True, slots=True)
class PublicCorpusSample:
    """One fully reconciled input for the later experiment loader."""

    partition: PublicPartition
    target_label_id: ExternalTargetLabel
    sample_id: str
    source_recording_id: str
    source_signer_id: str
    quality_disposition: Literal["pass", "warning"]
    feature: PortableFeatureSequenceV1


@dataclass(frozen=True, slots=True)
class PublicCorpusSplitFreezeResult:
    """Aggregate, path-free result returned by the CLI."""

    split_sha256: str
    attempt_count: int
    pass_count: int
    warning_count: int
    quarantine_count: int
    no_window_count: int
    selected_count: int


@dataclass(frozen=True, slots=True)
class _Attempt:
    path: Path
    content_sha256: str
    parquet_sha256: str
    size_bytes: int
    row_count: int


@dataclass(frozen=True, slots=True)
class _UsableAttempt:
    candidate: PublicSelectionCandidate
    artifact: _Attempt
    media: ExternalMediaRecordV1
    window: PopSignWindow


def public_corpus_split_digest(document: PublicCorpusSplitV1 | Mapping[str, object]) -> str:
    """Hash split content without its self-referential digest."""

    payload = (
        cast(dict[str, object], document.model_dump(mode="json", round_trip=True))
        if isinstance(document, PublicCorpusSplitV1)
        else dict(document)
    )
    payload.pop("split_sha256", None)
    return canonical_sha256(payload, domain=_FORMAT)


def validate_public_corpus_split(document: PublicSplitInput) -> PublicCorpusSplitV1:
    """Validate exact quotas, canonical identity, and leakage barriers."""

    try:
        payload = parse_json_object(document)
        return PublicCorpusSplitV1.model_validate_json(canonical_json_bytes(payload), strict=True)
    except (CanonicalizationError, ValidationError) as error:
        raise PublicCorpusSplitError("invalid public corpus split") from error


def _resolve_artifact(root: Path, path: str) -> Path:
    try:
        locator = WorkspaceRelativeLocatorV1(kind="workspace_relative", path=path)
        resolved = root.joinpath(*locator.path.split("/")).resolve(strict=True)
    except (OSError, ValidationError) as error:
        raise PublicCorpusSplitError("public split artifact is unavailable") from error
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise PublicCorpusSplitError("public split artifact escapes its root")
    return resolved


def reconcile_public_corpus_split(
    document: PublicSplitInput,
    *,
    external_manifest_document: ExternalManifestInput,
    corpus_root: str | Path,
    partitions: Sequence[PublicPartition] | None = None,
) -> tuple[PublicCorpusSample, ...]:
    """Verify all metadata and materialize features from only the requested partitions."""

    split = validate_public_corpus_split(document)
    requested = frozenset(("train", "validation", "test") if partitions is None else partitions)
    if not requested or not requested <= {"train", "validation", "test"}:
        raise PublicCorpusSplitError("public split partitions are invalid")
    try:
        manifest = validate_external_dataset_manifest(external_manifest_document)
    except (TypeError, ValueError) as error:
        raise PublicCorpusSplitError("public split external manifest is invalid") from error
    if manifest.content_sha256 != split.external_dataset_sha256:
        raise PublicCorpusSplitError("public split external dataset identity does not match")
    media_by_recording = {row.recording_id: row for row in manifest.media}
    if len(media_by_recording) != len(manifest.media):
        raise PublicCorpusSplitError("public split external recording identities repeat")
    try:
        root = Path(corpus_root).resolve(strict=True)
    except OSError as error:
        raise PublicCorpusSplitError("public split root is unavailable") from error
    if not root.is_dir():
        raise PublicCorpusSplitError("public split root is invalid")
    samples: list[PublicCorpusSample] = []
    for row in split.assignments:
        media = media_by_recording.get(row.source_recording_id)
        if media is None or (
            media.sample_id,
            media.participant_id,
            media.source_split,
            media.target_label_id,
        ) != (
            row.sample_id,
            row.source_signer_id,
            row.source_split,
            row.target_label_id,
        ):
            raise PublicCorpusSplitError("public split assignment metadata does not match")
        if row.partition not in requested:
            continue
        try:
            feature = validate_portable_feature_sequence(
                _resolve_artifact(root, row.feature_path).read_bytes()
            )
        except (OSError, TypeError, ValueError) as error:
            raise PublicCorpusSplitError("public split artifact is invalid") from error
        expected = (
            (feature.sequence_sha256, row.feature_sequence_sha256),
            (feature.source_recording_id, row.source_recording_id),
            (feature.source_media_sha256, media.sha256),
            (feature.extraction_config_sha256, split.extraction_config_sha256),
            (feature.quality_policy_sha256, split.quality_policy_sha256),
            (feature.feature_plan_sha256, split.feature_plan_sha256),
        )
        if any(actual != claimed for actual, claimed in expected):
            raise PublicCorpusSplitError("public split feature lineage does not match")
        samples.append(
            PublicCorpusSample(
                partition=row.partition,
                target_label_id=row.target_label_id,
                sample_id=row.sample_id,
                source_recording_id=row.source_recording_id,
                source_signer_id=row.source_signer_id,
                quality_disposition=row.quality_disposition,
                feature=feature,
            )
        )
    return tuple(samples)


def _source_corpus(path: Path) -> dict[str, object]:
    try:
        payload = cast(dict[str, object], parse_json_object(path.read_bytes()))
        claimed = payload.get("corpus_sha256")
        unhashed = dict(payload)
        unhashed.pop("corpus_sha256", None)
        if (
            payload.get("format") != "signlab-public-corpus/1"
            or claimed != _SOURCE_CORPUS_SHA256
            or canonical_sha256(unhashed, domain="signlab-public-corpus/1") != claimed
        ):
            raise ValueError
        return payload
    except (CanonicalizationError, OSError, ValueError) as error:
        raise PublicCorpusSplitError("source public corpus identity does not match") from error


def _attempt_inventory(root: Path) -> tuple[tuple[_Attempt, ...], str]:
    try:
        paths = tuple(sorted(root.resolve(strict=True).rglob("*.parquet")))
        artifacts: list[_Attempt] = []
        items: list[dict[str, object]] = []
        for path in paths:
            if path.is_symlink() or not path.is_file() or len(path.stem) != 64:
                raise ValueError
            int(path.stem, 16)
            captured = path.read_bytes()
            content_sha256 = f"sha256:{path.stem}"
            parquet_sha256 = f"sha256:{hashlib.sha256(captured).hexdigest()}"
            row_count = pq.ParquetFile(pa.BufferReader(captured)).metadata.num_rows
            artifacts.append(
                _Attempt(path, content_sha256, parquet_sha256, len(captured), row_count)
            )
            items.append(
                {
                    "content_sha256": content_sha256,
                    "parquet_sha256": parquet_sha256,
                    "row_count": row_count,
                    "size_bytes": len(captured),
                }
            )
        digest = canonical_sha256({"items": items}, domain=_INVENTORY_DOMAIN)
    except (OSError, pa.ArrowException, TypeError, ValueError) as error:
        raise PublicCorpusSplitError("attempt landmark inventory is invalid") from error
    if len(artifacts) != _ATTEMPT_COUNT or digest != _ATTEMPT_INVENTORY_SHA256:
        raise PublicCorpusSplitError("attempt landmark inventory identity does not match")
    return tuple(artifacts), digest


def _select_candidates(
    candidates: Sequence[PublicSelectionCandidate],
) -> tuple[PublicSelectionCandidate, ...]:
    grouped: dict[tuple[ExternalSplit, ExternalTargetLabel], list[PublicSelectionCandidate]] = (
        defaultdict(list)
    )
    for candidate in candidates:
        grouped[(candidate.source_split, candidate.target_label_id)].append(candidate)
    expected = {(source_split, target) for source_split in _QUOTAS for target in _TARGETS}
    if set(grouped) != expected:
        raise PublicCorpusSplitError("usable attempts do not cover every split-target group")
    selected: list[PublicSelectionCandidate] = []
    for source_split in ("train", "val", "test"):
        for target in _TARGETS:
            group: list[PublicSelectionCandidate] = []
            signers: set[str] = set()
            for candidate in sorted(
                grouped[(source_split, target)], key=lambda item: item.sample_id
            ):
                if candidate.source_signer_id in signers:
                    continue
                signers.add(candidate.source_signer_id)
                group.append(candidate)
                if len(group) == _QUOTAS[source_split]:
                    break
            if len(group) != _QUOTAS[source_split]:
                raise PublicCorpusSplitError("usable attempts cannot meet an exact quota")
            selected.extend(group)
    return tuple(selected)


def _evaluate_attempts(
    artifacts: Sequence[_Attempt],
    media_by_recording: Mapping[str, ExternalMediaRecordV1],
    policy: LandmarkQualityPolicyV1,
    progress: Callable[[int, int], None] | None,
) -> tuple[tuple[_UsableAttempt, ...], Counter[str]]:
    usable: list[_UsableAttempt] = []
    outcomes: Counter[str] = Counter()
    seen: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="signlab-popsign-window-") as scratch_name:
        scratch = Path(scratch_name) / "window.parquet"
        for index, artifact in enumerate(artifacts, start=1):
            table = read_landmark_frames(
                artifact.path,
                expected_size_bytes=artifact.size_bytes,
                expected_sha256=artifact.parquet_sha256,
                expected_content_sha256=artifact.content_sha256,
                expected_row_count=artifact.row_count,
            )
            window = select_popsign_window(table)
            recording_id = table.rows[0].source_recording_id
            media = media_by_recording.get(recording_id)
            if media is None or recording_id in seen:
                raise PublicCorpusSplitError("attempt recording identity is invalid")
            seen.add(recording_id)
            if not window.selected:
                outcomes["no_window"] += 1
            else:
                windowed = materialize_popsign_window(table, window)
                parquet = write_landmark_frames(windowed, scratch)
                quality = assess_landmark_source(
                    windowed,
                    policy,
                    source_recording_id=recording_id,
                    source_sequence_content_sha256=parquet.content_sha256,
                    source_landmark_parquet_sha256=parquet.sha256,
                    declared_duration_us=_duration_us(windowed.rows),
                    expected_hand_count=1,
                )
                outcomes[quality.disposition] += 1
                if quality.disposition in {"pass", "warning"}:
                    if quality.metrics.timestamp_discontinuity_count:
                        raise PublicCorpusSplitError("usable attempt is not feature-ready")
                    candidate = PublicSelectionCandidate(
                        media.sample_id,
                        media.recording_id,
                        media.participant_id,
                        media.source_split,
                        media.target_label_id,
                        quality.disposition,
                    )
                    usable.append(_UsableAttempt(candidate, artifact, media, window))
            if progress is not None:
                progress(index, len(artifacts))
    return tuple(usable), outcomes


def _materialize_assignment(
    attempt: _UsableAttempt,
    destination: Path,
    scratch: Path,
    policy: LandmarkQualityPolicyV1,
    feature_plan: LandmarkFeaturePlanV1,
    extraction_config_sha256: str,
) -> PublicSplitAssignmentV1:
    artifact = attempt.artifact
    table = read_landmark_frames(
        artifact.path,
        expected_size_bytes=artifact.size_bytes,
        expected_sha256=artifact.parquet_sha256,
        expected_content_sha256=artifact.content_sha256,
        expected_row_count=artifact.row_count,
    )
    window = select_popsign_window(table)
    if window != attempt.window:
        raise PublicCorpusSplitError("active window changed during materialization")
    windowed = materialize_popsign_window(table, window)
    content_sha256 = landmark_frames_table_digest(windowed)
    parquet = write_landmark_frames(windowed, scratch)
    quality = assess_landmark_source(
        windowed,
        policy,
        source_recording_id=attempt.media.recording_id,
        source_sequence_content_sha256=content_sha256,
        source_landmark_parquet_sha256=parquet.sha256,
        declared_duration_us=_duration_us(windowed.rows),
        expected_hand_count=1,
    )
    if quality.disposition != attempt.candidate.quality_disposition:
        raise PublicCorpusSplitError("quality result changed during materialization")
    feature = derive_feature_source(
        windowed,
        quality,
        feature_plan,
        source_recording_id=attempt.media.recording_id,
        source_media_sha256=attempt.media.sha256,
        source_landmarks_sha256=content_sha256,
        source_landmark_parquet_sha256=parquet.sha256,
        source_mirror_state=_SOURCE_MIRROR_STATE,
        extraction_config_sha256=extraction_config_sha256,
    )
    feature_path = _content_path(destination, "features", feature.sequence_sha256, ".json")
    _write_bytes(feature_path, canonical_json_bytes(feature) + b"\n")
    return PublicSplitAssignmentV1(
        sample_id=attempt.media.sample_id,
        source_recording_id=attempt.media.recording_id,
        source_signer_id=attempt.media.participant_id,
        source_split=attempt.media.source_split,
        partition=_PARTITION_BY_SOURCE[attempt.media.source_split],
        target_label_id=attempt.media.target_label_id,
        quality_disposition=quality.disposition,
        feature_sequence_sha256=feature.sequence_sha256,
        feature_path=feature_path.relative_to(destination).as_posix(),
    )


def _summary_markdown(summary: Mapping[str, object]) -> str:
    return "\n".join(
        (
            "# Frozen PopSign smoke split",
            "",
            "The immutable 750-attempt landmark inventory was replayed without MediaPipe.",
            "One label-blind window rule and one offline selection rule produced the exact",
            "feature-ready 50 train / 15 validation / 15 test split.",
            "",
            "| Result | Count |",
            "| --- | ---: |",
            f"| Pass | {summary['pass_count']} |",
            f"| Warning | {summary['warning_count']} |",
            f"| Quarantine | {summary['quarantine_count']} |",
            f"| No safe window | {summary['no_window_count']} |",
            f"| Usable | {summary['usable_count']} |",
            f"| Selected | {summary['selected_count']} |",
            "",
            f"- Source corpus: `{summary['source_corpus_sha256']}`",
            f"- Attempt inventory: `{summary['attempt_inventory_sha256']}`",
            f"- Frozen split: `{summary['split_sha256']}`",
            f"- Selection rule: `{summary['selection_rule']}`",
            "- MediaPipe executions: 0",
            "",
            "Pass and warning rows were equally eligible; quality was not used for ranking.",
            "Signer uniqueness is enforced within each split/gesture group, and the official",
            "PopSign signer-disjoint train, validation, and test assignments are preserved.",
            "",
            "This proves a bounded licensed isolated-sign data path, not model performance.",
            "",
        )
    )


def freeze_public_corpus_split(
    external_manifest_path: str | Path,
    *,
    source_root: str | Path,
    output_root: str | Path,
    progress: Callable[[int, int], None] | None = None,
) -> PublicCorpusSplitFreezeResult:
    """Create and verify the exact split without rerunning MediaPipe."""

    try:
        source = Path(source_root).resolve(strict=True)
        source_corpus = _source_corpus(source / "public-corpus.json")
        manifest = validate_external_dataset_manifest(Path(external_manifest_path).read_bytes())
        config_sha256 = mediapipe_extraction_config_digest(
            load_packaged_default_extraction_config()
        )
        policy = load_packaged_default_quality_policy()
        feature_plan = load_packaged_default_feature_plan("combined")
        policy_sha256 = landmark_quality_policy_digest(policy)
        feature_plan_sha256 = landmark_feature_plan_digest(feature_plan)
        expected = (
            ("external_dataset_sha256", manifest.content_sha256),
            ("extraction_config_sha256", config_sha256),
            ("quality_policy_sha256", policy_sha256),
            ("feature_plan_sha256", feature_plan_sha256),
        )
        if any(source_corpus.get(field) != value for field, value in expected):
            raise ValueError
        media_by_recording = {row.recording_id: row for row in manifest.media}
        if len(media_by_recording) != len(manifest.media):
            raise ValueError
    except (OSError, TypeError, ValueError) as error:
        raise PublicCorpusSplitError("public split inputs could not be verified") from error

    artifacts, inventory_sha256 = _attempt_inventory(source / "landmarks")
    usable, outcomes = _evaluate_attempts(artifacts, media_by_recording, policy, progress)
    observed = {key: outcomes[key] for key in _EXPECTED_REPLAY}
    expected_usable = _EXPECTED_REPLAY["pass"] + _EXPECTED_REPLAY["warning"]
    if observed != _EXPECTED_REPLAY or outcomes["reject"] or len(usable) != expected_usable:
        raise PublicCorpusSplitError("window and quality replay changed from frozen evidence")
    selected = _select_candidates(tuple(row.candidate for row in usable))
    by_sample = {row.candidate.sample_id: row for row in usable}
    destination = _fresh_output_root(output_root)
    with tempfile.TemporaryDirectory(prefix="signlab-popsign-selected-") as scratch_name:
        scratch = Path(scratch_name) / "window.parquet"
        assignments = [
            _materialize_assignment(
                by_sample[candidate.sample_id],
                destination,
                scratch,
                policy,
                feature_plan,
                config_sha256,
            )
            for candidate in selected
        ]
    assignments.sort(
        key=lambda row: (_PARTITION_ORDER[row.partition], row.target_label_id, row.sample_id)
    )
    payload: dict[str, object] = {
        "format": _FORMAT,
        "split_id": "popsign-five-isolated-smoke-v1",
        "version": "1.0.0",
        "source_corpus_sha256": _SOURCE_CORPUS_SHA256,
        "external_dataset_sha256": manifest.content_sha256,
        "attempt_inventory_sha256": inventory_sha256,
        "extraction_config_sha256": config_sha256,
        "quality_policy_sha256": policy_sha256,
        "feature_plan_id": feature_plan.plan_id,
        "feature_plan_sha256": feature_plan_sha256,
        "active_window_rule_id": POPSIGN_WINDOW_RULE_ID,
        "selection_rule": _SELECTION_RULE,
        "replay": {
            "pass_count": outcomes["pass"],
            "warning_count": outcomes["warning"],
            "quarantine_count": outcomes["quarantine"],
            "no_window_count": outcomes["no_window"],
            "usable_count": len(usable),
        },
        "assignments": [row.model_dump(mode="json", round_trip=True) for row in assignments],
    }
    payload["split_sha256"] = public_corpus_split_digest(payload)
    split = validate_public_corpus_split(payload)
    _write_bytes(destination / "public-corpus-split.json", canonical_json_bytes(split) + b"\n")
    summary: dict[str, object] = {
        "format": "signlab-public-corpus-split-summary/1",
        "source_corpus_sha256": _SOURCE_CORPUS_SHA256,
        "attempt_inventory_sha256": inventory_sha256,
        "split_sha256": split.split_sha256,
        "selection_rule": _SELECTION_RULE,
        "attempt_count": len(artifacts),
        "pass_count": outcomes["pass"],
        "warning_count": outcomes["warning"],
        "quarantine_count": outcomes["quarantine"],
        "no_window_count": outcomes["no_window"],
        "usable_count": len(usable),
        "selected_count": len(assignments),
        "mediapipe_execution_count": 0,
    }
    _write_bytes(
        destination / "public-corpus-split-summary.md",
        _summary_markdown(summary).encode(),
    )
    reconciled = reconcile_public_corpus_split(
        split.model_dump(mode="json"),
        external_manifest_document=manifest,
        corpus_root=destination,
    )
    if len(reconciled) != 80:
        raise PublicCorpusSplitError("frozen public split did not reconcile")
    return PublicCorpusSplitFreezeResult(
        split_sha256=split.split_sha256,
        attempt_count=len(artifacts),
        pass_count=outcomes["pass"],
        warning_count=outcomes["warning"],
        quarantine_count=outcomes["quarantine"],
        no_window_count=outcomes["no_window"],
        selected_count=len(assignments),
    )


__all__ = [
    "PublicCorpusSample",
    "PublicCorpusSplitError",
    "PublicCorpusSplitFreezeResult",
    "PublicCorpusSplitV1",
    "PublicSelectionCandidate",
    "PublicSplitAssignmentV1",
    "freeze_public_corpus_split",
    "public_corpus_split_digest",
    "reconcile_public_corpus_split",
    "validate_public_corpus_split",
]
