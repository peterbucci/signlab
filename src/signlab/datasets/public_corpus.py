"""Bounded PopSign vertical slice through SignLab's existing landmark pipeline."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

from signlab.contracts.canonical import canonical_json_bytes, canonical_sha256
from signlab.contracts.dataset import MirrorState
from signlab.contracts.external_dataset import (
    ExternalMediaRecordV1,
    ExternalSplit,
    ExternalTargetLabel,
    validate_external_dataset_manifest,
)
from signlab.contracts.extraction import (
    LandmarkFrameV1,
    landmark_frames_table_digest,
    mediapipe_extraction_config_digest,
)
from signlab.contracts.features import landmark_feature_plan_digest
from signlab.contracts.quality import landmark_quality_policy_digest
from signlab.datasets.external_resources import load_popsign_source
from signlab.datasets.popsign import validate_external_dataset_bundle
from signlab.extraction.batch import ExtractionBatchError, extract_media_landmarks
from signlab.extraction.parquet import write_landmark_frames
from signlab.extraction.resources import load_packaged_default_extraction_config
from signlab.extraction.runtime import verify_model_assets
from signlab.features.resources import load_packaged_default_feature_plan
from signlab.features.transforms import FeatureTransformError, derive_feature_source
from signlab.quality.policy import assess_landmark_source
from signlab.quality.resources import load_packaged_default_quality_policy

_SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}
_CORPUS_FILENAME = "public-corpus.json"
_SUMMARY_JSON_FILENAME = "public-corpus-summary.json"
_SUMMARY_MARKDOWN_FILENAME = "public-corpus-summary.md"
_SOURCE_MIRROR_STATE: MirrorState = "not_mirrored"
_SOURCE_ROTATION_DEGREES = 0
_SOURCE_ORIENTATION_BASIS = (
    "reviewed_popsign_v1_official_samples_upright_with_readable_unmirrored_scene_text"
)
_TARGET_LABELS: tuple[ExternalTargetLabel, ...] = (
    "hello",
    "no",
    "please",
    "thank_you",
    "yes",
)
_TRAINABLE_TARGETS: dict[ExternalSplit, int] = {"train": 10, "val": 3, "test": 3}
_TRAINABLE_TARGET_COUNT = len(_TARGET_LABELS) * sum(_TRAINABLE_TARGETS.values())
_TRAINABLE_ATTEMPT_LIMIT = 750


class PublicCorpusError(ValueError):
    """Raised when the bounded public corpus cannot be built."""


@dataclass(frozen=True, slots=True)
class PublicCorpusBuildResult:
    corpus_sha256: str
    selected_count: int
    group_count: int
    exclusion_count: int
    attempted_count: int
    target_count: int
    attempt_limit: int | None
    decision: str
    summary: dict[str, object]


@dataclass(slots=True)
class _TrainableGroupState:
    source_split: ExternalSplit
    target_label_id: ExternalTargetLabel
    candidates: tuple[ExternalMediaRecordV1, ...]
    target_count: int
    candidate_index: int = 0
    attempted_count: int = 0
    selected_count: int = 0
    selected_signers: set[str] = field(default_factory=set)


def _duration_us(media_rows: tuple[LandmarkFrameV1, ...]) -> int:
    timestamps = tuple(row.relative_timestamp_us for row in media_rows)
    deltas = tuple(right - left for left, right in pairwise(timestamps))
    if not deltas:
        return 1
    ordered = sorted(deltas)
    middle = len(ordered) // 2
    median = (
        ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle] + 1) // 2
    )
    return max(timestamps[-1] + median, 1)


def _content_path(root: Path, category: str, digest: str, suffix: str) -> Path:
    value = digest.removeprefix("sha256:")
    return root / category / "sha256" / value[:2] / f"{value}{suffix}"


def _verify_media_bytes(path: Path, media: ExternalMediaRecordV1) -> None:
    try:
        before = path.stat()
        if path.is_symlink() or not path.is_file() or before.st_size != media.size_bytes:
            raise PublicCorpusError("external media changed after bundle validation")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        after = path.stat()
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if (
            identity_before != identity_after
            or size != media.size_bytes
            or f"sha256:{digest.hexdigest()}" != media.sha256
        ):
            raise PublicCorpusError("external media changed after bundle validation")
    except PublicCorpusError:
        raise
    except OSError as error:
        raise PublicCorpusError("external media could not be verified") from error


def _fresh_output_root(output_root: str | Path) -> Path:
    destination = Path(output_root)
    try:
        if destination.exists():
            if not destination.is_dir() or any(destination.iterdir()):
                raise PublicCorpusError("public corpus output must be a new empty directory")
        else:
            destination.mkdir(parents=True)
        return destination.resolve(strict=True)
    except PublicCorpusError:
        raise
    except OSError as error:
        raise PublicCorpusError("public corpus output could not be created") from error


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as error:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise PublicCorpusError("public corpus report could not be written") from error


def _ordered_counts(values: Counter[str]) -> dict[str, int]:
    return {key: values[key] for key in sorted(values)}


def _signer_interleaved(
    candidates: tuple[ExternalMediaRecordV1, ...],
) -> tuple[ExternalMediaRecordV1, ...]:
    """Give every signer one stable attempt before any signer receives another."""

    by_signer: dict[str, list[ExternalMediaRecordV1]] = defaultdict(list)
    for media in candidates:
        by_signer[media.participant_id].append(media)
    ordered_signers = tuple(sorted(by_signer))
    for media_rows in by_signer.values():
        media_rows.sort(key=lambda item: item.sample_id)
    return tuple(
        by_signer[signer][depth]
        for depth in range(max((len(rows) for rows in by_signer.values()), default=0))
        for signer in ordered_signers
        if depth < len(by_signer[signer])
    )


def _trainable_group_results(
    states: tuple[_TrainableGroupState, ...],
    *,
    attempted_count: int,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for state in states:
        if state.selected_count >= state.target_count:
            terminal_reason = "target_reached"
        elif state.candidate_index >= len(state.candidates):
            terminal_reason = "candidates_exhausted"
        elif attempted_count >= _TRAINABLE_ATTEMPT_LIMIT:
            terminal_reason = "attempt_limit_reached"
        else:
            terminal_reason = "selection_stopped"
        results.append(
            {
                "source_split": state.source_split,
                "target_label_id": state.target_label_id,
                "target_count": state.target_count,
                "selected_count": state.selected_count,
                "selected_signer_count": len(state.selected_signers),
                "available_signer_count": len({media.participant_id for media in state.candidates}),
                "attempted_count": state.attempted_count,
                "shortfall_count": max(state.target_count - state.selected_count, 0),
                "terminal_reason": terminal_reason,
            }
        )
    return results


def _summary_markdown(summary: dict[str, object]) -> str:
    if summary.get("selection_mode") == "trainable_smoke":
        return _trainable_summary_markdown(summary)

    selected_by_split = summary["selected_by_split"]
    selected_by_target = summary["selected_by_target"]
    exclusions = summary["exclusions"]
    assert isinstance(selected_by_split, dict)
    assert isinstance(selected_by_target, dict)
    assert isinstance(exclusions, dict)
    lines = [
        "# PopSign public-data vertical slice",
        "",
        "This report proves that one bounded, licensed public corpus ran through the existing ",
        "SignLab extraction, quality, and `combined-64` feature pipeline.",
        "",
        f"- Candidate clips: {summary['candidate_count']}",
        f"- Split/target groups: {summary['group_count']}",
        f"- Selected usable clips: {summary['selected_count']}",
        f"- Groups without a usable clip: {summary['unfilled_group_count']}",
        f"- Corpus SHA-256: `{summary['corpus_sha256']}`",
        "",
        "## Selected clips",
        "",
        "| Source split | Count |",
        "| --- | ---: |",
        *[f"| {key} | {value} |" for key, value in selected_by_split.items()],
        "",
        "| Target | Count |",
        "| --- | ---: |",
        *[f"| {key} | {value} |" for key, value in selected_by_target.items()],
        "",
        "## Coded exclusions",
        "",
        "| Reason | Count |",
        "| --- | ---: |",
        *[f"| `{key}` | {value} |" for key, value in exclusions.items()],
        "",
        "## Reproducibility identities",
        "",
        f"- External dataset: `{summary['external_dataset_sha256']}`",
        f"- Extraction configuration: `{summary['extraction_config_sha256']}`",
        f"- Hand model: `{summary['hand_model_sha256']}`",
        f"- Pose model: `{summary['pose_model_sha256']}`",
        f"- Quality policy: `{summary['quality_policy_sha256']}`",
        f"- Feature plan: `{summary['feature_plan_sha256']}`",
        f"- Source orientation basis: `{summary['source_orientation_basis']}`",
        "",
        "## License and limits",
        "",
        f"{summary['attribution']}",
        "",
        f"License: [{summary['license_id']}]({summary['license_url']}).",
        "",
        f"**Limitation:** {summary['limitation']}",
        "",
    ]
    return "\n".join(lines)


def _trainable_summary_markdown(summary: dict[str, object]) -> str:
    groups = summary["selection_groups"]
    exclusions = summary["exclusions"]
    assert isinstance(groups, list)
    assert isinstance(exclusions, dict)
    lines = [
        "# PopSign trainable smoke-corpus decision",
        "",
        "This is the result of one predeclared, bounded run through SignLab's existing ",
        "extraction, quality, and `combined-64` feature pipeline.",
        "",
        f"- Decision: **{str(summary['decision']).upper()}**",
        f"- Selected usable clips: {summary['selected_count']}/{summary['target_count']}",
        f"- Attempted videos: {summary['attempted_count']}/{summary['attempt_limit']}",
        f"- Split/target groups below quota: {summary['unfilled_group_count']}",
        f"- Globally represented signers: {summary['selected_signer_count']}",
        f"- Corpus SHA-256: `{summary['corpus_sha256']}`",
        f"- Selection rule: `{summary['selection_rule']}`",
        "",
        "## Split and gesture quotas",
        "",
        "Signer uniqueness is enforced within each split/gesture group. PopSign's official ",
        "signer-disjoint train, validation, and test assignments are preserved.",
        "",
        "| Split | Gesture | Selected | Target | Distinct signers | Attempts | Result |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        *[
            "| {source_split} | {target_label_id} | {selected_count} | {target_count} | "
            "{selected_signer_count} | {attempted_count} | {terminal_reason} |".format(**item)
            for item in groups
            if isinstance(item, dict)
        ],
        "",
        "## Coded outcomes",
        "",
        "| Reason | Count |",
        "| --- | ---: |",
        *[f"| `{key}` | {value} |" for key, value in exclusions.items()],
        "",
        "## Reproducibility identities",
        "",
        f"- External dataset: `{summary['external_dataset_sha256']}`",
        f"- Extraction configuration: `{summary['extraction_config_sha256']}`",
        f"- Hand model: `{summary['hand_model_sha256']}`",
        f"- Pose model: `{summary['pose_model_sha256']}`",
        f"- Quality policy: `{summary['quality_policy_sha256']}`",
        f"- Feature plan: `{summary['feature_plan_sha256']}`",
        f"- Source orientation basis: `{summary['source_orientation_basis']}`",
        "",
        "## License and limits",
        "",
        str(summary["attribution"]),
        "",
        f"License: [{summary['license_id']}]({summary['license_url']}).",
        "",
        f"**Limitation:** {summary['limitation']}",
        "",
        "No model-performance metric is claimed by this data sufficiency run.",
        "",
    ]
    return "\n".join(lines)


def build_public_corpus(
    manifest_path: str | Path,
    *,
    external_root: str | Path,
    model_root: str | Path,
    output_root: str | Path,
    archive_root: str | Path | None = None,
    max_candidates_per_group: int = 5,
    trainable_smoke: bool = False,
    progress: Callable[[int, int, bool], None] | None = None,
) -> PublicCorpusBuildResult:
    """Build the legacy vertical slice or one fixed, capped trainable smoke corpus."""

    if type(max_candidates_per_group) is not int or max_candidates_per_group <= 0:
        raise PublicCorpusError("candidate attempt limit must be positive")
    if type(trainable_smoke) is not bool:
        raise PublicCorpusError("trainable smoke selection must be a boolean")
    try:
        manifest_bytes = Path(manifest_path).read_bytes()
        external_validation = validate_external_dataset_bundle(
            manifest_bytes,
            external_root,
            archive_root=archive_root,
        )
        manifest = validate_external_dataset_manifest(manifest_bytes)
        source = load_popsign_source()
        config = load_packaged_default_extraction_config()
        policy = load_packaged_default_quality_policy()
        feature_plan = load_packaged_default_feature_plan("combined")
        assets = verify_model_assets(model_root)
    except (OSError, TypeError, ValueError) as error:
        raise PublicCorpusError("public corpus inputs could not be verified") from error

    destination = _fresh_output_root(output_root)
    external_bundle = Path(external_root).resolve(strict=True)
    config_sha256 = mediapipe_extraction_config_digest(config)
    policy_sha256 = landmark_quality_policy_digest(policy)
    feature_plan_sha256 = landmark_feature_plan_digest(feature_plan)

    grouped: dict[tuple[ExternalSplit, ExternalTargetLabel], list[ExternalMediaRecordV1]] = (
        defaultdict(list)
    )
    for media in manifest.media:
        grouped[(media.source_split, media.target_label_id)].append(media)
    groups = tuple(
        (key, tuple(sorted(values, key=lambda item: item.sample_id)))
        for key, values in sorted(
            grouped.items(),
            key=lambda item: (_SPLIT_ORDER[item[0][0]], item[0][1]),
        )
    )
    if trainable_smoke:
        expected_groups = {
            (source_split, target_label_id)
            for source_split in _TRAINABLE_TARGETS
            for target_label_id in _TARGET_LABELS
        }
        if set(grouped) != expected_groups:
            raise PublicCorpusError("trainable smoke selection requires all split-target groups")

    exclusions: Counter[str] = Counter()
    selected: list[dict[str, object]] = []
    selected_by_split: Counter[str] = Counter()
    selected_by_target: Counter[str] = Counter()
    selected_by_disposition: Counter[str] = Counter()
    attempted_count = 0

    def attempt_candidate(
        media: ExternalMediaRecordV1,
    ) -> tuple[dict[str, object], str] | None:
        nonlocal attempted_count

        source_path = external_bundle.joinpath(*media.locator.path.split("/"))
        _verify_media_bytes(source_path, media)
        attempted_count += 1
        try:
            table = extract_media_landmarks(
                media.recording_id,
                source_path,
                assets=assets,
                config=config,
            )
        except ExtractionBatchError:
            exclusions["extraction.failed"] += 1
            return None
        _verify_media_bytes(source_path, media)

        content_sha256 = landmark_frames_table_digest(table)
        parquet_path = _content_path(destination, "landmarks", content_sha256, ".parquet")
        parquet = write_landmark_frames(table, parquet_path)
        quality = assess_landmark_source(
            table,
            policy,
            source_recording_id=media.recording_id,
            source_sequence_content_sha256=content_sha256,
            source_landmark_parquet_sha256=parquet.sha256,
            declared_duration_us=_duration_us(table.rows),
            expected_hand_count=1,
        )
        if quality.disposition in {"quarantine", "reject"}:
            exclusions[f"quality.{quality.disposition}"] += 1
            return None
        if quality.metrics.timestamp_discontinuity_count:
            exclusions["feature.timestamp_discontinuity"] += 1
            return None
        try:
            feature = derive_feature_source(
                table,
                quality,
                feature_plan,
                source_recording_id=media.recording_id,
                source_media_sha256=media.sha256,
                source_landmarks_sha256=content_sha256,
                source_landmark_parquet_sha256=parquet.sha256,
                source_mirror_state=_SOURCE_MIRROR_STATE,
                extraction_config_sha256=config_sha256,
            )
        except FeatureTransformError as error:
            raise PublicCorpusError("accepted landmarks could not produce features") from error
        feature_path = _content_path(
            destination,
            "features",
            feature.sequence_sha256,
            ".json",
        )
        _write_bytes(feature_path, canonical_json_bytes(feature) + b"\n")
        return (
            {
                "archive_id": media.archive_id,
                "source_member_fingerprint": media.source_member_fingerprint,
                "sample_id": media.sample_id,
                "source_recording_id": media.recording_id,
                "source_signer_id": media.participant_id,
                "source_split": media.source_split,
                "source_label": media.source_label,
                "target_label_id": media.target_label_id,
                "source_media_sha256": media.sha256,
                "source_media_size_bytes": media.size_bytes,
                "source_rotation_degrees": _SOURCE_ROTATION_DEGREES,
                "source_mirror_state": _SOURCE_MIRROR_STATE,
                "source_orientation_basis": _SOURCE_ORIENTATION_BASIS,
                "expected_hand_count": 1,
                "landmark_content_sha256": content_sha256,
                "landmark_parquet_sha256": parquet.sha256,
                "landmark_parquet_size_bytes": parquet.size_bytes,
                "landmark_frame_count": parquet.row_count,
                "landmark_path": parquet_path.relative_to(destination).as_posix(),
                "quality_report_sha256": quality.report_sha256,
                "quality_disposition": quality.disposition,
                "expected_hand_coverage_ppm": quality.metrics.expected_hand_coverage_ppm,
                "quality_finding_rule_ids": [item.rule_id for item in quality.findings],
                "feature_sequence_sha256": feature.sequence_sha256,
                "feature_path": feature_path.relative_to(destination).as_posix(),
            },
            quality.disposition,
        )

    def retain_candidate(
        media: ExternalMediaRecordV1,
        result: tuple[dict[str, object], str],
    ) -> None:
        record, disposition = result
        selected.append(record)
        selected_by_split[media.source_split] += 1
        selected_by_target[media.target_label_id] += 1
        selected_by_disposition[disposition] += 1

    trainable_states: tuple[_TrainableGroupState, ...] = ()
    if not trainable_smoke:
        for group_index, ((_split, _target), candidates) in enumerate(groups, start=1):
            accepted = False
            attempted = min(len(candidates), max_candidates_per_group)
            for candidate_index, media in enumerate(candidates[:attempted]):
                result = attempt_candidate(media)
                if result is None:
                    continue
                retain_candidate(media, result)
                remaining = len(candidates) - candidate_index - 1
                if remaining:
                    exclusions["selection.not_needed_after_accepted"] += remaining
                accepted = True
                break
            if not accepted and len(candidates) > attempted:
                exclusions["selection.attempt_limit"] += len(candidates) - attempted
            if progress is not None:
                progress(group_index, len(groups), accepted)
    else:
        trainable_states = tuple(
            _TrainableGroupState(
                source_split=source_split,
                target_label_id=target_label_id,
                candidates=_signer_interleaved(candidates),
                target_count=_TRAINABLE_TARGETS[source_split],
            )
            for (source_split, target_label_id), candidates in groups
        )
        while attempted_count < _TRAINABLE_ATTEMPT_LIMIT:
            active = tuple(
                state
                for state in trainable_states
                if state.selected_count < state.target_count
                and state.candidate_index < len(state.candidates)
            )
            if not active:
                break
            state = min(
                active,
                key=lambda item: (
                    Fraction(item.attempted_count, item.target_count),
                    _SPLIT_ORDER[item.source_split],
                    item.target_label_id,
                ),
            )
            candidate_media: ExternalMediaRecordV1 | None = None
            while state.candidate_index < len(state.candidates):
                candidate = state.candidates[state.candidate_index]
                state.candidate_index += 1
                if candidate.participant_id in state.selected_signers:
                    exclusions["selection.signer_already_selected"] += 1
                    continue
                candidate_media = candidate
                break
            if candidate_media is None:
                continue
            state.attempted_count += 1
            result = attempt_candidate(candidate_media)
            accepted = result is not None
            if result is not None:
                retain_candidate(candidate_media, result)
                state.selected_count += 1
                state.selected_signers.add(candidate_media.participant_id)
            if progress is not None:
                progress(attempted_count, _TRAINABLE_ATTEMPT_LIMIT, accepted)

        for state in trainable_states:
            remaining = len(state.candidates) - state.candidate_index
            if not remaining:
                continue
            reason = (
                "selection.not_needed_after_target"
                if state.selected_count >= state.target_count
                else "selection.total_attempt_limit"
            )
            exclusions[reason] += remaining
        selected.sort(
            key=lambda item: (
                _SPLIT_ORDER[str(item["source_split"])],
                str(item["target_label_id"]),
                str(item["sample_id"]),
            )
        )

    trainable_group_results = _trainable_group_results(
        trainable_states,
        attempted_count=attempted_count,
    )
    shortfall_group_count = sum(
        state.selected_count < state.target_count for state in trainable_states
    )
    decision = (
        ("ready" if shortfall_group_count == 0 else "insufficient")
        if trainable_smoke
        else "legacy_vertical_slice"
    )
    target_count = _TRAINABLE_TARGET_COUNT if trainable_smoke else len(groups)
    attempt_limit = _TRAINABLE_ATTEMPT_LIMIT if trainable_smoke else None

    corpus_payload: dict[str, object] = {
        "format": "signlab-public-corpus/1",
        "external_dataset_sha256": manifest.content_sha256,
        "archive_byte_integrity": external_validation.archive_byte_integrity,
        "archives": [
            {
                "archive_id": item.archive_id,
                "source_split": item.split,
                "source_label": item.source_label,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
                "member_count": item.member_count,
            }
            for item in manifest.archives
        ],
        "source_rotation_degrees": _SOURCE_ROTATION_DEGREES,
        "source_mirror_state": _SOURCE_MIRROR_STATE,
        "source_orientation_basis": _SOURCE_ORIENTATION_BASIS,
        "expected_hand_count": 1,
        "extraction_config_sha256": config_sha256,
        "hand_model_sha256": assets.hand_model_sha256,
        "pose_model_sha256": assets.pose_model_sha256,
        "quality_policy_sha256": policy_sha256,
        "feature_plan_id": feature_plan.plan_id,
        "feature_plan_sha256": feature_plan_sha256,
        "selected": selected,
        "exclusions": _ordered_counts(exclusions),
    }
    if trainable_smoke:
        corpus_payload.update(
            {
                "selection_mode": "trainable_smoke",
                "selection_rule": "signer_interleaved_fair_split_target_quota/1",
                "attempt_limit": _TRAINABLE_ATTEMPT_LIMIT,
                "attempted_count": attempted_count,
                "target_count": _TRAINABLE_TARGET_COUNT,
                "decision": decision,
                "selection_groups": trainable_group_results,
            }
        )
    else:
        corpus_payload.update(
            {
                "selection_rule": "first_usable_by_stable_sample_id_per_split_target/1",
                "max_candidates_per_group": max_candidates_per_group,
            }
        )
    corpus_payload["corpus_sha256"] = canonical_sha256(
        corpus_payload,
        domain="signlab-public-corpus/1",
    )

    group_count = len(groups)
    selected_count = len(selected)
    summary: dict[str, object] = {
        "format": "signlab-public-corpus-summary/1",
        "corpus_sha256": corpus_payload["corpus_sha256"],
        "external_dataset_sha256": manifest.content_sha256,
        "archive_byte_integrity": external_validation.archive_byte_integrity,
        "candidate_count": len(manifest.media),
        "group_count": group_count,
        "selected_count": selected_count,
        "unfilled_group_count": (
            shortfall_group_count if trainable_smoke else group_count - selected_count
        ),
        "selected_signer_count": len({item["source_signer_id"] for item in selected}),
        "selected_by_split": _ordered_counts(selected_by_split),
        "selected_by_target": _ordered_counts(selected_by_target),
        "selected_by_quality_disposition": _ordered_counts(selected_by_disposition),
        "exclusions": _ordered_counts(exclusions),
        "extraction_config_sha256": config_sha256,
        "hand_model_sha256": assets.hand_model_sha256,
        "pose_model_sha256": assets.pose_model_sha256,
        "quality_policy_sha256": policy_sha256,
        "feature_plan_id": feature_plan.plan_id,
        "feature_plan_sha256": feature_plan_sha256,
        "source_orientation_basis": _SOURCE_ORIENTATION_BASIS,
        "attribution": source.license.attribution_text,
        "license_id": source.license.license_id,
        "license_url": source.license.license_url,
        "limitation": (
            "Public isolated-sign data only; no participant, continuous-sign, or natural-use claim."
        ),
    }
    if trainable_smoke:
        summary.update(
            {
                "selection_mode": "trainable_smoke",
                "selection_rule": "signer_interleaved_fair_split_target_quota/1",
                "attempt_limit": _TRAINABLE_ATTEMPT_LIMIT,
                "attempted_count": attempted_count,
                "target_count": _TRAINABLE_TARGET_COUNT,
                "decision": decision,
                "selection_groups": trainable_group_results,
                "metric_claim": "none",
            }
        )
    summary["summary_sha256"] = canonical_sha256(
        summary,
        domain="signlab-public-corpus-summary/1",
    )

    _write_bytes(destination / _SUMMARY_JSON_FILENAME, canonical_json_bytes(summary) + b"\n")
    _write_bytes(
        destination / _SUMMARY_MARKDOWN_FILENAME,
        _summary_markdown(summary).encode("utf-8"),
    )
    _write_bytes(destination / _CORPUS_FILENAME, canonical_json_bytes(corpus_payload) + b"\n")
    return PublicCorpusBuildResult(
        corpus_sha256=str(corpus_payload["corpus_sha256"]),
        selected_count=selected_count,
        group_count=group_count,
        exclusion_count=sum(exclusions.values()),
        attempted_count=attempted_count,
        target_count=target_count,
        attempt_limit=attempt_limit,
        decision=decision,
        summary=summary,
    )


__all__ = [
    "PublicCorpusBuildResult",
    "PublicCorpusError",
    "build_public_corpus",
]
