from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest

from feature_fixtures import make_feature_fixture
from signlab.contracts.canonical import canonical_json_bytes, canonical_sha256
from signlab.contracts.core import WorkspaceRelativeLocatorV1
from signlab.contracts.external_dataset import (
    ExternalMediaRecordV1,
    ExternalSplit,
    ExternalTargetLabel,
)
from signlab.contracts.extraction import (
    LandmarkFramesTableV1,
    landmark_frames_table_digest,
    mediapipe_extraction_config_digest,
)
from signlab.contracts.features import landmark_feature_plan_digest
from signlab.contracts.quality import landmark_quality_policy_digest
from signlab.datasets import public_split
from signlab.extraction.parquet import write_landmark_frames
from signlab.extraction.resources import load_packaged_default_extraction_config
from signlab.features.resources import load_packaged_default_feature_plan
from signlab.quality.resources import load_packaged_default_quality_policy

_TARGETS: tuple[ExternalTargetLabel, ...] = ("hello", "no", "please", "thank_you", "yes")
_QUOTAS: dict[ExternalSplit, int] = {"train": 10, "val": 3, "test": 3}


def _candidate(
    index: int,
    *,
    source_split: ExternalSplit,
    target: ExternalTargetLabel,
    signer: int,
    disposition: Literal["pass", "warning"] = "pass",
) -> public_split.PublicSelectionCandidate:
    return public_split.PublicSelectionCandidate(
        sample_id=f"sample_{index:032x}",
        source_recording_id=f"recording_{index:032x}",
        source_signer_id=f"participant_{signer:032x}",
        source_split=source_split,
        target_label_id=target,
        quality_disposition=disposition,
    )


def test_offline_selector_is_exact_order_independent_and_quality_rank_blind() -> None:
    candidates: list[public_split.PublicSelectionCandidate] = []
    index = 1
    warning_id = ""
    for split_index, (source_split, quota) in enumerate(_QUOTAS.items()):
        for target in _TARGETS:
            for position in range(quota):
                disposition: Literal["pass", "warning"] = "warning" if index == 1 else "pass"
                candidate = _candidate(
                    index,
                    source_split=source_split,
                    target=target,
                    signer=split_index * 100 + position,
                    disposition=disposition,
                )
                if disposition == "warning":
                    warning_id = candidate.sample_id
                candidates.append(candidate)
                index += 1
                if position == 0:
                    candidates.append(
                        _candidate(
                            index,
                            source_split=source_split,
                            target=target,
                            signer=split_index * 100,
                        )
                    )
                    index += 1

    first = public_split._select_candidates(candidates)
    second = public_split._select_candidates(tuple(reversed(candidates)))

    assert first == second
    assert len(first) == 80
    assert warning_id in {row.sample_id for row in first}
    assert Counter(row.source_split for row in first) == {"train": 50, "val": 15, "test": 15}
    assert Counter(row.target_label_id for row in first) == {target: 16 for target in _TARGETS}
    assert Counter((row.source_split, row.target_label_id) for row in first) == {
        (source_split, target): quota
        for source_split, quota in _QUOTAS.items()
        for target in _TARGETS
    }


def _with_recording_id(table: LandmarkFramesTableV1, recording_id: str) -> LandmarkFramesTableV1:
    payload = table.model_dump(mode="json", round_trip=True)
    for row in payload["rows"]:
        row["source_recording_id"] = recording_id
    return LandmarkFramesTableV1.model_validate_json(canonical_json_bytes(payload), strict=True)


def _synthetic_source(source: Path) -> tuple[tuple[ExternalMediaRecordV1, ...], str]:
    base = make_feature_fixture().table
    stored: list[tuple[Path, dict[str, object]]] = []
    media: list[ExternalMediaRecordV1] = []
    index = 1
    split_offset: dict[ExternalSplit, int] = {"train": 100, "val": 200, "test": 300}
    for source_split, quota in _QUOTAS.items():
        for target in _TARGETS:
            for position in range(quota):
                recording_id = f"recording_{index:032x}"
                table = _with_recording_id(base, recording_id)
                content_sha256 = landmark_frames_table_digest(table)
                value = content_sha256.removeprefix("sha256:")
                path = source / "landmarks" / "sha256" / value[:2] / f"{value}.parquet"
                written = write_landmark_frames(table, path)
                stored.append(
                    (
                        path,
                        {
                            "content_sha256": content_sha256,
                            "parquet_sha256": written.sha256,
                            "row_count": written.row_count,
                            "size_bytes": written.size_bytes,
                        },
                    )
                )
                media_bytes = f"media-{index}".encode()
                media_sha256 = f"sha256:{hashlib.sha256(media_bytes).hexdigest()}"
                media_value = media_sha256.removeprefix("sha256:")
                media.append(
                    ExternalMediaRecordV1(
                        schema_version="external-media-record/1",
                        sample_id=f"sample_{index:032x}",
                        recording_id=recording_id,
                        participant_id=(
                            f"participant_{split_offset[source_split] + position:032x}"
                        ),
                        archive_id=f"fixture_{source_split}_{target}",
                        source_member_fingerprint=(
                            f"sha256:{hashlib.sha256(f'fingerprint-{index}'.encode()).hexdigest()}"
                        ),
                        category="game",
                        source_split=source_split,
                        source_label=target.replace("_", "-"),
                        target_label_id=target,
                        media_type="video/mp4",
                        sha256=media_sha256,
                        size_bytes=len(media_bytes),
                        locator=WorkspaceRelativeLocatorV1(
                            kind="workspace_relative",
                            path=f"media/sha256/{media_value[:2]}/{media_value}.mp4",
                        ),
                        eligible_for_extraction=True,
                    )
                )
                index += 1
    items = [item for _path, item in sorted(stored, key=lambda pair: pair[0])]
    inventory_sha256 = canonical_sha256(
        {"items": items},
        domain="signlab.popsign.quality-diagnostic.landmark-inventory/1",
    )
    return tuple(media), inventory_sha256


def test_freeze_is_deterministic_reconciled_and_rejects_contamination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    media, inventory_sha256 = _synthetic_source(source)
    external_sha256 = "sha256:" + "d" * 64
    config_sha256 = mediapipe_extraction_config_digest(load_packaged_default_extraction_config())
    policy_sha256 = landmark_quality_policy_digest(load_packaged_default_quality_policy())
    feature_sha256 = landmark_feature_plan_digest(load_packaged_default_feature_plan("combined"))
    monkeypatch.setattr(public_split, "_ATTEMPT_COUNT", 80)
    monkeypatch.setattr(public_split, "_ATTEMPT_INVENTORY_SHA256", inventory_sha256)
    monkeypatch.setattr(
        public_split,
        "_EXPECTED_REPLAY",
        {"no_window": 0, "pass": 80, "quarantine": 0, "warning": 0},
    )
    monkeypatch.setattr(
        public_split,
        "_source_corpus",
        lambda _path: {
            "external_dataset_sha256": external_sha256,
            "extraction_config_sha256": config_sha256,
            "quality_policy_sha256": policy_sha256,
            "feature_plan_sha256": feature_sha256,
        },
    )
    monkeypatch.setattr(
        public_split,
        "validate_external_dataset_manifest",
        lambda _document: SimpleNamespace(content_sha256=external_sha256, media=media),
    )
    manifest = tmp_path / "external.json"
    manifest.write_text("{}", encoding="utf-8")
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first = public_split.freeze_public_corpus_split(
        manifest, source_root=source, output_root=first_root
    )
    second = public_split.freeze_public_corpus_split(
        manifest, source_root=source, output_root=second_root
    )

    first_document = (first_root / "public-corpus-split.json").read_bytes()
    assert first.split_sha256 == second.split_sha256
    assert first_document == (second_root / "public-corpus-split.json").read_bytes()
    assert (first_root / "public-corpus-split-summary.md").read_bytes() == (
        second_root / "public-corpus-split-summary.md"
    ).read_bytes()
    samples = public_split.reconcile_public_corpus_split(
        first_document,
        external_manifest_document=b"{}",
        corpus_root=first_root,
    )
    assert Counter(row.partition for row in samples) == {
        "train": 50,
        "validation": 15,
        "test": 15,
    }
    opened_paths: list[str] = []
    resolve_artifact = public_split._resolve_artifact

    def record_artifact(root: Path, path: str) -> Path:
        opened_paths.append(path)
        return resolve_artifact(root, path)

    monkeypatch.setattr(public_split, "_resolve_artifact", record_artifact)
    development_samples = public_split.reconcile_public_corpus_split(
        first_document,
        external_manifest_document=b"{}",
        corpus_root=first_root,
        partitions=("train", "validation"),
    )
    assert len(development_samples) == 65
    assert opened_paths == [
        row["feature_path"]
        for row in json.loads(first_document)["assignments"]
        if row["partition"] != "test"
    ]
    with pytest.raises(public_split.PublicCorpusSplitError, match="partitions are invalid"):
        public_split.reconcile_public_corpus_split(
            first_document,
            external_manifest_document=b"{}",
            corpus_root=first_root,
            partitions=(),
        )

    original = json.loads(first_document)

    replay_drift = deepcopy(original)
    replay_drift["replay"]["usable_count"] = 79
    replay_drift["split_sha256"] = public_split.public_corpus_split_digest(replay_drift)

    partition_drift = deepcopy(original)
    partition_drift["assignments"][0]["partition"] = "validation"
    partition_drift["split_sha256"] = public_split.public_corpus_split_digest(partition_drift)

    order_drift = deepcopy(original)
    order_drift["assignments"][0], order_drift["assignments"][1] = (
        order_drift["assignments"][1],
        order_drift["assignments"][0],
    )
    order_drift["split_sha256"] = public_split.public_corpus_split_digest(order_drift)

    quota_drift = deepcopy(original)
    moved = quota_drift["assignments"].pop(0)
    moved["target_label_id"] = "no"
    quota_drift["assignments"].insert(9, moved)
    quota_drift["split_sha256"] = public_split.public_corpus_split_digest(quota_drift)

    signer_drift = deepcopy(original)
    signer_drift["assignments"][1]["source_signer_id"] = signer_drift["assignments"][0][
        "source_signer_id"
    ]
    signer_drift["split_sha256"] = public_split.public_corpus_split_digest(signer_drift)

    digest_drift = deepcopy(original)
    digest_drift["split_sha256"] = "sha256:" + "0" * 64

    for drifted in (
        replay_drift,
        partition_drift,
        order_drift,
        quota_drift,
        signer_drift,
        digest_drift,
    ):
        with pytest.raises(public_split.PublicCorpusSplitError):
            public_split.validate_public_corpus_split(drifted)

    with pytest.raises(public_split.PublicCorpusSplitError):
        public_split._select_candidates(
            [_candidate(1, source_split="train", target="hello", signer=1)]
        )
    one_candidate_per_group = [
        _candidate(index, source_split=source_split, target=target, signer=index)
        for index, (source_split, target) in enumerate(
            ((source_split, target) for source_split in _QUOTAS for target in _TARGETS),
            start=1,
        )
    ]
    with pytest.raises(public_split.PublicCorpusSplitError):
        public_split._select_candidates(one_candidate_per_group)

    with pytest.raises(public_split.PublicCorpusSplitError):
        public_split._resolve_artifact(tmp_path, "missing.json")
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(public_split.PublicCorpusSplitError):
        public_split._resolve_artifact(tmp_path, directory.name)

    train = next(row for row in original["assignments"] if row["partition"] == "train")
    validation_index = next(
        index
        for index, row in enumerate(original["assignments"])
        if row["partition"] == "validation"
    )
    for field in ("source_signer_id", "source_recording_id", "feature_sequence_sha256"):
        contaminated = deepcopy(original)
        row = contaminated["assignments"][validation_index]
        row[field] = train[field]
        if field == "feature_sequence_sha256":
            row["feature_path"] = train["feature_path"]
        contaminated["split_sha256"] = public_split.public_corpus_split_digest(contaminated)
        with pytest.raises(public_split.PublicCorpusSplitError):
            public_split.validate_public_corpus_split(contaminated)

    metadata_drift = deepcopy(original)
    left = metadata_drift["assignments"][0]
    right = metadata_drift["assignments"][validation_index]
    for field in ("source_recording_id", "feature_sequence_sha256", "feature_path"):
        left[field], right[field] = right[field], left[field]
    metadata_drift["split_sha256"] = public_split.public_corpus_split_digest(metadata_drift)
    public_split.validate_public_corpus_split(metadata_drift)
    with pytest.raises(public_split.PublicCorpusSplitError):
        public_split.reconcile_public_corpus_split(
            metadata_drift,
            external_manifest_document=b"{}",
            corpus_root=first_root,
        )

    lineage_drift = deepcopy(original)
    lineage_drift["extraction_config_sha256"] = "sha256:" + "c" * 64
    lineage_drift["split_sha256"] = public_split.public_corpus_split_digest(lineage_drift)
    public_split.validate_public_corpus_split(lineage_drift)
    with pytest.raises(public_split.PublicCorpusSplitError):
        public_split.reconcile_public_corpus_split(
            lineage_drift,
            external_manifest_document=b"{}",
            corpus_root=first_root,
        )

    unsafe_path = deepcopy(original)
    unsafe_path["assignments"][0]["feature_path"] = "../escaped.json"
    unsafe_path["split_sha256"] = public_split.public_corpus_split_digest(unsafe_path)
    with pytest.raises(public_split.PublicCorpusSplitError):
        public_split.validate_public_corpus_split(unsafe_path)

    feature_path = first_root.joinpath(*original["assignments"][0]["feature_path"].split("/"))
    feature_path.write_bytes(feature_path.read_bytes() + b"tampered")
    with pytest.raises(public_split.PublicCorpusSplitError):
        public_split.reconcile_public_corpus_split(
            first_document,
            external_manifest_document=b"{}",
            corpus_root=first_root,
        )

    source_parquet = next((source / "landmarks").rglob("*.parquet"))
    source_parquet.write_bytes(source_parquet.read_bytes() + b"tampered")
    rejected_output = tmp_path / "rejected"
    with pytest.raises(public_split.PublicCorpusSplitError):
        public_split.freeze_public_corpus_split(
            manifest, source_root=source, output_root=rejected_output
        )
    assert not rejected_output.exists()
