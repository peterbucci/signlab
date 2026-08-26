from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel, ValidationError

from signlab.contracts.core import (
    CURRENT_CONTRACT_SCHEMAS,
    SUPPORTED_CONTRACT_REFERENCE_SCHEMAS,
    ArtifactRefV1,
    ArtifactUriLocatorV1,
)
from signlab.contracts.dataset import (
    DATASET_TABLE_MODELS,
    DATASET_TABLE_SCHEMA_VERSIONS,
    AnnotationRowV1,
    AnnotationsTableV1,
    ClipRowV1,
    ClipsTableV1,
    DatasetContentV2,
    DatasetContractError,
    DatasetManifestV2,
    DatasetSampleIdentityV1,
    DatasetTable,
    DatasetTableRefV1,
    DatasetTableSetV1,
    DerivedArtifactRowV1,
    DerivedArtifactsTableV1,
    MediaIntervalV1,
    ParticipantRowV1,
    ParticipantsTableV1,
    RecordingRowV1,
    RecordingsTableV1,
    SessionRowV1,
    SessionsTableV1,
    TableName,
    dataset_content_digest,
    dataset_table_digest,
    validate_dataset_table,
)
from signlab.contracts.pipeline import (
    ContractVersionError,
    PipelineContractError,
    assert_split_compatible,
    contract_digest,
    contract_reference,
    validate_contract,
    validate_dataset_manifest,
    validate_dataset_manifest_v1,
    validate_dataset_manifest_v2,
)
from signlab.contracts.resources import (
    PUBLISHED_EXAMPLE_CONTRACT_DIGESTS,
    build_example_contract_chain,
    generated_contract_schemas,
)
from signlab.contracts.taxonomy import load_builtin_taxonomy, taxonomy_reference
from signlab.governance.resources import (
    build_example_inventory,
    build_example_recording_grant,
    build_governance_policy,
)

PARTICIPANT_ID = "participant_00000000000000000000000000000001"
SESSION_ID = "session_00000000000000000000000000000001"
DEVICE_ID = "device_00000000000000000000000000000001"
RECORDING_ID = "recording_00000000000000000000000000000031"
CLIP_ID = "clip_00000000000000000000000000000001"
ANNOTATION_ID = "annotation_00000000000000000000000000000001"
DERIVED_ID = "derived_artifact_00000000000000000000000000000001"
SAMPLE_ID = "sample_00000000000000000000000000000001"


def _sha(label: str) -> str:
    return "sha256:" + hashlib.sha256(f"synthetic dataset fixture: {label}\n".encode()).hexdigest()


def _artifact(
    artifact_id: str,
    role: str,
    media_type: str,
) -> ArtifactRefV1:
    sha256 = _sha(artifact_id)
    if role == "dataset_table":
        table_name = artifact_id.removesuffix("_table")
        locator = ArtifactUriLocatorV1(
            kind="artifact_uri",
            uri=f"signlab://tables/{table_name}",
        )
    else:
        digest = sha256.removeprefix("sha256:")
        locator = ArtifactUriLocatorV1(
            kind="artifact_uri",
            uri=(f"signlab://objects/sha256/p-{digest[:2]}/sha256-{digest}/{artifact_id}"),
        )
    return ArtifactRefV1(
        schema_version="artifact-reference/1",
        artifact_id=artifact_id,
        role=role,
        media_type=media_type,
        sha256=sha256,
        size_bytes=1000 + len(artifact_id),
        locator=locator,
    )


def _tables() -> tuple[DatasetTable, ...]:
    participant = ParticipantRowV1(participant_id=PARTICIPANT_ID, handedness="right")
    session = SessionRowV1(
        session_id=SESSION_ID,
        participant_id=PARTICIPANT_ID,
        device_id=DEVICE_ID,
        started_at="2026-08-26T12:09:00Z",
        finished_at="2026-08-26T12:20:00Z",
        capture_mode="isolated",
        capture_software_version="1.0.0",
        camera_facing="front",
        frame_width_px=1280,
        frame_height_px=720,
        frame_rate_numerator=30,
        frame_rate_denominator=1,
        rotation_degrees=0,
        mirror_state="mirrored",
    )
    recording = RecordingRowV1(
        recording_id=RECORDING_ID,
        participant_id=PARTICIPANT_ID,
        session_id=SESSION_ID,
        device_id=DEVICE_ID,
        captured_at="2026-08-26T12:10:00Z",
        duration_us=5_000_000,
        handedness="right",
        mirror_state="mirrored",
        rotation_degrees=0,
        audio_present=False,
        media=_artifact(RECORDING_ID, "raw_recording", "video/mp4"),
        consent_grant=build_example_recording_grant(),
    )
    clip = ClipRowV1(
        clip_id=CLIP_ID,
        participant_id=PARTICIPANT_ID,
        session_id=SESSION_ID,
        source_recording_id=RECORDING_ID,
        interval=MediaIntervalV1(
            schema_version="media-interval/1",
            start_us=500_000,
            end_us=2_500_000,
        ),
        handedness="right",
        mirror_state="mirrored",
        artifact=None,
    )
    annotation = AnnotationRowV1(
        annotation_id=ANNOTATION_ID,
        participant_id=PARTICIPANT_ID,
        session_id=SESSION_ID,
        source_recording_id=RECORDING_ID,
        clip_id=CLIP_ID,
        interval=MediaIntervalV1(
            schema_version="media-interval/1",
            start_us=600_000,
            end_us=2_400_000,
        ),
        disposition="class_label",
        label_id="hello",
        other_kind=None,
        reason_code=None,
        review_status="reviewed",
        eligible_for_training=True,
    )
    derived = DerivedArtifactRowV1(
        derived_artifact_id=DERIVED_ID,
        derivation_kind="crop",
        parent_artifact_ids=(RECORDING_ID,),
        participant_id=PARTICIPANT_ID,
        session_id=SESSION_ID,
        source_recording_id=RECORDING_ID,
        clip_id=CLIP_ID,
        annotation_id=ANNOTATION_ID,
        sample_id=SAMPLE_ID,
        label_id="hello",
        split_id="synthetic_grouped_split",
        partition="train",
        handedness="right",
        mirror_state="mirrored",
        operation_id="crop_interval",
        operation_version="1.0.0",
        artifact=_artifact(
            SAMPLE_ID,
            "sample_data",
            "application/vnd.signlab.landmarks+json",
        ),
    )
    return (
        ParticipantsTableV1(schema_version="participants-table/1", rows=(participant,)),
        SessionsTableV1(schema_version="sessions-table/1", rows=(session,)),
        RecordingsTableV1(schema_version="recordings-table/1", rows=(recording,)),
        ClipsTableV1(schema_version="clips-table/1", rows=(clip,)),
        AnnotationsTableV1(schema_version="annotations-table/1", rows=(annotation,)),
        DerivedArtifactsTableV1(
            schema_version="derived-artifacts-table/1",
            rows=(derived,),
        ),
    )


def _table_set(
    tables: tuple[DatasetTable, ...],
    *,
    derived_row_count: int | None = None,
) -> DatasetTableSetV1:
    references: dict[str, DatasetTableRefV1] = {}
    for table in tables:
        table_name = table.schema_version.removesuffix("-table/1")
        if table_name == "derived-artifacts":
            table_name = "derived_artifacts"
        row_count = len(table.rows)
        if table_name == "derived_artifacts" and derived_row_count is not None:
            row_count = derived_row_count
        references[table_name] = DatasetTableRefV1(
            schema_version="dataset-table-reference/1",
            table_name=cast(TableName, table_name),
            table_schema_version=table.schema_version,
            row_count=row_count,
            content_sha256=dataset_table_digest(table),
            artifact=_artifact(
                f"{table_name}_table",
                "dataset_table",
                "application/vnd.apache.parquet",
            ),
        )
    return DatasetTableSetV1(
        schema_version="dataset-table-set/1",
        **references,
    )


def _manifest() -> DatasetManifestV2:
    tables = _tables()
    derived = cast(DerivedArtifactsTableV1, tables[-1]).rows[0]
    content = DatasetContentV2(
        schema_version="dataset-content/2",
        taxonomy=taxonomy_reference(load_builtin_taxonomy()),
        governance_policy=build_governance_policy().policy_document,
        lineage_inventory_sha256=build_example_inventory().inventory_sha256,
        sample_schema_version="landmark-sequence/1",
        tables=_table_set(tables),
        samples=(
            DatasetSampleIdentityV1(
                sample_id=cast(str, derived.sample_id),
                participant_id=derived.participant_id,
                session_id=derived.session_id,
                source_recording_id=derived.source_recording_id,
                label_id=cast(Any, derived.label_id),
                artifact=derived.artifact,
            ),
        ),
    )
    return DatasetManifestV2(
        schema_version="dataset-manifest/2",
        dataset_id="synthetic_table_dataset",
        version="1.0.0",
        content=content,
        data_sha256=dataset_content_digest(content),
    )


def _json(model: BaseModel) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(model.model_dump_json(round_trip=True)))


def test_story_13_dataset_v1_schema_and_golden_identity_are_unchanged() -> None:
    dataset = build_example_contract_chain()[0]
    committed = json.loads(
        Path("src/signlab/resources/contracts/schemas/dataset-manifest-1.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert generated_contract_schemas()["dataset-manifest-1.schema.json"] == committed
    assert contract_digest(dataset) == PUBLISHED_EXAMPLE_CONTRACT_DIGESTS["dataset"]
    assert validate_dataset_manifest_v1(dataset) == dataset


def test_six_table_wrappers_round_trip_strictly_with_stable_digests() -> None:
    tables = _tables()

    assert set(DATASET_TABLE_MODELS) == set(DATASET_TABLE_SCHEMA_VERSIONS.values())
    for table in tables:
        encoded = table.model_dump_json(round_trip=True)
        assert validate_dataset_table(encoded) == table
        assert dataset_table_digest(encoded) == dataset_table_digest(table)


def test_table_digest_ignores_artifact_locator_but_binds_payload_bytes() -> None:
    table = cast(RecordingsTableV1, _tables()[2])
    document = _json(table)
    relocated = deepcopy(document)
    digest = table.rows[0].media.sha256.removeprefix("sha256:")
    relocated["rows"][0]["media"]["locator"] = {
        "kind": "workspace_relative",
        "path": f"objects/sha256/p-{digest[:2]}/sha256-{digest}/{RECORDING_ID}",
    }
    changed_bytes = deepcopy(relocated)
    changed_sha256 = _sha("different recording")
    changed_digest = changed_sha256.removeprefix("sha256:")
    changed_bytes["rows"][0]["media"]["sha256"] = changed_sha256
    changed_bytes["rows"][0]["media"]["locator"] = {
        "kind": "workspace_relative",
        "path": (f"objects/sha256/p-{changed_digest[:2]}/sha256-{changed_digest}/{RECORDING_ID}"),
    }

    assert dataset_table_digest(relocated) == dataset_table_digest(table)
    assert dataset_table_digest(changed_bytes) != dataset_table_digest(table)


def test_v2_manifest_round_trips_through_generic_and_exact_readers() -> None:
    manifest = _manifest()
    encoded = manifest.model_dump_json(round_trip=True)

    assert validate_contract(encoded) == manifest
    assert validate_dataset_manifest(encoded) == manifest
    assert validate_dataset_manifest_v2(encoded) == manifest
    assert contract_digest(encoded) == contract_digest(manifest)
    with pytest.raises(ContractVersionError):
        validate_dataset_manifest_v1(encoded)


def test_v2_data_hash_ignores_table_parquet_bytes_and_storage_location() -> None:
    manifest = _manifest()
    content = _json(manifest.content)
    moved = deepcopy(content)
    table = moved["tables"]["recordings"]["artifact"]
    table["locator"] = {
        "kind": "workspace_relative",
        "path": "tables/recordings.parquet",
    }
    table["sha256"] = _sha("reencoded parquet")
    table["size_bytes"] = 9999
    moved_content = DatasetContentV2.model_validate_json(json.dumps(moved), strict=True)

    assert dataset_content_digest(moved_content) == manifest.data_sha256
    moved_manifest = DatasetManifestV2(
        schema_version="dataset-manifest/2",
        dataset_id=manifest.dataset_id,
        version=manifest.version,
        content=moved_content,
        data_sha256=manifest.data_sha256,
    )
    assert contract_digest(moved_manifest) != contract_digest(manifest)


def test_v2_data_hash_binds_table_semantic_identity_and_sample_projection() -> None:
    manifest = _manifest()
    mutations: list[dict[str, object]] = []
    table = _json(manifest.content)
    table["tables"]["annotations"]["content_sha256"] = _sha("different rows")
    mutations.append(table)
    sample = _json(manifest.content)
    sample["samples"][0]["label_id"] = "no"
    mutations.append(sample)

    for mutation in mutations:
        changed = DatasetContentV2.model_validate_json(json.dumps(mutation), strict=True)
        assert dataset_content_digest(changed) != manifest.data_sha256


def test_current_dataset_writer_and_retained_readers_are_explicit() -> None:
    assert CURRENT_CONTRACT_SCHEMAS["dataset"] == "dataset-manifest/2"
    assert SUPPORTED_CONTRACT_REFERENCE_SCHEMAS["dataset"] == frozenset(
        {"dataset-manifest/1", "dataset-manifest/2"}
    )


def test_existing_split_contract_can_reference_a_v2_dataset() -> None:
    v1_dataset, split, *_ = build_example_contract_chain()
    table_set = _table_set(_tables(), derived_row_count=len(v1_dataset.content.samples))
    v2_samples = tuple(
        sample.model_copy(
            update={
                "artifact": sample.artifact.model_copy(
                    update={
                        "locator": ArtifactUriLocatorV1(
                            kind="artifact_uri",
                            uri=(
                                "signlab://objects/sha256/"
                                f"p-{sample.artifact.sha256[7:9]}/"
                                f"{sample.artifact.sha256.replace(':', '-')}/"
                                f"{sample.artifact.artifact_id}"
                            ),
                        )
                    }
                )
            }
        )
        for sample in v1_dataset.content.samples
    )
    content = DatasetContentV2(
        schema_version="dataset-content/2",
        taxonomy=v1_dataset.content.taxonomy,
        governance_policy=v1_dataset.content.governance_policy,
        lineage_inventory_sha256=v1_dataset.content.lineage_inventory_sha256,
        sample_schema_version=v1_dataset.content.sample_schema_version,
        tables=table_set,
        samples=v2_samples,
    )
    v2_dataset = DatasetManifestV2(
        schema_version="dataset-manifest/2",
        dataset_id=v1_dataset.dataset_id,
        version="2.0.0",
        content=content,
        data_sha256=dataset_content_digest(content),
    )
    split_document = _json(split)
    split_document["dataset"] = _json(contract_reference(v2_dataset, split.dataset.locator))
    split_document["dataset_data_sha256"] = v2_dataset.data_sha256
    migrated_split = type(split).model_validate_json(json.dumps(split_document), strict=True)

    assert_split_compatible(v2_dataset, migrated_split)


@pytest.mark.parametrize(
    ("start_us", "end_us"),
    [(0, 0), (2, 1), (-1, 1), (0, 2**53)],
)
def test_media_intervals_reject_empty_reversed_negative_and_unsafe_values(
    start_us: int,
    end_us: int,
) -> None:
    with pytest.raises(ValidationError):
        MediaIntervalV1(
            schema_version="media-interval/1",
            start_us=start_us,
            end_us=end_us,
        )


@pytest.mark.parametrize("bad_label", ["inactive", "abstain", "ambiguous", "thank you"])
def test_annotations_reject_non_taxonomy_classifier_labels(bad_label: str) -> None:
    annotation = _json(cast(AnnotationsTableV1, _tables()[4]).rows[0])
    annotation["label_id"] = bad_label

    with pytest.raises(ValidationError):
        AnnotationRowV1.model_validate(annotation, strict=True)


def test_annotation_disposition_and_training_eligibility_fail_closed() -> None:
    annotation = _json(cast(AnnotationsTableV1, _tables()[4]).rows[0])
    ambiguous = deepcopy(annotation)
    ambiguous.update(
        disposition="ambiguous",
        label_id=None,
        reason_code="uncertain_boundary",
        eligible_for_training=True,
    )
    invalid_other = deepcopy(annotation)
    invalid_other.update(label_id="other", other_kind=None)

    for document in (ambiguous, invalid_other):
        with pytest.raises(ValidationError):
            AnnotationRowV1.model_validate(document, strict=True)


def test_recording_rejects_consent_grouping_and_time_mismatches() -> None:
    recording = _json(cast(RecordingsTableV1, _tables()[2]).rows[0])
    for field, value in (
        ("participant_id", "participant_ffffffffffffffffffffffffffffffff"),
        ("captured_at", "2026-08-26T12:11:00Z"),
    ):
        mutated = deepcopy(recording)
        mutated[field] = value
        with pytest.raises(ValidationError):
            RecordingRowV1.model_validate(mutated, strict=True)


@pytest.mark.parametrize(
    "locator",
    [
        {
            "kind": "workspace_relative",
            "path": "data/recordings/john_smith.mp4",
        },
        {
            "kind": "workspace_relative",
            "path": f"data/recordings/{RECORDING_ID}.MP4",
        },
        {
            "kind": "artifact_uri",
            "uri": "signlab://synthetic/recordings/private_person_recording",
        },
    ],
    ids=["pii-filename", "non-lowercase-extension", "non-opaque-uri"],
)
def test_recording_rejects_nonopaque_media_locators(locator: dict[str, str]) -> None:
    recording = _json(cast(RecordingsTableV1, _tables()[2]).rows[0])
    recording["media"]["locator"] = locator

    with pytest.raises(ValidationError):
        RecordingRowV1.model_validate(recording, strict=True)


def test_every_v2_row_artifact_uses_the_exact_content_addressed_layout() -> None:
    tables = _tables()
    unsafe_path = {
        "kind": "workspace_relative",
        "path": "objects/private_person/artifact",
    }
    clip_payload = _json(cast(ClipsTableV1, tables[3]).rows[0])
    clip_payload["artifact"] = _json(_artifact(CLIP_ID, "clip_media", "video/mp4"))
    row_cases: tuple[tuple[dict[str, Any], type[BaseModel], str], ...] = (
        (_json(cast(RecordingsTableV1, tables[2]).rows[0]), RecordingRowV1, "media"),
        (clip_payload, ClipRowV1, "artifact"),
        (
            _json(cast(DerivedArtifactsTableV1, tables[5]).rows[0]),
            DerivedArtifactRowV1,
            "artifact",
        ),
    )
    for payload, model, field_name in row_cases:
        payload[field_name]["locator"] = unsafe_path
        with pytest.raises(ValidationError, match="canonical content address"):
            model.model_validate_json(json.dumps(payload), strict=True)

    manifest_content = _json(_manifest().content)
    manifest_content["samples"][0]["artifact"]["locator"] = unsafe_path
    with pytest.raises(ValidationError, match="canonical content address"):
        DatasetContentV2.model_validate_json(json.dumps(manifest_content), strict=True)


def test_retained_v1_sample_contract_does_not_gain_the_v2_locator_policy() -> None:
    sample = _manifest().content.samples[0]
    payload = _json(sample)
    payload["artifact"]["locator"] = {
        "kind": "artifact_uri",
        "uri": f"signlab://legacy/samples/{sample.sample_id}",
    }

    assert (
        DatasetSampleIdentityV1.model_validate_json(json.dumps(payload), strict=True).sample_id
        == sample.sample_id
    )


def test_derived_samples_require_complete_source_and_split_inheritance() -> None:
    derived = _json(cast(DerivedArtifactsTableV1, _tables()[-1]).rows[0])
    for missing in ("sample_id", "label_id", "split_id", "partition", "clip_id", "annotation_id"):
        mutated = deepcopy(derived)
        mutated[missing] = None
        with pytest.raises(ValidationError):
            DerivedArtifactRowV1.model_validate(mutated, strict=True)


def test_table_rows_and_derived_parents_must_be_sorted_and_unique() -> None:
    participant = cast(ParticipantsTableV1, _tables()[0]).rows[0]
    with pytest.raises(ValidationError, match="unique and sorted"):
        ParticipantsTableV1(
            schema_version="participants-table/1",
            rows=(participant, participant),
        )

    derived = _json(cast(DerivedArtifactsTableV1, _tables()[-1]).rows[0])
    derived["parent_artifact_ids"] = ("parent_b", "parent_a")
    with pytest.raises(ValidationError, match="unique and sorted"):
        DerivedArtifactRowV1.model_validate(derived, strict=True)


def test_table_set_requires_exact_named_nonempty_unique_parquet_references() -> None:
    table_set = _json(_table_set(_tables()))
    wrong_name = deepcopy(table_set)
    wrong_name["participants"]["table_name"] = "sessions"
    empty_recordings = deepcopy(table_set)
    empty_recordings["recordings"]["row_count"] = 0
    duplicate_artifact = deepcopy(table_set)
    duplicate_artifact["sessions"]["artifact"]["artifact_id"] = duplicate_artifact["participants"][
        "artifact"
    ]["artifact_id"]
    noncanonical_locator = deepcopy(table_set)
    noncanonical_locator["recordings"]["artifact"]["locator"] = {
        "kind": "workspace_relative",
        "path": "private_person/recordings.parquet",
    }

    for document in (wrong_name, empty_recordings, duplicate_artifact, noncanonical_locator):
        with pytest.raises(ValidationError):
            DatasetTableSetV1.model_validate(document, strict=True)


def test_dataset_table_reader_rejects_unknown_versions_without_migration() -> None:
    document = _json(_tables()[0])
    document["schema_version"] = "participants-table/2"

    with pytest.raises(DatasetContractError, match="unsupported"):
        validate_dataset_table(document)
    with pytest.raises(PipelineContractError):
        validate_contract({"schema_version": "dataset-manifest/3"})
