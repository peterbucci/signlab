from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from typing import Any, cast

import pytest

from signlab.contracts.core import (
    ArtifactRefV1,
    WorkspaceRelativeLocatorV1,
)
from signlab.contracts.dataset import (
    AnnotationRowV1,
    AnnotationsTableV1,
    ClipRowV1,
    ClipsTableV1,
    DatasetContentV2,
    DatasetManifestV2,
    DatasetSampleIdentityV1,
    DatasetTable,
    DatasetTableInput,
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
)
from signlab.contracts.governance import (
    PACKAGED_GOVERNANCE_DOCUMENTS,
    ConsentEventLogV1,
    ConsentEventV1,
    ConsentReceiptV1,
    ConsentScopeV1,
    DocumentRef,
    RecordingConsentGrantV1,
    consent_event_log_digest,
    consent_receipt_digest,
    consent_scope_digest,
)
from signlab.contracts.pipeline import (
    SplitManifestV1,
    SplitPartitionV1,
    contract_reference,
)
from signlab.contracts.taxonomy import (
    TaxonomyRef,
    load_builtin_taxonomy,
    taxonomy_reference,
)
from signlab.datasets import validation as dataset_validation
from signlab.datasets.validation import (
    DatasetValidationError,
    validate_dataset_manifest_tables,
)


@dataclass(frozen=True)
class DatasetBundle:
    manifest: DatasetManifestV2
    tables: dict[str, DatasetTableInput]
    split: SplitManifestV1
    consent_evidence: dict[str, tuple[ConsentReceiptV1, ConsentEventLogV1]]


def _identifier(prefix: str, number: int) -> str:
    return f"{prefix}_{number:032x}"


def _sha(number: int) -> str:
    return f"sha256:{number:064x}"


def _locator(path: str) -> WorkspaceRelativeLocatorV1:
    return WorkspaceRelativeLocatorV1(kind="workspace_relative", path=path)


def _artifact(
    artifact_id: str,
    *,
    role: str,
    media_type: str,
    number: int,
    path: str,
) -> ArtifactRefV1:
    sha256 = _sha(number)
    if role == "dataset_table":
        table_name = path.rsplit("/", maxsplit=1)[-1].removesuffix(".parquet")
        canonical_path = f"tables/{table_name}.parquet"
    else:
        digest = sha256.removeprefix("sha256:")
        canonical_path = f"objects/sha256/p-{digest[:2]}/sha256-{digest}/{artifact_id}"
    return ArtifactRefV1(
        schema_version="artifact-reference/1",
        artifact_id=artifact_id,
        role=role,
        media_type=media_type,
        sha256=sha256,
        size_bytes=number + 100,
        locator=_locator(canonical_path),
    )


def _document(document_type: str) -> DocumentRef:
    document_id, version, effective_at, uri, sha256 = PACKAGED_GOVERNANCE_DOCUMENTS[document_type]
    return DocumentRef(
        schema_version="document-reference/1",
        document_id=document_id,
        document_type=cast(Any, document_type),
        version=version,
        effective_at=effective_at,
        uri=uri,
        sha256=sha256,
    )


def _scope(**changes: object) -> ConsentScopeV1:
    values: dict[str, object] = {
        "schema_version": "consent-scope/1",
        "research_use": True,
        "raw_media_capture": True,
        "model_training": True,
        "model_evaluation": True,
        "public_demonstration": False,
        "model_weights_redistribution": False,
        "raw_media_retention": True,
        "raw_media_redistribution": False,
        "derived_features": True,
        "derived_features_redistribution": False,
        "evaluation_results_redistribution": False,
        "same_purpose_future_research": False,
        "withdrawal_supported": True,
        "audio_collection": False,
        "minor_participation": False,
        "identity_inference": False,
        "commercial_sale": False,
    }
    values.update(changes)
    return ConsentScopeV1.model_validate(values, strict=True)


def _consent_evidence(
    number: int,
    taxonomy: TaxonomyRef,
    scope: ConsentScopeV1,
) -> tuple[ConsentReceiptV1, ConsentEventLogV1]:
    participant_id = _identifier("participant", number)
    receipt = ConsentReceiptV1(
        schema_version="consent-receipt/1",
        receipt_id=_identifier("receipt", number),
        participant_id=participant_id,
        purpose_id=_identifier("purpose", number),
        study_id=_identifier("study", number),
        consent_form=_document("consent_form"),
        privacy_notice=_document("privacy_notice"),
        governance_policy=_document("governance_policy"),
        taxonomy=taxonomy,
        scope=scope,
        scope_sha256=consent_scope_digest(scope),
        granted_at="2026-08-26T00:00:01Z",
        valid_until="2027-08-26T00:00:01Z",
        completed_form_sha256=_sha(1000 + number),
        identity_vault_attestation_sha256=_sha(1100 + number),
        adult_attested=True,
    )
    event = ConsentEventV1(
        schema_version="consent-event/1",
        event_id=_identifier("event", number),
        receipt_id=receipt.receipt_id,
        participant_id=participant_id,
        event_type="granted",
        occurred_at=receipt.granted_at,
        scope_sha256=receipt.scope_sha256,
        reason_code=None,
        replacement_receipt_id=None,
    )
    event_log_values: dict[str, object] = {
        "schema_version": "consent-event-log/1",
        "event_log_id": _identifier("event_log", number),
        "receipt_id": receipt.receipt_id,
        "receipt_sha256": consent_receipt_digest(receipt),
        "participant_id": participant_id,
        "purpose_id": receipt.purpose_id,
        "study_id": receipt.study_id,
        "scope_sha256": receipt.scope_sha256,
        "generated_at": "2026-08-26T00:01:30Z",
        "complete_through": "2026-08-26T00:01:00Z",
        "completeness_attested": True,
        "identity_vault_attestation_sha256": receipt.identity_vault_attestation_sha256,
        "events": (event.model_dump(mode="json", round_trip=True),),
        "event_log_sha256": _sha(0),
    }
    event_log_values["event_log_sha256"] = consent_event_log_digest(event_log_values)
    return receipt, ConsentEventLogV1.model_validate(event_log_values, strict=True)


def _grant(
    number: int,
    taxonomy: TaxonomyRef,
    scope: ConsentScopeV1,
) -> RecordingConsentGrantV1:
    return RecordingConsentGrantV1(
        schema_version="recording-consent-grant/1",
        grant_id=_identifier("grant", number),
        recording_id=_identifier("recording", number),
        participant_id=_identifier("participant", number),
        receipt_id=_identifier("receipt", number),
        purpose_id=_identifier("purpose", number),
        study_id=_identifier("study", number),
        taxonomy=taxonomy,
        scope=scope,
        scope_sha256=consent_scope_digest(scope),
        receipt_scope_sha256=consent_scope_digest(scope),
        issued_at="2026-08-26T00:00:05Z",
        captured_at="2026-08-26T00:00:10Z",
    )


def _table_reference(table_name: TableName, table: DatasetTable, number: int) -> DatasetTableRefV1:
    return DatasetTableRefV1(
        schema_version="dataset-table-reference/1",
        table_name=table_name,
        table_schema_version=cast(Any, table.schema_version),
        row_count=len(table.rows),
        content_sha256=dataset_table_digest(table),
        artifact=_artifact(
            f"{table_name}_table",
            role="dataset_table",
            media_type="application/vnd.apache.parquet",
            number=2000 + number,
            path=f"fixtures/tables/{table_name}.parquet",
        ),
    )


def _build_split(
    manifest: DatasetManifestV2,
    derived: DerivedArtifactsTableV1,
) -> SplitManifestV1:
    rows_by_sample = {row.sample_id: row for row in derived.rows if row.sample_id is not None}
    partitions: list[SplitPartitionV1] = []
    for partition_name in ("train", "validation", "test"):
        sample_ids = tuple(
            sample.sample_id
            for sample in manifest.content.samples
            if rows_by_sample[sample.sample_id].partition == partition_name
        )
        members = tuple(rows_by_sample[sample_id] for sample_id in sample_ids)
        partitions.append(
            SplitPartitionV1(
                name=cast(Any, partition_name),
                sample_ids=sample_ids,
                participant_ids=tuple(sorted({row.participant_id for row in members})),
                session_ids=tuple(sorted({row.session_id for row in members})),
                source_recording_ids=tuple(sorted({row.source_recording_id for row in members})),
            )
        )
    return SplitManifestV1(
        schema_version="split-manifest/1",
        split_id="split_fixture",
        version="1.0.0",
        dataset=contract_reference(
            manifest,
            _locator("fixtures/contracts/dataset-manifest.json"),
        ),
        dataset_data_sha256=manifest.data_sha256,
        strategy="participant-and-session-grouped",
        random_seed=17,
        partitions=cast(Any, tuple(partitions)),
    )


def _bundle() -> DatasetBundle:
    taxonomy = taxonomy_reference(load_builtin_taxonomy())
    scope = _scope()
    participants = ParticipantsTableV1(
        schema_version="participants-table/1",
        rows=tuple(
            ParticipantRowV1(
                participant_id=_identifier("participant", number),
                handedness="left",
            )
            for number in range(1, 4)
        ),
    )
    sessions = SessionsTableV1(
        schema_version="sessions-table/1",
        rows=tuple(
            SessionRowV1(
                session_id=_identifier("session", number),
                participant_id=_identifier("participant", number),
                device_id=_identifier("device", number),
                started_at="2026-08-26T00:00:00Z",
                finished_at="2026-08-26T00:02:00Z",
                capture_mode="isolated",
                capture_software_version="1.0.0",
                camera_facing="front",
                frame_width_px=640,
                frame_height_px=480,
                frame_rate_numerator=30,
                frame_rate_denominator=1,
                rotation_degrees=0,
                mirror_state="not_mirrored",
            )
            for number in range(1, 4)
        ),
    )
    recordings = RecordingsTableV1(
        schema_version="recordings-table/1",
        rows=tuple(
            RecordingRowV1(
                recording_id=_identifier("recording", number),
                participant_id=_identifier("participant", number),
                session_id=_identifier("session", number),
                device_id=_identifier("device", number),
                captured_at="2026-08-26T00:00:10Z",
                duration_us=10_000_000,
                handedness="left",
                mirror_state="not_mirrored",
                rotation_degrees=0,
                audio_present=False,
                media=_artifact(
                    _identifier("recording", number),
                    role="raw_recording",
                    media_type="video/mp4",
                    number=100 + number,
                    path=f"fixtures/raw/{_identifier('recording', number)}.mp4",
                ),
                consent_grant=_grant(number, taxonomy, scope),
            )
            for number in range(1, 4)
        ),
    )
    clips = ClipsTableV1(
        schema_version="clips-table/1",
        rows=tuple(
            ClipRowV1(
                clip_id=_identifier("clip", number),
                participant_id=_identifier("participant", number),
                session_id=_identifier("session", number),
                source_recording_id=_identifier("recording", number),
                interval=MediaIntervalV1(
                    schema_version="media-interval/1",
                    start_us=1_000_000,
                    end_us=3_000_000,
                ),
                handedness="left",
                mirror_state="not_mirrored",
                artifact=_artifact(
                    _identifier("clip", number),
                    role="clip_media",
                    media_type="video/mp4",
                    number=200 + number,
                    path=f"fixtures/clips/{_identifier('clip', number)}.mp4",
                ),
            )
            for number in range(1, 4)
        ),
    )
    labels = ("hello", "no", "other")
    annotations = AnnotationsTableV1(
        schema_version="annotations-table/1",
        rows=tuple(
            AnnotationRowV1(
                annotation_id=_identifier("annotation", number),
                participant_id=_identifier("participant", number),
                session_id=_identifier("session", number),
                source_recording_id=_identifier("recording", number),
                clip_id=_identifier("clip", number),
                interval=MediaIntervalV1(
                    schema_version="media-interval/1",
                    start_us=1_250_000,
                    end_us=2_750_000,
                ),
                disposition="class_label",
                label_id=cast(Any, labels[number - 1]),
                other_kind="oov_gesture" if number == 3 else None,
                reason_code=None,
                review_status="reviewed",
                eligible_for_training=True,
            )
            for number in range(1, 4)
        ),
    )
    partitions = ("train", "validation", "test")
    derived = DerivedArtifactsTableV1(
        schema_version="derived-artifacts-table/1",
        rows=tuple(
            DerivedArtifactRowV1(
                derived_artifact_id=_identifier("derived_artifact", number),
                derivation_kind="crop",
                parent_artifact_ids=(_identifier("clip", number),),
                participant_id=_identifier("participant", number),
                session_id=_identifier("session", number),
                source_recording_id=_identifier("recording", number),
                clip_id=_identifier("clip", number),
                annotation_id=_identifier("annotation", number),
                sample_id=_identifier("sample", number),
                label_id=cast(Any, labels[number - 1]),
                split_id="split_fixture",
                partition=cast(Any, partitions[number - 1]),
                handedness="left",
                mirror_state="not_mirrored",
                operation_id="crop_interval",
                operation_version="1.0.0",
                artifact=_artifact(
                    _identifier("sample", number),
                    role="sample_data",
                    media_type="application/octet-stream",
                    number=300 + number,
                    path=f"fixtures/samples/{_identifier('sample', number)}.bin",
                ),
            )
            for number in range(1, 4)
        ),
    )
    typed_tables: dict[str, DatasetTableInput] = {
        "participants": participants,
        "sessions": sessions,
        "recordings": recordings,
        "clips": clips,
        "annotations": annotations,
        "derived_artifacts": derived,
    }
    table_set = DatasetTableSetV1(
        schema_version="dataset-table-set/1",
        participants=_table_reference("participants", participants, 1),
        sessions=_table_reference("sessions", sessions, 2),
        recordings=_table_reference("recordings", recordings, 3),
        clips=_table_reference("clips", clips, 4),
        annotations=_table_reference("annotations", annotations, 5),
        derived_artifacts=_table_reference("derived_artifacts", derived, 6),
    )
    content = DatasetContentV2(
        schema_version="dataset-content/2",
        taxonomy=taxonomy,
        governance_policy=_document("governance_policy"),
        lineage_inventory_sha256=_sha(5000),
        sample_schema_version="landmark-sequence/1",
        tables=table_set,
        samples=tuple(
            DatasetSampleIdentityV1(
                sample_id=cast(str, row.sample_id),
                participant_id=row.participant_id,
                session_id=row.session_id,
                source_recording_id=row.source_recording_id,
                label_id=cast(Any, row.label_id),
                artifact=row.artifact,
            )
            for row in derived.rows
        ),
    )
    manifest = DatasetManifestV2(
        schema_version="dataset-manifest/2",
        dataset_id="dataset_fixture",
        version="1.0.0",
        content=content,
        data_sha256=dataset_content_digest(content),
    )
    evidence = {
        _identifier("grant", number): _consent_evidence(number, taxonomy, scope)
        for number in range(1, 4)
    }
    return DatasetBundle(
        manifest=manifest,
        tables=typed_tables,
        split=_build_split(manifest, derived),
        consent_evidence=evidence,
    )


def _replace_table(
    bundle: DatasetBundle,
    table_name: TableName,
    table: DatasetTable,
) -> DatasetBundle:
    old_refs = bundle.manifest.content.tables
    new_ref = _table_reference(table_name, table, 20)
    table_set = DatasetTableSetV1.model_validate(
        {
            **old_refs.model_dump(mode="json", round_trip=True),
            table_name: new_ref.model_dump(mode="json", round_trip=True),
        },
        strict=True,
    )
    content = DatasetContentV2(
        schema_version="dataset-content/2",
        taxonomy=bundle.manifest.content.taxonomy,
        governance_policy=bundle.manifest.content.governance_policy,
        lineage_inventory_sha256=bundle.manifest.content.lineage_inventory_sha256,
        sample_schema_version=bundle.manifest.content.sample_schema_version,
        tables=table_set,
        samples=bundle.manifest.content.samples,
    )
    manifest = DatasetManifestV2(
        schema_version="dataset-manifest/2",
        dataset_id=bundle.manifest.dataset_id,
        version=bundle.manifest.version,
        content=content,
        data_sha256=dataset_content_digest(content),
    )
    tables = {**bundle.tables, table_name: table}
    derived = cast(DerivedArtifactsTableV1, tables["derived_artifacts"])
    return replace(
        bundle,
        manifest=manifest,
        tables=tables,
        split=_build_split(manifest, derived),
    )


def _replace_row[TableT: DatasetTable](
    bundle: DatasetBundle,
    table_name: TableName,
    table_type: type[TableT],
    index: int,
    row: object,
) -> DatasetBundle:
    old_table = cast(TableT, bundle.tables[table_name])
    rows = (*old_table.rows[:index], row, *old_table.rows[index + 1 :])
    table = table_type.model_validate(
        {"schema_version": old_table.schema_version, "rows": rows},
        strict=True,
    )
    return _replace_table(bundle, table_name, table)


def test_canonical_tables_validate_with_explicit_unchecked_external_states() -> None:
    bundle = _bundle()

    result = validate_dataset_manifest_tables(bundle.manifest, bundle.tables)

    assert result.semantic_integrity == "verified"
    assert result.artifact_byte_integrity == "not_checked"
    assert result.split_compatibility == "not_checked"
    assert result.consent_authorization == "not_checked"


def test_canonical_tables_validate_exact_split_and_authenticated_consent() -> None:
    bundle = _bundle()

    result = validate_dataset_manifest_tables(
        bundle.manifest,
        bundle.tables,
        split=bundle.split,
        consent_evidence_lookup=lambda grant: bundle.consent_evidence[grant.grant_id],
        consent_authorization_verifier=lambda _receipt, _grant, _event_log: True,
        authorization_permission="model_training",
        authorization_at="2026-08-26T00:01:00Z",
    )

    assert result.split_compatibility == "verified"
    assert result.consent_authorization == "verified"


def test_seeded_lineage_defect_is_rejected_after_digest_rebuild() -> None:
    bundle = _bundle()
    derived = cast(DerivedArtifactsTableV1, bundle.tables["derived_artifacts"])
    row = derived.rows[0].model_copy(update={"parent_artifact_ids": (_identifier("recording", 2),)})
    corrupted = _replace_row(bundle, "derived_artifacts", DerivedArtifactsTableV1, 0, row)

    with pytest.raises(DatasetValidationError) as captured:
        validate_dataset_manifest_tables(corrupted.manifest, corrupted.tables)

    assert captured.value.category == "lineage.invalid"


def test_seeded_consent_scope_defect_is_rejected_after_digest_rebuild() -> None:
    bundle = _bundle()
    recordings = cast(RecordingsTableV1, bundle.tables["recordings"])
    old = recordings.rows[0]
    narrowed_scope = _scope(model_training=False, model_evaluation=False, derived_features=False)
    grant = old.consent_grant.model_copy(
        update={
            "scope": narrowed_scope,
            "scope_sha256": consent_scope_digest(narrowed_scope),
        }
    )
    row = old.model_copy(update={"consent_grant": grant})
    corrupted = _replace_row(bundle, "recordings", RecordingsTableV1, 0, row)

    with pytest.raises(DatasetValidationError) as captured:
        validate_dataset_manifest_tables(corrupted.manifest, corrupted.tables)

    assert captured.value.category == "consent.binding"


def test_seeded_label_defect_is_rejected_after_digest_rebuild() -> None:
    bundle = _bundle()
    derived = cast(DerivedArtifactsTableV1, bundle.tables["derived_artifacts"])
    row = derived.rows[0].model_copy(update={"label_id": "no"})
    corrupted = _replace_row(bundle, "derived_artifacts", DerivedArtifactsTableV1, 0, row)

    with pytest.raises(DatasetValidationError) as captured:
        validate_dataset_manifest_tables(corrupted.manifest, corrupted.tables)

    assert captured.value.category == "label.invalid"


def test_seeded_interval_defect_is_rejected_after_digest_rebuild() -> None:
    bundle = _bundle()
    annotations = cast(AnnotationsTableV1, bundle.tables["annotations"])
    row = annotations.rows[0].model_copy(
        update={
            "interval": MediaIntervalV1(
                schema_version="media-interval/1",
                start_us=500_000,
                end_us=1_500_000,
            )
        }
    )
    corrupted = _replace_row(bundle, "annotations", AnnotationsTableV1, 0, row)

    with pytest.raises(DatasetValidationError) as captured:
        validate_dataset_manifest_tables(corrupted.manifest, corrupted.tables)

    assert captured.value.category == "interval.invalid"


def test_seeded_split_inheritance_defect_is_rejected_against_exact_split() -> None:
    bundle = _bundle()
    derived = cast(DerivedArtifactsTableV1, bundle.tables["derived_artifacts"])
    row = derived.rows[0].model_copy(update={"split_id": "split_other"})
    corrupted = _replace_row(bundle, "derived_artifacts", DerivedArtifactsTableV1, 0, row)

    with pytest.raises(DatasetValidationError) as captured:
        validate_dataset_manifest_tables(
            corrupted.manifest,
            corrupted.tables,
            split=corrupted.split,
        )

    assert captured.value.category == "split.invalid"


@pytest.mark.parametrize(
    ("split_id", "partition"),
    [("split_other", "validation"), ("split_fixture", "train")],
)
def test_derived_children_cannot_cross_parent_split_without_external_manifest(
    split_id: str,
    partition: str,
) -> None:
    bundle = _bundle()
    derived = cast(DerivedArtifactsTableV1, bundle.tables["derived_artifacts"])
    child = derived.rows[1]
    parent = DerivedArtifactRowV1(
        derived_artifact_id=_identifier("derived_artifact", 4),
        derivation_kind="crop",
        parent_artifact_ids=(cast(str, child.clip_id),),
        participant_id=child.participant_id,
        session_id=child.session_id,
        source_recording_id=child.source_recording_id,
        clip_id=child.clip_id,
        annotation_id=child.annotation_id,
        sample_id=_identifier("sample", 4),
        label_id=child.label_id,
        split_id=split_id,
        partition=cast(Any, partition),
        handedness=child.handedness,
        mirror_state=child.mirror_state,
        operation_id="parent_crop",
        operation_version="1.0.0",
        artifact=_artifact(
            _identifier("sample", 4),
            role="sample_data",
            media_type="application/octet-stream",
            number=304,
            path="fixtures/samples/sample_00000000000000000000000000000004.bin",
        ),
    )
    changed_child = child.model_copy(
        update={
            "derivation_kind": "augmentation",
            "parent_artifact_ids": (parent.artifact.artifact_id,),
        }
    )
    changed_table = DerivedArtifactsTableV1(
        schema_version="derived-artifacts-table/1",
        rows=(derived.rows[0], changed_child, derived.rows[2], parent),
    )
    corrupted = _replace_table(bundle, "derived_artifacts", changed_table)
    old_content = corrupted.manifest.content
    parent_projection = DatasetSampleIdentityV1(
        sample_id=cast(str, parent.sample_id),
        participant_id=parent.participant_id,
        session_id=parent.session_id,
        source_recording_id=parent.source_recording_id,
        label_id=cast(Any, parent.label_id),
        artifact=parent.artifact,
    )
    content = DatasetContentV2(
        schema_version="dataset-content/2",
        taxonomy=old_content.taxonomy,
        governance_policy=old_content.governance_policy,
        lineage_inventory_sha256=old_content.lineage_inventory_sha256,
        sample_schema_version=old_content.sample_schema_version,
        tables=old_content.tables,
        samples=(
            *old_content.samples,
            parent_projection,
        ),
    )
    manifest = DatasetManifestV2(
        schema_version="dataset-manifest/2",
        dataset_id=corrupted.manifest.dataset_id,
        version=corrupted.manifest.version,
        content=content,
        data_sha256=dataset_content_digest(content),
    )

    with pytest.raises(DatasetValidationError) as captured:
        validate_dataset_manifest_tables(manifest, corrupted.tables)

    assert captured.value.category == "split.invalid"


def test_first_split_assignment_may_follow_an_unassigned_intermediate() -> None:
    bundle = _bundle()
    derived = cast(DerivedArtifactsTableV1, bundle.tables["derived_artifacts"])
    sample = derived.rows[0]
    intermediate = DerivedArtifactRowV1(
        derived_artifact_id=_identifier("derived_artifact", 4),
        derivation_kind="feature_extraction",
        parent_artifact_ids=(cast(str, sample.clip_id),),
        participant_id=sample.participant_id,
        session_id=sample.session_id,
        source_recording_id=sample.source_recording_id,
        clip_id=sample.clip_id,
        annotation_id=None,
        sample_id=None,
        label_id=None,
        split_id=None,
        partition=None,
        handedness=sample.handedness,
        mirror_state=sample.mirror_state,
        operation_id="unassigned_intermediate",
        operation_version="1.0.0",
        artifact=_artifact(
            _identifier("derived_artifact", 4),
            role="derived_data",
            media_type="application/octet-stream",
            number=404,
            path="fixtures/derived/derived_artifact_00000000000000000000000000000004.bin",
        ),
    )
    changed_sample = sample.model_copy(
        update={"parent_artifact_ids": (intermediate.artifact.artifact_id,)}
    )
    table = DerivedArtifactsTableV1(
        schema_version="derived-artifacts-table/1",
        rows=(changed_sample, *derived.rows[1:], intermediate),
    )
    changed = _replace_table(bundle, "derived_artifacts", table)

    result = validate_dataset_manifest_tables(changed.manifest, changed.tables)

    assert result.semantic_integrity == "verified"


def test_group_metadata_reconciles_handedness_and_mirroring() -> None:
    bundle = _bundle()
    derived = cast(DerivedArtifactsTableV1, bundle.tables["derived_artifacts"])
    row = derived.rows[0].model_copy(update={"mirror_state": "mirrored"})
    corrupted = _replace_row(bundle, "derived_artifacts", DerivedArtifactsTableV1, 0, row)

    with pytest.raises(DatasetValidationError) as captured:
        validate_dataset_manifest_tables(corrupted.manifest, corrupted.tables)

    assert captured.value.category == "group.invalid"


def test_recording_level_annotation_used_by_clip_must_be_contained() -> None:
    bundle = _bundle()
    annotations = cast(AnnotationsTableV1, bundle.tables["annotations"])
    contained_annotation = annotations.rows[0].model_copy(update={"clip_id": None})
    contained = _replace_row(
        bundle,
        "annotations",
        AnnotationsTableV1,
        0,
        contained_annotation,
    )

    result = validate_dataset_manifest_tables(contained.manifest, contained.tables)
    assert result.semantic_integrity == "verified"

    nonoverlapping_annotation = contained_annotation.model_copy(
        update={
            "interval": MediaIntervalV1(
                schema_version="media-interval/1",
                start_us=5_000_000,
                end_us=6_000_000,
            ),
        }
    )
    corrupted = _replace_row(
        contained,
        "annotations",
        AnnotationsTableV1,
        0,
        nonoverlapping_annotation,
    )

    with pytest.raises(DatasetValidationError) as captured:
        validate_dataset_manifest_tables(corrupted.manifest, corrupted.tables)

    assert captured.value.category == "interval.invalid"


def test_duplicate_primary_keys_fail_before_reference_checks() -> None:
    bundle = _bundle()
    participants = cast(ParticipantsTableV1, bundle.tables["participants"])
    unsafe = participants.model_copy(update={"rows": (participants.rows[0],) * 2})
    tables = {**bundle.tables, "participants": unsafe}

    with pytest.raises(DatasetValidationError) as captured:
        validate_dataset_manifest_tables(bundle.manifest, tables)

    assert captured.value.category == "identity.invalid"


def test_duplicate_consent_grant_ids_fail_after_table_identities_are_rebuilt() -> None:
    bundle = _bundle()
    recordings = cast(RecordingsTableV1, bundle.tables["recordings"])
    duplicate_grant = recordings.rows[1].consent_grant.model_copy(
        update={"grant_id": recordings.rows[0].consent_grant.grant_id}
    )
    duplicate_recording = recordings.rows[1].model_copy(update={"consent_grant": duplicate_grant})
    corrupted = _replace_row(
        bundle,
        "recordings",
        RecordingsTableV1,
        1,
        duplicate_recording,
    )

    with pytest.raises(DatasetValidationError) as captured:
        validate_dataset_manifest_tables(corrupted.manifest, corrupted.tables)

    assert captured.value.category == "identity.invalid"


def test_table_and_row_artifact_ids_cannot_collide_across_identity_boundaries() -> None:
    bundle = _bundle()
    recordings = cast(RecordingsTableV1, bundle.tables["recordings"])
    old_tables = bundle.manifest.content.tables
    participants_artifact = old_tables.participants.artifact.model_copy(
        update={"artifact_id": recordings.rows[0].media.artifact_id}
    )
    participants_reference = old_tables.participants.model_copy(
        update={"artifact": participants_artifact}
    )
    table_set = DatasetTableSetV1.model_validate(
        {
            **old_tables.model_dump(mode="json", round_trip=True),
            "participants": participants_reference.model_dump(mode="json", round_trip=True),
        },
        strict=True,
    )
    content = bundle.manifest.content.model_copy(update={"tables": table_set})
    manifest = DatasetManifestV2(
        schema_version="dataset-manifest/2",
        dataset_id=bundle.manifest.dataset_id,
        version=bundle.manifest.version,
        content=content,
        data_sha256=dataset_content_digest(content),
    )

    with pytest.raises(DatasetValidationError) as captured:
        validate_dataset_manifest_tables(manifest, bundle.tables)

    assert captured.value.category == "identity.invalid"


def test_table_reference_count_and_digest_are_exact() -> None:
    bundle = _bundle()
    annotations = cast(AnnotationsTableV1, bundle.tables["annotations"])
    changed = AnnotationsTableV1(
        schema_version="annotations-table/1",
        rows=(annotations.rows[0],),
    )
    tables = {**bundle.tables, "annotations": changed}

    with pytest.raises(DatasetValidationError) as captured:
        validate_dataset_manifest_tables(bundle.manifest, tables)

    assert captured.value.category == "table.reference"


def test_sample_projection_must_equal_every_derived_sample() -> None:
    bundle = _bundle()
    sample = bundle.manifest.content.samples[0].model_copy(update={"label_id": "yes"})
    content = bundle.manifest.content.model_copy(
        update={"samples": (sample, *bundle.manifest.content.samples[1:])}
    )
    manifest = bundle.manifest.model_copy(
        update={"content": content, "data_sha256": dataset_content_digest(content)}
    )

    with pytest.raises(DatasetValidationError) as captured:
        validate_dataset_manifest_tables(manifest, bundle.tables)

    assert captured.value.category == "sample_projection.invalid"


def test_partial_or_denied_authorization_never_reports_success() -> None:
    bundle = _bundle()

    with pytest.raises(DatasetValidationError) as partial:
        validate_dataset_manifest_tables(
            bundle.manifest,
            bundle.tables,
            consent_evidence_lookup=lambda grant: bundle.consent_evidence[grant.grant_id],
        )
    assert partial.value.category == "authorization.dependencies"

    with pytest.raises(DatasetValidationError) as denied:
        validate_dataset_manifest_tables(
            bundle.manifest,
            bundle.tables,
            consent_evidence_lookup=lambda grant: bundle.consent_evidence[grant.grant_id],
            consent_authorization_verifier=lambda _receipt, _grant, _event_log: False,
            authorization_permission="model_training",
            authorization_at="2026-08-26T00:01:00Z",
        )
    assert denied.value.category == "authorization.denied"


def test_validation_errors_are_stable_and_do_not_echo_untrusted_values() -> None:
    bundle = _bundle()
    sensitive_marker = "private-user-controlled-value"
    tables = {**bundle.tables, "annotations": {"schema_version": sensitive_marker}}

    with pytest.raises(DatasetValidationError) as captured:
        validate_dataset_manifest_tables(bundle.manifest, tables)

    assert captured.value.code == "dataset.table.reference"
    assert sensitive_marker not in str(captured.value)


def test_manifest_and_table_inventory_fail_closed() -> None:
    bundle = _bundle()

    with pytest.raises(DatasetValidationError) as invalid_manifest:
        validate_dataset_manifest_tables(
            {"schema_version": "dataset-manifest/99"},
            bundle.tables,
        )
    assert invalid_manifest.value.category == "contract.invalid"

    missing_table = dict(bundle.tables)
    del missing_table["annotations"]
    with pytest.raises(DatasetValidationError) as missing:
        validate_dataset_manifest_tables(bundle.manifest, missing_table)
    assert missing.value.category == "table.inventory"

    wrong_type = {**bundle.tables, "sessions": bundle.tables["participants"]}
    with pytest.raises(DatasetValidationError) as mismatched:
        validate_dataset_manifest_tables(bundle.manifest, wrong_type)
    assert mismatched.value.category == "table.reference"


@pytest.mark.parametrize(
    ("table_name", "table_type", "updates", "category"),
    [
        (
            "sessions",
            SessionsTableV1,
            {"participant_id": _identifier("participant", 99)},
            "foreign_key.invalid",
        ),
        (
            "recordings",
            RecordingsTableV1,
            {"device_id": _identifier("device", 99)},
            "group.invalid",
        ),
        (
            "clips",
            ClipsTableV1,
            {"source_recording_id": _identifier("recording", 99)},
            "foreign_key.invalid",
        ),
        (
            "clips",
            ClipsTableV1,
            {"handedness": "right"},
            "group.invalid",
        ),
        (
            "annotations",
            AnnotationsTableV1,
            {"clip_id": _identifier("clip", 99)},
            "foreign_key.invalid",
        ),
        (
            "derived_artifacts",
            DerivedArtifactsTableV1,
            {"annotation_id": _identifier("annotation", 99)},
            "foreign_key.invalid",
        ),
    ],
)
def test_foreign_keys_and_group_metadata_are_reconciled(
    table_name: TableName,
    table_type: type[DatasetTable],
    updates: dict[str, object],
    category: str,
) -> None:
    bundle = _bundle()
    table = cast(Any, bundle.tables[table_name])
    row = table.rows[0].model_copy(update=updates)
    corrupted = _replace_row(
        bundle,
        table_name,
        cast(Any, table_type),
        0,
        row,
    )

    with pytest.raises(DatasetValidationError) as captured:
        validate_dataset_manifest_tables(corrupted.manifest, corrupted.tables)

    assert captured.value.category == category


def test_session_and_clip_bounds_are_checked_against_source_time() -> None:
    bundle = _bundle()
    sessions = cast(SessionsTableV1, bundle.tables["sessions"])
    short_session = sessions.rows[0].model_copy(update={"finished_at": "2026-08-26T00:00:15Z"})
    recording_outside = _replace_row(bundle, "sessions", SessionsTableV1, 0, short_session)
    with pytest.raises(DatasetValidationError) as session_error:
        validate_dataset_manifest_tables(recording_outside.manifest, recording_outside.tables)
    assert session_error.value.category == "interval.invalid"

    clips = cast(ClipsTableV1, bundle.tables["clips"])
    long_clip = clips.rows[0].model_copy(
        update={
            "interval": MediaIntervalV1(
                schema_version="media-interval/1",
                start_us=1,
                end_us=10_000_001,
            )
        }
    )
    clip_outside = _replace_row(bundle, "clips", ClipsTableV1, 0, long_clip)
    with pytest.raises(DatasetValidationError) as clip_error:
        validate_dataset_manifest_tables(clip_outside.manifest, clip_outside.tables)
    assert clip_error.value.category == "interval.invalid"


def test_timestamp_boundary_overflow_is_reported_as_invalid_interval() -> None:
    bundle = _bundle()
    sessions = cast(SessionsTableV1, bundle.tables["sessions"])
    boundary_session = sessions.rows[0].model_copy(update={"finished_at": "9999-12-31T23:59:59Z"})
    bundle = _replace_row(bundle, "sessions", SessionsTableV1, 0, boundary_session)
    recordings = cast(RecordingsTableV1, bundle.tables["recordings"])
    boundary_grant = recordings.rows[0].consent_grant.model_copy(
        update={"captured_at": "9999-12-31T23:59:59Z"}
    )
    boundary_recording = recordings.rows[0].model_copy(
        update={
            "captured_at": "9999-12-31T23:59:59Z",
            "duration_us": 2_000_000,
            "consent_grant": boundary_grant,
        }
    )
    corrupted = _replace_row(
        bundle,
        "recordings",
        RecordingsTableV1,
        0,
        boundary_recording,
    )

    with pytest.raises(DatasetValidationError) as captured:
        validate_dataset_manifest_tables(corrupted.manifest, corrupted.tables)

    assert captured.value.category == "interval.invalid"


def test_retained_media_requires_explicit_retention_scope() -> None:
    bundle = _bundle()
    recordings = cast(RecordingsTableV1, bundle.tables["recordings"])
    old = recordings.rows[0]
    scope = _scope(raw_media_retention=False)
    grant = old.consent_grant.model_copy(
        update={"scope": scope, "scope_sha256": consent_scope_digest(scope)}
    )
    row = old.model_copy(update={"consent_grant": grant})
    corrupted = _replace_row(bundle, "recordings", RecordingsTableV1, 0, row)

    with pytest.raises(DatasetValidationError) as captured:
        validate_dataset_manifest_tables(corrupted.manifest, corrupted.tables)

    assert captured.value.category == "consent.binding"


def test_orphan_and_cyclic_lineage_are_rejected() -> None:
    bundle = _bundle()
    derived = cast(DerivedArtifactsTableV1, bundle.tables["derived_artifacts"])
    orphan = derived.rows[0].model_copy(update={"parent_artifact_ids": ("missing_parent",)})
    orphaned = _replace_row(bundle, "derived_artifacts", DerivedArtifactsTableV1, 0, orphan)
    with pytest.raises(DatasetValidationError) as orphan_error:
        validate_dataset_manifest_tables(orphaned.manifest, orphaned.tables)
    assert orphan_error.value.category == "lineage.invalid"

    first = derived.rows[0].model_copy(
        update={"parent_artifact_ids": (_identifier("derived_artifact", 4),)}
    )
    intermediate = DerivedArtifactRowV1(
        derived_artifact_id=_identifier("derived_artifact", 4),
        derivation_kind="feature_extraction",
        parent_artifact_ids=(_identifier("sample", 1),),
        participant_id=first.participant_id,
        session_id=first.session_id,
        source_recording_id=first.source_recording_id,
        clip_id=first.clip_id,
        annotation_id=None,
        sample_id=None,
        label_id=None,
        split_id=None,
        partition=None,
        handedness=first.handedness,
        mirror_state=first.mirror_state,
        operation_id="feature_passthrough",
        operation_version="1.0.0",
        artifact=_artifact(
            _identifier("derived_artifact", 4),
            role="derived_data",
            media_type="application/octet-stream",
            number=404,
            path="fixtures/derived/derived_artifact_00000000000000000000000000000004.bin",
        ),
    )
    cyclic_table = DerivedArtifactsTableV1(
        schema_version="derived-artifacts-table/1",
        rows=(first, *derived.rows[1:], intermediate),
    )
    cyclic = _replace_table(bundle, "derived_artifacts", cyclic_table)
    with pytest.raises(DatasetValidationError) as cycle_error:
        validate_dataset_manifest_tables(cyclic.manifest, cyclic.tables)
    assert cycle_error.value.category == "lineage.invalid"


def test_lineage_deeper_than_the_python_recursion_limit_validates_iteratively() -> None:
    bundle = _bundle()
    derived = cast(DerivedArtifactsTableV1, bundle.tables["derived_artifacts"])
    source = derived.rows[0]
    chain_numbers = tuple(range(1000, 1100 + sys.getrecursionlimit()))
    chain_rows = tuple(
        DerivedArtifactRowV1(
            derived_artifact_id=_identifier("derived_artifact", number),
            derivation_kind="feature_extraction",
            parent_artifact_ids=(
                _identifier("derived_artifact", chain_numbers[index + 1])
                if index + 1 < len(chain_numbers)
                else _identifier("recording", 1),
            ),
            participant_id=source.participant_id,
            session_id=source.session_id,
            source_recording_id=source.source_recording_id,
            clip_id=source.clip_id,
            annotation_id=None,
            sample_id=None,
            label_id=None,
            split_id=None,
            partition=None,
            handedness=source.handedness,
            mirror_state=source.mirror_state,
            operation_id="deep_feature_lineage",
            operation_version="1.0.0",
            artifact=_artifact(
                _identifier("derived_artifact", number),
                role="derived_data",
                media_type="application/octet-stream",
                number=10_000 + number,
                path=(f"fixtures/derived/{_identifier('derived_artifact', number)}.bin"),
            ),
        )
        for index, number in enumerate(chain_numbers)
    )
    deep_table = DerivedArtifactsTableV1(
        schema_version="derived-artifacts-table/1",
        rows=(*derived.rows, *chain_rows),
    )
    deep_bundle = _replace_table(bundle, "derived_artifacts", deep_table)

    result = validate_dataset_manifest_tables(deep_bundle.manifest, deep_bundle.tables)

    assert result.semantic_integrity == "verified"


def test_sample_projection_binds_exact_artifact_bytes() -> None:
    bundle = _bundle()
    sample = bundle.manifest.content.samples[0]
    changed_sha256 = _sha(9999)
    changed_digest = changed_sha256.removeprefix("sha256:")
    changed_artifact = sample.artifact.model_copy(
        update={
            "sha256": changed_sha256,
            "locator": _locator(
                f"objects/sha256/p-{changed_digest[:2]}/sha256-{changed_digest}/"
                f"{sample.artifact.artifact_id}"
            ),
        }
    )
    changed_sample = sample.model_copy(update={"artifact": changed_artifact})
    content = bundle.manifest.content.model_copy(
        update={"samples": (changed_sample, *bundle.manifest.content.samples[1:])}
    )
    manifest = bundle.manifest.model_copy(
        update={"content": content, "data_sha256": dataset_content_digest(content)}
    )

    with pytest.raises(DatasetValidationError) as captured:
        validate_dataset_manifest_tables(manifest, bundle.tables)

    assert captured.value.category == "sample_projection.invalid"


def test_stale_split_reference_and_test_transition_fragment_are_rejected() -> None:
    bundle = _bundle()
    derived = cast(DerivedArtifactsTableV1, bundle.tables["derived_artifacts"])
    first = derived.rows[0].model_copy(update={"operation_version": "2.0.0"})
    changed = _replace_row(bundle, "derived_artifacts", DerivedArtifactsTableV1, 0, first)
    with pytest.raises(DatasetValidationError) as stale:
        validate_dataset_manifest_tables(changed.manifest, changed.tables, split=bundle.split)
    assert stale.value.category == "split.invalid"

    annotations = cast(AnnotationsTableV1, bundle.tables["annotations"])
    transition = annotations.rows[2].model_copy(update={"other_kind": "transition_fragment"})
    transitioned = _replace_row(bundle, "annotations", AnnotationsTableV1, 2, transition)
    with pytest.raises(DatasetValidationError) as unchecked_transition_error:
        validate_dataset_manifest_tables(transitioned.manifest, transitioned.tables)
    assert unchecked_transition_error.value.category == "split.invalid"

    with pytest.raises(DatasetValidationError) as transition_error:
        validate_dataset_manifest_tables(
            transitioned.manifest,
            transitioned.tables,
            split=transitioned.split,
        )
    assert transition_error.value.category == "split.invalid"


def test_authorization_lookup_exceptions_and_pre_capture_use_are_denied() -> None:
    bundle = _bundle()

    def unavailable(_grant: RecordingConsentGrantV1) -> tuple[ConsentReceiptV1, ConsentEventLogV1]:
        raise RuntimeError("private backend detail")

    with pytest.raises(DatasetValidationError) as lookup_error:
        validate_dataset_manifest_tables(
            bundle.manifest,
            bundle.tables,
            consent_evidence_lookup=unavailable,
            consent_authorization_verifier=lambda _receipt, _grant, _event_log: True,
            authorization_permission="model_training",
            authorization_at="2026-08-26T00:01:00Z",
        )
    assert lookup_error.value.category == "authorization.denied"
    assert "private backend detail" not in str(lookup_error.value)

    with pytest.raises(DatasetValidationError) as too_early:
        validate_dataset_manifest_tables(
            bundle.manifest,
            bundle.tables,
            consent_evidence_lookup=lambda grant: bundle.consent_evidence[grant.grant_id],
            consent_authorization_verifier=lambda _receipt, _grant, _event_log: True,
            authorization_permission="model_training",
            authorization_at="2026-08-26T00:00:09Z",
        )
    assert too_early.value.category == "authorization.denied"


def test_authenticated_consent_must_match_manifest_governance_policy() -> None:
    bundle = _bundle()
    old_content = bundle.manifest.content
    different_policy = old_content.governance_policy.model_copy(
        update={"version": "2.0.0", "sha256": _sha(7000)}
    )
    unsafe_content = old_content.model_copy(update={"governance_policy": different_policy})
    unsafe_manifest = bundle.manifest.model_copy(update={"content": unsafe_content})

    with pytest.raises(DatasetValidationError) as public_boundary:
        validate_dataset_manifest_tables(unsafe_manifest, bundle.tables)
    assert public_boundary.value.category == "contract.invalid"

    recordings_table = cast(RecordingsTableV1, bundle.tables["recordings"])
    recordings = {recording.recording_id: recording for recording in recordings_table.rows}
    with pytest.raises(DatasetValidationError) as authorization:
        dataset_validation._authorize_current_consent(
            recordings,
            governance_policy=different_policy,
            evidence_lookup=lambda grant: bundle.consent_evidence[grant.grant_id],
            authorization_verifier=lambda _receipt, _grant, _event_log: True,
            permission="model_training",
            authorization_at="2026-08-26T00:01:00Z",
        )

    assert authorization.value.category == "authorization.denied"
