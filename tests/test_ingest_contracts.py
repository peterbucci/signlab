from __future__ import annotations

import copy
import json
from typing import Any, cast

import pytest
from pydantic import ValidationError

from signlab.contracts.dataset import (
    DatasetTableSetV1,
    ParticipantRowV1,
    RecordingRowV1,
    SessionRowV1,
)
from signlab.contracts.governance import RecordingConsentGrantV1
from signlab.contracts.ingest import (
    AMBIGUOUS_REASON_CODES,
    ANNOTATION_REASON_CODES,
    ATTEMPT_REASON_CODES,
    CAPTURE_CHECKLIST_IDS,
    CONSENT_CHECKLIST_IDS,
    IGNORE_REASON_CODES,
    SKIP_REASON_CODES,
    AnnotationProposalV1,
    CaptureAnnotationV1,
    CaptureAttemptV1,
    CaptureIdentifierSetV1,
    ChecklistResultV1,
    CollectionSidecarV1,
    IngestContractError,
    PromptOccurrenceV1,
    RawDatasetContentV1,
    RawDatasetManifestV1,
    capture_identifier_set_digest,
    collection_sidecar_digest,
    project_annotation_rows,
    raw_dataset_content_digest,
    require_importable_sidecar,
    validate_capture_identifier_set,
    validate_collection_sidecar,
    validate_raw_dataset_manifest,
)
from signlab.datasets.resources import (
    build_example_dataset_manifest,
    build_example_dataset_tables,
)
from signlab.governance.resources import (
    build_example_recording_grant,
    build_governance_policy,
)

ZERO_DIGEST = "sha256:" + "0" * 64


def _opaque(prefix: str, number: int) -> str:
    separator = "-" if prefix == "store" else "_"
    return f"{prefix}{separator}{number:032x}"


def _identifier_payload() -> dict[str, object]:
    return {
        "schema_version": "capture-identifier-set/1",
        "collection_id": _opaque("collection", 1),
        "participant_id": _opaque("participant", 1),
        "visit_id": _opaque("visit", 1),
        "session_id": _opaque("session", 1),
        "device_id": _opaque("device", 1),
        "recording_id": _opaque("recording", 31),
        "attempt_id": _opaque("attempt", 1),
        "source_key": _opaque("source", 1),
        "prompt_occurrence_id": _opaque("occurrence", 1),
        "annotation_id": _opaque("annotation", 1),
        "annotator_actor_id": _opaque("actor", 1),
        "reviewer_actor_id": _opaque("actor", 2),
        "adjudicator_actor_id": _opaque("actor", 3),
        "annotator_decision_id": _opaque("decision", 1),
        "reviewer_decision_id": _opaque("decision", 2),
        "adjudicator_decision_id": _opaque("decision", 3),
        "dataset_id": _opaque("dataset", 1),
        "store_id": _opaque("store", 1),
        "inventory_id": _opaque("inventory", 1),
    }


def _proposal(
    *,
    start_us: int = 100_000,
    end_us: int = 900_000,
    label_id: str = "hello",
) -> dict[str, object]:
    return {
        "schema_version": "annotation-proposal/1",
        "interval": {
            "schema_version": "media-interval/1",
            "start_us": start_us,
            "end_us": end_us,
        },
        "disposition": "class_label",
        "label_id": label_id,
        "other_kind": None,
        "reason_code": None,
    }


def _decision(
    number: int,
    role: str,
    decided_at: str,
    proposal: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "annotation-decision/1",
        "decision_id": _opaque("decision", number),
        "actor_id": _opaque("actor", number),
        "role": role,
        "decided_at": decided_at,
        "proposal": copy.deepcopy(proposal),
    }


def _grant(
    *,
    recording_number: int = 31,
    grant_number: int = 21,
    captured_at: str = "2026-08-26T12:10:00Z",
) -> dict[str, object]:
    payload = build_example_recording_grant().model_dump(mode="json", round_trip=True)
    payload["recording_id"] = _opaque("recording", recording_number)
    payload["grant_id"] = _opaque("grant", grant_number)
    payload["captured_at"] = captured_at
    return RecordingConsentGrantV1.model_validate(payload, strict=True).model_dump(
        mode="json", round_trip=True
    )


def _accepted_occurrence(*, number: int = 1, ordinal: int = 1) -> dict[str, object]:
    tables = build_example_dataset_tables()
    recording = cast(RecordingRowV1, tables["recordings"].rows[0])
    return {
        "schema_version": "prompt-occurrence/1",
        "prompt_occurrence_id": _opaque("occurrence", number),
        "ordinal": ordinal,
        "repetition": 1,
        "prompt_label_id": "hello" if number == 1 else "no",
        "participant_id": _opaque("participant", 1),
        "session_id": _opaque("session", 1),
        "state": "accepted",
        "skip_reason_code": None,
        "attempts": [
            {
                "schema_version": "capture-attempt/1",
                "attempt_id": _opaque("attempt", number),
                "recording_id": _opaque("recording", 30 + number),
                "source_key": _opaque("source", number),
                "outcome": "accepted",
                "reason_code": None,
                "retry_of_attempt_id": None,
                "recorded_at": "2026-08-26T12:10:00Z",
                "media_type": recording.media.media_type,
                "expected_sha256": recording.media.sha256,
                "expected_size_bytes": recording.media.size_bytes,
                "duration_us": recording.duration_us,
                "handedness": recording.handedness,
                "mirror_state": recording.mirror_state,
                "rotation_degrees": recording.rotation_degrees,
                "audio_present": False,
                "consent_grant": _grant(
                    recording_number=30 + number,
                    grant_number=20 + number,
                ),
            }
        ],
    }


def _sidecar_payload(*, fixture_only: bool = True) -> dict[str, object]:
    tables = build_example_dataset_tables()
    participant = cast(ParticipantRowV1, tables["participants"].rows[0])
    session = cast(SessionRowV1, tables["sessions"].rows[0])
    proposal = _proposal()
    payload: dict[str, object] = {
        "schema_version": "collection-sidecar/1",
        "collection_id": _opaque("collection", 1),
        "dataset_id": _opaque("dataset", 1),
        "dataset_version": "1.0.0",
        "store_id": _opaque("store", 1),
        "inventory_id": _opaque("inventory", 1),
        "generated_at": "2026-08-26T11:00:00Z",
        "updated_at": "2026-08-26T13:00:00Z",
        "finalized_at": "2026-08-26T13:00:00Z",
        "state": "complete",
        "fixture_only": fixture_only,
        "taxonomy": build_governance_policy().taxonomy.model_dump(mode="json", round_trip=True),
        "protocol": {
            "schema_version": "collection-protocol-reference/1",
            "protocol_id": "consent_aware_collection",
            "version": "0.1.0",
            "sha256": ("sha256:15e90df9d0940fbbc83d089f4b4ef5e2aa95cec9c015ee1be6b95d259600a5cd"),
        },
        "governance_policy": build_governance_policy().model_dump(mode="json", round_trip=True),
        "participants": [participant.model_dump(mode="json", round_trip=True)],
        "sessions": [session.model_dump(mode="json", round_trip=True)],
        "session_plans": [
            {
                "schema_version": "collection-session-plan/1",
                "visit_id": _opaque("visit", 1),
                "session_id": _opaque("session", 1),
                "condition_profile_id": "baseline_front_camera",
                "prompt_randomization": {
                    "schema_version": "prompt-randomization/1",
                    "algorithm_id": "balanced_non_adjacent",
                    "algorithm_version": "1.0.0",
                    "seed_sha256": "sha256:" + "1" * 64,
                    "realized_order_authoritative": True,
                    "rerolled_for_performance": False,
                },
                "consent_checklist": [
                    {
                        "schema_version": "checklist-result/1",
                        "check_id": "authenticated_receipt_is_current",
                        "status": "not_applicable",
                        "reason_code": "synthetic_no_person_no_camera",
                    },
                    {
                        "schema_version": "checklist-result/1",
                        "check_id": "collection_readiness_is_ready",
                        "status": "not_applicable",
                        "reason_code": "synthetic_no_person_no_camera",
                    },
                    {
                        "schema_version": "checklist-result/1",
                        "check_id": "purpose_is_authorized_before_capture",
                        "status": "not_applicable",
                        "reason_code": "synthetic_no_person_no_camera",
                    },
                ],
                "capture_checklist": [
                    {
                        "schema_version": "checklist-result/1",
                        "check_id": "camera_and_lens_ready",
                        "status": "not_applicable",
                        "reason_code": "synthetic_no_person_no_camera",
                    },
                    {
                        "schema_version": "checklist-result/1",
                        "check_id": "framing_and_lighting_usable",
                        "status": "not_applicable",
                        "reason_code": "synthetic_no_person_no_camera",
                    },
                    {
                        "schema_version": "checklist-result/1",
                        "check_id": "no_third_party_present",
                        "status": "not_applicable",
                        "reason_code": "synthetic_no_person_no_camera",
                    },
                    {
                        "schema_version": "checklist-result/1",
                        "check_id": "orientation_and_mirror_recorded",
                        "status": "not_applicable",
                        "reason_code": "synthetic_no_person_no_camera",
                    },
                    {
                        "schema_version": "checklist-result/1",
                        "check_id": "timing_and_playback_checked",
                        "status": "not_applicable",
                        "reason_code": "synthetic_no_person_no_camera",
                    },
                ],
            }
        ],
        "occurrences": [_accepted_occurrence()],
        "annotations": [
            {
                "schema_version": "capture-annotation/1",
                "annotation_id": _opaque("annotation", 1),
                "source_recording_id": _opaque("recording", 31),
                "decisions": [
                    _decision(1, "annotator", "2026-08-26T12:40:00Z", proposal),
                    _decision(2, "reviewer", "2026-08-26T12:41:00Z", proposal),
                ],
            }
        ],
        "collection_sidecar_sha256": ZERO_DIGEST,
    }
    payload["collection_sidecar_sha256"] = collection_sidecar_digest(payload)
    return payload


def _redigest(payload: dict[str, object]) -> dict[str, object]:
    payload["collection_sidecar_sha256"] = collection_sidecar_digest(payload)
    return payload


def test_identifier_set_is_strict_opaque_and_has_stable_identity() -> None:
    payload = _identifier_payload()
    checked = validate_capture_identifier_set(json.dumps(payload))

    assert isinstance(checked, CaptureIdentifierSetV1)
    assert capture_identifier_set_digest(checked) == capture_identifier_set_digest(payload)

    payload["source_key"] = "source-not-opaque"
    with pytest.raises(IngestContractError, match="invalid capture identifier set"):
        validate_capture_identifier_set(payload)


@pytest.mark.parametrize(
    ("duplicate_field", "original_field"),
    [
        ("reviewer_actor_id", "annotator_actor_id"),
        ("reviewer_decision_id", "annotator_decision_id"),
    ],
)
def test_identifier_set_rejects_reused_workflow_ids(
    duplicate_field: str,
    original_field: str,
) -> None:
    payload = _identifier_payload()
    payload[duplicate_field] = payload[original_field]
    with pytest.raises(IngestContractError, match="invalid capture identifier set"):
        validate_capture_identifier_set(payload)


def test_complete_sidecar_validates_and_projects_one_reviewed_row() -> None:
    payload = _sidecar_payload()
    checked = require_importable_sidecar(payload)
    rows = project_annotation_rows(payload)

    assert isinstance(checked, CollectionSidecarV1)
    assert checked.occurrences[0].accepted_recording_id == _opaque("recording", 31)
    assert len(rows) == 1
    assert rows[0].source_recording_id == _opaque("recording", 31)
    assert rows[0].review_status == "reviewed"
    assert rows[0].eligible_for_training is True
    assert rows[0].clip_id is None


def test_annotation_proposals_enforce_coded_disposition_shapes() -> None:
    ambiguous = _proposal()
    ambiguous.update(
        {
            "disposition": "ambiguous",
            "label_id": None,
            "reason_code": "boundary_unclear",
        }
    )
    checked = AnnotationProposalV1.model_validate(ambiguous, strict=True)
    assert checked.disposition == "ambiguous"

    for invalid in (
        {**_proposal(), "label_id": None},
        {**_proposal(label_id="other"), "other_kind": None},
        {**ambiguous, "reason_code": None},
    ):
        with pytest.raises(ValidationError):
            AnnotationProposalV1.model_validate(invalid, strict=True)


@pytest.mark.parametrize(
    ("disposition", "reason_code"),
    [
        ("ambiguous", "consent_exclusion"),
        ("ignore", "boundary_unclear"),
    ],
)
def test_annotation_reasons_cannot_cross_disposition_allowlists(
    disposition: str,
    reason_code: str,
) -> None:
    proposal = _proposal()
    proposal.update(
        {
            "disposition": disposition,
            "label_id": None,
            "reason_code": reason_code,
        }
    )

    with pytest.raises(ValidationError):
        AnnotationProposalV1.model_validate(proposal, strict=True)


@pytest.mark.parametrize(
    ("disposition", "reason_code"),
    [
        ("ambiguous", "unusable_occlusion"),
        ("ambiguous", "boundary_unclear"),
        ("ignore", "consent_exclusion"),
        ("ignore", "camera_setup"),
        ("ignore", "third_party_presence"),
        ("ignore", "unresolved_conflict"),
        ("ignore", "unusable_occlusion"),
    ],
)
def test_annotation_reason_is_valid_for_its_documented_disposition(
    disposition: str,
    reason_code: str,
) -> None:
    proposal = _proposal()
    proposal.update(
        {
            "disposition": disposition,
            "label_id": None,
            "reason_code": reason_code,
        }
    )

    assert AnnotationProposalV1.model_validate(proposal, strict=True).reason_code == reason_code


def test_reason_and_checklist_allowlists_are_exact_and_reject_ad_hoc_codes() -> None:
    assert ANNOTATION_REASON_CODES == (
        "unusable_occlusion",
        "boundary_unclear",
        "consent_exclusion",
        "camera_setup",
        "third_party_presence",
        "unresolved_conflict",
    )
    assert ATTEMPT_REASON_CODES == (
        "camera_interruption",
        "prompt_display_failure",
        "third_party_presence",
        "framing_failure",
        "corrupt_source",
        "missing_source",
        "legacy_consent_unknown",
        "legacy_label_unknown",
    )
    assert AMBIGUOUS_REASON_CODES == ("unusable_occlusion", "boundary_unclear")
    assert IGNORE_REASON_CODES == (
        "consent_exclusion",
        "camera_setup",
        "third_party_presence",
        "unresolved_conflict",
        "unusable_occlusion",
    )
    assert SKIP_REASON_CODES == ("participant_skip", "session_stopped")
    assert CONSENT_CHECKLIST_IDS == (
        "authenticated_receipt_is_current",
        "collection_readiness_is_ready",
        "purpose_is_authorized_before_capture",
    )
    assert CAPTURE_CHECKLIST_IDS == (
        "camera_and_lens_ready",
        "framing_and_lighting_usable",
        "no_third_party_present",
        "orientation_and_mirror_recorded",
        "timing_and_playback_checked",
    )

    ambiguous = _proposal()
    ambiguous.update(
        {
            "disposition": "ambiguous",
            "label_id": None,
            "reason_code": "unregistered_reason",
        }
    )
    with pytest.raises(ValidationError):
        AnnotationProposalV1.model_validate(ambiguous, strict=True)

    retry = cast(list[dict[str, Any]], _accepted_occurrence()["attempts"])[0]
    retry.update(
        {
            "outcome": "retry",
            "reason_code": "unregistered_reason",
            "consent_grant": None,
        }
    )
    with pytest.raises(ValidationError):
        CaptureAttemptV1.model_validate(retry, strict=True)


def test_annotation_history_enforces_roles_actors_times_and_adjudication() -> None:
    proposal = _proposal()
    base = {
        "schema_version": "capture-annotation/1",
        "annotation_id": _opaque("annotation", 1),
        "source_recording_id": _opaque("recording", 31),
        "decisions": [_decision(1, "annotator", "2026-08-26T12:40:00Z", proposal)],
    }
    draft = CaptureAnnotationV1.model_validate_json(json.dumps(base), strict=True)
    assert draft.review_status == "draft"

    invalid_histories = []
    wrong_role = copy.deepcopy(base)
    wrong_role["decisions"] = [_decision(1, "reviewer", "2026-08-26T12:40:00Z", proposal)]
    invalid_histories.append(wrong_role)

    reused_actor = copy.deepcopy(base)
    reused_actor["decisions"] = [
        _decision(1, "annotator", "2026-08-26T12:40:00Z", proposal),
        _decision(2, "reviewer", "2026-08-26T12:41:00Z", proposal),
    ]
    cast(list[dict[str, Any]], reused_actor["decisions"])[1]["actor_id"] = _opaque("actor", 1)
    invalid_histories.append(reused_actor)

    repeated_time = copy.deepcopy(base)
    repeated_time["decisions"] = [
        _decision(1, "annotator", "2026-08-26T12:40:00Z", proposal),
        _decision(2, "reviewer", "2026-08-26T12:40:00Z", proposal),
    ]
    invalid_histories.append(repeated_time)

    unnecessary_adjudication = copy.deepcopy(base)
    unnecessary_adjudication["decisions"] = [
        _decision(1, "annotator", "2026-08-26T12:40:00Z", proposal),
        _decision(2, "reviewer", "2026-08-26T12:41:00Z", proposal),
        _decision(3, "adjudicator", "2026-08-26T12:42:00Z", proposal),
    ]
    invalid_histories.append(unnecessary_adjudication)

    for invalid in invalid_histories:
        with pytest.raises(ValidationError):
            CaptureAnnotationV1.model_validate_json(json.dumps(invalid), strict=True)


def test_attempt_outcome_requires_consent_only_for_accepted_bytes() -> None:
    occurrence = _accepted_occurrence()
    accepted = cast(list[dict[str, Any]], occurrence["attempts"])[0]

    assert CaptureAttemptV1.model_validate(accepted, strict=True).outcome == "accepted"

    invalid_accepted = copy.deepcopy(accepted)
    invalid_accepted["reason_code"] = "camera_interruption"
    invalid_retry = copy.deepcopy(accepted)
    invalid_retry.update({"outcome": "retry", "reason_code": None, "consent_grant": None})
    ambidextrous = copy.deepcopy(accepted)
    ambidextrous["handedness"] = "ambidextrous"
    mismatched_time = copy.deepcopy(accepted)
    mismatched_time["recorded_at"] = "2026-08-26T12:11:00Z"
    for invalid in (invalid_accepted, invalid_retry, ambidextrous, mismatched_time):
        with pytest.raises(ValidationError):
            CaptureAttemptV1.model_validate(invalid, strict=True)


def test_checklist_results_are_coded_and_canonically_ordered() -> None:
    assert (
        ChecklistResultV1(
            schema_version="checklist-result/1",
            check_id="optional_check",
            status="not_applicable",
            reason_code="synthetic_no_person_no_camera",
        ).status
        == "not_applicable"
    )
    with pytest.raises(ValidationError):
        ChecklistResultV1.model_validate(
            {
                "schema_version": "checklist-result/1",
                "check_id": "authenticated_receipt_is_current",
                "status": "passed",
                "reason_code": "unexpected_reason",
            },
            strict=True,
        )

    payload = _sidecar_payload()
    plan = cast(list[dict[str, Any]], payload["session_plans"])[0]
    checklist = cast(list[dict[str, Any]], plan["capture_checklist"])
    checklist.insert(
        0,
        {
            "schema_version": "checklist-result/1",
            "check_id": "z_last_check",
            "status": "not_applicable",
            "reason_code": "synthetic_no_person_no_camera",
        },
    )
    with pytest.raises(IngestContractError, match="invalid collection sidecar"):
        validate_collection_sidecar(_redigest(payload))


@pytest.mark.parametrize("state", ["active", "paused"])
def test_incomplete_sidecars_validate_but_are_not_importable(state: str) -> None:
    payload = _sidecar_payload()
    payload["state"] = state
    payload["finalized_at"] = None
    checked = validate_collection_sidecar(_redigest(payload))

    assert checked.state == state
    with pytest.raises(IngestContractError, match="not complete"):
        require_importable_sidecar(checked)


def test_non_fixture_sidecar_is_structurally_valid_without_claiming_readiness() -> None:
    checked = require_importable_sidecar(_sidecar_payload(fixture_only=False))

    assert checked.fixture_only is False


def test_fixture_sidecar_cannot_claim_real_consent_or_camera_checks_passed() -> None:
    payload = _sidecar_payload()
    plan = cast(list[dict[str, Any]], payload["session_plans"])[0]
    consent_checklist = cast(list[dict[str, Any]], plan["consent_checklist"])
    consent_checklist[0].update({"status": "passed", "reason_code": None})

    with pytest.raises(IngestContractError, match="invalid collection sidecar"):
        validate_collection_sidecar(_redigest(payload))


def test_skipped_occurrence_preserves_participant_stop_without_fake_media() -> None:
    payload = _sidecar_payload()
    occurrences = cast(list[dict[str, Any]], payload["occurrences"])
    occurrences.append(
        {
            "schema_version": "prompt-occurrence/1",
            "prompt_occurrence_id": _opaque("occurrence", 2),
            "ordinal": 2,
            "repetition": 1,
            "prompt_label_id": "no",
            "participant_id": _opaque("participant", 1),
            "session_id": _opaque("session", 1),
            "state": "skipped",
            "skip_reason_code": "session_stopped",
            "attempts": [],
        }
    )

    checked = require_importable_sidecar(_redigest(payload))
    assert checked.occurrences[1].state == "skipped"
    assert checked.occurrences[1].attempts == ()

    occurrences[1]["attempts"] = copy.deepcopy(occurrences[0]["attempts"])
    with pytest.raises(IngestContractError, match="invalid collection sidecar"):
        validate_collection_sidecar(_redigest(payload))


def test_retry_history_allocates_a_new_recording_and_preserves_acceptance() -> None:
    payload = _sidecar_payload()
    occurrence = cast(list[dict[str, Any]], payload["occurrences"])[0]
    accepted = cast(list[dict[str, Any]], occurrence["attempts"])[0]
    retry = copy.deepcopy(accepted)
    retry.update(
        {
            "attempt_id": _opaque("attempt", 2),
            "recording_id": _opaque("recording", 32),
            "source_key": _opaque("source", 2),
            "outcome": "retry",
            "reason_code": "camera_interruption",
            "recorded_at": "2026-08-26T12:09:00Z",
            "expected_sha256": "sha256:" + "2" * 64,
            "consent_grant": None,
        }
    )
    accepted["retry_of_attempt_id"] = retry["attempt_id"]
    occurrence["attempts"] = [retry, accepted]

    checked = require_importable_sidecar(_redigest(payload))
    assert checked.occurrences[0].attempts[0].recording_id != (
        checked.occurrences[0].accepted_recording_id
    )

    accepted["retry_of_attempt_id"] = None
    with pytest.raises(IngestContractError, match="invalid collection sidecar"):
        validate_collection_sidecar(_redigest(payload))


def _retry_occurrence() -> dict[str, object]:
    occurrence = _accepted_occurrence()
    accepted = cast(list[dict[str, Any]], occurrence["attempts"])[0]
    retry = copy.deepcopy(accepted)
    retry.update(
        {
            "attempt_id": _opaque("attempt", 2),
            "recording_id": _opaque("recording", 32),
            "source_key": _opaque("source", 2),
            "outcome": "retry",
            "reason_code": "camera_interruption",
            "recorded_at": "2026-08-26T12:09:00Z",
            "expected_sha256": "sha256:" + "2" * 64,
            "consent_grant": None,
        }
    )
    accepted["retry_of_attempt_id"] = retry["attempt_id"]
    occurrence["attempts"] = [retry, accepted]
    return occurrence


def test_occurrence_rejects_duplicate_or_misordered_attempt_history() -> None:
    mutations = ("attempt_id", "source_key", "expected_sha256")
    for field in mutations:
        occurrence = _retry_occurrence()
        attempts = cast(list[dict[str, Any]], occurrence["attempts"])
        attempts[1][field] = attempts[0][field]
        with pytest.raises(ValidationError):
            PromptOccurrenceV1.model_validate_json(json.dumps(occurrence), strict=True)

    occurrence = _retry_occurrence()
    attempts = cast(list[dict[str, Any]], occurrence["attempts"])
    attempts.reverse()
    with pytest.raises(ValidationError):
        PromptOccurrenceV1.model_validate_json(json.dumps(occurrence), strict=True)

    occurrence = _accepted_occurrence()
    first = cast(list[dict[str, Any]], occurrence["attempts"])[0]
    first["retry_of_attempt_id"] = _opaque("attempt", 2)
    with pytest.raises(ValidationError):
        PromptOccurrenceV1.model_validate_json(json.dumps(occurrence), strict=True)

    occurrence = _accepted_occurrence()
    occurrence["state"] = "pending"
    with pytest.raises(ValidationError):
        PromptOccurrenceV1.model_validate_json(json.dumps(occurrence), strict=True)


def test_sidecar_rejects_envelope_plan_and_sequence_conflicts() -> None:
    payload = _sidecar_payload()
    payload["generated_at"] = "2026-08-26T14:00:00Z"
    with pytest.raises(IngestContractError):
        validate_collection_sidecar(_redigest(payload))

    payload = _sidecar_payload()
    payload["finalized_at"] = None
    with pytest.raises(IngestContractError):
        validate_collection_sidecar(_redigest(payload))

    payload = _sidecar_payload()
    payload["state"] = "active"
    with pytest.raises(IngestContractError):
        validate_collection_sidecar(_redigest(payload))

    payload = _sidecar_payload()
    plan = cast(list[dict[str, Any]], payload["session_plans"])[0]
    plan["session_id"] = _opaque("session", 2)
    with pytest.raises(IngestContractError):
        validate_collection_sidecar(_redigest(payload))

    payload = _sidecar_payload()
    occurrence = cast(list[dict[str, Any]], payload["occurrences"])[0]
    occurrence["ordinal"] = 2
    with pytest.raises(IngestContractError):
        validate_collection_sidecar(_redigest(payload))

    payload = _sidecar_payload()
    occurrence = cast(list[dict[str, Any]], payload["occurrences"])[0]
    occurrence["repetition"] = 2
    with pytest.raises(IngestContractError):
        validate_collection_sidecar(_redigest(payload))

    payload = _sidecar_payload()
    payload["collection_sidecar_sha256"] = ZERO_DIGEST
    with pytest.raises(IngestContractError):
        validate_collection_sidecar(payload)


def test_sidecar_rejects_capture_and_annotation_time_conflicts() -> None:
    payload = _sidecar_payload()
    occurrence = cast(list[dict[str, Any]], payload["occurrences"])[0]
    attempt = cast(list[dict[str, Any]], occurrence["attempts"])[0]
    attempt["recorded_at"] = "2026-08-26T13:01:00Z"
    cast(dict[str, Any], attempt["consent_grant"])["captured_at"] = attempt["recorded_at"]
    with pytest.raises(IngestContractError):
        validate_collection_sidecar(_redigest(payload))

    payload = _sidecar_payload()
    occurrence = cast(list[dict[str, Any]], payload["occurrences"])[0]
    attempt = cast(list[dict[str, Any]], occurrence["attempts"])[0]
    attempt["recorded_at"] = "2026-08-26T12:31:00Z"
    cast(dict[str, Any], attempt["consent_grant"])["captured_at"] = attempt["recorded_at"]
    with pytest.raises(IngestContractError):
        validate_collection_sidecar(_redigest(payload))

    payload = _sidecar_payload()
    occurrence = cast(list[dict[str, Any]], payload["occurrences"])[0]
    attempt = cast(list[dict[str, Any]], occurrence["attempts"])[0]
    attempt["recorded_at"] = "2026-08-26T12:29:59Z"
    cast(dict[str, Any], attempt["consent_grant"])["captured_at"] = attempt["recorded_at"]
    with pytest.raises(IngestContractError):
        validate_collection_sidecar(_redigest(payload))

    for decided_at in ("2026-08-26T12:09:00Z", "2026-08-26T13:01:00Z"):
        payload = _sidecar_payload()
        annotation = cast(list[dict[str, Any]], payload["annotations"])[0]
        decisions = cast(list[dict[str, Any]], annotation["decisions"])
        decisions[0]["decided_at"] = decided_at
        with pytest.raises(IngestContractError):
            validate_collection_sidecar(_redigest(payload))

    payload = _sidecar_payload()
    annotation = cast(list[dict[str, Any]], payload["annotations"])[0]
    for decision in cast(list[dict[str, Any]], annotation["decisions"]):
        proposal = cast(dict[str, Any], decision["proposal"])
        interval = cast(dict[str, Any], proposal["interval"])
        interval["end_us"] = 6_000_000
    with pytest.raises(IngestContractError):
        validate_collection_sidecar(_redigest(payload))


def test_complete_sidecar_requires_at_least_one_accepted_occurrence() -> None:
    payload = _sidecar_payload()
    occurrence = cast(list[dict[str, Any]], payload["occurrences"])[0]
    attempt = cast(list[dict[str, Any]], occurrence["attempts"])[0]
    occurrence["state"] = "quarantined"
    attempt.update(
        {
            "outcome": "quarantined",
            "reason_code": "legacy_consent_unknown",
            "consent_grant": None,
        }
    )
    payload["annotations"] = []
    with pytest.raises(IngestContractError):
        validate_collection_sidecar(_redigest(payload))


def test_sidecar_rejects_duplicate_media_across_all_attempt_outcomes() -> None:
    payload = _sidecar_payload()
    occurrences = cast(list[dict[str, Any]], payload["occurrences"])
    duplicate = copy.deepcopy(occurrences[0])
    duplicate.update(
        {
            "prompt_occurrence_id": _opaque("occurrence", 2),
            "ordinal": 2,
            "prompt_label_id": "no",
            "state": "quarantined",
        }
    )
    attempt = cast(list[dict[str, Any]], duplicate["attempts"])[0]
    attempt.update(
        {
            "attempt_id": _opaque("attempt", 2),
            "recording_id": _opaque("recording", 32),
            "source_key": _opaque("source", 2),
            "outcome": "quarantined",
            "reason_code": "corrupt_source",
            "consent_grant": None,
        }
    )
    occurrences.append(duplicate)

    with pytest.raises(IngestContractError, match="invalid collection sidecar"):
        validate_collection_sidecar(_redigest(payload))


def test_sidecar_rejects_attempt_id_reused_across_occurrences() -> None:
    payload = _sidecar_payload()
    occurrences = cast(list[dict[str, Any]], payload["occurrences"])
    duplicate = copy.deepcopy(occurrences[0])
    duplicate.update(
        {
            "prompt_occurrence_id": _opaque("occurrence", 2),
            "ordinal": 2,
            "prompt_label_id": "no",
            "state": "quarantined",
        }
    )
    attempt = cast(list[dict[str, Any]], duplicate["attempts"])[0]
    attempt.update(
        {
            "recording_id": _opaque("recording", 32),
            "source_key": _opaque("source", 2),
            "expected_sha256": "sha256:" + "2" * 64,
            "outcome": "quarantined",
            "reason_code": "corrupt_source",
            "consent_grant": None,
        }
    )
    occurrences.append(duplicate)

    with pytest.raises(IngestContractError, match="invalid collection sidecar"):
        validate_collection_sidecar(_redigest(payload))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("participant_id", _opaque("participant", 2)),
        ("session_id", _opaque("session", 2)),
    ],
)
def test_sidecar_rejects_conflicting_group_metadata(field: str, value: str) -> None:
    payload = _sidecar_payload()
    occurrence = cast(list[dict[str, Any]], payload["occurrences"])[0]
    occurrence[field] = value

    with pytest.raises(IngestContractError, match="invalid collection sidecar"):
        validate_collection_sidecar(_redigest(payload))


def test_sidecar_rejects_camera_and_consent_identity_conflicts() -> None:
    payload = _sidecar_payload()
    occurrence = cast(list[dict[str, Any]], payload["occurrences"])[0]
    attempt = cast(list[dict[str, Any]], occurrence["attempts"])[0]
    attempt["mirror_state"] = "not_mirrored"
    with pytest.raises(IngestContractError, match="invalid collection sidecar"):
        validate_collection_sidecar(_redigest(payload))

    payload = _sidecar_payload()
    occurrence = cast(list[dict[str, Any]], payload["occurrences"])[0]
    attempt = cast(list[dict[str, Any]], occurrence["attempts"])[0]
    grant = cast(dict[str, Any], attempt["consent_grant"])
    grant["recording_id"] = _opaque("recording", 32)
    with pytest.raises(IngestContractError, match="invalid collection sidecar"):
        validate_collection_sidecar(_redigest(payload))


def test_annotation_disagreement_requires_adjudication_and_projects_final_choice() -> None:
    payload = _sidecar_payload()
    annotation = cast(list[dict[str, Any]], payload["annotations"])[0]
    decisions = cast(list[dict[str, Any]], annotation["decisions"])
    reviewer_proposal = cast(dict[str, Any], decisions[1]["proposal"])
    reviewer_proposal["label_id"] = "no"

    with pytest.raises(IngestContractError, match="invalid collection sidecar"):
        validate_collection_sidecar(_redigest(payload))

    decisions.append(_decision(3, "adjudicator", "2026-08-26T12:42:00Z", reviewer_proposal))
    checked = require_importable_sidecar(_redigest(payload))
    row = project_annotation_rows(checked)[0]
    assert row.review_status == "adjudicated"
    assert row.label_id == "no"


def test_prompt_label_may_differ_from_observed_annotation() -> None:
    payload = _sidecar_payload()
    occurrence = cast(list[dict[str, Any]], payload["occurrences"])[0]
    occurrence["prompt_label_id"] = "no"

    checked = require_importable_sidecar(_redigest(payload))
    assert checked.occurrences[0].prompt_label_id == "no"
    assert project_annotation_rows(checked)[0].label_id == "hello"


def test_sidecar_rejects_overlapping_annotation_intervals_and_decision_ids() -> None:
    payload = _sidecar_payload()
    annotations = cast(list[dict[str, Any]], payload["annotations"])
    second = copy.deepcopy(annotations[0])
    second["annotation_id"] = _opaque("annotation", 2)
    second["decisions"] = [
        _decision(3, "annotator", "2026-08-26T12:43:00Z", _proposal(start_us=800_000)),
    ]
    annotations.append(second)
    with pytest.raises(IngestContractError, match="invalid collection sidecar"):
        validate_collection_sidecar(_redigest(payload))

    second_decisions = cast(list[dict[str, Any]], second["decisions"])
    second_decisions[0]["proposal"] = _proposal(start_us=900_000, end_us=1_000_000)
    second_decisions[0]["decision_id"] = _opaque("decision", 1)
    with pytest.raises(IngestContractError, match="invalid collection sidecar"):
        validate_collection_sidecar(_redigest(payload))


def test_complete_sidecar_rejects_pending_capture_or_blocked_checklist() -> None:
    payload = _sidecar_payload()
    occurrence = cast(list[dict[str, Any]], payload["occurrences"])[0]
    occurrence.update({"state": "pending", "attempts": []})
    with pytest.raises(IngestContractError, match="invalid collection sidecar"):
        validate_collection_sidecar(_redigest(payload))

    payload = _sidecar_payload(fixture_only=False)
    session_plan = cast(list[dict[str, Any]], payload["session_plans"])[0]
    checklist = cast(list[dict[str, Any]], session_plan["consent_checklist"])
    checklist[0].update({"status": "blocked", "reason_code": "consent_unverified"})
    with pytest.raises(IngestContractError, match="invalid collection sidecar"):
        validate_collection_sidecar(_redigest(payload))


def _raw_content(tables: DatasetTableSetV1 | None = None) -> RawDatasetContentV1:
    policy = build_governance_policy()
    return RawDatasetContentV1(
        schema_version="raw-dataset-content/1",
        taxonomy=policy.taxonomy,
        governance_policy=policy.policy_document,
        lineage_inventory_sha256="sha256:" + "3" * 64,
        collection_sidecar_sha256=cast(str, _sidecar_payload()["collection_sidecar_sha256"]),
        tables=tables or build_example_dataset_manifest().content.tables,
    )


def test_raw_manifest_binds_semantic_tables_without_sample_projection() -> None:
    content = _raw_content()
    manifest = RawDatasetManifestV1(
        schema_version="raw-dataset-manifest/1",
        dataset_id=_opaque("dataset", 1),
        version="1.0.0",
        content=content,
        raw_data_sha256=raw_dataset_content_digest(content),
    )

    checked = validate_raw_dataset_manifest(manifest.model_dump_json(round_trip=True))
    assert checked.raw_data_sha256 == raw_dataset_content_digest(checked.content)
    assert "samples" not in type(checked.content).model_fields

    bad = checked.model_dump(mode="json", round_trip=True)
    bad["raw_data_sha256"] = ZERO_DIGEST
    with pytest.raises(IngestContractError, match="invalid raw dataset manifest"):
        validate_raw_dataset_manifest(bad)


def test_raw_identity_ignores_table_locators_and_parquet_bytes() -> None:
    original = _raw_content()
    table_payload = original.tables.model_dump(mode="json", round_trip=True)
    for index, table_name in enumerate(
        ("participants", "sessions", "recordings", "clips", "annotations", "derived_artifacts"),
        start=1,
    ):
        reference = cast(dict[str, Any], table_payload[table_name])
        artifact = cast(dict[str, Any], reference["artifact"])
        artifact["sha256"] = f"sha256:{index:064x}"
        artifact["size_bytes"] = 1000 + index
        artifact["locator"] = {
            "kind": "artifact_uri",
            "uri": f"signlab://tables/{table_name}",
        }
    relocated = DatasetTableSetV1.model_validate(table_payload, strict=True)
    assert raw_dataset_content_digest(original) == raw_dataset_content_digest(
        _raw_content(relocated)
    )

    changed_payload = relocated.model_dump(mode="json", round_trip=True)
    participant_ref = cast(dict[str, Any], changed_payload["participants"])
    participant_ref["content_sha256"] = "sha256:" + "f" * 64
    changed = DatasetTableSetV1.model_validate(changed_payload, strict=True)
    assert raw_dataset_content_digest(original) != raw_dataset_content_digest(_raw_content(changed))


def test_public_validators_reject_extra_fields_and_noncanonical_json() -> None:
    payload = _sidecar_payload()
    payload["participant_name"] = "not allowed"
    with pytest.raises(IngestContractError, match="invalid collection sidecar"):
        validate_collection_sidecar(payload)

    duplicate_key_json = '{"schema_version":"capture-identifier-set/1","schema_version":"x"}'
    with pytest.raises(IngestContractError, match="invalid capture identifier set"):
        validate_capture_identifier_set(duplicate_key_json)

    with pytest.raises(ValidationError):
        CaptureIdentifierSetV1.model_validate(
            {**_identifier_payload(), "unexpected": "forbidden"}, strict=True
        )
