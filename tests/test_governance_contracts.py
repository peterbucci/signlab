from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import cast

import pytest
from pydantic import ValidationError

from signlab.contracts.governance import (
    CollectionReadinessV1,
    ConsentEventLogV1,
    ConsentEventV1,
    ConsentReceiptV1,
    ConsentScopeV1,
    DocumentRef,
    GovernanceAssetV1,
    GovernanceContractError,
    GovernancePolicyV1,
    LineageInventoryV1,
    RecordingConsentGrantV1,
    WithdrawalImpactV1,
    WithdrawalReportV1,
    WithdrawalRequestV1,
    assert_receipt_grant_consistent,
    consent_event_log_digest,
    consent_receipt_digest,
    consent_scope_digest,
    grant_authorizes_at,
    lineage_inventory_digest,
    new_participant_id,
    receipt_is_active_at,
    recording_consent_grant_digest,
    validate_collection_readiness,
    validate_consent_event,
    validate_consent_event_log,
    validate_consent_receipt,
    validate_consent_scope,
    validate_document_ref,
    validate_governance_asset,
    validate_governance_policy,
    validate_lineage_inventory,
    validate_recording_consent_grant,
    validate_withdrawal_impact,
    validate_withdrawal_report,
    validate_withdrawal_request,
    withdrawal_impact_digest,
    withdrawal_report_digest,
    withdrawal_request_digest,
)
from signlab.contracts.taxonomy import load_builtin_taxonomy, taxonomy_reference

ZERO_DIGEST = "sha256:" + "0" * 64
TAXONOMY = taxonomy_reference(load_builtin_taxonomy()).model_dump(mode="json")
PURPOSE_ID = "purpose_00000000000000000000000000000001"
STUDY_ID = "study_00000000000000000000000000000001"
FUTURE_STUDY_ID = "study_00000000000000000000000000000002"
ALL_BLOCKERS = [
    "access_controls",
    "adult_jurisdictions",
    "affiliation_sponsorship",
    "backup_deletion",
    "consent_documents",
    "encrypted_storage",
    "ethics_legal_institutional",
    "identity_vault_operations",
    "legacy_media_quarantine",
    "lineage_tracking",
    "participant_contact_process",
    "pseudonymous_ids",
    "retention_publication_scope",
    "withdrawal_dry_run",
]


def _identifier(prefix: str, digit: str) -> str:
    return f"{prefix}_{digit * 32}"


def _document(document_type: str, digit: str) -> dict[str, object]:
    published = {
        "consent_form": (
            "consent-form",
            "5ec3cad0f28d407f90278f61dc9f56457ce65ec7ae732c9afa7a26770ef05417",
        ),
        "privacy_notice": (
            "privacy-notice",
            "1b6e4d6ed9ddb7d67bb819e4c66ecc3325c4a3eaeee82a01b17180e50baf8762",
        ),
        "governance_policy": (
            "data-governance-policy",
            "5a29272296198c03f4c118d27ee19955174f4b26141284380a653eb874d4dfa9",
        ),
        "withdrawal_procedure": (
            "withdrawal-runbook",
            "d583a387d2debfcb1c49746cfa51a32f694e39410c6bd900c3a292a090cb8075",
        ),
    }
    uri_name, digest = published[document_type]
    return {
        "schema_version": "document-reference/1",
        "document_id": f"document_{int(digit):032x}",
        "document_type": document_type,
        "version": "1.0.0",
        "effective_at": "2026-08-26T00:00:00Z",
        "uri": f"signlab://governance/{uri_name}/1.0.0",
        "sha256": f"sha256:{digest}",
    }


def _scope(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "consent-scope/1",
        "research_use": True,
        "raw_media_capture": True,
        "model_training": True,
        "model_evaluation": True,
        "public_demonstration": False,
        "model_weights_redistribution": True,
        "raw_media_retention": True,
        "raw_media_redistribution": False,
        "derived_features": True,
        "derived_features_redistribution": True,
        "evaluation_results_redistribution": True,
        "same_purpose_future_research": True,
        "withdrawal_supported": True,
        "audio_collection": False,
        "minor_participation": False,
        "identity_inference": False,
        "commercial_sale": False,
    }
    payload.update(changes)
    return payload


def _receipt(**changes: object) -> dict[str, object]:
    scope = _scope()
    payload: dict[str, object] = {
        "schema_version": "consent-receipt/1",
        "receipt_id": _identifier("receipt", "1"),
        "participant_id": _identifier("participant", "1"),
        "purpose_id": PURPOSE_ID,
        "study_id": STUDY_ID,
        "consent_form": _document("consent_form", "1"),
        "privacy_notice": _document("privacy_notice", "2"),
        "governance_policy": _document("governance_policy", "3"),
        "taxonomy": TAXONOMY,
        "scope": scope,
        "scope_sha256": consent_scope_digest(scope),
        "granted_at": "2026-08-26T00:00:00Z",
        "valid_until": "2027-08-26T00:00:00Z",
        "completed_form_sha256": "sha256:" + "a" * 64,
        "identity_vault_attestation_sha256": "sha256:" + "b" * 64,
        "adult_attested": True,
    }
    payload.update(changes)
    return payload


def _grant(**changes: object) -> dict[str, object]:
    scope = _scope()
    payload: dict[str, object] = {
        "schema_version": "recording-consent-grant/1",
        "grant_id": _identifier("grant", "1"),
        "recording_id": _identifier("recording", "1"),
        "participant_id": _identifier("participant", "1"),
        "receipt_id": _identifier("receipt", "1"),
        "purpose_id": PURPOSE_ID,
        "study_id": STUDY_ID,
        "taxonomy": TAXONOMY,
        "scope": scope,
        "scope_sha256": consent_scope_digest(scope),
        "receipt_scope_sha256": consent_scope_digest(scope),
        "issued_at": "2026-08-26T00:00:00Z",
        "captured_at": "2026-08-27T00:00:00Z",
    }
    payload.update(changes)
    return payload


def _event(event_type: str = "granted", **changes: object) -> dict[str, object]:
    reasons: dict[str, str | None] = {
        "granted": None,
        "withdrawn": "participant_request",
        "expired": "consent_expired",
        "superseded": "scope_replaced",
    }
    payload: dict[str, object] = {
        "schema_version": "consent-event/1",
        "event_id": _identifier("event", "1"),
        "receipt_id": _identifier("receipt", "1"),
        "participant_id": _identifier("participant", "1"),
        "event_type": event_type,
        "occurred_at": (
            "2026-08-26T00:00:00Z" if event_type == "granted" else "2026-09-01T00:00:00Z"
        ),
        "scope_sha256": consent_scope_digest(_scope()),
        "reason_code": reasons[event_type],
        "replacement_receipt_id": (
            _identifier("receipt", "2") if event_type == "superseded" else None
        ),
    }
    payload.update(changes)
    return payload


def _event_log(*events: dict[str, object], **changes: object) -> dict[str, object]:
    lifecycle = events or (_event(),)
    receipt = _receipt()
    payload: dict[str, object] = {
        "schema_version": "consent-event-log/1",
        "event_log_id": _identifier("event_log", "1"),
        "receipt_id": _identifier("receipt", "1"),
        "receipt_sha256": consent_receipt_digest(receipt),
        "participant_id": _identifier("participant", "1"),
        "purpose_id": PURPOSE_ID,
        "study_id": STUDY_ID,
        "scope_sha256": consent_scope_digest(_scope()),
        "generated_at": "2026-10-02T00:00:00Z",
        "complete_through": "2026-10-01T00:00:00Z",
        "completeness_attested": True,
        "identity_vault_attestation_sha256": "sha256:" + "d" * 64,
        "events": list(lifecycle),
        "event_log_sha256": ZERO_DIGEST,
    }
    payload.update(changes)
    payload["event_log_sha256"] = consent_event_log_digest(payload)
    return payload


def _trusted_event_log(
    _receipt: ConsentReceiptV1,
    _event_log: ConsentEventLogV1,
) -> bool:
    return True


def _trusted_authorization(
    _receipt: ConsentReceiptV1,
    _grant: RecordingConsentGrantV1,
    _event_log: ConsentEventLogV1,
) -> bool:
    return True


def _policy(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "governance-policy/1",
        "policy_id": _identifier("policy", "1"),
        "version": "1.0.0",
        "taxonomy": TAXONOMY,
        "policy_document": _document("governance_policy", "3"),
        "withdrawal_procedure": _document("withdrawal_procedure", "4"),
        "effective_at": "2026-08-26T00:00:00Z",
        "permitted_roles": ["data_steward", "model_operator", "release_reviewer", "researcher"],
        "raw_media_roles": ["data_steward"],
        "raw_media_retention_days": 90,
        "derived_feature_retention_days": 365,
        "evaluation_result_retention_days": 365,
        "backup_retention_days": 30,
        "withdrawal_impact_inventory_days": 5,
        "withdrawal_response_days": 14,
        "storage_encrypted": True,
        "backups_encrypted": True,
        "backups_include_raw_media": True,
        "access_least_privilege": True,
        "access_audit_logged": True,
        "deletion_includes_backups": True,
        "withdrawal_invalidates_downstream": True,
        "prohibited_uses": [
            "audio_collection",
            "commercial_sale",
            "identity_inference",
            "minor_participation",
            "participant_level_public_ranking",
            "surveillance",
        ],
    }
    payload.update(changes)
    return payload


def _readiness(*, ready: bool = False, **changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "collection-readiness/1",
        "readiness_id": _identifier("readiness", "1"),
        "policy_id": _identifier("policy", "1"),
        "taxonomy": TAXONOMY,
        "assessed_at": "2026-02-01T00:00:00Z",
        "status": "ready" if ready else "blocked",
        "consent_documents_ready": ready,
        "pseudonymous_ids_ready": ready,
        "encrypted_storage_ready": ready,
        "access_controls_ready": ready,
        "adult_jurisdictions_ready": ready,
        "affiliation_sponsorship_ready": ready,
        "backup_deletion_ready": ready,
        "ethics_legal_institutional_ready": ready,
        "identity_vault_operations_ready": ready,
        "withdrawal_dry_run_ready": ready,
        "lineage_tracking_ready": ready,
        "legacy_media_quarantined": ready,
        "participant_contact_process_ready": ready,
        "retention_publication_scope_ready": ready,
        "blockers": [] if ready else ALL_BLOCKERS,
    }
    payload.update(changes)
    return payload


def _asset(digit: str, *, parent: str | None = None, **changes: object) -> dict[str, object]:
    is_root = parent is None
    payload: dict[str, object] = {
        "schema_version": "governance-asset/1",
        "asset_id": _identifier("asset", digit),
        "asset_kind": "raw_recording" if is_root else "derived_features",
        "logical_uri": "",
        "sha256": f"sha256:{digit * 64}",
        "taxonomy": TAXONOMY,
        "created_at": "2026-08-27T00:00:00Z",
        "participant_ids": [_identifier("participant", "1")],
        "recording_ids": [_identifier("recording", "1")],
        "receipt_ids": [_identifier("receipt", "1")],
        "grant_ids": [_identifier("grant", "1")],
        "parent_asset_ids": [] if parent is None else [_identifier("asset", parent)],
        "lifecycle_state": "active",
        "invalidated_at": None,
    }
    payload.update(changes)
    if "logical_uri" not in changes:
        payload["logical_uri"] = (
            f"signlab://store-{'a' * 32}/{payload['asset_kind']}/{payload['asset_id']}"
        )
    return payload


def _inventory(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "lineage-inventory/1",
        "inventory_id": _identifier("inventory", "1"),
        "taxonomy": TAXONOMY,
        "generated_at": "2026-09-01T00:00:00Z",
        "assets": [_asset("1"), _asset("2", parent="1")],
        "inventory_sha256": ZERO_DIGEST,
    }
    payload.update(changes)
    payload["inventory_sha256"] = lineage_inventory_digest(payload)
    return payload


def _request(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "withdrawal-request/1",
        "request_id": _identifier("withdrawal", "1"),
        "participant_id": _identifier("participant", "1"),
        "receipt_ids": [_identifier("receipt", "1")],
        "requested_at": "2026-09-02T00:00:00Z",
        "effective_at": "2026-09-02T00:00:00Z",
        "target": "all_participant_data",
        "identity_verification_attestation_sha256": "sha256:" + "c" * 64,
        "request_sha256": ZERO_DIGEST,
    }
    payload.update(changes)
    payload["request_sha256"] = withdrawal_request_digest(payload)
    return payload


def _impact(digit: str, *, relationship: str, **changes: object) -> dict[str, object]:
    asset = _asset(digit, parent=None if digit == "1" else "1")
    actions = (
        ["delete_backups", "delete_primary", "invalidate", "revoke_access"]
        if relationship == "direct"
        else ["invalidate", "retrain"]
    )
    payload: dict[str, object] = {
        "schema_version": "withdrawal-impact/1",
        "impact_id": _identifier("impact", digit),
        "asset_id": asset["asset_id"],
        "logical_uri": asset["logical_uri"],
        "asset_sha256": asset["sha256"],
        "asset_kind": asset["asset_kind"],
        "relationship": relationship,
        "planned_actions": actions,
        "impact_sha256": ZERO_DIGEST,
    }
    payload.update(changes)
    if "logical_uri" not in changes:
        payload["logical_uri"] = (
            f"signlab://store-{'a' * 32}/{payload['asset_kind']}/{payload['asset_id']}"
        )
    payload["impact_sha256"] = withdrawal_impact_digest(payload)
    return payload


def _report(**changes: object) -> dict[str, object]:
    inventory = _inventory()
    payload: dict[str, object] = {
        "schema_version": "withdrawal-report/1",
        "report_id": _identifier("report", "1"),
        "mode": "dry_run",
        "request": _request(),
        "inventory_id": inventory["inventory_id"],
        "inventory_sha256": inventory["inventory_sha256"],
        "generated_at": "2026-09-03T00:00:00Z",
        "status": "complete",
        "impacts": [_impact("1", relationship="direct"), _impact("2", relationship="downstream")],
        "affected_asset_count": 2,
        "direct_asset_count": 1,
        "downstream_asset_count": 1,
        "unresolved_asset_ids": [],
        "report_sha256": ZERO_DIGEST,
    }
    payload.update(changes)
    payload["report_sha256"] = withdrawal_report_digest(payload)
    return payload


def test_participant_id_generator_is_128_bit_prefixed_and_collision_free() -> None:
    generated = {new_participant_id() for _ in range(512)}
    assert len(generated) == 512
    assert all(re.fullmatch(r"participant_[0-9a-f]{32}", value) for value in generated)


def test_all_core_contracts_validate_as_strict_frozen_closed_models() -> None:
    scope = validate_consent_scope(_scope())
    receipt = validate_consent_receipt(_receipt())
    event = validate_consent_event(_event())
    event_log = validate_consent_event_log(_event_log(_event()))
    grant = validate_recording_consent_grant(_grant())
    policy = validate_governance_policy(_policy())
    blocked = validate_collection_readiness(_readiness())
    asset = validate_governance_asset(_asset("1"))
    inventory = validate_lineage_inventory(_inventory())
    request = validate_withdrawal_request(_request())
    impact = validate_withdrawal_impact(_impact("1", relationship="direct"))
    report = validate_withdrawal_report(_report())

    assert isinstance(scope, ConsentScopeV1)
    assert isinstance(receipt, ConsentReceiptV1)
    assert isinstance(event, ConsentEventV1)
    assert isinstance(event_log, ConsentEventLogV1)
    assert isinstance(grant, RecordingConsentGrantV1)
    assert isinstance(policy, GovernancePolicyV1)
    assert blocked.status == "blocked"
    assert isinstance(blocked, CollectionReadinessV1)
    assert isinstance(asset, GovernanceAssetV1)
    assert isinstance(inventory, LineageInventoryV1)
    assert isinstance(request, WithdrawalRequestV1)
    assert isinstance(impact, WithdrawalImpactV1)
    assert isinstance(report, WithdrawalReportV1)
    with pytest.raises(ValidationError, match="frozen"):
        receipt.participant_id = _identifier("participant", "2")

    extra = _scope(unreviewed=True)
    with pytest.raises(GovernanceContractError, match="Extra inputs"):
        validate_consent_scope(extra)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"research_use": False}, "research_use"),
        ({"raw_media_capture": False}, "capture"),
        ({"model_training": True, "derived_features": False}, "derived_features"),
        ({"model_weights_redistribution": True, "model_training": False}, "model training"),
        ({"public_demonstration": True, "raw_media_retention": False}, "raw-media display"),
        ({"raw_media_redistribution": True, "raw_media_retention": False}, "retention"),
        (
            {
                "model_training": False,
                "model_evaluation": False,
                "model_weights_redistribution": False,
                "evaluation_results_redistribution": False,
                "derived_features_redistribution": True,
                "derived_features": False,
            },
            "generation",
        ),
        ({"evaluation_results_redistribution": True, "model_evaluation": False}, "evaluation"),
        ({"audio_collection": True}, "False"),
        ({"minor_participation": True}, "False"),
        ({"identity_inference": True}, "False"),
        ({"commercial_sale": True}, "False"),
    ],
)
def test_consent_scope_is_explicit_and_fail_closed(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(GovernanceContractError, match=message):
        validate_consent_scope(_scope(**changes))


def test_consent_scope_requires_every_permission_and_has_a_stable_digest() -> None:
    payload = _scope()
    expected = consent_scope_digest(payload)
    assert expected == consent_scope_digest(validate_consent_scope(payload))
    assert expected.startswith("sha256:")
    assert len(expected) == 71
    payload.pop("raw_media_redistribution")
    with pytest.raises(GovernanceContractError, match="Field required"):
        validate_consent_scope(payload)
    for required in ("raw_media_capture", "same_purpose_future_research"):
        missing = _scope()
        missing.pop(required)
        with pytest.raises(GovernanceContractError, match="Field required"):
            validate_consent_scope(missing)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(scope_sha256=ZERO_DIGEST), "scope_sha256"),
        (lambda value: value.update(valid_until="2026-01-01T00:00:00Z"), "valid_until"),
        (
            lambda value: value["consent_form"].update(document_type="privacy_notice"),
            "published registry",
        ),
        (
            lambda value: value["privacy_notice"].update(effective_at="2026-03-01T00:00:00Z"),
            "published registry",
        ),
        (lambda value: value.update(adult_attested=False), "True"),
    ],
)
def test_receipt_rejects_incompatible_or_tampered_evidence(
    mutation: Callable[[dict[str, object]], object], message: str
) -> None:
    payload = _receipt()
    mutation(payload)
    with pytest.raises(GovernanceContractError, match=message):
        validate_consent_receipt(payload)


@pytest.mark.parametrize(
    "changes",
    [
        {"valid_until": None},
        {"valid_until": "2028-08-26T00:00:00Z"},
        {"completed_form_sha256": None},
        {"identity_vault_attestation_sha256": None},
    ],
)
def test_receipt_requires_bounded_expiration_and_both_evidence_digests(
    changes: dict[str, object],
) -> None:
    with pytest.raises(GovernanceContractError, match=r"valid_until|730|completed|identity"):
        validate_consent_receipt(_receipt(**changes))


def test_receipt_and_recording_grant_digests_bind_complete_documents() -> None:
    receipt_document = _receipt()
    grant_document = _grant()

    assert consent_receipt_digest(receipt_document) == consent_receipt_digest(
        validate_consent_receipt(receipt_document)
    )
    assert recording_consent_grant_digest(grant_document) == recording_consent_grant_digest(
        validate_recording_consent_grant(grant_document)
    )

    changed_grant = _grant(captured_at="2026-08-28T00:00:00Z")
    assert recording_consent_grant_digest(changed_grant) != recording_consent_grant_digest(
        grant_document
    )


@pytest.mark.parametrize("event_type", ["granted", "withdrawn", "expired", "superseded"])
def test_consent_event_shapes_and_receipt_time_authorization(event_type: str) -> None:
    receipt = validate_consent_receipt(_receipt())
    events = [_event()]
    if event_type != "granted":
        events.append(_event(event_type, event_id=_identifier("event", "2")))
    at = "2026-10-01T00:00:00Z"
    expected = event_type == "granted"
    assert (
        receipt_is_active_at(
            receipt,
            validate_consent_event_log(_event_log(*events)),
            at,
            event_log_verifier=_trusted_event_log,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("event_type", "changes"),
    [
        ("granted", {"reason_code": "participant_request"}),
        ("withdrawn", {"reason_code": None}),
        ("expired", {"reason_code": "participant_request"}),
        ("superseded", {"replacement_receipt_id": None}),
        ("superseded", {"replacement_receipt_id": _identifier("receipt", "1")}),
    ],
)
def test_consent_event_rejects_invalid_state_combinations(
    event_type: str, changes: dict[str, object]
) -> None:
    with pytest.raises(GovernanceContractError):
        validate_consent_event(_event(event_type, **changes))


def test_consent_event_log_is_digest_bound_complete_and_has_one_lifecycle() -> None:
    granted = _event()
    withdrawn = _event(
        "withdrawn",
        event_id=_identifier("event", "2"),
        occurred_at="2026-09-01T00:00:00Z",
    )
    checked = validate_consent_event_log(_event_log(granted, withdrawn))
    assert checked.completeness_attested is True
    assert checked.identity_vault_attestation_sha256.startswith("sha256:")
    assert checked.event_log_sha256 == consent_event_log_digest(checked)
    assert checked.event_log_id.startswith("event_log_")

    with pytest.raises(GovernanceContractError, match="sorted"):
        validate_consent_event_log(_event_log(withdrawn, granted))
    with pytest.raises(GovernanceContractError, match="unique"):
        validate_consent_event_log(_event_log(granted, granted))
    with pytest.raises(GovernanceContractError, match="bind"):
        validate_consent_event_log(
            _event_log(_event(participant_id=_identifier("participant", "2")))
        )
    with pytest.raises(GovernanceContractError, match="complete-through"):
        validate_consent_event_log(
            _event_log(
                granted,
                withdrawn,
                complete_through="2026-08-31T00:00:00Z",
            )
        )
    with pytest.raises(GovernanceContractError, match="generation"):
        validate_consent_event_log(_event_log(generated_at="2026-09-01T00:00:00Z"))
    with pytest.raises(GovernanceContractError, match="True"):
        validate_consent_event_log(_event_log(completeness_attested=False))

    with pytest.raises(GovernanceContractError, match="at least 1"):
        validate_consent_event_log(_event_log(events=[]))
    with pytest.raises(GovernanceContractError, match="exactly one granted"):
        validate_consent_event_log(_event_log(granted, _event(event_id=_identifier("event", "2"))))
    with pytest.raises(GovernanceContractError, match="at most 2"):
        validate_consent_event_log(
            _event_log(
                granted,
                withdrawn,
                _event(
                    "expired",
                    event_id=_identifier("event", "3"),
                    occurred_at="2026-09-02T00:00:00Z",
                ),
            )
        )
    with pytest.raises(GovernanceContractError, match="after the granted"):
        validate_consent_event_log(
            _event_log(
                granted,
                _event(
                    "withdrawn",
                    event_id=_identifier("event", "2"),
                    occurred_at=granted["occurred_at"],
                ),
            )
        )

    tampered_digest = _event_log(granted)
    tampered_digest["event_log_sha256"] = ZERO_DIGEST
    with pytest.raises(GovernanceContractError, match="event_log_sha256"):
        validate_consent_event_log(tampered_digest)

    missing_attestation = _event_log(granted)
    missing_attestation.pop("identity_vault_attestation_sha256")
    missing_attestation["event_log_sha256"] = consent_event_log_digest(missing_attestation)
    with pytest.raises(GovernanceContractError, match="identity_vault_attestation"):
        validate_consent_event_log(missing_attestation)


def test_receipt_time_helper_uses_half_open_window_and_validates_events() -> None:
    receipt = validate_consent_receipt(_receipt())
    long_log = validate_consent_event_log(
        _event_log(
            complete_through="2027-08-26T00:00:00Z",
            generated_at="2027-08-27T00:00:00Z",
        )
    )
    assert not receipt_is_active_at(
        receipt,
        long_log,
        "2026-08-25T23:59:59Z",
        event_log_verifier=_trusted_event_log,
    )
    assert receipt_is_active_at(
        receipt,
        long_log,
        "2026-08-26T00:00:00Z",
        event_log_verifier=_trusted_event_log,
    )
    assert not receipt_is_active_at(
        receipt,
        long_log,
        "2027-08-26T00:00:00Z",
        event_log_verifier=_trusted_event_log,
    )

    with pytest.raises(GovernanceContractError, match="unique"):
        receipt_is_active_at(
            receipt,
            _event_log(_event(), _event()),
            "2026-09-01T00:00:00Z",
            event_log_verifier=_trusted_event_log,
        )
    with pytest.raises(GovernanceContractError, match="bind"):
        receipt_is_active_at(
            receipt,
            _event_log(_event(participant_id=_identifier("participant", "2"))),
            "2026-09-01T00:00:00Z",
            event_log_verifier=_trusted_event_log,
        )
    with pytest.raises(GovernanceContractError, match="event time"):
        receipt_is_active_at(
            receipt,
            _event_log(_event(occurred_at="2026-08-27T00:00:00Z")),
            "2026-09-01T00:00:00Z",
            event_log_verifier=_trusted_event_log,
        )


def test_recording_grant_may_narrow_but_never_expand_receipt_scope() -> None:
    receipt = validate_consent_receipt(_receipt())
    event_log = validate_consent_event_log(_event_log())
    narrow_scope = _scope(model_weights_redistribution=False)
    grant = validate_recording_consent_grant(
        _grant(scope=narrow_scope, scope_sha256=consent_scope_digest(narrow_scope))
    )
    assert_receipt_grant_consistent(receipt, grant, event_log)
    assert grant_authorizes_at(
        grant,
        receipt,
        event_log,
        "model_training",
        "2026-09-01T00:00:00Z",
        purpose_id=PURPOSE_ID,
        study_id=STUDY_ID,
        authorization_verifier=_trusted_authorization,
    )
    assert not grant_authorizes_at(
        grant,
        receipt,
        event_log,
        "model_weights_redistribution",
        "2026-09-01T00:00:00Z",
        purpose_id=PURPOSE_ID,
        study_id=STUDY_ID,
        authorization_verifier=_trusted_authorization,
    )
    assert not grant_authorizes_at(
        grant,
        receipt,
        event_log,
        "model_training",
        "2026-08-26T12:00:00Z",
        purpose_id=PURPOSE_ID,
        study_id=STUDY_ID,
        authorization_verifier=_trusted_authorization,
    )

    expanded_receipt_scope = _scope(model_weights_redistribution=False)
    narrower_receipt = validate_consent_receipt(
        _receipt(
            scope=expanded_receipt_scope,
            scope_sha256=consent_scope_digest(expanded_receipt_scope),
        )
    )
    expanded_grant = validate_recording_consent_grant(
        _grant(receipt_scope_sha256=narrower_receipt.scope_sha256)
    )
    with pytest.raises(GovernanceContractError, match="exceeds"):
        assert_receipt_grant_consistent(
            narrower_receipt,
            expanded_grant,
            validate_consent_event_log(
                _event_log(
                    _event(scope_sha256=narrower_receipt.scope_sha256),
                    receipt_sha256=consent_receipt_digest(narrower_receipt),
                    scope_sha256=narrower_receipt.scope_sha256,
                )
            ),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"participant_id": _identifier("participant", "2")}, "supplied receipt"),
        ({"purpose_id": _identifier("purpose", "2")}, "purpose and study"),
        ({"study_id": _identifier("study", "2")}, "purpose and study"),
        ({"receipt_scope_sha256": ZERO_DIGEST}, "receipt scope"),
        ({"issued_at": "2026-01-01T00:00:00Z"}, "predates"),
        ({"captured_at": "2027-09-01T00:00:00Z"}, "validity window"),
    ],
)
def test_receipt_grant_consistency_rejects_mismatches(
    changes: dict[str, object], message: str
) -> None:
    receipt = validate_consent_receipt(_receipt())
    grant = validate_recording_consent_grant(_grant(**changes))
    with pytest.raises(GovernanceContractError, match=message):
        assert_receipt_grant_consistent(
            receipt,
            grant,
            validate_consent_event_log(
                _event_log(
                    complete_through="2028-09-01T00:00:00Z",
                    generated_at="2028-09-02T00:00:00Z",
                )
            ),
        )


def test_recording_grant_rejects_bad_time_and_scope_hash() -> None:
    with pytest.raises(GovernanceContractError, match="capture"):
        validate_recording_consent_grant(
            _grant(issued_at="2026-08-28T00:00:00Z", captured_at="2026-08-27T00:00:00Z")
        )
    with pytest.raises(GovernanceContractError, match="scope_sha256"):
        validate_recording_consent_grant(_grant(scope_sha256=ZERO_DIGEST))

    no_capture_scope = _scope(
        raw_media_capture=False,
        model_training=False,
        model_evaluation=False,
        model_weights_redistribution=False,
        raw_media_retention=False,
        derived_features=False,
        derived_features_redistribution=False,
        evaluation_results_redistribution=False,
    )
    with pytest.raises(GovernanceContractError, match="raw_media_capture"):
        validate_recording_consent_grant(
            _grant(
                scope=no_capture_scope,
                scope_sha256=consent_scope_digest(no_capture_scope),
            )
        )


def test_authorization_requires_complete_log_and_rejects_pre_capture_withdrawal() -> None:
    receipt = validate_consent_receipt(_receipt())
    grant = validate_recording_consent_grant(_grant())
    incomplete = validate_consent_event_log(
        _event_log(
            generated_at="2026-08-27T00:00:00Z",
            complete_through="2026-08-26T23:59:59Z",
        )
    )
    with pytest.raises(GovernanceContractError, match="complete through"):
        assert_receipt_grant_consistent(receipt, grant, incomplete)

    capture_complete_only = validate_consent_event_log(
        _event_log(
            generated_at="2026-08-28T00:00:00Z",
            complete_through="2026-08-27T00:00:00Z",
        )
    )
    with pytest.raises(GovernanceContractError, match="complete through"):
        grant_authorizes_at(
            grant,
            receipt,
            capture_complete_only,
            "model_training",
            "2026-09-01T00:00:00Z",
            purpose_id=PURPOSE_ID,
            study_id=STUDY_ID,
            authorization_verifier=_trusted_authorization,
        )

    withdrawn_before_capture = validate_consent_event_log(
        _event_log(
            _event(),
            _event(
                "withdrawn",
                event_id=_identifier("event", "2"),
                occurred_at="2026-08-26T12:00:00Z",
            ),
        )
    )
    with pytest.raises(GovernanceContractError, match="validity window"):
        assert_receipt_grant_consistent(receipt, grant, withdrawn_before_capture)

    with pytest.raises(GovernanceContractError, match="sorted"):
        validate_consent_event_log(
            _event_log(
                _event(),
                _event(
                    "withdrawn",
                    event_id=_identifier("event", "2"),
                    occurred_at="2026-08-25T23:59:59Z",
                ),
            )
        )


def test_positive_authorization_requires_a_sanitized_authenticated_log() -> None:
    receipt = validate_consent_receipt(_receipt())
    grant = validate_recording_consent_grant(_grant())
    event_log = validate_consent_event_log(_event_log())

    with pytest.raises(GovernanceContractError, match="authentication failed"):
        grant_authorizes_at(
            grant,
            receipt,
            event_log,
            "model_training",
            "2026-09-01T00:00:00Z",
            purpose_id=PURPOSE_ID,
            study_id=STUDY_ID,
            authorization_verifier=lambda _receipt, _grant, _log: False,
        )

    def broken_verifier(
        _receipt: ConsentReceiptV1,
        _log: ConsentEventLogV1,
    ) -> bool:
        raise RuntimeError("person@example.invalid")

    with pytest.raises(GovernanceContractError, match="authentication failed") as caught:
        receipt_is_active_at(
            receipt,
            event_log,
            "2026-09-01T00:00:00Z",
            event_log_verifier=broken_verifier,
        )
    assert "person@example.invalid" not in str(caught.value)


def test_authorization_binds_purpose_and_requires_future_study_permission() -> None:
    current_only_scope = _scope(same_purpose_future_research=False)
    receipt = validate_consent_receipt(
        _receipt(
            scope=current_only_scope,
            scope_sha256=consent_scope_digest(current_only_scope),
        )
    )
    grant = validate_recording_consent_grant(
        _grant(
            scope=current_only_scope,
            scope_sha256=consent_scope_digest(current_only_scope),
            receipt_scope_sha256=receipt.scope_sha256,
        )
    )
    event_log = validate_consent_event_log(
        _event_log(
            _event(scope_sha256=receipt.scope_sha256),
            receipt_sha256=consent_receipt_digest(receipt),
            scope_sha256=receipt.scope_sha256,
        )
    )
    assert not grant_authorizes_at(
        grant,
        receipt,
        event_log,
        "model_training",
        "2026-09-01T00:00:00Z",
        purpose_id=_identifier("purpose", "2"),
        study_id=STUDY_ID,
        authorization_verifier=_trusted_authorization,
    )
    assert not grant_authorizes_at(
        grant,
        receipt,
        event_log,
        "model_training",
        "2026-09-01T00:00:00Z",
        purpose_id=PURPOSE_ID,
        study_id=FUTURE_STUDY_ID,
        authorization_verifier=_trusted_authorization,
    )

    future_scope = _scope(same_purpose_future_research=True)
    future_receipt = validate_consent_receipt(
        _receipt(
            scope=future_scope,
            scope_sha256=consent_scope_digest(future_scope),
        )
    )
    future_grant = validate_recording_consent_grant(
        _grant(
            scope=future_scope,
            scope_sha256=consent_scope_digest(future_scope),
            receipt_scope_sha256=future_receipt.scope_sha256,
        )
    )
    future_log = validate_consent_event_log(
        _event_log(
            _event(scope_sha256=future_receipt.scope_sha256),
            receipt_sha256=consent_receipt_digest(future_receipt),
            scope_sha256=future_receipt.scope_sha256,
        )
    )
    assert grant_authorizes_at(
        future_grant,
        future_receipt,
        future_log,
        "model_training",
        "2026-09-01T00:00:00Z",
        purpose_id=PURPOSE_ID,
        study_id=FUTURE_STUDY_ID,
        authorization_verifier=_trusted_authorization,
    )

    with pytest.raises(GovernanceContractError, match="purpose identifier"):
        grant_authorizes_at(
            grant,
            receipt,
            event_log,
            "model_training",
            "2026-09-01T00:00:00Z",
            purpose_id="purpose_invalid",
            study_id=STUDY_ID,
            authorization_verifier=_trusted_authorization,
        )


def test_authorization_revalidates_unsafe_model_copies() -> None:
    receipt = validate_consent_receipt(_receipt())
    grant = validate_recording_consent_grant(_grant())
    event_log = validate_consent_event_log(_event_log())
    forged_scope = grant.scope.model_copy(update={"research_use": False})
    forged_grant = grant.model_copy(update={"scope": forged_scope})
    with pytest.raises(GovernanceContractError, match="research_use"):
        grant_authorizes_at(
            forged_grant,
            receipt,
            event_log,
            "model_training",
            "2026-09-01T00:00:00Z",
            purpose_id=PURPOSE_ID,
            study_id=STUDY_ID,
            authorization_verifier=_trusted_authorization,
        )

    forged_log = event_log.model_copy(update={"completeness_attested": False})
    with pytest.raises(GovernanceContractError, match="True"):
        assert_receipt_grant_consistent(receipt, grant, forged_log)

    forged_receipt = receipt.model_copy(update={"valid_until": None})
    with pytest.raises(GovernanceContractError, match="valid_until"):
        receipt_is_active_at(
            forged_receipt,
            event_log,
            "2026-09-01T00:00:00Z",
            event_log_verifier=_trusted_event_log,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"permitted_roles": ["researcher", "data_steward"]}, "sorted"),
        ({"raw_media_roles": ["researcher"]}, "data_steward"),
        (
            {
                "permitted_roles": ["data_steward"],
                "raw_media_roles": ["data_steward", "model_operator"],
            },
            "subset",
        ),
        ({"prohibited_uses": ["audio_collection"] * 6}, "restricted uses"),
        ({"backup_retention_days": 0}, "nonzero"),
        ({"backup_retention_days": 100}, "30 days"),
        ({"backups_include_raw_media": False}, "must be zero"),
    ],
)
def test_governance_policy_is_explicit_and_fail_closed(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(GovernanceContractError, match=message):
        validate_governance_policy(_policy(**changes))


def test_policy_allows_explicit_no_raw_backup_and_checks_document_types() -> None:
    policy = validate_governance_policy(
        _policy(backups_include_raw_media=False, backup_retention_days=0)
    )
    assert not policy.backups_include_raw_media
    with pytest.raises(GovernanceContractError, match="governance policy"):
        validate_governance_policy(_policy(policy_document=_document("consent_form", "1")))
    with pytest.raises(GovernanceContractError, match="withdrawal_procedure"):
        validate_governance_policy(_policy(withdrawal_procedure=_document("privacy_notice", "2")))
    future = _document("governance_policy", "3")
    future["effective_at"] = "2026-08-27T00:00:00Z"
    with pytest.raises(GovernanceContractError, match="publish a new reviewed version"):
        validate_governance_policy(_policy(policy_document=future))


@pytest.mark.parametrize(
    "changes",
    [
        {"raw_media_retention_days": 731},
        {"derived_feature_retention_days": 731},
        {"evaluation_result_retention_days": 731},
        {"backup_retention_days": 31},
        {"withdrawal_impact_inventory_days": 6},
        {"withdrawal_response_days": 31},
    ],
)
def test_governance_policy_enforces_all_hard_retention_and_withdrawal_caps(
    changes: dict[str, object],
) -> None:
    with pytest.raises(GovernanceContractError, match=r"730|30 days|5 days"):
        validate_governance_policy(_policy(**changes))


def test_policy_exposes_release_reviewer_and_full_prohibited_use_registry() -> None:
    policy = validate_governance_policy(_policy())
    assert "release_reviewer" in policy.permitted_roles
    assert policy.prohibited_uses == (
        "audio_collection",
        "commercial_sale",
        "identity_inference",
        "minor_participation",
        "participant_level_public_ranking",
        "surveillance",
    )


def test_collection_readiness_v1_is_permanently_blocked_without_authenticated_verifier() -> None:
    blocked = validate_collection_readiness(_readiness())
    assert blocked.blockers == tuple(ALL_BLOCKERS)
    assert blocked.status == "blocked"
    with pytest.raises(GovernanceContractError, match="blocked"):
        validate_collection_readiness(_readiness(ready=True))

    with pytest.raises(GovernanceContractError, match="exactly"):
        validate_collection_readiness(_readiness(blockers=[]))
    with pytest.raises(GovernanceContractError, match="blocked"):
        validate_collection_readiness(_readiness(status="ready"))


def test_lineage_inventory_round_trips_with_verified_hash_and_acyclic_order() -> None:
    payload = _inventory()
    inventory = validate_lineage_inventory(json.dumps(payload))
    assert inventory.inventory_sha256 == lineage_inventory_digest(inventory)
    assert [asset.asset_kind for asset in inventory.assets] == [
        "raw_recording",
        "derived_features",
    ]
    assert validate_lineage_inventory(inventory) == inventory


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value["assets"].reverse(), "sorted"),
        (
            lambda value: value["assets"][1].update(logical_uri=value["assets"][0]["logical_uri"]),
            "locator",
        ),
        (
            lambda value: value["assets"][1].update(parent_asset_ids=[_identifier("asset", "9")]),
            "parent_asset_id",
        ),
        (
            lambda value: value["assets"][0].update(
                asset_kind="derived_features",
                parent_asset_ids=[_identifier("asset", "2")],
                logical_uri=(
                    f"signlab://store-{'a' * 32}/derived_features/{_identifier('asset', '1')}"
                ),
            ),
            "acyclic",
        ),
        (lambda value: value.update(inventory_sha256=ZERO_DIGEST), "inventory_sha256"),
    ],
)
def test_lineage_inventory_rejects_order_locator_parent_cycle_and_hash_errors(
    mutator: Callable[[dict[str, object]], object], message: str
) -> None:
    payload = _inventory()
    mutator(payload)
    if message != "inventory_sha256":
        payload["inventory_sha256"] = lineage_inventory_digest(payload)
    with pytest.raises(GovernanceContractError, match=message):
        validate_lineage_inventory(payload)


def test_governance_assets_enforce_sorted_links_root_shape_and_lifecycle() -> None:
    duplicate = _asset("1", participant_ids=[_identifier("participant", "1")] * 2)
    with pytest.raises(GovernanceContractError, match="participant_ids"):
        validate_governance_asset(duplicate)
    with pytest.raises(GovernanceContractError, match="lineage root"):
        validate_governance_asset(_asset("1", parent_asset_ids=[_identifier("asset", "2")]))
    with pytest.raises(GovernanceContractError, match="at least one parent"):
        validate_governance_asset(_asset("2", asset_kind="model_artifact"))
    with pytest.raises(GovernanceContractError, match="invalidated_at"):
        validate_governance_asset(_asset("1", lifecycle_state="invalidated"))
    with pytest.raises(GovernanceContractError, match="only an invalidated"):
        validate_governance_asset(_asset("1", invalidated_at="2026-03-01T00:00:00Z"))
    with pytest.raises(GovernanceContractError, match="cannot precede"):
        validate_governance_asset(
            _asset(
                "1",
                lifecycle_state="invalidated",
                invalidated_at="2026-01-01T00:00:00Z",
            )
        )
    assert (
        validate_governance_asset(
            _asset(
                "1",
                lifecycle_state="invalidated",
                invalidated_at="2026-09-01T00:00:00Z",
            )
        ).lifecycle_state
        == "invalidated"
    )


@pytest.mark.parametrize("asset_kind", ["backup_copy", "cache", "withdrawal_tombstone"])
def test_lineage_can_enumerate_backup_cache_and_tombstone_nodes(asset_kind: str) -> None:
    asset = validate_governance_asset(_asset("2", parent="1", asset_kind=asset_kind))
    assert asset.asset_kind == asset_kind


def test_lineage_rejects_raw_cardinality_metadata_drift_and_future_parents() -> None:
    with pytest.raises(GovernanceContractError, match="exactly one"):
        validate_governance_asset(
            _asset(
                "1",
                participant_ids=[
                    _identifier("participant", "1"),
                    _identifier("participant", "2"),
                ],
            )
        )

    metadata_drift = _inventory()
    metadata_assets = cast(list[dict[str, object]], metadata_drift["assets"])
    metadata_assets[1]["participant_ids"] = [_identifier("participant", "2")]
    metadata_drift["inventory_sha256"] = lineage_inventory_digest(metadata_drift)
    with pytest.raises(GovernanceContractError, match="sorted union"):
        validate_lineage_inventory(metadata_drift)

    future_parent = _inventory()
    future_assets = cast(list[dict[str, object]], future_parent["assets"])
    future_assets[0]["created_at"] = "2026-08-29T00:00:00Z"
    future_parent["inventory_sha256"] = lineage_inventory_digest(future_parent)
    with pytest.raises(GovernanceContractError, match="created after"):
        validate_lineage_inventory(future_parent)


def test_withdrawal_contracts_verify_every_nested_hash_and_count() -> None:
    request = validate_withdrawal_request(_request())
    impact = validate_withdrawal_impact(_impact("1", relationship="direct"))
    report = validate_withdrawal_report(_report())
    assert request.request_sha256 == withdrawal_request_digest(request)
    assert impact.impact_sha256 == withdrawal_impact_digest(impact)
    assert report.report_sha256 == withdrawal_report_digest(report)
    assert report.mode == "dry_run"
    assert report.status == "complete"


def test_withdrawal_request_requires_identity_verification_attestation() -> None:
    payload = _request()
    payload.pop("identity_verification_attestation_sha256")
    payload["request_sha256"] = withdrawal_request_digest(payload)
    with pytest.raises(GovernanceContractError, match="Field required"):
        validate_withdrawal_request(payload)


@pytest.mark.parametrize(
    ("payload_factory", "field", "message"),
    [
        (_request, "request_sha256", "request_sha256"),
        (lambda: _impact("1", relationship="direct"), "impact_sha256", "impact_sha256"),
        (_report, "report_sha256", "report_sha256"),
    ],
)
def test_withdrawal_contracts_reject_tampered_hashes(
    payload_factory: Callable[[], dict[str, object]], field: str, message: str
) -> None:
    payload = payload_factory()
    payload[field] = ZERO_DIGEST
    validator = {
        "request_sha256": validate_withdrawal_request,
        "impact_sha256": validate_withdrawal_impact,
        "report_sha256": validate_withdrawal_report,
    }[field]
    with pytest.raises(GovernanceContractError, match=message):
        validator(payload)


def test_withdrawal_request_impact_and_report_state_invariants() -> None:
    bad_request = _request(effective_at="2026-05-01T00:00:00Z")
    with pytest.raises(GovernanceContractError, match="cannot precede"):
        validate_withdrawal_request(bad_request)
    duplicate_receipts = _request(receipt_ids=[_identifier("receipt", "1")] * 2)
    with pytest.raises(GovernanceContractError, match="receipt_ids"):
        validate_withdrawal_request(duplicate_receipts)

    unsorted_actions = _impact(
        "1", relationship="direct", planned_actions=["invalidate", "delete_primary"]
    )
    with pytest.raises(GovernanceContractError, match="sorted"):
        validate_withdrawal_impact(unsorted_actions)
    missing_invalidation = _impact("1", relationship="direct", planned_actions=["delete_primary"])
    with pytest.raises(GovernanceContractError, match="invalidate"):
        validate_withdrawal_impact(missing_invalidation)

    tombstone = _impact(
        "3",
        relationship="downstream",
        asset_kind="withdrawal_tombstone",
        planned_actions=["retain"],
    )
    assert validate_withdrawal_impact(tombstone).planned_actions == ("retain",)
    bad_tombstone = _impact(
        "3",
        relationship="downstream",
        asset_kind="withdrawal_tombstone",
        planned_actions=["invalidate"],
    )
    with pytest.raises(GovernanceContractError, match="only be retained"):
        validate_withdrawal_impact(bad_tombstone)
    retained_data = _impact(
        "2",
        relationship="downstream",
        planned_actions=["invalidate", "retain"],
    )
    with pytest.raises(GovernanceContractError, match="cannot retain"):
        validate_withdrawal_impact(retained_data)

    wrong_counts = _report(affected_asset_count=1)
    with pytest.raises(GovernanceContractError, match="affected_asset_count"):
        validate_withdrawal_report(wrong_counts)
    blocked = _report(
        status="blocked",
        unresolved_asset_ids=[_identifier("asset", "9")],
    )
    assert validate_withdrawal_report(blocked).status == "blocked"
    wrong_status = _report(status="blocked")
    with pytest.raises(GovernanceContractError, match="must be complete"):
        validate_withdrawal_report(wrong_status)


@pytest.mark.parametrize(
    "prohibited",
    [
        {"Email": "person@example.invalid"},
        {"nested": {"full_name": "private"}},
        {"nested": [{"signature": "private"}]},
        {"notes": "private"},
        {"nested": {"path": "C" + ":" + "\\Users\\person\\recording.mp4"}},
        {"nested": {"path": "/" + "home/person/recording.mp4"}},
        {"nested": {"path": "file" + ":///private/recording.mp4"}},
    ],
)
def test_validation_helpers_recursively_reject_personal_fields_and_machine_paths(
    prohibited: dict[str, object],
) -> None:
    payload = _scope()
    payload.update(prohibited)
    error_fragment = "machine-specific path" if "path" in str(prohibited) else "prohibited"
    with pytest.raises(GovernanceContractError, match=error_fragment) as caught:
        validate_consent_scope(payload)
    assert "person@example.invalid" not in str(caught.value)
    assert "recording.mp4" not in str(caught.value)


def test_validation_errors_never_echo_unknown_user_controlled_keys() -> None:
    unexpected_field = "unexpected_field_marker"
    payload = _scope()
    payload[unexpected_field] = False
    with pytest.raises(GovernanceContractError) as caught:
        validate_consent_scope(payload)
    assert unexpected_field not in str(caught.value)


@pytest.mark.parametrize(
    "uri",
    [
        "C" + ":" + "\\data\\asset.json",
        "/" + "data/asset.json",
        "file" + ":///data/asset.json",
        "https://example.invalid/asset",
        "signlab://assets/../private",
        "signlab://Assets/recording/one",
        "signlab://assets/recording?token=secret",
    ],
)
def test_document_references_accept_only_portable_logical_uris(uri: str) -> None:
    payload = _document("consent_form", "1")
    payload["uri"] = uri
    with pytest.raises(GovernanceContractError):
        validate_document_ref(payload)


@pytest.mark.parametrize("field", ["document_id", "version", "effective_at", "uri", "sha256"])
def test_document_reference_registry_is_immutable(field: str) -> None:
    payload = _document("consent_form", "1")
    payload[field] = {
        "document_id": _identifier("document", "f"),
        "version": "1.0.1",
        "effective_at": "2026-08-27T00:00:00Z",
        "uri": "signlab://governance/consent-form/1.0.1",
        "sha256": "sha256:" + "f" * 64,
    }[field]
    with pytest.raises(GovernanceContractError, match="publish a new reviewed version"):
        validate_document_ref(payload)


def test_asset_and_impact_locators_are_opaque_and_cross_field_bound() -> None:
    asset = _asset("1")
    asset["logical_uri"] = f"signlab://store-{'a' * 32}/cache/{_identifier('asset', '1')}"
    with pytest.raises(GovernanceContractError, match="structural locator"):
        validate_governance_asset(asset)

    impact = _impact("1", relationship="direct")
    impact["logical_uri"] = f"signlab://store-{'a' * 32}/raw_recording/{_identifier('asset', '2')}"
    impact["impact_sha256"] = withdrawal_impact_digest(impact)
    with pytest.raises(GovernanceContractError, match="structural locator"):
        validate_withdrawal_impact(impact)


def test_validation_helpers_reject_duplicate_keys_nonfinite_numbers_and_bad_utf8() -> None:
    with pytest.raises(GovernanceContractError, match="duplicate"):
        validate_consent_scope('{"schema_version":"consent-scope/1","schema_version":"x"}')
    with pytest.raises(GovernanceContractError, match="non-finite"):
        validate_consent_scope('{"value":NaN}')
    with pytest.raises(GovernanceContractError, match="UTF-8"):
        validate_consent_scope(b"\xff")
    with pytest.raises(GovernanceContractError, match="valid JSON"):
        validate_consent_scope("{")
    with pytest.raises(GovernanceContractError, match="JSON object"):
        validate_consent_scope("[]")


def test_raw_mapping_digest_normalizes_integral_json_floats_recursively() -> None:
    report = _report()
    report["affected_asset_count"] = 2.0
    assert withdrawal_report_digest(report) == report["report_sha256"]
    assert validate_withdrawal_report(report).report_sha256 == report["report_sha256"]


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-02-01T00:00:00+00:00",
        "2026-02-01T00:00:00.000Z",
        "2026-02-30T00:00:00Z",
        "2026-2-01T00:00:00Z",
    ],
)
def test_timestamps_require_exact_semantically_valid_utc_z_strings(timestamp: str) -> None:
    payload = _document("consent_form", "1")
    payload["effective_at"] = timestamp
    with pytest.raises(GovernanceContractError, match=r"timestamp|String.*pattern"):
        validate_document_ref(payload)


def test_json_schemas_are_closed_and_expose_v1_discriminators() -> None:
    models = (
        DocumentRef,
        ConsentScopeV1,
        ConsentReceiptV1,
        ConsentEventV1,
        ConsentEventLogV1,
        RecordingConsentGrantV1,
        GovernancePolicyV1,
        CollectionReadinessV1,
        GovernanceAssetV1,
        LineageInventoryV1,
        WithdrawalRequestV1,
        WithdrawalImpactV1,
        WithdrawalReportV1,
    )
    for model in models:
        schema = model.model_json_schema(mode="validation")
        assert schema["additionalProperties"] is False
        assert schema["$id"].startswith("https://signlab.dev/schemas/")
        assert schema["properties"]["schema_version"]["const"].endswith("/1")
