"""Packaged governance policy, schemas, synthetic examples, and evidence."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from importlib.resources import files
from typing import Final

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel

from signlab.contracts.governance import (
    PACKAGED_GOVERNANCE_DOCUMENTS,
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
    lineage_inventory_digest,
    validate_collection_readiness,
    validate_consent_event_log,
    validate_consent_receipt,
    validate_document_ref,
    validate_governance_asset,
    validate_governance_policy,
    validate_lineage_inventory,
    validate_recording_consent_grant,
    validate_withdrawal_report,
    validate_withdrawal_request,
    withdrawal_request_digest,
)
from signlab.contracts.taxonomy import load_builtin_taxonomy, taxonomy_reference
from signlab.governance.withdrawal import (
    WithdrawalPlanningError,
    plan_withdrawal_dry_run,
    render_withdrawal_report_markdown,
)

RESOURCE_PACKAGE: Final = "signlab.resources.governance"
EFFECTIVE_AT: Final = "2026-08-26T00:00:00Z"
REPORT_GENERATED_AT: Final = "2026-08-26T14:00:00Z"

DOCUMENT_RESOURCE_NAMES: Final = {
    "consent_form": "consent-form-1.0.0.md",
    "privacy_notice": "privacy-notice-1.0.0.md",
    "governance_policy": "data-governance-policy-1.0.0.md",
    "withdrawal_procedure": "withdrawal-runbook-1.0.0.md",
}
DOCUMENT_IDS: Final = {
    document_type: f"document_{index:032x}"
    for index, document_type in enumerate(DOCUMENT_RESOURCE_NAMES, start=1)
}
DOCUMENT_URIS: Final = {
    "consent_form": "signlab://governance/consent-form/1.0.0",
    "privacy_notice": "signlab://governance/privacy-notice/1.0.0",
    "governance_policy": "signlab://governance/data-governance-policy/1.0.0",
    "withdrawal_procedure": "signlab://governance/withdrawal-runbook/1.0.0",
}

GOVERNANCE_SCHEMA_MODELS: Final[dict[str, type[BaseModel]]] = {
    "collection-readiness-1.schema.json": CollectionReadinessV1,
    "consent-event-log-1.schema.json": ConsentEventLogV1,
    "consent-event-1.schema.json": ConsentEventV1,
    "consent-receipt-1.schema.json": ConsentReceiptV1,
    "consent-scope-1.schema.json": ConsentScopeV1,
    "governance-asset-1.schema.json": GovernanceAssetV1,
    "governance-document-reference-1.schema.json": DocumentRef,
    "governance-policy-1.schema.json": GovernancePolicyV1,
    "lineage-inventory-1.schema.json": LineageInventoryV1,
    "recording-consent-grant-1.schema.json": RecordingConsentGrantV1,
    "withdrawal-impact-1.schema.json": WithdrawalImpactV1,
    "withdrawal-report-1.schema.json": WithdrawalReportV1,
    "withdrawal-request-1.schema.json": WithdrawalRequestV1,
}

GENERATED_JSON_RESOURCE_NAMES: Final = {
    "collection-readiness.template.json",
    "examples/consent-receipt.example.json",
    "examples/consent-event-log.example.json",
    "examples/lineage-inventory.example.json",
    "examples/recording-consent-grant.example.json",
    "examples/withdrawal-request.example.json",
    "evidence/withdrawal-dry-run-v1.json",
    "governance-policy-1.0.0.json",
    *(f"schemas/{name}" for name in GOVERNANCE_SCHEMA_MODELS),
}
GENERATED_TEXT_RESOURCE_NAMES: Final = {
    "evidence/withdrawal-dry-run-v1.md",
}
GENERATED_RESOURCE_NAMES: Final = GENERATED_JSON_RESOURCE_NAMES | GENERATED_TEXT_RESOURCE_NAMES

_SCHEMA_BOUNDARY_COMMENT: Final = (
    "This JSON Schema enforces portable structure and expressible local invariants. "
    "SignLab Pydantic and application validation remains authoritative for canonical "
    "digests, timestamp ordering, DAG closure, cross-record bindings, event-log "
    "completeness, cross-field locator equality, and withdrawal closure."
)
_SET_LIKE_ARRAY_FIELDS: Final = {
    "assets",
    "blockers",
    "events",
    "grant_ids",
    "impacts",
    "parent_asset_ids",
    "participant_ids",
    "permitted_roles",
    "planned_actions",
    "prohibited_uses",
    "raw_media_roles",
    "receipt_ids",
    "recording_ids",
    "unresolved_asset_ids",
}


class GovernanceResourceError(GovernanceContractError):
    """Raised when packaged governance evidence is missing, invalid, or stale."""


def _schema_nodes(value: object) -> Iterator[dict[str, object]]:
    """Yield schema-object nodes so nested Pydantic ``$defs`` receive the same rules."""

    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _schema_nodes(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _schema_nodes(nested)


def _when_true_requires_true(field: str, *required_fields: str) -> dict[str, object]:
    return {
        "if": {"properties": {field: {"const": True}}, "required": [field]},
        "then": {
            "properties": {required_field: {"const": True} for required_field in required_fields},
            "required": list(required_fields),
        },
    }


def _append_rules(schema: dict[str, object], *rules: dict[str, object]) -> None:
    all_of = schema.setdefault("allOf", [])
    if not isinstance(all_of, list):  # pragma: no cover - generated Pydantic shape is stable.
        raise GovernanceResourceError("generated schema has an incompatible allOf value")
    all_of.extend(rules)


def _harden_consent_scope(schema: dict[str, object]) -> None:
    research_uses = (
        "raw_media_capture",
        "raw_media_retention",
        "raw_media_redistribution",
        "derived_features",
        "model_training",
        "model_evaluation",
        "public_demonstration",
        "model_weights_redistribution",
        "derived_features_redistribution",
        "evaluation_results_redistribution",
        "same_purpose_future_research",
    )
    rules = [_when_true_requires_true(field, "research_use") for field in research_uses]
    rules.extend(
        (
            _when_true_requires_true("raw_media_retention", "raw_media_capture"),
            _when_true_requires_true(
                "raw_media_redistribution", "raw_media_capture", "raw_media_retention"
            ),
            _when_true_requires_true("derived_features", "raw_media_capture"),
            _when_true_requires_true("model_training", "derived_features"),
            _when_true_requires_true("model_evaluation", "derived_features"),
            _when_true_requires_true(
                "public_demonstration", "raw_media_capture", "raw_media_retention"
            ),
            _when_true_requires_true("model_weights_redistribution", "model_training"),
            _when_true_requires_true("derived_features_redistribution", "derived_features"),
            _when_true_requires_true("evaluation_results_redistribution", "model_evaluation"),
        )
    )
    _append_rules(schema, *rules)


def _harden_consent_event(schema: dict[str, object]) -> None:
    terminal_without_replacement = {
        "granted": None,
        "withdrawn": "participant_request",
        "expired": "consent_expired",
    }
    rules: list[dict[str, object]] = []
    for event_type, reason in terminal_without_replacement.items():
        rules.append(
            {
                "if": {
                    "properties": {"event_type": {"const": event_type}},
                    "required": ["event_type"],
                },
                "then": {
                    "properties": {
                        "reason_code": {"const": reason},
                        "replacement_receipt_id": {"type": "null"},
                    }
                },
            }
        )
    rules.append(
        {
            "if": {
                "properties": {"event_type": {"const": "superseded"}},
                "required": ["event_type"],
            },
            "then": {
                "properties": {
                    "reason_code": {"enum": ["policy_replaced", "scope_replaced"]},
                    "replacement_receipt_id": {
                        "type": "string",
                        "pattern": r"^receipt_[0-9a-f]{32}$",
                    },
                }
            },
        }
    )
    _append_rules(schema, *rules)


def _harden_policy(schema: dict[str, object]) -> None:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise GovernanceResourceError("governance policy schema has no properties")
    maxima = {
        "raw_media_retention_days": 730,
        "derived_feature_retention_days": 730,
        "evaluation_result_retention_days": 730,
        "backup_retention_days": 30,
        "withdrawal_impact_inventory_days": 5,
        "withdrawal_response_days": 30,
    }
    for field, maximum in maxima.items():
        field_schema = properties.get(field)
        if not isinstance(field_schema, dict):
            raise GovernanceResourceError(f"governance policy schema is missing {field}")
        field_schema["maximum"] = maximum
    prohibited_schema = properties.get("prohibited_uses")
    if not isinstance(prohibited_schema, dict):
        raise GovernanceResourceError("governance policy schema is missing prohibited uses")
    prohibited_schema["const"] = [
        "audio_collection",
        "commercial_sale",
        "identity_inference",
        "minor_participation",
        "participant_level_public_ranking",
        "surveillance",
    ]
    _append_rules(
        schema,
        {
            "if": {
                "properties": {"backups_include_raw_media": {"const": True}},
                "required": ["backups_include_raw_media"],
            },
            "then": {"properties": {"backup_retention_days": {"minimum": 1}}},
            "else": {"properties": {"backup_retention_days": {"const": 0}}},
        },
    )


def _harden_readiness(schema: dict[str, object]) -> None:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise GovernanceResourceError("collection readiness schema has no properties")
    status_schema = properties.get("status")
    blockers_schema = properties.get("blockers")
    if not isinstance(status_schema, dict) or not isinstance(blockers_schema, dict):
        raise GovernanceResourceError("collection readiness schema is incomplete")
    status_schema["const"] = "blocked"
    blockers_schema["minItems"] = 1
    checks = {
        "access_controls_ready": "access_controls",
        "adult_jurisdictions_ready": "adult_jurisdictions",
        "affiliation_sponsorship_ready": "affiliation_sponsorship",
        "backup_deletion_ready": "backup_deletion",
        "consent_documents_ready": "consent_documents",
        "encrypted_storage_ready": "encrypted_storage",
        "ethics_legal_institutional_ready": "ethics_legal_institutional",
        "identity_vault_operations_ready": "identity_vault_operations",
        "legacy_media_quarantined": "legacy_media_quarantine",
        "lineage_tracking_ready": "lineage_tracking",
        "participant_contact_process_ready": "participant_contact_process",
        "pseudonymous_ids_ready": "pseudonymous_ids",
        "retention_publication_scope_ready": "retention_publication_scope",
        "withdrawal_dry_run_ready": "withdrawal_dry_run",
    }
    check_rules = tuple(
        {
            "if": {"properties": {field: {"const": True}}, "required": [field]},
            "then": {"properties": {"blockers": {"not": {"contains": {"const": blocker}}}}},
            "else": {"properties": {"blockers": {"contains": {"const": blocker}}}},
        }
        for field, blocker in checks.items()
    )
    _append_rules(schema, *check_rules)


def _harden_document_ref(schema: dict[str, object]) -> None:
    registry_rules: list[dict[str, object]] = []
    fields = ("document_id", "version", "effective_at", "uri", "sha256")
    for document_type, values in PACKAGED_GOVERNANCE_DOCUMENTS.items():
        exact = dict(zip(fields, values, strict=True))
        exact["document_type"] = document_type
        registry_rules.append(
            {
                "properties": {field: {"const": value} for field, value in exact.items()},
                "required": ["document_type", *fields],
            }
        )
    schema["oneOf"] = registry_rules


def _harden_asset_locator(schema: dict[str, object]) -> tuple[dict[str, object], ...]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise GovernanceResourceError("governed locator schema has no properties")
    kind_schema = properties.get("asset_kind")
    if not isinstance(kind_schema, dict) or not isinstance(kind_schema.get("enum"), list):
        raise GovernanceResourceError("governed locator schema has no kind registry")
    kinds = tuple(str(kind) for kind in kind_schema["enum"])
    uri_schema = properties.get("logical_uri")
    if not isinstance(uri_schema, dict):
        raise GovernanceResourceError("governed locator schema has no logical URI")
    uri_schema["pattern"] = (
        r"^signlab://store-[0-9a-f]{32}/(?:"
        + "|".join(re.escape(kind) for kind in kinds)
        + r")/asset_[0-9a-f]{32}$"
    )
    return tuple(
        {
            "if": {
                "properties": {"asset_kind": {"const": kind}},
                "required": ["asset_kind"],
            },
            "then": {
                "properties": {
                    "logical_uri": {
                        "pattern": (
                            rf"^signlab://store-[0-9a-f]{{32}}/{re.escape(kind)}/"
                            r"asset_[0-9a-f]{32}$"
                        )
                    }
                }
            },
        }
        for kind in kinds
    )


def _harden_governance_asset(schema: dict[str, object]) -> None:
    kind_uri_rules = _harden_asset_locator(schema)
    _append_rules(
        schema,
        {
            "if": {
                "properties": {"asset_kind": {"const": "raw_recording"}},
                "required": ["asset_kind"],
            },
            "then": {
                "properties": {
                    "parent_asset_ids": {"maxItems": 0},
                    "participant_ids": {"maxItems": 1},
                    "recording_ids": {"maxItems": 1},
                    "receipt_ids": {"maxItems": 1},
                    "grant_ids": {"maxItems": 1},
                }
            },
            "else": {"properties": {"parent_asset_ids": {"minItems": 1}}},
        },
        {
            "if": {
                "properties": {"lifecycle_state": {"const": "invalidated"}},
                "required": ["lifecycle_state"],
            },
            "then": {"properties": {"invalidated_at": {"type": "string"}}},
            "else": {"properties": {"invalidated_at": {"type": "null"}}},
        },
        *kind_uri_rules,
    )


def _harden_withdrawal_impact(schema: dict[str, object]) -> None:
    kind_uri_rules = _harden_asset_locator(schema)
    _append_rules(
        schema,
        {
            "if": {
                "properties": {"asset_kind": {"const": "withdrawal_tombstone"}},
                "required": ["asset_kind"],
            },
            "then": {
                "properties": {
                    "planned_actions": {
                        "const": ["retain"],
                    }
                }
            },
            "else": {
                "properties": {
                    "planned_actions": {
                        "contains": {"const": "invalidate"},
                        "not": {"contains": {"const": "retain"}},
                    }
                }
            },
        },
        *kind_uri_rules,
    )


def _harden_withdrawal_report(schema: dict[str, object]) -> None:
    _append_rules(
        schema,
        {
            "if": {
                "properties": {"unresolved_asset_ids": {"maxItems": 0}},
                "required": ["unresolved_asset_ids"],
            },
            "then": {"properties": {"status": {"const": "complete"}}},
            "else": {"properties": {"status": {"const": "blocked"}}},
        },
    )


def _harden_generated_schema(schema: dict[str, object]) -> None:
    """Add portable constraints omitted by Pydantic's generated JSON Schema.

    JSON Schema cannot compare hashes, timestamps, graph ancestry, IDs embedded in
    locators, or records in an attested event log. Those checks intentionally remain
    in the authoritative contract and application validators named in ``$comment``.
    """

    schema["$comment"] = _SCHEMA_BOUNDARY_COMMENT
    for node in _schema_nodes(schema):
        properties = node.get("properties")
        if not isinstance(properties, dict):
            continue
        for field in _SET_LIKE_ARRAY_FIELDS.intersection(properties):
            array_schema = properties[field]
            if isinstance(array_schema, dict):
                array_schema["uniqueItems"] = True
        version = properties.get("schema_version")
        if not isinstance(version, dict):
            continue
        discriminator = version.get("const")
        if discriminator == "document-reference/1":
            _harden_document_ref(node)
        elif discriminator == "consent-scope/1":
            _harden_consent_scope(node)
        elif discriminator == "consent-event/1":
            _harden_consent_event(node)
        elif discriminator == "governance-policy/1":
            _harden_policy(node)
        elif discriminator == "collection-readiness/1":
            _harden_readiness(node)
        elif discriminator == "governance-asset/1":
            _harden_governance_asset(node)
        elif discriminator == "withdrawal-impact/1":
            _harden_withdrawal_impact(node)
        elif discriminator == "withdrawal-report/1":
            _harden_withdrawal_report(node)


def _digest_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _synthetic_digest(label: str) -> str:
    return _digest_bytes(f"SignLab synthetic governance fixture: {label}\n".encode())


def _resource_bytes(name: str) -> bytes:
    try:
        return files(RESOURCE_PACKAGE).joinpath(name).read_bytes()
    except OSError as error:
        raise GovernanceResourceError("a packaged governance resource is missing") from error


def _document_references() -> dict[str, DocumentRef]:
    references: dict[str, DocumentRef] = {}
    for document_type, resource_name in DOCUMENT_RESOURCE_NAMES.items():
        references[document_type] = validate_document_ref(
            {
                "schema_version": "document-reference/1",
                "document_id": DOCUMENT_IDS[document_type],
                "document_type": document_type,
                "version": "1.0.0",
                "effective_at": EFFECTIVE_AT,
                "uri": DOCUMENT_URIS[document_type],
                "sha256": _digest_bytes(_resource_bytes(resource_name)),
            }
        )
    return references


def generated_governance_schemas() -> dict[str, dict[str, object]]:
    """Generate Draft 2020-12 schemas from the authoritative Pydantic models."""

    def keep_only_root_resource_identity(node: object, *, root: bool = False) -> None:
        # Pydantic carries each model's standalone $id into nested $defs. In JSON
        # Schema, that starts a new resource boundary and makes local #/$defs refs
        # resolve against the nested model rather than this complete document.
        if isinstance(node, dict):
            if not root:
                node.pop("$id", None)
            for value in node.values():
                keep_only_root_resource_identity(value)
        elif isinstance(node, list):
            for value in node:
                keep_only_root_resource_identity(value)

    generated: dict[str, dict[str, object]] = {}
    for filename, model in GOVERNANCE_SCHEMA_MODELS.items():
        schema = model.model_json_schema(mode="validation")
        keep_only_root_resource_identity(schema, root=True)
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        _harden_generated_schema(schema)
        generated[filename] = schema
    return generated


def build_governance_policy() -> GovernancePolicyV1:
    """Build the immutable public engineering policy from reviewed documents."""

    references = _document_references()
    return validate_governance_policy(
        {
            "schema_version": "governance-policy/1",
            "policy_id": "policy_00000000000000000000000000000001",
            "version": "1.0.0",
            "taxonomy": taxonomy_reference(load_builtin_taxonomy()).model_dump(mode="json"),
            "policy_document": references["governance_policy"].model_dump(mode="json"),
            "withdrawal_procedure": references["withdrawal_procedure"].model_dump(mode="json"),
            "effective_at": EFFECTIVE_AT,
            "permitted_roles": [
                "data_steward",
                "model_operator",
                "release_reviewer",
                "researcher",
            ],
            "raw_media_roles": ["data_steward", "researcher"],
            "raw_media_retention_days": 730,
            "derived_feature_retention_days": 730,
            "evaluation_result_retention_days": 730,
            "backup_retention_days": 30,
            "withdrawal_impact_inventory_days": 5,
            "withdrawal_response_days": 30,
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
    )


def build_collection_readiness() -> CollectionReadinessV1:
    """Build the truthful pre-collection gate; unresolved human checks stay blocked."""

    policy = build_governance_policy()
    blockers = (
        "access_controls",
        "adult_jurisdictions",
        "affiliation_sponsorship",
        "backup_deletion",
        "consent_documents",
        "encrypted_storage",
        "ethics_legal_institutional",
        "identity_vault_operations",
        "participant_contact_process",
        "retention_publication_scope",
    )
    return validate_collection_readiness(
        {
            "schema_version": "collection-readiness/1",
            "readiness_id": "readiness_00000000000000000000000000000001",
            "policy_id": policy.policy_id,
            "taxonomy": policy.taxonomy.model_dump(mode="json"),
            "assessed_at": REPORT_GENERATED_AT,
            "status": "blocked",
            "consent_documents_ready": False,
            "pseudonymous_ids_ready": True,
            "encrypted_storage_ready": False,
            "access_controls_ready": False,
            "adult_jurisdictions_ready": False,
            "affiliation_sponsorship_ready": False,
            "backup_deletion_ready": False,
            "ethics_legal_institutional_ready": False,
            "identity_vault_operations_ready": False,
            "withdrawal_dry_run_ready": True,
            "lineage_tracking_ready": True,
            "legacy_media_quarantined": True,
            "participant_contact_process_ready": False,
            "retention_publication_scope_ready": False,
            "blockers": blockers,
        }
    )


def _example_scope() -> ConsentScopeV1:
    return ConsentScopeV1.model_validate(
        {
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
            "derived_features_redistribution": False,
            "evaluation_results_redistribution": True,
            "same_purpose_future_research": False,
            "withdrawal_supported": True,
            "audio_collection": False,
            "minor_participation": False,
            "identity_inference": False,
            "commercial_sale": False,
        },
        strict=True,
    )


def build_example_receipt() -> ConsentReceiptV1:
    """Build a fully synthetic, identity-free consent receipt example."""

    references = _document_references()
    scope = _example_scope()
    return validate_consent_receipt(
        {
            "schema_version": "consent-receipt/1",
            "receipt_id": "receipt_00000000000000000000000000000011",
            "participant_id": "participant_00000000000000000000000000000001",
            "purpose_id": "purpose_00000000000000000000000000000051",
            "study_id": "study_00000000000000000000000000000061",
            "consent_form": references["consent_form"].model_dump(mode="json"),
            "privacy_notice": references["privacy_notice"].model_dump(mode="json"),
            "governance_policy": references["governance_policy"].model_dump(mode="json"),
            "taxonomy": taxonomy_reference(load_builtin_taxonomy()).model_dump(mode="json"),
            "scope": scope.model_dump(mode="json"),
            "scope_sha256": consent_scope_digest(scope),
            "granted_at": "2026-08-26T12:00:00Z",
            "valid_until": "2028-08-25T12:00:00Z",
            "completed_form_sha256": _synthetic_digest("completed-consent-form"),
            "identity_vault_attestation_sha256": _synthetic_digest(
                "identity-vault-consent-attestation"
            ),
            "adult_attested": True,
        }
    )


def build_example_consent_event_log() -> ConsentEventLogV1:
    """Build an attested synthetic event-log snapshot complete through capture."""

    receipt = build_example_receipt()
    payload: dict[str, object] = {
        "schema_version": "consent-event-log/1",
        "event_log_id": "event_log_00000000000000000000000000000041",
        "receipt_id": receipt.receipt_id,
        "receipt_sha256": consent_receipt_digest(receipt),
        "participant_id": receipt.participant_id,
        "purpose_id": receipt.purpose_id,
        "study_id": receipt.study_id,
        "scope_sha256": receipt.scope_sha256,
        "generated_at": "2026-08-26T12:30:00Z",
        "complete_through": "2026-08-26T12:20:00Z",
        "completeness_attested": True,
        "identity_vault_attestation_sha256": _synthetic_digest(
            "identity-vault-event-log-attestation"
        ),
        "events": [
            {
                "schema_version": "consent-event/1",
                "event_id": "event_00000000000000000000000000000041",
                "receipt_id": receipt.receipt_id,
                "participant_id": receipt.participant_id,
                "event_type": "granted",
                "occurred_at": receipt.granted_at,
                "scope_sha256": receipt.scope_sha256,
                "reason_code": None,
                "replacement_receipt_id": None,
            }
        ],
        "event_log_sha256": "sha256:" + "0" * 64,
    }
    payload["event_log_sha256"] = consent_event_log_digest(payload)
    return validate_consent_event_log(payload)


def build_example_recording_grant() -> RecordingConsentGrantV1:
    """Build a per-recording scope snapshot no broader than its synthetic receipt."""

    receipt = build_example_receipt()
    scope = receipt.scope.model_copy(update={"model_weights_redistribution": False})
    grant = validate_recording_consent_grant(
        {
            "schema_version": "recording-consent-grant/1",
            "grant_id": "grant_00000000000000000000000000000021",
            "recording_id": "recording_00000000000000000000000000000031",
            "participant_id": receipt.participant_id,
            "receipt_id": receipt.receipt_id,
            "purpose_id": receipt.purpose_id,
            "study_id": receipt.study_id,
            "taxonomy": receipt.taxonomy.model_dump(mode="json"),
            "scope": scope.model_dump(mode="json"),
            "scope_sha256": consent_scope_digest(scope),
            "receipt_scope_sha256": receipt.scope_sha256,
            "issued_at": "2026-08-26T12:05:00Z",
            "captured_at": "2026-08-26T12:10:00Z",
        }
    )
    assert_receipt_grant_consistent(receipt, grant, build_example_consent_event_log())
    return grant


def _asset(
    index: int,
    kind: str,
    *,
    participants: tuple[str, ...],
    recordings: tuple[str, ...],
    receipts: tuple[str, ...],
    grants: tuple[str, ...],
    parents: tuple[str, ...] = (),
    created_at: str = "2026-08-26T12:00:00Z",
) -> GovernanceAssetV1:
    asset_id = f"asset_{index:032x}"
    return validate_governance_asset(
        {
            "schema_version": "governance-asset/1",
            "asset_id": asset_id,
            "asset_kind": kind,
            "logical_uri": (f"signlab://store-00000000000000000000000000000001/{kind}/{asset_id}"),
            "sha256": _synthetic_digest(f"asset-{index}"),
            "taxonomy": taxonomy_reference(load_builtin_taxonomy()).model_dump(mode="json"),
            "created_at": created_at,
            "participant_ids": participants,
            "recording_ids": recordings,
            "receipt_ids": receipts,
            "grant_ids": grants,
            "parent_asset_ids": parents,
            "lifecycle_state": "active",
            "invalidated_at": None,
        }
    )


def build_example_inventory() -> LineageInventoryV1:
    """Build a synthetic two-participant graph with shared and isolated descendants."""

    participant_a = "participant_00000000000000000000000000000001"
    participant_b = "participant_00000000000000000000000000000002"
    recording_a = "recording_00000000000000000000000000000031"
    recording_b = "recording_00000000000000000000000000000032"
    receipt_a = "receipt_00000000000000000000000000000011"
    receipt_b = "receipt_00000000000000000000000000000012"
    grant_a = "grant_00000000000000000000000000000021"
    grant_b = "grant_00000000000000000000000000000022"

    a = (participant_a,), (recording_a,), (receipt_a,), (grant_a,)
    b = (participant_b,), (recording_b,), (receipt_b,), (grant_b,)
    shared = (
        (participant_a, participant_b),
        (recording_a, recording_b),
        (receipt_a, receipt_b),
        (grant_a, grant_b),
    )
    assets = (
        _asset(1, "raw_recording", participants=a[0], recordings=a[1], receipts=a[2], grants=a[3]),
        _asset(2, "raw_recording", participants=b[0], recordings=b[1], receipts=b[2], grants=b[3]),
        _asset(
            3,
            "derived_features",
            participants=a[0],
            recordings=a[1],
            receipts=a[2],
            grants=a[3],
            parents=(f"asset_{1:032x}",),
        ),
        _asset(
            4,
            "derived_features",
            participants=b[0],
            recordings=b[1],
            receipts=b[2],
            grants=b[3],
            parents=(f"asset_{2:032x}",),
        ),
        _asset(
            5,
            "annotation",
            participants=a[0],
            recordings=a[1],
            receipts=a[2],
            grants=a[3],
            parents=(f"asset_{1:032x}",),
        ),
        _asset(
            6,
            "dataset_version",
            participants=shared[0],
            recordings=shared[1],
            receipts=shared[2],
            grants=shared[3],
            parents=(f"asset_{3:032x}", f"asset_{4:032x}", f"asset_{5:032x}"),
        ),
        _asset(
            7,
            "dataset_version",
            participants=b[0],
            recordings=b[1],
            receipts=b[2],
            grants=b[3],
            parents=(f"asset_{4:032x}",),
        ),
        _asset(
            8,
            "split_version",
            participants=shared[0],
            recordings=shared[1],
            receipts=shared[2],
            grants=shared[3],
            parents=(f"asset_{6:032x}",),
        ),
        _asset(
            9,
            "experiment_run",
            participants=shared[0],
            recordings=shared[1],
            receipts=shared[2],
            grants=shared[3],
            parents=(f"asset_{8:032x}",),
        ),
        _asset(
            10,
            "model_artifact",
            participants=shared[0],
            recordings=shared[1],
            receipts=shared[2],
            grants=shared[3],
            parents=(f"asset_{9:032x}",),
        ),
        _asset(
            11,
            "evaluation_report",
            participants=shared[0],
            recordings=shared[1],
            receipts=shared[2],
            grants=shared[3],
            parents=(f"asset_{8:032x}", f"asset_{10:032x}"),
        ),
        _asset(
            12,
            "public_demo",
            participants=shared[0],
            recordings=shared[1],
            receipts=shared[2],
            grants=shared[3],
            parents=(f"asset_{10:032x}", f"asset_{11:032x}"),
        ),
        _asset(
            13,
            "experiment_run",
            participants=b[0],
            recordings=b[1],
            receipts=b[2],
            grants=b[3],
            parents=(f"asset_{7:032x}",),
        ),
        _asset(
            14,
            "model_artifact",
            participants=b[0],
            recordings=b[1],
            receipts=b[2],
            grants=b[3],
            parents=(f"asset_{13:032x}",),
        ),
        _asset(
            15,
            "cache",
            participants=a[0],
            recordings=a[1],
            receipts=a[2],
            grants=a[3],
            parents=(f"asset_{3:032x}",),
        ),
        _asset(
            16,
            "backup_copy",
            participants=a[0],
            recordings=a[1],
            receipts=a[2],
            grants=a[3],
            parents=(f"asset_{1:032x}",),
        ),
        _asset(
            17,
            "withdrawal_tombstone",
            participants=a[0],
            recordings=a[1],
            receipts=a[2],
            grants=a[3],
            parents=(f"asset_{1:032x}",),
            created_at="2026-08-26T13:10:00Z",
        ),
    )
    payload: dict[str, object] = {
        "schema_version": "lineage-inventory/1",
        "inventory_id": "inventory_00000000000000000000000000000001",
        "taxonomy": taxonomy_reference(load_builtin_taxonomy()).model_dump(mode="json"),
        "generated_at": "2026-08-26T13:30:00Z",
        "assets": [asset.model_dump(mode="json") for asset in assets],
        "inventory_sha256": "sha256:" + "0" * 64,
    }
    payload["inventory_sha256"] = lineage_inventory_digest(payload)
    return validate_lineage_inventory(payload)


def build_example_withdrawal_request() -> WithdrawalRequestV1:
    """Build a synthetic all-data request for the shared-graph participant."""

    payload: dict[str, object] = {
        "schema_version": "withdrawal-request/1",
        "request_id": "withdrawal_00000000000000000000000000000001",
        "participant_id": "participant_00000000000000000000000000000001",
        "receipt_ids": ["receipt_00000000000000000000000000000011"],
        "requested_at": "2026-08-26T13:00:00Z",
        "effective_at": "2026-08-26T13:05:00Z",
        "target": "all_participant_data",
        "identity_verification_attestation_sha256": _synthetic_digest(
            "withdrawal-identity-verification-attestation"
        ),
        "request_sha256": "sha256:" + "0" * 64,
    }
    payload["request_sha256"] = withdrawal_request_digest(payload)
    return validate_withdrawal_request(payload)


def build_example_withdrawal_report() -> WithdrawalReportV1:
    """Recompute the golden, read-only withdrawal closure from synthetic inputs."""

    return plan_withdrawal_dry_run(
        build_example_withdrawal_request(),
        build_example_inventory(),
        generated_at=REPORT_GENERATED_AT,
    )


def render_json_document(value: BaseModel | Mapping[str, object]) -> str:
    """Return stable, reviewable JSON text with a single trailing newline."""

    payload: object
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json", round_trip=True)
    else:
        payload = dict(value)
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def generated_governance_resource_texts() -> dict[str, str]:
    """Render every generated governance package resource deterministically."""

    report = build_example_withdrawal_report()
    rendered: dict[str, str] = {
        "governance-policy-1.0.0.json": render_json_document(build_governance_policy()),
        "collection-readiness.template.json": render_json_document(build_collection_readiness()),
        "examples/consent-event-log.example.json": render_json_document(
            build_example_consent_event_log()
        ),
        "examples/consent-receipt.example.json": render_json_document(build_example_receipt()),
        "examples/recording-consent-grant.example.json": render_json_document(
            build_example_recording_grant()
        ),
        "examples/lineage-inventory.example.json": render_json_document(build_example_inventory()),
        "examples/withdrawal-request.example.json": render_json_document(
            build_example_withdrawal_request()
        ),
        "evidence/withdrawal-dry-run-v1.json": render_json_document(report),
        "evidence/withdrawal-dry-run-v1.md": render_withdrawal_report_markdown(report),
    }
    rendered.update(
        {
            f"schemas/{filename}": render_json_document(schema)
            for filename, schema in generated_governance_schemas().items()
        }
    )
    if set(rendered) != GENERATED_RESOURCE_NAMES:
        raise GovernanceResourceError("generated governance resource registry is incomplete")
    return rendered


def validate_packaged_governance_resources() -> None:
    """Validate document integrity, schemas, examples, and golden dry-run evidence."""

    try:
        generated = generated_governance_resource_texts()
        for name, expected_text in generated.items():
            actual = _resource_bytes(name).decode("utf-8")
            if actual != expected_text:
                raise GovernanceResourceError(
                    "packaged governance resource drift detected; regenerate resources"
                )
        for filename, schema in generated_governance_schemas().items():
            Draft202012Validator.check_schema(schema)
            packaged_schema = json.loads(_resource_bytes(f"schemas/{filename}"))
            if packaged_schema != schema:
                raise GovernanceResourceError("packaged governance schema drift detected")

        receipt = validate_consent_receipt(_resource_bytes("examples/consent-receipt.example.json"))
        event_log = validate_consent_event_log(
            _resource_bytes("examples/consent-event-log.example.json")
        )
        grant = validate_recording_consent_grant(
            _resource_bytes("examples/recording-consent-grant.example.json")
        )
        assert_receipt_grant_consistent(receipt, grant, event_log)
        inventory = validate_lineage_inventory(
            _resource_bytes("examples/lineage-inventory.example.json")
        )
        request = validate_withdrawal_request(
            _resource_bytes("examples/withdrawal-request.example.json")
        )
        report = validate_withdrawal_report(_resource_bytes("evidence/withdrawal-dry-run-v1.json"))
        recomputed = plan_withdrawal_dry_run(
            request,
            inventory,
            generated_at=report.generated_at,
        )
        if recomputed != report:
            raise GovernanceResourceError("withdrawal evidence omits or changes an impact")
        readiness = validate_collection_readiness(
            _resource_bytes("collection-readiness.template.json")
        )
        if readiness.status != "blocked" or not readiness.blockers:
            raise GovernanceResourceError("real collection readiness must fail closed")
    except GovernanceResourceError:
        raise
    except (
        GovernanceContractError,
        WithdrawalPlanningError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        SchemaError,
    ) as error:
        raise GovernanceResourceError(
            "packaged governance resources are missing, invalid, or inconsistent"
        ) from error


__all__ = [
    "DOCUMENT_RESOURCE_NAMES",
    "GENERATED_RESOURCE_NAMES",
    "GOVERNANCE_SCHEMA_MODELS",
    "GovernanceResourceError",
    "build_collection_readiness",
    "build_example_consent_event_log",
    "build_example_inventory",
    "build_example_receipt",
    "build_example_recording_grant",
    "build_example_withdrawal_report",
    "build_example_withdrawal_request",
    "build_governance_policy",
    "generated_governance_resource_texts",
    "generated_governance_schemas",
    "render_json_document",
    "validate_packaged_governance_resources",
]
