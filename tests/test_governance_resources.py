from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from scripts.generate_governance_resources import write_resources

from signlab.contracts.governance import GovernanceContractError, validate_lineage_inventory
from signlab.governance.resources import (
    DOCUMENT_RESOURCE_NAMES,
    GENERATED_RESOURCE_NAMES,
    GOVERNANCE_SCHEMA_MODELS,
    build_collection_readiness,
    build_example_consent_event_log,
    build_example_inventory,
    build_example_receipt,
    build_example_recording_grant,
    build_example_withdrawal_report,
    build_example_withdrawal_request,
    build_governance_policy,
    generated_governance_resource_texts,
    generated_governance_schemas,
    validate_packaged_governance_resources,
)


def test_every_governance_schema_is_valid_and_committed_without_drift() -> None:
    generated = generated_governance_schemas()

    assert set(generated) == set(GOVERNANCE_SCHEMA_MODELS)
    for filename, schema in generated.items():
        Draft202012Validator.check_schema(schema)
        definitions = schema.get("$defs", {})
        assert isinstance(definitions, dict)
        assert all(
            isinstance(definition, dict) and "$id" not in definition
            for definition in definitions.values()
        )
        packaged = json.loads(
            files("signlab.resources.governance")
            .joinpath("schemas", filename)
            .read_text(encoding="utf-8")
        )
        assert packaged == schema


def test_json_schemas_state_the_semantic_boundary_and_reject_expressible_bypasses() -> None:
    schemas = generated_governance_schemas()
    for schema in schemas.values():
        comment = schema.get("$comment")
        assert isinstance(comment, str)
        assert "Pydantic and application validation remains authoritative" in comment
    policy_schema = schemas["governance-policy-1.schema.json"]
    policy_properties = policy_schema["properties"]
    assert isinstance(policy_properties, dict)
    permitted_roles = policy_properties["permitted_roles"]
    assert isinstance(permitted_roles, dict)
    assert permitted_roles["uniqueItems"] is True

    receipt = build_example_receipt().model_dump(mode="json")
    receipt["scope"]["raw_media_capture"] = False
    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(schemas["consent-receipt-1.schema.json"]).validate(receipt)

    inventory = build_example_inventory().model_dump(mode="json")
    inventory["assets"][0]["logical_uri"] = "signlab://recordings/alice-private"
    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(schemas["lineage-inventory-1.schema.json"]).validate(inventory)

    policy = build_governance_policy().model_dump(mode="json")
    policy["permitted_roles"] = ["data_steward", "researcher", "researcher"]
    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(policy_schema).validate(policy)


@pytest.mark.parametrize("field", ["document_id", "sha256"])
def test_json_schema_rejects_unregistered_governance_document_references(field: str) -> None:
    schema = generated_governance_schemas()["consent-receipt-1.schema.json"]
    receipt = build_example_receipt().model_dump(mode="json")
    receipt["consent_form"][field] = (
        "document_ffffffffffffffffffffffffffffffff"
        if field == "document_id"
        else "sha256:" + "f" * 64
    )

    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(schema).validate(receipt)


def test_collection_readiness_schema_only_represents_permanently_blocked_v1_state() -> None:
    schema = generated_governance_schemas()["collection-readiness-1.schema.json"]
    properties = schema["properties"]
    assert isinstance(properties, dict)
    status = properties["status"]
    blockers = properties["blockers"]
    assert isinstance(status, dict)
    assert isinstance(blockers, dict)
    assert status["const"] == "blocked"
    assert blockers["minItems"] == 1

    ready = build_collection_readiness().model_dump(mode="json")
    ready["status"] = "ready"
    ready["blockers"] = []
    for field in tuple(ready):
        if field.endswith("_ready") or field == "legacy_media_quarantined":
            ready[field] = True
    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(schema).validate(ready)


def test_json_schema_does_not_claim_non_expressible_digest_or_graph_validation() -> None:
    schema = generated_governance_schemas()["lineage-inventory-1.schema.json"]
    inventory = build_example_inventory().model_dump(mode="json")
    inventory["inventory_sha256"] = "sha256:" + "f" * 64

    # JSON Schema deliberately accepts structurally valid hashes. The authoritative
    # application validator performs canonical digest, DAG, ancestry, and closure checks.
    Draft202012Validator(schema).validate(inventory)
    with pytest.raises(GovernanceContractError, match="inventory_sha256"):
        validate_lineage_inventory(inventory)


def test_generated_examples_satisfy_their_portable_json_schemas() -> None:
    schemas = generated_governance_schemas()
    examples = {
        "consent-receipt-1.schema.json": build_example_receipt(),
        "consent-event-log-1.schema.json": build_example_consent_event_log(),
        "recording-consent-grant-1.schema.json": build_example_recording_grant(),
        "governance-policy-1.schema.json": build_governance_policy(),
        "collection-readiness-1.schema.json": build_collection_readiness(),
        "lineage-inventory-1.schema.json": build_example_inventory(),
        "withdrawal-request-1.schema.json": build_example_withdrawal_request(),
        "withdrawal-report-1.schema.json": build_example_withdrawal_report(),
    }

    for filename, model in examples.items():
        Draft202012Validator(schemas[filename]).validate(model.model_dump(mode="json"))


def test_resource_generator_is_complete_and_byte_stable(tmp_path: Path) -> None:
    expected = generated_governance_resource_texts()

    assert set(expected) == GENERATED_RESOURCE_NAMES
    written = write_resources(tmp_path)
    assert {path.relative_to(tmp_path).as_posix() for path in written} == set(expected)
    for relative_name, content in expected.items():
        assert tmp_path.joinpath(*relative_name.split("/")).read_bytes() == content.encode()


def test_packaged_resources_reproduce_and_validate_without_external_state() -> None:
    validate_packaged_governance_resources()


def test_collection_readiness_truthfully_blocks_every_unverified_real_world_gate() -> None:
    readiness = build_collection_readiness()

    assert readiness.status == "blocked"
    assert readiness.pseudonymous_ids_ready
    assert readiness.withdrawal_dry_run_ready
    assert readiness.lineage_tracking_ready
    assert readiness.legacy_media_quarantined
    assert set(readiness.blockers) == {
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
    }


def test_synthetic_consent_evidence_is_attested_and_complete_through_capture() -> None:
    receipt = build_example_receipt()
    event_log = build_example_consent_event_log()
    grant = build_example_recording_grant()

    assert receipt.completed_form_sha256.startswith("sha256:")
    assert receipt.identity_vault_attestation_sha256.startswith("sha256:")
    assert receipt.purpose_id == grant.purpose_id
    assert receipt.study_id == grant.study_id
    assert event_log.completeness_attested
    assert event_log.event_log_sha256.startswith("sha256:")
    assert event_log.identity_vault_attestation_sha256.startswith("sha256:")
    assert event_log.complete_through >= grant.captured_at
    assert event_log.receipt_id == receipt.receipt_id == grant.receipt_id
    assert event_log.participant_id == receipt.participant_id == grant.participant_id
    assert event_log.purpose_id == receipt.purpose_id == grant.purpose_id
    assert event_log.study_id == receipt.study_id == grant.study_id
    assert event_log.receipt_sha256.startswith("sha256:")


def test_blank_participant_documents_cover_scope_risk_retention_and_withdrawal() -> None:
    root = files("signlab.resources.governance")
    consent = root.joinpath(DOCUMENT_RESOURCE_NAMES["consent_form"]).read_text(encoding="utf-8")
    privacy = root.joinpath(DOCUMENT_RESOURCE_NAMES["privacy_notice"]).read_text(encoding="utf-8")
    policy = root.joinpath(DOCUMENT_RESOURCE_NAMES["governance_policy"]).read_text(encoding="utf-8")
    runbook = root.joinpath(DOCUMENT_RESOURCE_NAMES["withdrawal_procedure"]).read_text(
        encoding="utf-8"
    )

    for phrase in (
        "NOT APPROVED FOR REAL COLLECTION",
        "not a sign-language translator",
        "derived features",
        "model weights",
        "redistributed",
        "24 months",
        "Withdrawal",
        "Participant signature",
        "{{STUDY_CONTACT}}",
        "{{JURISDICTIONS_AND_REQUIRED_REVIEW}}",
    ):
        assert phrase in consent
    assert "Pseudonymous data is not guaranteed anonymous" in privacy
    assert "Missing permission fails closed" in policy
    assert "does not delete" in runbook


def test_synthetic_evidence_enumerates_shared_backups_caches_and_tombstone() -> None:
    report = build_example_withdrawal_report()
    kinds = {impact.asset_kind for impact in report.impacts}
    actions = {impact.asset_kind: impact.planned_actions for impact in report.impacts}

    assert report.status == "complete"
    assert report.direct_asset_count == 1
    assert report.downstream_asset_count == 11
    assert report.affected_asset_count == 12
    assert {
        "raw_recording",
        "backup_copy",
        "cache",
        "derived_features",
        "annotation",
        "dataset_version",
        "split_version",
        "experiment_run",
        "model_artifact",
        "evaluation_report",
        "public_demo",
        "withdrawal_tombstone",
    } == kinds
    assert actions["backup_copy"] == ("invalidate", "purge_backup")
    assert actions["cache"] == ("delete_primary", "invalidate", "rebuild")
    assert actions["dataset_version"] == ("invalidate", "rebuild")
    assert actions["experiment_run"] == ("invalidate", "rerun")
    assert actions["withdrawal_tombstone"] == ("retain",)


def test_synthetic_lineage_uses_opaque_locators_and_propagates_parent_metadata() -> None:
    inventory = build_example_inventory()
    by_id = {asset.asset_id: asset for asset in inventory.assets}

    for asset in inventory.assets:
        assert asset.logical_uri == (
            f"signlab://store-00000000000000000000000000000001/{asset.asset_kind}/{asset.asset_id}"
        )
        if not asset.parent_asset_ids:
            continue
        parents = tuple(by_id[parent_id] for parent_id in asset.parent_asset_ids)
        for field in ("participant_ids", "recording_ids", "receipt_ids", "grant_ids"):
            expected = tuple(
                sorted({item for parent in parents for item in getattr(parent, field)})
            )
            assert getattr(asset, field) == expected


def test_public_json_examples_have_no_identity_values_or_machine_paths() -> None:
    rendered = "\n".join(
        content
        for name, content in generated_governance_resource_texts().items()
        if name.endswith(".json")
    )

    windows_user_root = "C:" + chr(92) + "Users" + chr(92)
    posix_home_root = "/" + "home" + "/"
    for forbidden in (
        windows_user_root,
        posix_home_root,
        "@example.",
        "participant name",
        "phone number",
        "signed by",
    ):
        assert forbidden.casefold() not in rendered.casefold()
