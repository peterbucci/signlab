"""Strict, privacy-preserving governance and consent contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Annotated, Final, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from signlab.contracts.taxonomy import (
    JsonNonNegativeInteger,
    JsonPositiveInteger,
    SemanticVersion,
    Sha256Digest,
    StrictFrozenModel,
    TaxonomyRef,
    canonical_json_bytes,
)

SCHEMA_BASE: Final = "https://signlab.dev/schemas/"
PROHIBITED_USES: Final = (
    "audio_collection",
    "commercial_sale",
    "identity_inference",
    "minor_participation",
    "participant_level_public_ranking",
    "surveillance",
)
SCOPE_PERMISSION_FIELDS: Final = (
    "research_use",
    "raw_media_capture",
    "model_training",
    "model_evaluation",
    "public_demonstration",
    "model_weights_redistribution",
    "raw_media_retention",
    "raw_media_redistribution",
    "derived_features",
    "derived_features_redistribution",
    "evaluation_results_redistribution",
    "same_purpose_future_research",
)


class GovernanceContractError(ValueError):
    """Raised when governance data violates the public, privacy-safe contract."""


def _governance_config(schema_name: str) -> ConfigDict:
    return ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        hide_input_in_errors=True,
        json_schema_extra={"$id": f"{SCHEMA_BASE}{schema_name}"},
    )


def _validate_utc_timestamp(value: str) -> str:
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError(
            "timestamp must be an exact UTC value in YYYY-MM-DDTHH:MM:SSZ form"
        ) from error
    return value


def _as_datetime(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _validate_logical_uri(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "signlab"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
    ):
        raise ValueError("locator must be a portable signlab:// logical URI")
    if re.fullmatch(r"[a-z][a-z0-9-]*", parsed.netloc) is None:
        raise ValueError("logical URI authority must use lowercase portable characters")
    segments = parsed.path.removeprefix("/").split("/")
    if (
        not parsed.path.startswith("/")
        or not segments
        or any(re.fullmatch(r"[a-z0-9][a-z0-9._-]*", segment) is None for segment in segments)
        or any(segment in {".", ".."} for segment in segments)
    ):
        raise ValueError("logical URI path must contain canonical lowercase segments")
    return value


def _is_asset_logical_uri(value: str, asset_kind: str, asset_id: str) -> bool:
    parsed = urlsplit(value)
    return (
        re.fullmatch(r"store-[0-9a-f]{32}", parsed.netloc) is not None
        and parsed.path == f"/{asset_kind}/{asset_id}"
    )


UtcTimestamp = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"),
    AfterValidator(_validate_utc_timestamp),
]
LogicalUri = Annotated[
    str,
    StringConstraints(min_length=12, max_length=512),
    AfterValidator(_validate_logical_uri),
]

ParticipantId = Annotated[str, StringConstraints(pattern=r"^participant_[0-9a-f]{32}$")]
PurposeId = Annotated[str, StringConstraints(pattern=r"^purpose_[0-9a-f]{32}$")]
StudyId = Annotated[str, StringConstraints(pattern=r"^study_[0-9a-f]{32}$")]
DocumentId = Annotated[str, StringConstraints(pattern=r"^document_[0-9a-f]{32}$")]
ReceiptId = Annotated[str, StringConstraints(pattern=r"^receipt_[0-9a-f]{32}$")]
EventId = Annotated[str, StringConstraints(pattern=r"^event_[0-9a-f]{32}$")]
EventLogId = Annotated[str, StringConstraints(pattern=r"^event_log_[0-9a-f]{32}$")]
RecordingId = Annotated[str, StringConstraints(pattern=r"^recording_[0-9a-f]{32}$")]
GrantId = Annotated[str, StringConstraints(pattern=r"^grant_[0-9a-f]{32}$")]
PolicyId = Annotated[str, StringConstraints(pattern=r"^policy_[0-9a-f]{32}$")]
ReadinessId = Annotated[str, StringConstraints(pattern=r"^readiness_[0-9a-f]{32}$")]
AssetId = Annotated[str, StringConstraints(pattern=r"^asset_[0-9a-f]{32}$")]
InventoryId = Annotated[str, StringConstraints(pattern=r"^inventory_[0-9a-f]{32}$")]
WithdrawalId = Annotated[str, StringConstraints(pattern=r"^withdrawal_[0-9a-f]{32}$")]
ImpactId = Annotated[str, StringConstraints(pattern=r"^impact_[0-9a-f]{32}$")]
ReportId = Annotated[str, StringConstraints(pattern=r"^report_[0-9a-f]{32}$")]

DocumentType = Literal[
    "consent_form",
    "privacy_notice",
    "governance_policy",
    "withdrawal_procedure",
]
ConsentEventType = Literal["granted", "withdrawn", "expired", "superseded"]
ConsentEventReason = Literal[
    "participant_request",
    "consent_expired",
    "scope_replaced",
    "policy_replaced",
]
GovernanceRole = Literal["data_steward", "model_operator", "release_reviewer", "researcher"]
ProhibitedUse = Literal[
    "audio_collection",
    "commercial_sale",
    "identity_inference",
    "minor_participation",
    "participant_level_public_ranking",
    "surveillance",
]
ReadinessStatus = Literal["blocked"]
ReadinessBlocker = Literal[
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
AssetKind = Literal[
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
]
AssetLifecycleState = Literal["active", "invalidation_pending", "invalidated"]
WithdrawalRelationship = Literal["direct", "downstream"]
WithdrawalAction = Literal[
    "delete_backups",
    "delete_primary",
    "invalidate",
    "purge_backup",
    "rebuild",
    "reevaluate",
    "remove_public_copy",
    "republish",
    "rerun",
    "retire",
    "retract",
    "retain",
    "retrain",
    "revoke_access",
]
ScopePermission = Literal[
    "research_use",
    "raw_media_capture",
    "model_training",
    "model_evaluation",
    "public_demonstration",
    "model_weights_redistribution",
    "raw_media_retention",
    "raw_media_redistribution",
    "derived_features",
    "derived_features_redistribution",
    "evaluation_results_redistribution",
    "same_purpose_future_research",
]

PACKAGED_GOVERNANCE_DOCUMENTS: Final[Mapping[str, tuple[str, str, str, str, str]]] = (
    MappingProxyType(
        {
            "consent_form": (
                "document_00000000000000000000000000000001",
                "1.0.0",
                "2026-08-26T00:00:00Z",
                "signlab://governance/consent-form/1.0.0",
                "sha256:5ec3cad0f28d407f90278f61dc9f56457ce65ec7ae732c9afa7a26770ef05417",
            ),
            "privacy_notice": (
                "document_00000000000000000000000000000002",
                "1.0.0",
                "2026-08-26T00:00:00Z",
                "signlab://governance/privacy-notice/1.0.0",
                "sha256:1b6e4d6ed9ddb7d67bb819e4c66ecc3325c4a3eaeee82a01b17180e50baf8762",
            ),
            "governance_policy": (
                "document_00000000000000000000000000000003",
                "1.0.0",
                "2026-08-26T00:00:00Z",
                "signlab://governance/data-governance-policy/1.0.0",
                "sha256:5a29272296198c03f4c118d27ee19955174f4b26141284380a653eb874d4dfa9",
            ),
            "withdrawal_procedure": (
                "document_00000000000000000000000000000004",
                "1.0.0",
                "2026-08-26T00:00:00Z",
                "signlab://governance/withdrawal-runbook/1.0.0",
                "sha256:d583a387d2debfcb1c49746cfa51a32f694e39410c6bd900c3a292a090cb8075",
            ),
        }
    )
)


class DocumentRef(StrictFrozenModel):
    """Immutable reference to a reviewed governance document, never its signatures."""

    model_config = _governance_config("governance-document-reference-1.schema.json")

    schema_version: Literal["document-reference/1"]
    document_id: DocumentId
    document_type: DocumentType
    version: SemanticVersion
    effective_at: UtcTimestamp
    uri: LogicalUri
    sha256: Sha256Digest

    @model_validator(mode="after")
    def _verify_published_document(self) -> Self:
        expected = PACKAGED_GOVERNANCE_DOCUMENTS[self.document_type]
        actual = (
            self.document_id,
            self.version,
            self.effective_at,
            self.uri,
            self.sha256,
        )
        if actual != expected:
            raise ValueError(
                "governance document reference is not in the immutable published registry; "
                "publish a new reviewed version before using it"
            )
        return self


class ConsentScopeV1(StrictFrozenModel):
    """An explicit permission ceiling; every permission is required on the wire."""

    model_config = _governance_config("consent-scope-1.schema.json")

    schema_version: Literal["consent-scope/1"]
    research_use: bool
    raw_media_capture: bool
    model_training: bool
    model_evaluation: bool
    public_demonstration: bool
    model_weights_redistribution: bool
    raw_media_retention: bool
    raw_media_redistribution: bool
    derived_features: bool
    derived_features_redistribution: bool
    evaluation_results_redistribution: bool
    same_purpose_future_research: bool
    withdrawal_supported: Literal[True]
    audio_collection: Literal[False]
    minor_participation: Literal[False]
    identity_inference: Literal[False]
    commercial_sale: Literal[False]

    @model_validator(mode="after")
    def _enforce_dependencies(self) -> Self:
        uses = (
            self.raw_media_capture,
            self.model_training,
            self.model_evaluation,
            self.public_demonstration,
            self.model_weights_redistribution,
            self.raw_media_retention,
            self.raw_media_redistribution,
            self.derived_features,
            self.derived_features_redistribution,
            self.evaluation_results_redistribution,
            self.same_purpose_future_research,
        )
        if any(uses) and not self.research_use:
            raise ValueError("research_use must authorize every downstream research permission")
        if (self.model_training or self.model_evaluation) and not self.derived_features:
            raise ValueError("training and evaluation require derived_features permission")
        if self.derived_features and not self.raw_media_capture:
            raise ValueError("derived features require raw media capture permission")
        if self.raw_media_retention and not self.raw_media_capture:
            raise ValueError("raw media retention requires raw media capture permission")
        if self.model_weights_redistribution and not self.model_training:
            raise ValueError("model weight redistribution requires model training permission")
        if self.public_demonstration and not self.raw_media_retention:
            raise ValueError(
                "public demonstration is raw-media display permission and requires retention"
            )
        if self.raw_media_redistribution and not self.raw_media_retention:
            raise ValueError("raw media redistribution requires raw media retention")
        if self.derived_features_redistribution and not self.derived_features:
            raise ValueError("derived feature redistribution requires derived feature generation")
        if self.evaluation_results_redistribution and not self.model_evaluation:
            raise ValueError("evaluation result redistribution requires model evaluation")
        return self


class ConsentReceiptV1(StrictFrozenModel):
    """Pseudonymous evidence of an explicit participant consent decision."""

    model_config = _governance_config("consent-receipt-1.schema.json")

    schema_version: Literal["consent-receipt/1"]
    receipt_id: ReceiptId
    participant_id: ParticipantId
    purpose_id: PurposeId
    study_id: StudyId
    consent_form: DocumentRef
    privacy_notice: DocumentRef
    governance_policy: DocumentRef
    taxonomy: TaxonomyRef
    scope: ConsentScopeV1
    scope_sha256: Sha256Digest
    granted_at: UtcTimestamp
    valid_until: UtcTimestamp
    completed_form_sha256: Sha256Digest
    identity_vault_attestation_sha256: Sha256Digest
    adult_attested: Literal[True]

    @model_validator(mode="after")
    def _verify_receipt(self) -> Self:
        document_types = (
            self.consent_form.document_type,
            self.privacy_notice.document_type,
            self.governance_policy.document_type,
        )
        if document_types != ("consent_form", "privacy_notice", "governance_policy"):
            raise ValueError("receipt document references have incompatible document types")
        if any(
            _as_datetime(document.effective_at) > _as_datetime(self.granted_at)
            for document in (self.consent_form, self.privacy_notice, self.governance_policy)
        ):
            raise ValueError("receipt cannot precede an attached document's effective time")
        if _as_datetime(self.valid_until) <= _as_datetime(self.granted_at):
            raise ValueError("valid_until must be later than granted_at")
        if _as_datetime(self.valid_until) - _as_datetime(self.granted_at) > timedelta(days=730):
            raise ValueError("valid_until cannot exceed the 730-day consent ceiling")
        if self.scope_sha256 != consent_scope_digest(self.scope):
            raise ValueError("scope_sha256 does not match the canonical consent scope")
        return self


class ConsentEventV1(StrictFrozenModel):
    """Append-only consent lifecycle event without identity or free-text fields."""

    model_config = _governance_config("consent-event-1.schema.json")

    schema_version: Literal["consent-event/1"]
    event_id: EventId
    receipt_id: ReceiptId
    participant_id: ParticipantId
    event_type: ConsentEventType
    occurred_at: UtcTimestamp
    scope_sha256: Sha256Digest
    reason_code: ConsentEventReason | None
    replacement_receipt_id: ReceiptId | None

    @model_validator(mode="after")
    def _verify_event_shape(self) -> Self:
        if self.event_type == "granted":
            if self.reason_code is not None or self.replacement_receipt_id is not None:
                raise ValueError("a granted event cannot have a reason or replacement receipt")
        elif self.event_type == "withdrawn":
            if self.reason_code != "participant_request" or self.replacement_receipt_id is not None:
                raise ValueError("a withdrawn event requires participant_request only")
        elif self.event_type == "expired":
            if self.reason_code != "consent_expired" or self.replacement_receipt_id is not None:
                raise ValueError("an expired event requires consent_expired only")
        elif (
            self.reason_code not in {"scope_replaced", "policy_replaced"}
            or self.replacement_receipt_id is None
            or self.replacement_receipt_id == self.receipt_id
        ):
            raise ValueError("a superseded event requires a distinct replacement receipt")
        return self


class ConsentEventLogV1(StrictFrozenModel):
    """Attested complete consent-event history through an explicit UTC boundary."""

    model_config = _governance_config("consent-event-log-1.schema.json")

    schema_version: Literal["consent-event-log/1"]
    event_log_id: EventLogId
    receipt_id: ReceiptId
    receipt_sha256: Sha256Digest
    participant_id: ParticipantId
    purpose_id: PurposeId
    study_id: StudyId
    scope_sha256: Sha256Digest
    generated_at: UtcTimestamp
    complete_through: UtcTimestamp
    completeness_attested: Literal[True]
    identity_vault_attestation_sha256: Sha256Digest
    events: tuple[ConsentEventV1, ...] = Field(min_length=1, max_length=2)
    event_log_sha256: Sha256Digest

    @model_validator(mode="after")
    def _verify_event_log(self) -> Self:
        if _as_datetime(self.generated_at) < _as_datetime(self.complete_through):
            raise ValueError("event log generation cannot precede its complete-through time")
        ordering = tuple((event.occurred_at, event.event_id) for event in self.events)
        if ordering != tuple(sorted(ordering)):
            raise ValueError("consent events must be sorted by occurred_at and event_id")
        event_ids = tuple(event.event_id for event in self.events)
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("consent event IDs must be unique")
        initial, *terminal = self.events
        if initial.event_type != "granted":
            raise ValueError("a consent event log must begin with exactly one granted event")
        if terminal:
            if terminal[0].event_type == "granted":
                raise ValueError("a consent event log must contain exactly one granted event")
            if _as_datetime(terminal[0].occurred_at) <= _as_datetime(initial.occurred_at):
                raise ValueError("a terminal consent event must occur after the granted event")
        for event in self.events:
            if (
                event.receipt_id != self.receipt_id
                or event.participant_id != self.participant_id
                or event.scope_sha256 != self.scope_sha256
            ):
                raise ValueError(
                    "every consent event must bind to the event log identity and scope"
                )
            if _as_datetime(event.occurred_at) > _as_datetime(self.complete_through):
                raise ValueError("consent event occurs after the log's complete-through time")
        if self.event_log_sha256 != consent_event_log_digest(self):
            raise ValueError("event_log_sha256 does not match the canonical consent event log")
        return self


type ConsentEventLogVerifier = Callable[[ConsentReceiptV1, ConsentEventLogV1], bool]


class RecordingConsentGrantV1(StrictFrozenModel):
    """The explicit, receipt-bounded consent scope attached to one recording."""

    model_config = _governance_config("recording-consent-grant-1.schema.json")

    schema_version: Literal["recording-consent-grant/1"]
    grant_id: GrantId
    recording_id: RecordingId
    participant_id: ParticipantId
    receipt_id: ReceiptId
    purpose_id: PurposeId
    study_id: StudyId
    taxonomy: TaxonomyRef
    scope: ConsentScopeV1
    scope_sha256: Sha256Digest
    receipt_scope_sha256: Sha256Digest
    issued_at: UtcTimestamp
    captured_at: UtcTimestamp

    @model_validator(mode="after")
    def _verify_grant(self) -> Self:
        if _as_datetime(self.issued_at) > _as_datetime(self.captured_at):
            raise ValueError("recording capture cannot precede consent grant issuance")
        if self.scope_sha256 != consent_scope_digest(self.scope):
            raise ValueError("scope_sha256 does not match the recording consent scope")
        if not self.scope.raw_media_capture:
            raise ValueError("a recording consent grant requires raw_media_capture permission")
        return self


type ConsentAuthorizationVerifier = Callable[
    [ConsentReceiptV1, RecordingConsentGrantV1, ConsentEventLogV1], bool
]


class GovernancePolicyV1(StrictFrozenModel):
    """Reviewed access, retention, backup, deletion, and withdrawal rules."""

    model_config = _governance_config("governance-policy-1.schema.json")

    schema_version: Literal["governance-policy/1"]
    policy_id: PolicyId
    version: SemanticVersion
    taxonomy: TaxonomyRef
    policy_document: DocumentRef
    withdrawal_procedure: DocumentRef
    effective_at: UtcTimestamp
    permitted_roles: tuple[GovernanceRole, ...] = Field(min_length=1)
    raw_media_roles: tuple[GovernanceRole, ...] = Field(min_length=1)
    raw_media_retention_days: JsonPositiveInteger
    derived_feature_retention_days: JsonPositiveInteger
    evaluation_result_retention_days: JsonPositiveInteger
    backup_retention_days: JsonNonNegativeInteger
    withdrawal_impact_inventory_days: JsonPositiveInteger
    withdrawal_response_days: JsonPositiveInteger
    storage_encrypted: Literal[True]
    backups_encrypted: Literal[True]
    backups_include_raw_media: bool
    access_least_privilege: Literal[True]
    access_audit_logged: Literal[True]
    deletion_includes_backups: Literal[True]
    withdrawal_invalidates_downstream: Literal[True]
    prohibited_uses: tuple[ProhibitedUse, ...] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def _verify_policy(self) -> Self:
        if self.policy_document.document_type != "governance_policy":
            raise ValueError("policy_document must reference a governance policy")
        if self.withdrawal_procedure.document_type != "withdrawal_procedure":
            raise ValueError("withdrawal_procedure has an incompatible document type")
        if _as_datetime(self.policy_document.effective_at) > _as_datetime(self.effective_at):
            raise ValueError("policy cannot be effective before its reviewed document")
        if tuple(sorted(set(self.permitted_roles))) != self.permitted_roles:
            raise ValueError("permitted_roles must be unique and sorted")
        if tuple(sorted(set(self.raw_media_roles))) != self.raw_media_roles:
            raise ValueError("raw_media_roles must be unique and sorted")
        if not set(self.raw_media_roles).issubset(self.permitted_roles):
            raise ValueError("raw_media_roles must be a subset of permitted_roles")
        if "data_steward" not in self.raw_media_roles:
            raise ValueError("raw media requires accountable data_steward access")
        if self.prohibited_uses != PROHIBITED_USES:
            raise ValueError("prohibited_uses must explicitly ban all restricted uses")
        if any(
            value > 730
            for value in (
                self.raw_media_retention_days,
                self.derived_feature_retention_days,
                self.evaluation_result_retention_days,
            )
        ):
            raise ValueError("participant-data retention cannot exceed 730 days")
        if self.backup_retention_days > 30:
            raise ValueError("backup retention cannot exceed 30 days")
        if self.withdrawal_impact_inventory_days > 5:
            raise ValueError("withdrawal impact inventory cannot exceed 5 days")
        if self.withdrawal_response_days > 30:
            raise ValueError("withdrawal response cannot exceed 30 days")
        if self.backups_include_raw_media:
            if self.backup_retention_days == 0:
                raise ValueError("raw-media backup retention must be explicit and nonzero")
            if self.backup_retention_days > self.raw_media_retention_days:
                raise ValueError("raw-media backups cannot outlive raw-media retention")
        elif self.backup_retention_days != 0:
            raise ValueError("backup_retention_days must be zero when raw media is not backed up")
        return self


class CollectionReadinessV1(StrictFrozenModel):
    """Fail-closed collection gate that may truthfully remain blocked."""

    model_config = _governance_config("collection-readiness-1.schema.json")

    schema_version: Literal["collection-readiness/1"]
    readiness_id: ReadinessId
    policy_id: PolicyId
    taxonomy: TaxonomyRef
    assessed_at: UtcTimestamp
    status: ReadinessStatus
    consent_documents_ready: bool
    pseudonymous_ids_ready: bool
    encrypted_storage_ready: bool
    access_controls_ready: bool
    adult_jurisdictions_ready: bool
    affiliation_sponsorship_ready: bool
    backup_deletion_ready: bool
    ethics_legal_institutional_ready: bool
    identity_vault_operations_ready: bool
    withdrawal_dry_run_ready: bool
    lineage_tracking_ready: bool
    legacy_media_quarantined: bool
    participant_contact_process_ready: bool
    retention_publication_scope_ready: bool
    blockers: tuple[ReadinessBlocker, ...]

    @model_validator(mode="after")
    def _verify_readiness(self) -> Self:
        checks = {
            "access_controls": self.access_controls_ready,
            "adult_jurisdictions": self.adult_jurisdictions_ready,
            "affiliation_sponsorship": self.affiliation_sponsorship_ready,
            "backup_deletion": self.backup_deletion_ready,
            "consent_documents": self.consent_documents_ready,
            "encrypted_storage": self.encrypted_storage_ready,
            "ethics_legal_institutional": self.ethics_legal_institutional_ready,
            "identity_vault_operations": self.identity_vault_operations_ready,
            "legacy_media_quarantine": self.legacy_media_quarantined,
            "lineage_tracking": self.lineage_tracking_ready,
            "participant_contact_process": self.participant_contact_process_ready,
            "pseudonymous_ids": self.pseudonymous_ids_ready,
            "retention_publication_scope": self.retention_publication_scope_ready,
            "withdrawal_dry_run": self.withdrawal_dry_run_ready,
        }
        expected_blockers = tuple(key for key, ready in sorted(checks.items()) if not ready)
        if not expected_blockers:
            raise ValueError(
                "collection-readiness/1 cannot become ready without an authenticated approval "
                "verifier; keep it blocked or migrate to a future schema version"
            )
        if self.blockers != expected_blockers:
            raise ValueError("blockers must exactly identify every failed readiness check")
        expected_status = "ready" if not expected_blockers else "blocked"
        if self.status != expected_status:
            raise ValueError(f"collection status must be {expected_status} for these checks")
        return self


class GovernanceAssetV1(StrictFrozenModel):
    """One content-addressed lineage node affected by consent withdrawal."""

    model_config = _governance_config("governance-asset-1.schema.json")

    schema_version: Literal["governance-asset/1"]
    asset_id: AssetId
    asset_kind: AssetKind
    logical_uri: LogicalUri
    sha256: Sha256Digest
    taxonomy: TaxonomyRef
    created_at: UtcTimestamp
    participant_ids: tuple[ParticipantId, ...] = Field(min_length=1)
    recording_ids: tuple[RecordingId, ...] = Field(min_length=1)
    receipt_ids: tuple[ReceiptId, ...] = Field(min_length=1)
    grant_ids: tuple[GrantId, ...] = Field(min_length=1)
    parent_asset_ids: tuple[AssetId, ...]
    lifecycle_state: AssetLifecycleState
    invalidated_at: UtcTimestamp | None

    @model_validator(mode="after")
    def _verify_asset(self) -> Self:
        ordered_fields = (
            ("participant_ids", self.participant_ids),
            ("recording_ids", self.recording_ids),
            ("receipt_ids", self.receipt_ids),
            ("grant_ids", self.grant_ids),
            ("parent_asset_ids", self.parent_asset_ids),
        )
        for field_name, values in ordered_fields:
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{field_name} must be unique and sorted")
        if self.asset_id in self.parent_asset_ids:
            raise ValueError("an asset cannot be its own parent")
        if self.asset_kind == "raw_recording" and self.parent_asset_ids:
            raise ValueError("a raw recording must be a lineage root")
        if self.asset_kind == "raw_recording" and any(
            len(values) != 1
            for values in (
                self.participant_ids,
                self.recording_ids,
                self.receipt_ids,
                self.grant_ids,
            )
        ):
            raise ValueError(
                "a raw recording requires exactly one participant, recording, receipt, and grant"
            )
        if self.asset_kind != "raw_recording" and not self.parent_asset_ids:
            raise ValueError("a derived governance asset must identify at least one parent")
        if not _is_asset_logical_uri(self.logical_uri, self.asset_kind, self.asset_id):
            raise ValueError(
                "asset logical_uri must be the opaque store/kind/asset structural locator"
            )
        if self.lifecycle_state == "invalidated":
            if self.invalidated_at is None:
                raise ValueError("an invalidated asset requires invalidated_at")
            if _as_datetime(self.invalidated_at) < _as_datetime(self.created_at):
                raise ValueError("asset invalidation cannot precede creation")
        elif self.invalidated_at is not None:
            raise ValueError("only an invalidated asset may carry invalidated_at")
        return self


class LineageInventoryV1(StrictFrozenModel):
    """A complete, deterministic, acyclic inventory of governed assets."""

    model_config = _governance_config("lineage-inventory-1.schema.json")

    schema_version: Literal["lineage-inventory/1"]
    inventory_id: InventoryId
    taxonomy: TaxonomyRef
    generated_at: UtcTimestamp
    assets: tuple[GovernanceAssetV1, ...] = Field(min_length=1)
    inventory_sha256: Sha256Digest

    @model_validator(mode="after")
    def _verify_inventory(self) -> Self:
        asset_ids = tuple(asset.asset_id for asset in self.assets)
        if asset_ids != tuple(sorted(set(asset_ids))):
            raise ValueError("assets must have unique IDs and be sorted by asset_id")
        locators = tuple(asset.logical_uri for asset in self.assets)
        if len(locators) != len(set(locators)):
            raise ValueError("asset logical locators must be unique")
        by_id = {asset.asset_id: asset for asset in self.assets}
        for asset in self.assets:
            if asset.taxonomy != self.taxonomy:
                raise ValueError("every asset must use the inventory taxonomy reference")
            missing = set(asset.parent_asset_ids).difference(by_id)
            if missing:
                raise ValueError("every parent_asset_id must exist in the same inventory")
            if asset.parent_asset_ids:
                parents = tuple(by_id[parent_id] for parent_id in asset.parent_asset_ids)
                if any(
                    _as_datetime(parent.created_at) > _as_datetime(asset.created_at)
                    for parent in parents
                ):
                    raise ValueError("a parent asset cannot be created after its child")
                for field_name in (
                    "participant_ids",
                    "recording_ids",
                    "receipt_ids",
                    "grant_ids",
                ):
                    expected = tuple(
                        sorted({item for parent in parents for item in getattr(parent, field_name)})
                    )
                    if getattr(asset, field_name) != expected:
                        raise ValueError(
                            f"{field_name} must exactly equal the sorted union of parent metadata"
                        )

        states: dict[str, int] = {}

        def visit(asset_id: str) -> None:
            state = states.get(asset_id, 0)
            if state == 1:
                raise ValueError("asset parent relationships must form an acyclic DAG")
            if state == 2:
                return
            states[asset_id] = 1
            for parent_id in by_id[asset_id].parent_asset_ids:
                visit(parent_id)
            states[asset_id] = 2

        for asset_id in by_id:
            visit(asset_id)
        if self.inventory_sha256 != lineage_inventory_digest(self):
            raise ValueError("inventory_sha256 does not match the canonical inventory")
        return self


class WithdrawalRequestV1(StrictFrozenModel):
    """A pseudonymous, whole-participant withdrawal request."""

    model_config = _governance_config("withdrawal-request-1.schema.json")

    schema_version: Literal["withdrawal-request/1"]
    request_id: WithdrawalId
    participant_id: ParticipantId
    receipt_ids: tuple[ReceiptId, ...] = Field(min_length=1)
    requested_at: UtcTimestamp
    effective_at: UtcTimestamp
    target: Literal["all_participant_data"]
    identity_verification_attestation_sha256: Sha256Digest
    request_sha256: Sha256Digest

    @model_validator(mode="after")
    def _verify_request(self) -> Self:
        if self.receipt_ids != tuple(sorted(set(self.receipt_ids))):
            raise ValueError("receipt_ids must be unique and sorted")
        if _as_datetime(self.effective_at) < _as_datetime(self.requested_at):
            raise ValueError("withdrawal effective_at cannot precede requested_at")
        if self.request_sha256 != withdrawal_request_digest(self):
            raise ValueError("request_sha256 does not match the canonical request")
        return self


class WithdrawalImpactV1(StrictFrozenModel):
    """One planned dry-run consequence; it never performs deletion itself."""

    model_config = _governance_config("withdrawal-impact-1.schema.json")

    schema_version: Literal["withdrawal-impact/1"]
    impact_id: ImpactId
    asset_id: AssetId
    logical_uri: LogicalUri
    asset_sha256: Sha256Digest
    asset_kind: AssetKind
    relationship: WithdrawalRelationship
    planned_actions: tuple[WithdrawalAction, ...] = Field(min_length=1)
    impact_sha256: Sha256Digest

    @model_validator(mode="after")
    def _verify_impact(self) -> Self:
        if not _is_asset_logical_uri(self.logical_uri, self.asset_kind, self.asset_id):
            raise ValueError(
                "withdrawal impact logical_uri must match its opaque asset structural locator"
            )
        if self.planned_actions != tuple(sorted(set(self.planned_actions))):
            raise ValueError("planned_actions must be unique and sorted")
        if self.asset_kind == "withdrawal_tombstone":
            if self.planned_actions != ("retain",):
                raise ValueError("a withdrawal tombstone must only be retained")
        elif "invalidate" not in self.planned_actions or "retain" in self.planned_actions:
            raise ValueError("non-tombstone withdrawal impacts must invalidate and cannot retain")
        if self.impact_sha256 != withdrawal_impact_digest(self):
            raise ValueError("impact_sha256 does not match the canonical impact")
        return self


class WithdrawalReportV1(StrictFrozenModel):
    """Verified output of a withdrawal dry run, without destructive side effects."""

    model_config = _governance_config("withdrawal-report-1.schema.json")

    schema_version: Literal["withdrawal-report/1"]
    report_id: ReportId
    mode: Literal["dry_run"]
    request: WithdrawalRequestV1
    inventory_id: InventoryId
    inventory_sha256: Sha256Digest
    generated_at: UtcTimestamp
    status: Literal["blocked", "complete"]
    impacts: tuple[WithdrawalImpactV1, ...]
    affected_asset_count: JsonNonNegativeInteger
    direct_asset_count: JsonNonNegativeInteger
    downstream_asset_count: JsonNonNegativeInteger
    unresolved_asset_ids: tuple[AssetId, ...]
    report_sha256: Sha256Digest

    @model_validator(mode="after")
    def _verify_report(self) -> Self:
        impact_asset_ids = tuple(impact.asset_id for impact in self.impacts)
        if impact_asset_ids != tuple(sorted(set(impact_asset_ids))):
            raise ValueError("impacts must have unique assets and be sorted by asset_id")
        impact_ids = tuple(impact.impact_id for impact in self.impacts)
        if len(impact_ids) != len(set(impact_ids)):
            raise ValueError("impact IDs must be unique")
        if self.unresolved_asset_ids != tuple(sorted(set(self.unresolved_asset_ids))):
            raise ValueError("unresolved_asset_ids must be unique and sorted")
        if set(impact_asset_ids).intersection(self.unresolved_asset_ids):
            raise ValueError("an asset cannot be both planned and unresolved")
        direct_count = sum(impact.relationship == "direct" for impact in self.impacts)
        downstream_count = sum(impact.relationship == "downstream" for impact in self.impacts)
        if self.affected_asset_count != len(self.impacts):
            raise ValueError("affected_asset_count does not match impacts")
        if self.direct_asset_count != direct_count:
            raise ValueError("direct_asset_count does not match impacts")
        if self.downstream_asset_count != downstream_count:
            raise ValueError("downstream_asset_count does not match impacts")
        expected_status = "blocked" if self.unresolved_asset_ids else "complete"
        if self.status != expected_status:
            raise ValueError(f"withdrawal report status must be {expected_status}")
        if _as_datetime(self.generated_at) < _as_datetime(self.request.requested_at):
            raise ValueError("withdrawal report cannot precede its request")
        if self.report_sha256 != withdrawal_report_digest(self):
            raise ValueError("report_sha256 does not match the canonical report")
        return self


type GovernanceInput = BaseModel | str | bytes | bytearray | Mapping[str, object]

_FORBIDDEN_KEY_TOKENS: Final = {
    "address",
    "caption",
    "comment",
    "comments",
    "contact",
    "description",
    "details",
    "email",
    "free_text",
    "freetext",
    "identity",
    "message",
    "name",
    "note",
    "notes",
    "phone",
    "signature",
    "telephone",
    "text",
}
_FORBIDDEN_COMPACT_KEYS: Final = {
    "birthdate",
    "contactdetails",
    "dateofbirth",
    "firstname",
    "fullname",
    "governmentid",
    "identitydocument",
    "lastname",
    "signedby",
}
_SAFE_SPECIAL_KEYS: Final = {
    "identity_inference",
    "identity_vault_attestation_sha256",
    "identity_vault_operations_ready",
    "identity_verification_attestation_sha256",
    "participant_contact_process_ready",
    "reason_code",
}


def _is_machine_path(value: str) -> bool:
    if value.startswith("signlab://"):
        return False
    lowered = value.casefold()
    if lowered.startswith("file:") or value.startswith(("~/", "~\\")):
        return True
    if re.search(r"(?:^|\s)[a-zA-Z]:[\\/]", value) is not None:
        return True
    if value.startswith(("\\\\", "//")):
        return True
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _scan_public_value(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise GovernanceContractError("governance object keys must be strings")
            normalized = key.casefold()
            compact = re.sub(r"[^a-z0-9]", "", normalized)
            tokens = set(re.split(r"[^a-z0-9]+", normalized))
            if key not in _SAFE_SPECIAL_KEYS and (
                compact in _FORBIDDEN_COMPACT_KEYS or tokens.intersection(_FORBIDDEN_KEY_TOKENS)
            ):
                raise GovernanceContractError(
                    "governance input contains a prohibited identity, contact, signature, "
                    "or free-text field; use only pseudonymous IDs and coded values"
                )
            _scan_public_value(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _scan_public_value(nested)
    elif isinstance(value, str) and _is_machine_path(value):
        raise GovernanceContractError(
            "governance input contains a machine-specific path; use a signlab:// logical URI"
        )


def _reject_constant(_value: str) -> None:
    raise GovernanceContractError("governance input contains a non-finite JSON number")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise GovernanceContractError("governance input contains a duplicate object key")
        result[key] = value
    return result


def _json_object(document: GovernanceInput) -> Mapping[str, object]:
    if isinstance(document, BaseModel):
        payload: object = document.model_dump(mode="json", round_trip=True)
    else:
        if isinstance(document, Mapping):
            try:
                encoded = json.dumps(document, ensure_ascii=False, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise GovernanceContractError("governance input is not valid JSON data") from error
        elif isinstance(document, str):
            encoded = document
        elif isinstance(document, (bytes, bytearray)):
            try:
                encoded = bytes(document).decode("utf-8")
            except UnicodeDecodeError as error:
                raise GovernanceContractError("governance input must use valid UTF-8") from error
        else:
            raise GovernanceContractError("governance input must be a JSON object")
        try:
            payload = json.loads(
                encoded,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except GovernanceContractError:
            raise
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise GovernanceContractError("governance input is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise GovernanceContractError("governance input must be a JSON object")
    _scan_public_value(payload)
    return payload


def _validation_message(error: ValidationError) -> str:
    details: list[str] = []
    for item in error.errors(include_input=False, include_url=False):
        location = (
            "document"
            if item["type"] == "extra_forbidden"
            else ".".join(str(part) for part in item["loc"]) or "document"
        )
        details.append(f"{location}: {item['msg']}")
    return "; ".join(details)


def _validate_contract[ModelT: BaseModel](
    document: GovernanceInput,
    model: type[ModelT],
    label: str,
) -> ModelT:
    try:
        payload = _json_object(document)
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        return model.model_validate_json(encoded, strict=True)
    except GovernanceContractError:
        raise
    except ValidationError as error:
        raise GovernanceContractError(f"invalid {label}: {_validation_message(error)}") from error


def _canonical_digest(
    value: BaseModel | Mapping[str, object],
    *,
    exclude: str | None = None,
) -> str:
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json", round_trip=True)
    else:
        payload = dict(value)
    if exclude is not None:
        payload.pop(exclude, None)
    normalized = _normalize_integral_json_numbers(payload)
    if not isinstance(normalized, Mapping):  # pragma: no cover - payload is always a dict.
        raise GovernanceContractError("canonical governance payload must be a JSON object")
    return f"sha256:{hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()}"


def _normalize_integral_json_numbers(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _normalize_integral_json_numbers(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_integral_json_numbers(nested) for nested in value]
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def consent_scope_digest(scope: ConsentScopeV1 | Mapping[str, object]) -> str:
    """Return the deterministic digest used to bind receipts and recording grants."""

    return _canonical_digest(scope)


def consent_receipt_digest(
    receipt: ConsentReceiptV1 | Mapping[str, object],
) -> str:
    """Return the complete canonical identity of a consent receipt."""

    return _canonical_digest(receipt)


def recording_consent_grant_digest(
    grant: RecordingConsentGrantV1 | Mapping[str, object],
) -> str:
    """Return the complete canonical identity of a recording consent grant."""

    return _canonical_digest(grant)


def consent_event_log_digest(
    event_log: ConsentEventLogV1 | Mapping[str, object],
) -> str:
    """Return an event-log digest excluding its self-referential digest field."""

    return _canonical_digest(event_log, exclude="event_log_sha256")


def new_participant_id() -> str:
    """Generate a cryptographically random pseudonymous 128-bit participant ID."""

    return f"participant_{secrets.token_hex(16)}"


def lineage_inventory_digest(inventory: LineageInventoryV1 | Mapping[str, object]) -> str:
    """Return the inventory digest, excluding its self-referential digest field."""

    return _canonical_digest(inventory, exclude="inventory_sha256")


def withdrawal_request_digest(request: WithdrawalRequestV1 | Mapping[str, object]) -> str:
    """Return the withdrawal request digest, excluding its digest field."""

    return _canonical_digest(request, exclude="request_sha256")


def withdrawal_impact_digest(impact: WithdrawalImpactV1 | Mapping[str, object]) -> str:
    """Return one withdrawal impact digest, excluding its digest field."""

    return _canonical_digest(impact, exclude="impact_sha256")


def withdrawal_report_digest(report: WithdrawalReportV1 | Mapping[str, object]) -> str:
    """Return the withdrawal report digest, excluding its digest field."""

    return _canonical_digest(report, exclude="report_sha256")


def _checked_at(value: str) -> datetime:
    try:
        return _as_datetime(_validate_utc_timestamp(value))
    except (TypeError, ValueError) as error:
        raise GovernanceContractError("checked time must be an exact UTC timestamp") from error


def _receipt_is_structurally_active_at(
    receipt: ConsentReceiptV1,
    event_log: ConsentEventLogV1,
    checked_at: datetime,
) -> bool:
    if (
        event_log.receipt_id != receipt.receipt_id
        or event_log.receipt_sha256 != consent_receipt_digest(receipt)
        or event_log.participant_id != receipt.participant_id
        or event_log.purpose_id != receipt.purpose_id
        or event_log.study_id != receipt.study_id
        or event_log.scope_sha256 != receipt.scope_sha256
    ):
        raise GovernanceContractError("consent event log does not match the supplied receipt")
    if _as_datetime(event_log.complete_through) < checked_at:
        raise GovernanceContractError("consent event log is not complete through the checked time")
    granted_event = event_log.events[0]
    if granted_event.occurred_at != receipt.granted_at:
        raise GovernanceContractError("granted event time does not match its receipt")
    if checked_at < _as_datetime(receipt.granted_at):
        return False
    if checked_at >= _as_datetime(receipt.valid_until):
        return False
    return not (
        len(event_log.events) == 2 and _as_datetime(event_log.events[1].occurred_at) <= checked_at
    )


def _authenticate_event_log(
    receipt: ConsentReceiptV1,
    event_log: ConsentEventLogV1,
    verifier: ConsentEventLogVerifier,
) -> None:
    try:
        authenticated = verifier(receipt, event_log)
    except Exception as error:
        raise GovernanceContractError("consent event log authentication failed") from error
    if authenticated is not True:
        raise GovernanceContractError("consent event log authentication failed")


def _authenticate_authorization(
    receipt: ConsentReceiptV1,
    grant: RecordingConsentGrantV1,
    event_log: ConsentEventLogV1,
    verifier: ConsentAuthorizationVerifier,
) -> None:
    try:
        authenticated = verifier(receipt, grant, event_log)
    except Exception as error:
        raise GovernanceContractError("consent authorization authentication failed") from error
    if authenticated is not True:
        raise GovernanceContractError("consent authorization authentication failed")


def assert_receipt_grant_consistent(
    receipt: GovernanceInput,
    grant: GovernanceInput,
    event_log: GovernanceInput,
) -> None:
    """Check structural consistency without claiming the evidence is authenticated."""

    receipt = validate_consent_receipt(receipt)
    grant = validate_recording_consent_grant(grant)
    event_log = validate_consent_event_log(event_log)
    if grant.receipt_id != receipt.receipt_id or grant.participant_id != receipt.participant_id:
        raise GovernanceContractError("recording grant does not reference the supplied receipt")
    if grant.purpose_id != receipt.purpose_id or grant.study_id != receipt.study_id:
        raise GovernanceContractError("recording grant purpose and study do not match its receipt")
    if grant.taxonomy != receipt.taxonomy:
        raise GovernanceContractError("recording grant and receipt taxonomy references differ")
    if grant.receipt_scope_sha256 != receipt.scope_sha256:
        raise GovernanceContractError("recording grant does not bind the supplied receipt scope")
    for permission in SCOPE_PERMISSION_FIELDS:
        if getattr(grant.scope, permission) and not getattr(receipt.scope, permission):
            raise GovernanceContractError("recording grant exceeds its receipt consent scope")
    if _as_datetime(grant.issued_at) < _as_datetime(receipt.granted_at):
        raise GovernanceContractError("recording grant predates its receipt")
    if not _receipt_is_structurally_active_at(
        receipt,
        event_log,
        _checked_at(grant.captured_at),
    ):
        raise GovernanceContractError("recording was captured outside the receipt validity window")


def receipt_is_active_at(
    receipt: GovernanceInput,
    event_log: GovernanceInput,
    at: UtcTimestamp,
    *,
    event_log_verifier: ConsentEventLogVerifier,
) -> bool:
    """Return active state only after authenticating the complete lifecycle log."""

    receipt = validate_consent_receipt(receipt)
    event_log = validate_consent_event_log(event_log)
    if not _receipt_is_structurally_active_at(receipt, event_log, _checked_at(at)):
        return False
    _authenticate_event_log(receipt, event_log, event_log_verifier)
    return True


def grant_authorizes_at(
    grant: GovernanceInput,
    receipt: GovernanceInput,
    event_log: GovernanceInput,
    permission: ScopePermission,
    at: UtcTimestamp,
    *,
    purpose_id: PurposeId,
    study_id: StudyId,
    authorization_verifier: ConsentAuthorizationVerifier,
) -> bool:
    """Return authorization only for authenticated evidence and the requested context."""

    if permission not in SCOPE_PERMISSION_FIELDS:
        raise GovernanceContractError("unknown consent permission")
    grant = validate_recording_consent_grant(grant)
    receipt = validate_consent_receipt(receipt)
    event_log = validate_consent_event_log(event_log)
    assert_receipt_grant_consistent(receipt, grant, event_log)
    if re.fullmatch(r"purpose_[0-9a-f]{32}", purpose_id) is None:
        raise GovernanceContractError("requested purpose identifier is invalid")
    if re.fullmatch(r"study_[0-9a-f]{32}", study_id) is None:
        raise GovernanceContractError("requested study identifier is invalid")
    if purpose_id != receipt.purpose_id:
        return False
    if study_id != receipt.study_id and not (
        receipt.scope.same_purpose_future_research and grant.scope.same_purpose_future_research
    ):
        return False
    if _checked_at(at) < _as_datetime(grant.captured_at):
        return False
    if not (getattr(grant.scope, permission) and getattr(receipt.scope, permission)):
        return False
    if not _receipt_is_structurally_active_at(receipt, event_log, _checked_at(at)):
        return False
    _authenticate_authorization(receipt, grant, event_log, authorization_verifier)
    return True


def validate_document_ref(document: GovernanceInput) -> DocumentRef:
    return _validate_contract(document, DocumentRef, "governance document reference")


def validate_consent_scope(document: GovernanceInput) -> ConsentScopeV1:
    return _validate_contract(document, ConsentScopeV1, "consent scope")


def validate_consent_receipt(document: GovernanceInput) -> ConsentReceiptV1:
    return _validate_contract(document, ConsentReceiptV1, "consent receipt")


def validate_consent_event(document: GovernanceInput) -> ConsentEventV1:
    return _validate_contract(document, ConsentEventV1, "consent event")


def validate_consent_event_log(document: GovernanceInput) -> ConsentEventLogV1:
    return _validate_contract(document, ConsentEventLogV1, "consent event log")


def validate_recording_consent_grant(document: GovernanceInput) -> RecordingConsentGrantV1:
    return _validate_contract(document, RecordingConsentGrantV1, "recording consent grant")


def validate_governance_policy(document: GovernanceInput) -> GovernancePolicyV1:
    return _validate_contract(document, GovernancePolicyV1, "governance policy")


def validate_collection_readiness(document: GovernanceInput) -> CollectionReadinessV1:
    return _validate_contract(document, CollectionReadinessV1, "collection readiness")


def validate_governance_asset(document: GovernanceInput) -> GovernanceAssetV1:
    return _validate_contract(document, GovernanceAssetV1, "governance asset")


def validate_lineage_inventory(document: GovernanceInput) -> LineageInventoryV1:
    return _validate_contract(document, LineageInventoryV1, "lineage inventory")


def validate_withdrawal_request(document: GovernanceInput) -> WithdrawalRequestV1:
    return _validate_contract(document, WithdrawalRequestV1, "withdrawal request")


def validate_withdrawal_impact(document: GovernanceInput) -> WithdrawalImpactV1:
    return _validate_contract(document, WithdrawalImpactV1, "withdrawal impact")


def validate_withdrawal_report(document: GovernanceInput) -> WithdrawalReportV1:
    return _validate_contract(document, WithdrawalReportV1, "withdrawal report")


__all__ = [
    "AssetKind",
    "CollectionReadinessV1",
    "ConsentAuthorizationVerifier",
    "ConsentEventLogV1",
    "ConsentEventLogVerifier",
    "ConsentEventV1",
    "ConsentReceiptV1",
    "ConsentScopeV1",
    "DocumentRef",
    "GovernanceAssetV1",
    "GovernanceContractError",
    "GovernancePolicyV1",
    "LineageInventoryV1",
    "LogicalUri",
    "PurposeId",
    "RecordingConsentGrantV1",
    "ScopePermission",
    "StudyId",
    "UtcTimestamp",
    "WithdrawalImpactV1",
    "WithdrawalReportV1",
    "WithdrawalRequestV1",
    "assert_receipt_grant_consistent",
    "consent_event_log_digest",
    "consent_receipt_digest",
    "consent_scope_digest",
    "grant_authorizes_at",
    "lineage_inventory_digest",
    "new_participant_id",
    "receipt_is_active_at",
    "recording_consent_grant_digest",
    "validate_collection_readiness",
    "validate_consent_event",
    "validate_consent_event_log",
    "validate_consent_receipt",
    "validate_consent_scope",
    "validate_document_ref",
    "validate_governance_asset",
    "validate_governance_policy",
    "validate_lineage_inventory",
    "validate_recording_consent_grant",
    "validate_withdrawal_impact",
    "validate_withdrawal_report",
    "validate_withdrawal_request",
    "withdrawal_impact_digest",
    "withdrawal_report_digest",
    "withdrawal_request_digest",
]
