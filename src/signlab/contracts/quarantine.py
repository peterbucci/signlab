"""Strict inventory for retained capture attempts without consent evidence.

Retry and quarantined attempt bytes cannot truthfully be represented as
``GovernanceAssetV1`` raw recordings: that contract requires a receipt and a
recording grant.  This separate inventory keeps those bytes discoverable for
withdrawal and deletion while preserving their explicitly unconsented state.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal, Self, cast

from pydantic import BaseModel, TypeAdapter, ValidationError, model_validator

from signlab.contracts.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    canonical_sha256,
    parse_json_object,
)
from signlab.contracts.core import (
    ArtifactRefV1,
    UtcTimestamp,
    WorkspaceRelativeLocatorV1,
    contract_config,
)
from signlab.contracts.dataset import SessionId
from signlab.contracts.governance import ParticipantId, RecordingId
from signlab.contracts.ingest import (
    AttemptReasonCode,
    CaptureAttemptId,
    CaptureMediaType,
    CaptureSourceKey,
    CollectionId,
    StoreId,
)
from signlab.contracts.taxonomy import Sha256Digest, TaxonomyRef

type QuarantineInventoryInput = (
    QuarantineInventoryV1 | str | bytes | bytearray | Mapping[str, object]
)
type QuarantinedAttemptOutcome = Literal["retry", "quarantined"]

_PARTICIPANT_ID_ADAPTER: Final = TypeAdapter(ParticipantId)
_CAPTURE_MEDIA_TYPES: Final[frozenset[CaptureMediaType]] = frozenset(
    {"video/mp4", "video/quicktime", "video/webm"}
)


class QuarantineContractError(ValueError):
    """Raised when a quarantine inventory is malformed or internally inconsistent."""


def quarantine_attempt_workspace_path(attempt_id: str, sha256: str) -> str:
    """Return the sole portable bundle path for one retained attempt's bytes."""

    digest = sha256.removeprefix("sha256:")
    return f"quarantine/sha256/p-{digest[:2]}/sha256-{digest}/{attempt_id}"


class QuarantinedAttemptAssetV1(BaseModel):
    """One retained retry or quarantine byte object with no invented consent evidence."""

    model_config = contract_config("quarantined-attempt-asset-1.schema.json")

    schema_version: Literal["quarantined-attempt-asset/1"]
    attempt_id: CaptureAttemptId
    recording_id: RecordingId
    source_key: CaptureSourceKey
    participant_id: ParticipantId
    session_id: SessionId
    outcome: QuarantinedAttemptOutcome
    reason_code: AttemptReasonCode
    recorded_at: UtcTimestamp
    artifact: ArtifactRefV1
    lifecycle_state: Literal["quarantined"]
    consent_evidence_status: Literal["absent"]

    @model_validator(mode="after")
    def _require_quarantine_artifact(self) -> Self:
        locator = self.artifact.locator
        if (
            self.artifact.artifact_id != self.attempt_id
            or self.artifact.role != "quarantined_capture_attempt"
            or self.artifact.media_type not in _CAPTURE_MEDIA_TYPES
            or self.artifact.size_bytes <= 0
            or not isinstance(locator, WorkspaceRelativeLocatorV1)
            or locator.path
            != quarantine_attempt_workspace_path(self.attempt_id, self.artifact.sha256)
        ):
            raise ValueError("quarantine artifact reference is not canonical")
        return self


class QuarantineInventoryV1(BaseModel):
    """Complete participant-addressable inventory of retained nonaccepted bytes."""

    model_config = contract_config("quarantine-inventory-1.schema.json")

    schema_version: Literal["quarantine-inventory/1"]
    collection_id: CollectionId
    store_id: StoreId
    taxonomy: TaxonomyRef
    collection_sidecar_sha256: Sha256Digest
    generated_at: UtcTimestamp
    assets: tuple[QuarantinedAttemptAssetV1, ...]
    quarantine_inventory_sha256: Sha256Digest

    @model_validator(mode="after")
    def _require_complete_canonical_inventory(self) -> Self:
        ordering = tuple(
            (asset.participant_id, asset.recording_id, asset.attempt_id) for asset in self.assets
        )
        if ordering != tuple(sorted(ordering)):
            raise ValueError("quarantine assets must use canonical participant/recording order")
        for values in (
            tuple(asset.attempt_id for asset in self.assets),
            tuple(asset.recording_id for asset in self.assets),
            tuple(asset.source_key for asset in self.assets),
            tuple(
                cast(WorkspaceRelativeLocatorV1, asset.artifact.locator).path
                for asset in self.assets
            ),
        ):
            if len(values) != len(set(values)):
                raise ValueError("quarantine asset identities and locations must be unique")
        if self.quarantine_inventory_sha256 != quarantine_inventory_digest(self):
            raise ValueError("quarantine_inventory_sha256 does not match canonical content")
        return self


def quarantine_inventory_digest(
    inventory: QuarantineInventoryV1 | Mapping[str, object],
) -> str:
    """Hash an inventory while excluding its self-referential digest field."""

    if isinstance(inventory, BaseModel):
        payload = cast(
            dict[str, object],
            inventory.model_dump(mode="json", round_trip=True),
        )
    else:
        payload = dict(inventory)
    payload.pop("quarantine_inventory_sha256", None)
    try:
        return canonical_sha256(payload, domain="quarantine-inventory/1")
    except CanonicalizationError as error:
        raise QuarantineContractError("quarantine inventory cannot be canonicalized") from error


def validate_quarantine_inventory(document: QuarantineInventoryInput) -> QuarantineInventoryV1:
    """Validate one strict quarantine inventory without accepting coercion."""

    try:
        if isinstance(document, BaseModel):
            payload = cast(
                Mapping[str, object],
                document.model_dump(mode="json", round_trip=True),
            )
        else:
            payload = cast(Mapping[str, object], parse_json_object(document))
        return QuarantineInventoryV1.model_validate_json(
            canonical_json_bytes(payload),
            strict=True,
        )
    except (CanonicalizationError, ValidationError) as error:
        raise QuarantineContractError("invalid quarantine inventory") from error


def discover_quarantined_recording_ids(
    inventory: QuarantineInventoryInput,
    participant_id: str,
) -> tuple[str, ...]:
    """Return every retained recording identity discoverable for one participant."""

    checked = validate_quarantine_inventory(inventory)
    try:
        checked_participant_id = _PARTICIPANT_ID_ADAPTER.validate_python(
            participant_id,
            strict=True,
        )
    except ValidationError as error:
        raise QuarantineContractError("invalid participant identity") from error
    return tuple(
        asset.recording_id
        for asset in checked.assets
        if asset.participant_id == checked_participant_id
    )


__all__ = [
    "QuarantineContractError",
    "QuarantineInventoryInput",
    "QuarantineInventoryV1",
    "QuarantinedAttemptAssetV1",
    "QuarantinedAttemptOutcome",
    "discover_quarantined_recording_ids",
    "quarantine_attempt_workspace_path",
    "quarantine_inventory_digest",
    "validate_quarantine_inventory",
]
