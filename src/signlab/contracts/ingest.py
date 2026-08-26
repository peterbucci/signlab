"""Strict capture-ledger and raw-dataset handoff contracts."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from itertools import pairwise
from typing import Annotated, Final, Literal, Self, cast

from pydantic import BaseModel, Field, StringConstraints, ValidationError, model_validator

from signlab.contracts.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    canonical_sha256,
    parse_json_object,
)
from signlab.contracts.core import (
    PositiveSafeInteger,
    SemanticVersion,
    StableId,
    StrictContractModel,
    UtcTimestamp,
    contract_config,
)
from signlab.contracts.dataset import (
    DATASET_TABLE_SCHEMA_VERSIONS,
    AnnotationId,
    AnnotationRowV1,
    DatasetTableRefV1,
    DatasetTableSetV1,
    DeviceId,
    Handedness,
    LabelId,
    MediaIntervalV1,
    MirrorState,
    OtherKind,
    ParticipantRowV1,
    SessionId,
    SessionRowV1,
)
from signlab.contracts.governance import (
    DocumentRef,
    GovernancePolicyV1,
    InventoryId,
    ParticipantId,
    RecordingConsentGrantV1,
    RecordingId,
    validate_recording_consent_grant,
)
from signlab.contracts.taxonomy import Sha256Digest, TaxonomyRef

CollectionId = Annotated[str, StringConstraints(pattern=r"^collection_[0-9a-f]{32}$")]
CaptureAttemptId = Annotated[str, StringConstraints(pattern=r"^attempt_[0-9a-f]{32}$")]
PromptOccurrenceId = Annotated[str, StringConstraints(pattern=r"^occurrence_[0-9a-f]{32}$")]
CaptureSourceKey = Annotated[str, StringConstraints(pattern=r"^source_[0-9a-f]{32}$")]
AnnotationDecisionId = Annotated[str, StringConstraints(pattern=r"^decision_[0-9a-f]{32}$")]
CaptureActorId = Annotated[str, StringConstraints(pattern=r"^actor_[0-9a-f]{32}$")]
VisitId = Annotated[str, StringConstraints(pattern=r"^visit_[0-9a-f]{32}$")]
DatasetId = Annotated[str, StringConstraints(pattern=r"^dataset_[0-9a-f]{32}$")]
StoreId = Annotated[str, StringConstraints(pattern=r"^store-[0-9a-f]{32}$")]

SidecarState = Literal["active", "paused", "complete"]
OccurrenceState = Literal["pending", "accepted", "quarantined", "skipped"]
AttemptOutcome = Literal["accepted", "retry", "quarantined"]
AnnotationRole = Literal["annotator", "reviewer", "adjudicator"]
CaptureMediaType = Literal["video/mp4", "video/quicktime", "video/webm"]
ChecklistStatus = Literal["passed", "not_applicable", "blocked"]
AnnotationReasonCode = Literal[
    "unusable_occlusion",
    "boundary_unclear",
    "consent_exclusion",
    "camera_setup",
    "third_party_presence",
    "unresolved_conflict",
]
AttemptReasonCode = Literal[
    "camera_interruption",
    "prompt_display_failure",
    "third_party_presence",
    "framing_failure",
    "corrupt_source",
    "missing_source",
    "legacy_consent_unknown",
    "legacy_label_unknown",
]
SkipReasonCode = Literal["participant_skip", "session_stopped"]
ChecklistReasonCode = Literal[
    "synthetic_no_person_no_camera",
    "consent_unverified",
    "collection_not_ready",
    "purpose_not_authorized",
    "camera_unavailable",
    "framing_unusable",
    "third_party_presence",
    "orientation_unverified",
    "playback_unverified",
]
ANNOTATION_REASON_CODES: Final[tuple[AnnotationReasonCode, ...]] = (
    "unusable_occlusion",
    "boundary_unclear",
    "consent_exclusion",
    "camera_setup",
    "third_party_presence",
    "unresolved_conflict",
)
AMBIGUOUS_REASON_CODES: Final[tuple[AnnotationReasonCode, ...]] = (
    "unusable_occlusion",
    "boundary_unclear",
)
IGNORE_REASON_CODES: Final[tuple[AnnotationReasonCode, ...]] = (
    "consent_exclusion",
    "camera_setup",
    "third_party_presence",
    "unresolved_conflict",
    "unusable_occlusion",
)
ATTEMPT_REASON_CODES: Final[tuple[AttemptReasonCode, ...]] = (
    "camera_interruption",
    "prompt_display_failure",
    "third_party_presence",
    "framing_failure",
    "corrupt_source",
    "missing_source",
    "legacy_consent_unknown",
    "legacy_label_unknown",
)
SKIP_REASON_CODES: Final[tuple[SkipReasonCode, ...]] = (
    "participant_skip",
    "session_stopped",
)
CHECKLIST_REASON_CODES: Final[tuple[ChecklistReasonCode, ...]] = (
    "synthetic_no_person_no_camera",
    "consent_unverified",
    "collection_not_ready",
    "purpose_not_authorized",
    "camera_unavailable",
    "framing_unusable",
    "third_party_presence",
    "orientation_unverified",
    "playback_unverified",
)
CONSENT_CHECKLIST_IDS: Final = (
    "authenticated_receipt_is_current",
    "collection_readiness_is_ready",
    "purpose_is_authorized_before_capture",
)
CAPTURE_CHECKLIST_IDS: Final = (
    "camera_and_lens_ready",
    "framing_and_lighting_usable",
    "no_third_party_present",
    "orientation_and_mirror_recorded",
    "timing_and_playback_checked",
)


class IngestContractError(ValueError):
    """Raised when a capture or raw-dataset contract is invalid."""


class CaptureIdentifierSetV1(StrictContractModel):
    """One preallocated, pseudonymous identifier set for an import fixture or visit."""

    model_config = contract_config("capture-identifier-set-1.schema.json")

    schema_version: Literal["capture-identifier-set/1"]
    collection_id: CollectionId
    participant_id: ParticipantId
    visit_id: VisitId
    session_id: SessionId
    device_id: DeviceId
    recording_id: RecordingId
    attempt_id: CaptureAttemptId
    source_key: CaptureSourceKey
    prompt_occurrence_id: PromptOccurrenceId
    annotation_id: AnnotationId
    annotator_actor_id: CaptureActorId
    reviewer_actor_id: CaptureActorId
    adjudicator_actor_id: CaptureActorId
    annotator_decision_id: AnnotationDecisionId
    reviewer_decision_id: AnnotationDecisionId
    adjudicator_decision_id: AnnotationDecisionId
    dataset_id: DatasetId
    store_id: StoreId
    inventory_id: InventoryId

    @model_validator(mode="after")
    def _require_distinct_annotation_actors(self) -> Self:
        actors = (
            self.annotator_actor_id,
            self.reviewer_actor_id,
            self.adjudicator_actor_id,
        )
        if len(set(actors)) != len(actors):
            raise ValueError("annotation actor identifiers must be distinct")
        decisions = (
            self.annotator_decision_id,
            self.reviewer_decision_id,
            self.adjudicator_decision_id,
        )
        if len(set(decisions)) != len(decisions):
            raise ValueError("annotation decision identifiers must be distinct")
        return self


class CollectionProtocolRefV1(StrictContractModel):
    """Immutable identity of the reviewed protocol used for a collection."""

    schema_version: Literal["collection-protocol-reference/1"]
    protocol_id: StableId
    version: SemanticVersion
    sha256: Sha256Digest


class PromptRandomizationV1(StrictContractModel):
    """Reproducibility facts for one authoritative realized prompt order."""

    schema_version: Literal["prompt-randomization/1"]
    algorithm_id: StableId
    algorithm_version: SemanticVersion
    seed_sha256: Sha256Digest
    realized_order_authoritative: Literal[True]
    rerolled_for_performance: Literal[False]


class ChecklistResultV1(StrictContractModel):
    """One coded checklist result without notes or participant-identifying text."""

    schema_version: Literal["checklist-result/1"]
    check_id: StableId
    status: ChecklistStatus
    reason_code: ChecklistReasonCode | None

    @model_validator(mode="after")
    def _require_coded_result(self) -> Self:
        if (self.status == "passed") != (self.reason_code is None):
            raise ValueError("passed checks have no reason; other results require a coded reason")
        return self


class CollectionSessionPlanV1(StrictContractModel):
    """Lean operational plan binding one visit and its reproducibility checklist."""

    schema_version: Literal["collection-session-plan/1"]
    visit_id: VisitId
    session_id: SessionId
    condition_profile_id: StableId
    prompt_randomization: PromptRandomizationV1
    consent_checklist: tuple[ChecklistResultV1, ...] = Field(min_length=1)
    capture_checklist: tuple[ChecklistResultV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_canonical_checklists(self) -> Self:
        consent_ids = tuple(result.check_id for result in self.consent_checklist)
        capture_ids = tuple(result.check_id for result in self.capture_checklist)
        if consent_ids != CONSENT_CHECKLIST_IDS:
            raise ValueError("consent checklist must cover the exact reviewed checks")
        if capture_ids != CAPTURE_CHECKLIST_IDS:
            raise ValueError("capture checklist must cover the exact reviewed checks")
        return self


class AnnotationProposalV1(StrictContractModel):
    """One coded, timestamped annotation proposal without free text."""

    schema_version: Literal["annotation-proposal/1"]
    interval: MediaIntervalV1
    disposition: Literal["class_label", "ambiguous", "ignore"]
    label_id: LabelId | None
    other_kind: OtherKind | None
    reason_code: AnnotationReasonCode | None

    @model_validator(mode="after")
    def _require_disposition_shape(self) -> Self:
        if self.disposition == "class_label":
            if self.label_id is None or self.reason_code is not None:
                raise ValueError("class proposals require a label and no exclusion reason")
            if (self.label_id == "other") != (self.other_kind is not None):
                raise ValueError("only the other label requires a registered other kind")
        elif self.label_id is not None or self.other_kind is not None or self.reason_code is None:
            raise ValueError("ambiguous and ignored proposals require only a coded reason")
        elif self.disposition == "ambiguous" and self.reason_code not in AMBIGUOUS_REASON_CODES:
            raise ValueError("ambiguous proposals require an approved ambiguous reason")
        elif self.disposition == "ignore" and self.reason_code not in IGNORE_REASON_CODES:
            raise ValueError("ignored proposals require an approved ignore reason")
        return self


class AnnotationDecisionV1(StrictContractModel):
    """One immutable actor decision in an annotation review history."""

    schema_version: Literal["annotation-decision/1"]
    decision_id: AnnotationDecisionId
    actor_id: CaptureActorId
    role: AnnotationRole
    decided_at: UtcTimestamp
    proposal: AnnotationProposalV1


class CaptureAnnotationV1(StrictContractModel):
    """An append-only annotation decision history for one accepted recording."""

    schema_version: Literal["capture-annotation/1"]
    annotation_id: AnnotationId
    source_recording_id: RecordingId
    decisions: tuple[AnnotationDecisionV1, ...] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def _require_valid_review_sequence(self) -> Self:
        roles = tuple(decision.role for decision in self.decisions)
        expected_roles = {
            1: ("annotator",),
            2: ("annotator", "reviewer"),
            3: ("annotator", "reviewer", "adjudicator"),
        }[len(self.decisions)]
        if roles != expected_roles:
            raise ValueError("annotation decisions must follow annotator, reviewer, adjudicator")

        decision_ids = tuple(decision.decision_id for decision in self.decisions)
        actor_ids = tuple(decision.actor_id for decision in self.decisions)
        ordering = tuple((decision.decided_at, decision.decision_id) for decision in self.decisions)
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("annotation decision identifiers must be unique")
        if len(actor_ids) != len(set(actor_ids)):
            raise ValueError("annotation review roles must use distinct opaque actors")
        if ordering != tuple(sorted(ordering)) or len({item[0] for item in ordering}) != len(
            ordering
        ):
            raise ValueError("annotation decisions must use strictly increasing timestamps")

        proposals = tuple(decision.proposal for decision in self.decisions)
        if len(proposals) == 2 and proposals[0] != proposals[1]:
            raise ValueError("a reviewer disagreement requires an adjudicator decision")
        if len(proposals) == 3 and proposals[0] == proposals[1]:
            raise ValueError("adjudication is only valid after an annotator-reviewer disagreement")
        return self

    @property
    def final_proposal(self) -> AnnotationProposalV1:
        """Return the proposal selected by the immutable decision sequence."""

        return self.decisions[-1].proposal

    @property
    def review_status(self) -> Literal["draft", "reviewed", "adjudicated"]:
        """Derive the normalized review state without storing redundant state."""

        if len(self.decisions) == 1:
            return "draft"
        if len(self.decisions) == 2:
            return "reviewed"
        return "adjudicated"


class CaptureAttemptV1(StrictContractModel):
    """One immutable capture result addressed only by opaque source identity."""

    schema_version: Literal["capture-attempt/1"]
    attempt_id: CaptureAttemptId
    recording_id: RecordingId
    source_key: CaptureSourceKey
    outcome: AttemptOutcome
    reason_code: AttemptReasonCode | None
    retry_of_attempt_id: CaptureAttemptId | None
    recorded_at: UtcTimestamp
    media_type: CaptureMediaType
    expected_sha256: Sha256Digest
    expected_size_bytes: PositiveSafeInteger
    duration_us: PositiveSafeInteger
    handedness: Handedness
    mirror_state: MirrorState
    rotation_degrees: Literal[0, 90, 180, 270]
    audio_present: Literal[False]
    consent_grant: RecordingConsentGrantV1 | None

    @model_validator(mode="after")
    def _require_outcome_shape(self) -> Self:
        if self.handedness == "ambidextrous":
            raise ValueError("capture handedness must describe the observed performance")
        if self.outcome == "accepted":
            if self.reason_code is not None or self.consent_grant is None:
                raise ValueError("accepted attempts require a consent grant and no failure reason")
            validate_recording_consent_grant(self.consent_grant)
            if self.consent_grant.captured_at != self.recorded_at:
                raise ValueError("accepted attempt time must match its consent grant")
            if not self.consent_grant.scope.raw_media_retention:
                raise ValueError("accepted attempts require raw-media retention permission")
        elif self.reason_code is None or self.consent_grant is not None:
            raise ValueError("retry and quarantine attempts require only a coded reason")
        return self


class PromptOccurrenceV1(StrictContractModel):
    """One preallocated protocol occurrence with its immutable attempt history."""

    schema_version: Literal["prompt-occurrence/1"]
    prompt_occurrence_id: PromptOccurrenceId
    ordinal: PositiveSafeInteger
    repetition: PositiveSafeInteger
    prompt_label_id: LabelId
    participant_id: ParticipantId
    session_id: SessionId
    state: OccurrenceState
    skip_reason_code: SkipReasonCode | None
    attempts: tuple[CaptureAttemptV1, ...]

    @model_validator(mode="after")
    def _require_consistent_attempt_history(self) -> Self:
        if self.state == "skipped":
            if self.attempts or self.skip_reason_code is None:
                raise ValueError("skipped occurrences require no attempts and a coded reason")
            return self
        if self.skip_reason_code is not None:
            raise ValueError("only a skipped occurrence may carry a skip reason")

        attempt_ids = tuple(attempt.attempt_id for attempt in self.attempts)
        source_keys = tuple(attempt.source_key for attempt in self.attempts)
        media_digests = tuple(attempt.expected_sha256 for attempt in self.attempts)
        ordering = tuple((attempt.recorded_at, attempt.attempt_id) for attempt in self.attempts)
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("capture attempt identifiers must be unique")
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("capture source keys must be unique")
        if len(media_digests) != len(set(media_digests)):
            raise ValueError("capture attempts must not duplicate media bytes")
        if ordering != tuple(sorted(ordering)):
            raise ValueError("capture attempts must be ordered by recorded time and identifier")

        for index, attempt in enumerate(self.attempts):
            if index == 0:
                if attempt.retry_of_attempt_id is not None:
                    raise ValueError("the first capture attempt cannot retry another attempt")
                continue
            previous = self.attempts[index - 1]
            if attempt.retry_of_attempt_id != previous.attempt_id or previous.outcome != "retry":
                raise ValueError("each later attempt must directly retry the preceding retry")

        accepted = tuple(attempt for attempt in self.attempts if attempt.outcome == "accepted")
        if len(accepted) > 1:
            raise ValueError("a prompt occurrence may have at most one accepted attempt")
        terminal = self.attempts[-1].outcome if self.attempts else None
        expected_state: OccurrenceState
        if terminal == "accepted":
            expected_state = "accepted"
        elif terminal == "quarantined":
            expected_state = "quarantined"
        else:
            expected_state = "pending"
        if self.state != expected_state:
            raise ValueError(f"prompt occurrence state must be {expected_state}")
        if accepted:
            grant = accepted[0].consent_grant
            if grant is None:  # pragma: no cover - guarded by CaptureAttemptV1.
                raise ValueError("accepted capture attempt is missing its consent grant")
            if (
                grant.recording_id != accepted[0].recording_id
                or grant.participant_id != self.participant_id
            ):
                raise ValueError("accepted consent grant conflicts with occurrence identity")
        return self

    @property
    def accepted_attempt(self) -> CaptureAttemptV1 | None:
        """Return the sole accepted attempt, if capture has succeeded."""

        return next(
            (attempt for attempt in self.attempts if attempt.outcome == "accepted"),
            None,
        )

    @property
    def accepted_recording_id(self) -> str | None:
        """Return the accepted attempt's preallocated recording ID, if any."""

        attempt = self.accepted_attempt
        return attempt.recording_id if attempt is not None else None


class CollectionSidecarV1(StrictContractModel):
    """A resumable capture ledger and its immutable annotation decisions."""

    model_config = contract_config("collection-sidecar-1.schema.json")

    schema_version: Literal["collection-sidecar/1"]
    collection_id: CollectionId
    dataset_id: DatasetId
    dataset_version: SemanticVersion
    store_id: StoreId
    inventory_id: InventoryId
    generated_at: UtcTimestamp
    updated_at: UtcTimestamp
    finalized_at: UtcTimestamp | None
    state: SidecarState
    fixture_only: bool
    taxonomy: TaxonomyRef
    protocol: CollectionProtocolRefV1
    governance_policy: GovernancePolicyV1
    participants: tuple[ParticipantRowV1, ...] = Field(min_length=1)
    sessions: tuple[SessionRowV1, ...] = Field(min_length=1)
    session_plans: tuple[CollectionSessionPlanV1, ...] = Field(min_length=1)
    occurrences: tuple[PromptOccurrenceV1, ...] = Field(min_length=1)
    annotations: tuple[CaptureAnnotationV1, ...]
    collection_sidecar_sha256: Sha256Digest

    @model_validator(mode="after")
    def _require_consistent_ledger(self) -> Self:
        if self.generated_at > self.updated_at:
            raise ValueError("sidecar update time cannot precede generation")
        if self.state == "complete":
            if self.finalized_at is None or self.finalized_at < self.updated_at:
                raise ValueError("a complete sidecar requires a finalization at or after update")
        elif self.finalized_at is not None:
            raise ValueError("only a complete sidecar may carry a finalization timestamp")
        if self.governance_policy.taxonomy != self.taxonomy:
            raise ValueError("governance policy and collection must use the same taxonomy")
        if self.fixture_only and any(
            result.status != "not_applicable"
            or result.reason_code != "synthetic_no_person_no_camera"
            for plan in self.session_plans
            for result in (*plan.consent_checklist, *plan.capture_checklist)
        ):
            raise ValueError(
                "fixture checklists must be not applicable with a synthetic coded reason"
            )

        participant_ids = tuple(row.participant_id for row in self.participants)
        session_ids = tuple(row.session_id for row in self.sessions)
        plan_session_ids = tuple(plan.session_id for plan in self.session_plans)
        occurrence_ids = tuple(item.prompt_occurrence_id for item in self.occurrences)
        attempt_ids = tuple(
            attempt.attempt_id for item in self.occurrences for attempt in item.attempts
        )
        recording_ids = tuple(
            attempt.recording_id for item in self.occurrences for attempt in item.attempts
        )
        annotation_ids = tuple(item.annotation_id for item in self.annotations)
        for field_name, values in (
            ("participant_id", participant_ids),
            ("session_id", session_ids),
            ("session_plan.session_id", plan_session_ids),
            ("prompt_occurrence_id", occurrence_ids),
            ("annotation_id", annotation_ids),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field_name} values must be unique and sorted")
        if len(recording_ids) != len(set(recording_ids)):
            raise ValueError("every capture attempt must use a unique recording_id")
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("every capture attempt must use a unique attempt_id")
        if plan_session_ids != session_ids:
            raise ValueError("sidecar requires exactly one session plan per session")

        participants = {row.participant_id: row for row in self.participants}
        sessions = {row.session_id: row for row in self.sessions}
        occurrences = {
            item.accepted_recording_id: item
            for item in self.occurrences
            if item.accepted_recording_id is not None
        }
        for session in self.sessions:
            if session.participant_id not in participants:
                raise ValueError("every session participant must exist in the sidecar")
        ordinals_by_session: dict[str, set[int]] = {}
        repetitions_by_session_label: dict[tuple[str, str], set[int]] = {}
        source_keys: list[str] = []
        media_digests: list[str] = []
        grant_ids: list[str] = []
        for occurrence in self.occurrences:
            session_row = sessions.get(occurrence.session_id)
            if session_row is None or session_row.participant_id != occurrence.participant_id:
                raise ValueError("occurrence participant and session grouping conflicts")
            used_ordinals = ordinals_by_session.setdefault(occurrence.session_id, set())
            if occurrence.ordinal in used_ordinals:
                raise ValueError("prompt ordinals must be unique within each session")
            used_ordinals.add(occurrence.ordinal)
            repetition_key = (occurrence.session_id, occurrence.prompt_label_id)
            used_repetitions = repetitions_by_session_label.setdefault(repetition_key, set())
            if occurrence.repetition in used_repetitions:
                raise ValueError("prompt repetitions must be unique by session and label")
            used_repetitions.add(occurrence.repetition)
            for attempt in occurrence.attempts:
                source_keys.append(attempt.source_key)
                media_digests.append(attempt.expected_sha256)
                if attempt.recorded_at > self.updated_at:
                    raise ValueError("capture attempt cannot occur after the sidecar update")
                captured_at = _parse_utc(attempt.recorded_at)
                session_started = _parse_utc(session_row.started_at)
                session_finished = _parse_utc(session_row.finished_at)
                if not session_started <= captured_at < session_finished:
                    raise ValueError("capture attempt time falls outside its session")
                remaining_us = int((session_finished - captured_at).total_seconds() * 1_000_000)
                if attempt.duration_us > remaining_us:
                    raise ValueError("capture attempt duration exceeds its session boundary")
                if (
                    attempt.mirror_state != session_row.mirror_state
                    or attempt.rotation_degrees != session_row.rotation_degrees
                ):
                    raise ValueError("capture attempt camera facts conflict with its session")
                if attempt.outcome == "accepted":
                    grant = attempt.consent_grant
                    if grant is None:  # pragma: no cover - guarded by CaptureAttemptV1.
                        raise ValueError("accepted attempt is missing consent")
                    grant_ids.append(grant.grant_id)
                    if grant.taxonomy != self.taxonomy:
                        raise ValueError(
                            "accepted consent grant taxonomy conflicts with collection"
                        )
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("source keys must be unique across the collection")
        if len(media_digests) != len(set(media_digests)):
            raise ValueError("capture attempts must not duplicate media bytes")
        if len(grant_ids) != len(set(grant_ids)):
            raise ValueError("accepted recordings must use distinct consent grants")
        for session_id, ordinals in ordinals_by_session.items():
            if ordinals != set(range(1, len(ordinals) + 1)):
                raise ValueError(f"prompt ordinals must be contiguous for session {session_id}")
        for repetitions in repetitions_by_session_label.values():
            if repetitions != set(range(1, len(repetitions) + 1)):
                raise ValueError("prompt repetitions must be contiguous by session and label")

        for annotation in self.annotations:
            occurrence_row = occurrences.get(annotation.source_recording_id)
            if occurrence_row is None or occurrence_row.state != "accepted":
                raise ValueError("annotations may reference only accepted collection recordings")
            accepted_attempt = occurrence_row.accepted_attempt
            if accepted_attempt is None:  # pragma: no cover - guarded by PromptOccurrenceV1.
                raise ValueError("accepted occurrence is missing its accepted attempt")
            if any(decision.decided_at > self.updated_at for decision in annotation.decisions):
                raise ValueError("annotation decision cannot occur after the sidecar update")
            if any(
                decision.decided_at < accepted_attempt.recorded_at
                for decision in annotation.decisions
            ):
                raise ValueError("annotation decisions cannot precede recording capture")
            if annotation.final_proposal.interval.end_us > accepted_attempt.duration_us:
                raise ValueError("annotation interval exceeds the accepted recording duration")
            _project_annotation_row(self, annotation)

        annotations_by_recording: dict[str, list[CaptureAnnotationV1]] = {}
        decision_ids = tuple(
            decision.decision_id
            for annotation in self.annotations
            for decision in annotation.decisions
        )
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("annotation decision IDs must be unique across the collection")
        for annotation in self.annotations:
            annotations_by_recording.setdefault(annotation.source_recording_id, []).append(
                annotation
            )
        for recording_annotations in annotations_by_recording.values():
            temporal_order = sorted(
                recording_annotations,
                key=lambda item: (
                    item.final_proposal.interval.start_us,
                    item.final_proposal.interval.end_us,
                    item.annotation_id,
                ),
            )
            for previous, current in pairwise(temporal_order):
                if (
                    current.final_proposal.interval.start_us
                    < previous.final_proposal.interval.end_us
                ):
                    raise ValueError("annotation intervals for one recording must not overlap")

        if self.state == "complete":
            if any(item.state == "pending" for item in self.occurrences):
                raise ValueError("a complete sidecar cannot contain pending occurrences")
            if not any(item.state == "accepted" for item in self.occurrences):
                raise ValueError("a complete sidecar requires at least one accepted occurrence")
            accepted_session_ids = {
                item.session_id for item in self.occurrences if item.state == "accepted"
            }
            for plan in self.session_plans:
                if plan.session_id in accepted_session_ids and any(
                    result.status == "blocked"
                    for result in (*plan.consent_checklist, *plan.capture_checklist)
                ):
                    raise ValueError(
                        "accepted collection sessions cannot have blocked checklist results"
                    )
        if self.collection_sidecar_sha256 != collection_sidecar_digest(self):
            raise ValueError("collection_sidecar_sha256 does not match canonical sidecar content")
        return self


class RawDatasetContentV1(StrictContractModel):
    """Storage-independent identity of normalized raw dataset tables."""

    schema_version: Literal["raw-dataset-content/1"]
    taxonomy: TaxonomyRef
    governance_policy: DocumentRef
    lineage_inventory_sha256: Sha256Digest
    collection_sidecar_sha256: Sha256Digest
    tables: DatasetTableSetV1

    @model_validator(mode="after")
    def _require_governance_policy(self) -> Self:
        if self.governance_policy.document_type != "governance_policy":
            raise ValueError("raw dataset content must bind the governance policy document")
        return self


class RawDatasetManifestV1(StrictContractModel):
    """Pre-sample dataset handoff with no trainable-sample projection."""

    model_config = contract_config("raw-dataset-manifest-1.schema.json")

    schema_version: Literal["raw-dataset-manifest/1"]
    dataset_id: DatasetId
    version: SemanticVersion
    content: RawDatasetContentV1
    raw_data_sha256: Sha256Digest

    @model_validator(mode="after")
    def _require_raw_data_identity(self) -> Self:
        if self.raw_data_sha256 != raw_dataset_content_digest(self.content):
            raise ValueError("raw_data_sha256 does not match storage-independent raw content")
        return self


type IngestInput = BaseModel | str | bytes | bytearray | Mapping[str, object]


def _validate_model[ModelT: BaseModel](
    document: IngestInput,
    model: type[ModelT],
    label: str,
) -> ModelT:
    try:
        if isinstance(document, BaseModel):
            payload = cast(
                Mapping[str, object],
                document.model_dump(mode="json", round_trip=True),
            )
        else:
            payload = cast(Mapping[str, object], parse_json_object(document))
        return model.model_validate_json(canonical_json_bytes(payload), strict=True)
    except (CanonicalizationError, ValidationError) as error:
        raise IngestContractError(f"invalid {label}") from error


def validate_capture_identifier_set(document: IngestInput) -> CaptureIdentifierSetV1:
    """Validate one strict preallocated identifier set."""

    return _validate_model(document, CaptureIdentifierSetV1, "capture identifier set")


def capture_identifier_set_digest(document: IngestInput) -> str:
    """Return the deterministic identity of a validated identifier set."""

    checked = validate_capture_identifier_set(document)
    return canonical_sha256(checked, domain=checked.schema_version)


def collection_sidecar_digest(
    sidecar: CollectionSidecarV1 | Mapping[str, object],
) -> str:
    """Hash the complete ledger while excluding its self-referential digest field."""

    if isinstance(sidecar, BaseModel):
        payload = cast(
            dict[str, object],
            sidecar.model_dump(mode="json", round_trip=True),
        )
    else:
        payload = dict(sidecar)
    payload.pop("collection_sidecar_sha256", None)
    try:
        return canonical_sha256(payload, domain="collection-sidecar/1")
    except CanonicalizationError as error:
        raise IngestContractError("collection sidecar cannot be canonicalized") from error


def validate_collection_sidecar(document: IngestInput) -> CollectionSidecarV1:
    """Validate one resumable capture ledger without accepting coercion."""

    return _validate_model(document, CollectionSidecarV1, "collection sidecar")


def require_importable_sidecar(document: IngestInput) -> CollectionSidecarV1:
    """Validate a sidecar and reject any state that is not explicitly complete."""

    checked = validate_collection_sidecar(document)
    if checked.state != "complete":
        raise IngestContractError("collection sidecar is not complete")
    return checked


def _project_annotation_row(
    sidecar: CollectionSidecarV1,
    annotation: CaptureAnnotationV1,
) -> AnnotationRowV1:
    occurrence = next(
        item
        for item in sidecar.occurrences
        if item.accepted_recording_id == annotation.source_recording_id
    )
    proposal = annotation.final_proposal
    review_status = annotation.review_status
    return AnnotationRowV1(
        annotation_id=annotation.annotation_id,
        participant_id=occurrence.participant_id,
        session_id=occurrence.session_id,
        source_recording_id=cast(str, occurrence.accepted_recording_id),
        clip_id=None,
        interval=proposal.interval,
        disposition=proposal.disposition,
        label_id=proposal.label_id,
        other_kind=proposal.other_kind,
        reason_code=proposal.reason_code,
        review_status=review_status,
        eligible_for_training=(proposal.disposition == "class_label" and review_status != "draft"),
    )


def project_annotation_rows(
    sidecar: CollectionSidecarV1 | IngestInput,
) -> tuple[AnnotationRowV1, ...]:
    """Project each decision history into exactly one normalized annotation row."""

    checked = (
        sidecar
        if isinstance(sidecar, CollectionSidecarV1)
        else validate_collection_sidecar(sidecar)
    )
    return tuple(_project_annotation_row(checked, annotation) for annotation in checked.annotations)


def _semantic_table_reference(reference: DatasetTableRefV1) -> dict[str, object]:
    return {
        "table_name": reference.table_name,
        "table_schema_version": reference.table_schema_version,
        "row_count": reference.row_count,
        "content_sha256": reference.content_sha256,
    }


def _raw_dataset_semantic_payload(content: RawDatasetContentV1) -> dict[str, object]:
    return {
        "schema_version": content.schema_version,
        "taxonomy": content.taxonomy.model_dump(mode="json", round_trip=True),
        "governance_policy": content.governance_policy.model_dump(mode="json", round_trip=True),
        "lineage_inventory_sha256": content.lineage_inventory_sha256,
        "collection_sidecar_sha256": content.collection_sidecar_sha256,
        "tables": {
            table_name: _semantic_table_reference(
                cast(DatasetTableRefV1, getattr(content.tables, table_name))
            )
            for table_name in DATASET_TABLE_SCHEMA_VERSIONS
        },
    }


def validate_raw_dataset_content(document: IngestInput) -> RawDatasetContentV1:
    """Validate the storage-independent portion of a raw dataset manifest."""

    return _validate_model(document, RawDatasetContentV1, "raw dataset content")


def raw_dataset_content_digest(
    content: RawDatasetContentV1 | IngestInput,
) -> str:
    """Hash semantic raw tables and bindings, never locators or Parquet bytes."""

    checked = (
        content
        if isinstance(content, RawDatasetContentV1)
        else validate_raw_dataset_content(content)
    )
    try:
        return canonical_sha256(
            _raw_dataset_semantic_payload(checked),
            domain=checked.schema_version,
        )
    except CanonicalizationError as error:
        raise IngestContractError("raw dataset content cannot be canonicalized") from error


def validate_raw_dataset_manifest(document: IngestInput) -> RawDatasetManifestV1:
    """Validate the raw pre-sample handoff without schema migration or coercion."""

    return _validate_model(document, RawDatasetManifestV1, "raw dataset manifest")


def _parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "AMBIGUOUS_REASON_CODES",
    "ANNOTATION_REASON_CODES",
    "ATTEMPT_REASON_CODES",
    "CAPTURE_CHECKLIST_IDS",
    "CHECKLIST_REASON_CODES",
    "CONSENT_CHECKLIST_IDS",
    "IGNORE_REASON_CODES",
    "SKIP_REASON_CODES",
    "AnnotationDecisionId",
    "AnnotationDecisionV1",
    "AnnotationProposalV1",
    "AnnotationReasonCode",
    "AnnotationRole",
    "AttemptOutcome",
    "AttemptReasonCode",
    "CaptureActorId",
    "CaptureAnnotationV1",
    "CaptureAttemptId",
    "CaptureAttemptV1",
    "CaptureIdentifierSetV1",
    "CaptureMediaType",
    "CaptureSourceKey",
    "ChecklistReasonCode",
    "ChecklistResultV1",
    "ChecklistStatus",
    "CollectionId",
    "CollectionProtocolRefV1",
    "CollectionSessionPlanV1",
    "CollectionSidecarV1",
    "DatasetId",
    "IngestContractError",
    "OccurrenceState",
    "PromptOccurrenceId",
    "PromptOccurrenceV1",
    "PromptRandomizationV1",
    "RawDatasetContentV1",
    "RawDatasetManifestV1",
    "SidecarState",
    "SkipReasonCode",
    "StoreId",
    "VisitId",
    "capture_identifier_set_digest",
    "collection_sidecar_digest",
    "project_annotation_rows",
    "raw_dataset_content_digest",
    "require_importable_sidecar",
    "validate_capture_identifier_set",
    "validate_collection_sidecar",
    "validate_raw_dataset_content",
    "validate_raw_dataset_manifest",
]
