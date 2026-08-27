"""Strict contracts for licensed external datasets and offline acquisition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, Self, cast
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

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
    WorkspaceRelativeLocatorV1,
    contract_config,
)
from signlab.contracts.taxonomy import Sha256Digest, TaxonomyRef

ExternalSplit = Literal["train", "val", "test"]
ExternalCategory = Literal["game", "non-game"]
ExternalTargetLabel = Literal["hello", "no", "please", "thank_you", "yes"]

BoundedText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=1000, pattern=r"^\S(?:.*\S)?$"),
]
SourceLabel = Annotated[
    str,
    StringConstraints(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9-]*$"),
]
OpaqueParticipantId = Annotated[
    str,
    StringConstraints(pattern=r"^participant_[0-9a-f]{32}$"),
]
OpaqueRecordingId = Annotated[
    str,
    StringConstraints(pattern=r"^recording_[0-9a-f]{32}$"),
]
OpaqueSampleId = Annotated[
    str,
    StringConstraints(pattern=r"^sample_[0-9a-f]{32}$"),
]


class ExternalDatasetContractError(ValueError):
    """Raised when licensed external-data metadata is invalid or incompatible."""


def _validate_https_url(value: str) -> str:
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("URL contains an invalid authority") from error
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or parsed.fragment
        or "\\" in value
    ):
        raise ValueError("URL must be a credential-free HTTPS URL without a fragment")
    return value


HttpsUrl = Annotated[
    str,
    StringConstraints(min_length=12, max_length=2048, pattern=r"^https://"),
    AfterValidator(_validate_https_url),
]


def _validate_download_template(value: str) -> str:
    placeholders = ("{download_id}", "{category}", "{split}", "{source_label}")
    if any(value.count(placeholder) != 1 for placeholder in placeholders):
        raise ValueError("download template must contain each registered placeholder once")
    _validate_https_url(
        value.format(
            download_id="dataset",
            category="category",
            split="split",
            source_label="label",
        )
    )
    return value


DownloadUrlTemplate = Annotated[
    str,
    StringConstraints(min_length=12, max_length=2048, pattern=r"^https://"),
    AfterValidator(_validate_download_template),
]


def _split_sort_key(split: ExternalSplit) -> int:
    return {"train": 0, "val": 1, "test": 2}[split]


class ExternalLicenseV1(StrictContractModel):
    """Machine-readable obligations for one externally governed dataset release."""

    schema_version: Literal["external-license/1"]
    license_id: Literal["CC-BY-4.0"]
    license_url: HttpsUrl
    attribution_text: BoundedText
    attribution_required: Literal[True]
    change_notice_required: Literal[True]
    redistribution_permitted: Literal[True]


class LicensedDatasetSourceV1(StrictContractModel):
    """Reviewed source facts that remain distinct from SignLab participant consent."""

    model_config = contract_config("licensed-dataset-source-1.schema.json")

    schema_version: Literal["licensed-dataset-source/1"]
    source_id: StableId
    version: SemanticVersion
    title: BoundedText
    publishers: tuple[BoundedText, ...] = Field(min_length=1, max_length=16)
    dataset_url: HttpsUrl
    download_guide_url: HttpsUrl
    download_id: StableId
    download_url_template: DownloadUrlTemplate
    license: ExternalLicenseV1
    categories: tuple[ExternalCategory, ...] = Field(min_length=1, max_length=2)
    splits: tuple[ExternalSplit, ...] = Field(min_length=1, max_length=3)
    total_videos: PositiveSafeInteger
    total_signs: PositiveSafeInteger
    total_signers: PositiveSafeInteger
    contains_identifiable_human_video: Literal[True]
    provider_reports_participant_consent: Literal[True]
    signlab_participant_consent_applicable: Literal[False]
    publisher_checksums_available: Literal[False]
    website_preview_media_permitted: Literal[False]
    suitable_uses: tuple[StableId, ...] = Field(min_length=1, max_length=16)
    unsuitable_uses: tuple[StableId, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def _require_canonical_source_lists(self) -> Self:
        for values, label in (
            (self.publishers, "publishers"),
            (self.categories, "categories"),
            (self.suitable_uses, "suitable uses"),
            (self.unsuitable_uses, "unsuitable uses"),
        ):
            if tuple(values) != tuple(sorted(set(values))):
                raise ValueError(f"{label} must be unique and in canonical order")
        if self.splits != ("train", "val", "test"):
            raise ValueError("external source splits must use train, val, test order")
        if set(self.suitable_uses) & set(self.unsuitable_uses):
            raise ValueError("suitable and unsuitable uses must not overlap")
        return self


class ExternalLabelMappingV1(StrictContractModel):
    """One reviewed source-gloss mapping into the narrower SignLab taxonomy."""

    schema_version: Literal["external-label-mapping/1"]
    source_label: SourceLabel
    target_label_id: ExternalTargetLabel
    review_status: Literal["reviewed_gloss_alignment"]
    language_equivalence_claimed: Literal[False]


class ExternalDatasetSelectionV1(StrictContractModel):
    """Frozen source subset and explicit label mappings for one SignLab use."""

    model_config = contract_config("external-dataset-selection-1.schema.json")

    schema_version: Literal["external-dataset-selection/1"]
    selection_id: StableId
    version: SemanticVersion
    source_id: StableId
    source_version: SemanticVersion
    taxonomy: TaxonomyRef
    category: ExternalCategory
    splits: tuple[ExternalSplit, ExternalSplit, ExternalSplit]
    mappings: tuple[ExternalLabelMappingV1, ...] = Field(min_length=1, max_length=64)
    learned_negative_included: Literal[False]
    claim_scope: Literal["signlab_predefined_gestures_only"]

    @model_validator(mode="after")
    def _require_canonical_selection(self) -> Self:
        if self.splits != ("train", "val", "test"):
            raise ValueError("selection splits must use train, val, test order")
        keys = tuple((item.target_label_id, item.source_label) for item in self.mappings)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("label mappings must be unique and in canonical target order")
        if len({item.source_label for item in self.mappings}) != len(self.mappings):
            raise ValueError("source labels must map at most once")
        if len({item.target_label_id for item in self.mappings}) != len(self.mappings):
            raise ValueError("target labels must map at most once")
        return self


class ExternalResourceRefV1(StrictContractModel):
    """Digest-bound reference to one immutable external-data resource."""

    schema_version: Literal["external-resource-reference/1"]
    resource_kind: Literal["source", "selection"]
    resource_id: StableId
    version: SemanticVersion
    sha256: Sha256Digest


class ExternalArchivePlanV1(StrictContractModel):
    """One official archive that an operator downloads outside SignLab."""

    schema_version: Literal["external-archive-plan/1"]
    archive_id: StableId
    category: ExternalCategory
    split: ExternalSplit
    source_label: SourceLabel
    download_url: HttpsUrl
    local_archive: WorkspaceRelativeLocatorV1
    archive_format: Literal["tar"]
    publisher_sha256: None
    integrity_basis: Literal["trust_on_first_use_then_sha256"]

    @model_validator(mode="after")
    def _require_bound_archive_location(self) -> Self:
        suffix = f"/{self.category}/{self.split}/{self.source_label}.tar"
        if not self.download_url.endswith(suffix):
            raise ValueError("archive URL does not match its category, split, and label")
        expected_path = f"archives/{self.category}/{self.split}/{self.source_label}.tar"
        if self.local_archive.path != expected_path:
            raise ValueError("archive locator does not match its category, split, and label")
        return self


class ExternalAcquisitionPlanV1(StrictContractModel):
    """Deterministic, network-free handoff from reviewed metadata to local archives."""

    model_config = contract_config("external-acquisition-plan-1.schema.json")

    schema_version: Literal["external-acquisition-plan/1"]
    plan_id: StableId
    version: SemanticVersion
    source: ExternalResourceRefV1
    selection: ExternalResourceRefV1
    network_access: Literal["forbidden"]
    preview_media: Literal["forbidden"]
    required_license_acknowledgement: Literal["CC-BY-4.0"]
    archives: tuple[ExternalArchivePlanV1, ...] = Field(min_length=1, max_length=4096)
    plan_sha256: Sha256Digest

    @model_validator(mode="after")
    def _require_canonical_plan(self) -> Self:
        if self.source.resource_kind != "source" or self.selection.resource_kind != "selection":
            raise ValueError("plan resource references have incorrect roles")
        keys = tuple(
            (_split_sort_key(item.split), item.source_label, item.archive_id)
            for item in self.archives
        )
        if keys != tuple(sorted(set(keys))):
            raise ValueError("archive plans must be unique and in canonical split/label order")
        if len({item.download_url for item in self.archives}) != len(self.archives):
            raise ValueError("archive download URLs must be unique")
        if len({item.local_archive.path for item in self.archives}) != len(self.archives):
            raise ValueError("local archive locators must be unique")
        if self.plan_sha256 != external_acquisition_plan_digest(self):
            raise ValueError("plan_sha256 does not match canonical acquisition plan content")
        return self


class ExternalLicenseAcknowledgementV1(StrictContractModel):
    """Explicit dataset-license acknowledgement without a consent claim."""

    schema_version: Literal["external-license-acknowledgement/1"]
    license_id: Literal["CC-BY-4.0"]
    accepted: Literal[True]
    authorization_basis: Literal["licensed_public_dataset"]
    signlab_participant_consent: Literal["not_applicable"]


class ExternalArchiveRecordV1(StrictContractModel):
    """Exact local identity and aggregate evidence for one imported archive."""

    schema_version: Literal["external-archive-record/1"]
    archive_id: StableId
    category: ExternalCategory
    split: ExternalSplit
    source_label: SourceLabel
    local_archive: WorkspaceRelativeLocatorV1
    sha256: Sha256Digest
    size_bytes: PositiveSafeInteger
    member_count: PositiveSafeInteger
    uncompressed_size_bytes: PositiveSafeInteger
    publisher_checksum_available: Literal[False]
    integrity_basis: Literal["local_sha256_after_download"]


class ExternalMediaRecordV1(StrictContractModel):
    """One copied media member with opaque identity and no upstream filename."""

    schema_version: Literal["external-media-record/1"]
    sample_id: OpaqueSampleId
    recording_id: OpaqueRecordingId
    participant_id: OpaqueParticipantId
    archive_id: StableId
    source_member_fingerprint: Sha256Digest
    category: ExternalCategory
    source_split: ExternalSplit
    source_label: SourceLabel
    target_label_id: ExternalTargetLabel
    media_type: Literal["video/mp4"]
    sha256: Sha256Digest
    size_bytes: PositiveSafeInteger
    locator: WorkspaceRelativeLocatorV1
    eligible_for_extraction: Literal[True]

    @model_validator(mode="after")
    def _require_content_addressed_media_locator(self) -> Self:
        digest = self.sha256.removeprefix("sha256:")
        expected = f"media/sha256/{digest[:2]}/{digest}.mp4"
        if self.locator.path != expected:
            raise ValueError("external media locator must be content-addressed")
        return self


class ExternalDatasetManifestV1(StrictContractModel):
    """License-authorized raw external media, distinct from participant raw data."""

    model_config = contract_config("external-dataset-manifest-1.schema.json")

    schema_version: Literal["external-dataset-manifest/1"]
    dataset_id: StableId
    version: SemanticVersion
    source: ExternalResourceRefV1
    selection: ExternalResourceRefV1
    acquisition_plan_sha256: Sha256Digest
    taxonomy: TaxonomyRef
    license_acknowledgement: ExternalLicenseAcknowledgementV1
    contains_identifiable_human_video: Literal[True]
    source_metadata_retained: Literal[False]
    claim_scope: Literal["isolated_predefined_gesture_research_only"]
    archives: tuple[ExternalArchiveRecordV1, ...] = Field(min_length=1, max_length=4096)
    media: tuple[ExternalMediaRecordV1, ...] = Field(min_length=1, max_length=1_000_000)
    content_sha256: Sha256Digest

    @model_validator(mode="after")
    def _require_canonical_manifest(self) -> Self:
        if self.source.resource_kind != "source" or self.selection.resource_kind != "selection":
            raise ValueError("manifest resource references have incorrect roles")
        archive_keys = tuple(
            (_split_sort_key(item.split), item.source_label, item.archive_id)
            for item in self.archives
        )
        if archive_keys != tuple(sorted(set(archive_keys))):
            raise ValueError("archive records must be unique and in canonical split/label order")
        media_keys = tuple(item.sample_id for item in self.media)
        if media_keys != tuple(sorted(set(media_keys))):
            raise ValueError("external media records must be unique and in sample order")
        archives = {item.archive_id: item for item in self.archives}
        counts = {archive_id: 0 for archive_id in archives}
        signer_splits: dict[str, ExternalSplit] = {}
        for item in self.media:
            archive = archives.get(item.archive_id)
            if archive is None:
                raise ValueError("external media references an unknown archive")
            if (
                item.category,
                item.source_split,
                item.source_label,
            ) != (
                archive.category,
                archive.split,
                archive.source_label,
            ):
                raise ValueError("external media provenance does not match its archive")
            counts[item.archive_id] += 1
            prior_split = signer_splits.setdefault(item.participant_id, item.source_split)
            if prior_split != item.source_split:
                raise ValueError("one external signer appears in more than one source split")
        if any(counts[item.archive_id] != item.member_count for item in self.archives):
            raise ValueError("archive member counts do not match external media records")
        if self.content_sha256 != external_dataset_manifest_digest(self):
            raise ValueError("content_sha256 does not match canonical external dataset content")
        return self


type ExternalDatasetInput = BaseModel | str | bytes | bytearray | Mapping[str, object]


def _payload_without_digest(
    document: BaseModel | Mapping[str, object],
    field_name: str,
) -> dict[str, object]:
    if isinstance(document, BaseModel):
        payload = cast(dict[str, object], document.model_dump(mode="json", round_trip=True))
    else:
        payload = dict(document)
    payload.pop(field_name, None)
    return payload


def licensed_dataset_source_digest(document: LicensedDatasetSourceV1) -> str:
    """Return the immutable semantic identity of a reviewed source record."""

    return canonical_sha256(document, domain=document.schema_version)


def external_dataset_selection_digest(document: ExternalDatasetSelectionV1) -> str:
    """Return the immutable semantic identity of a reviewed source selection."""

    return canonical_sha256(document, domain=document.schema_version)


def external_acquisition_plan_digest(
    document: ExternalAcquisitionPlanV1 | Mapping[str, object],
) -> str:
    """Hash an acquisition plan while excluding its self-referential digest."""

    try:
        return canonical_sha256(
            _payload_without_digest(document, "plan_sha256"),
            domain="external-acquisition-plan/1",
        )
    except CanonicalizationError as error:
        raise ExternalDatasetContractError("acquisition plan cannot be canonicalized") from error


def external_dataset_manifest_digest(
    document: ExternalDatasetManifestV1 | Mapping[str, object],
) -> str:
    """Hash manifest semantics while excluding the self-referential content digest."""

    try:
        return canonical_sha256(
            _payload_without_digest(document, "content_sha256"),
            domain="external-dataset-manifest/1",
        )
    except CanonicalizationError as error:
        raise ExternalDatasetContractError("external manifest cannot be canonicalized") from error


def _validate_model[ModelT: BaseModel](
    document: ExternalDatasetInput,
    model: type[ModelT],
    label: str,
) -> ModelT:
    try:
        if isinstance(document, BaseModel):
            payload = cast(Mapping[str, object], document.model_dump(mode="json", round_trip=True))
        else:
            payload = cast(Mapping[str, object], parse_json_object(document))
        return model.model_validate_json(canonical_json_bytes(payload), strict=True)
    except (CanonicalizationError, ValidationError) as error:
        raise ExternalDatasetContractError(f"invalid {label}") from error


def validate_licensed_dataset_source(document: ExternalDatasetInput) -> LicensedDatasetSourceV1:
    """Validate one strict licensed-dataset source record."""

    return _validate_model(document, LicensedDatasetSourceV1, "licensed dataset source")


def validate_external_dataset_selection(
    document: ExternalDatasetInput,
) -> ExternalDatasetSelectionV1:
    """Validate one strict source selection and label map."""

    return _validate_model(document, ExternalDatasetSelectionV1, "external dataset selection")


def validate_external_acquisition_plan(
    document: ExternalDatasetInput,
) -> ExternalAcquisitionPlanV1:
    """Validate one strict offline acquisition plan."""

    return _validate_model(document, ExternalAcquisitionPlanV1, "external acquisition plan")


def validate_external_dataset_manifest(
    document: ExternalDatasetInput,
) -> ExternalDatasetManifestV1:
    """Validate one strict license-authorized external dataset manifest."""

    return _validate_model(document, ExternalDatasetManifestV1, "external dataset manifest")


__all__ = [
    "ExternalAcquisitionPlanV1",
    "ExternalArchivePlanV1",
    "ExternalArchiveRecordV1",
    "ExternalDatasetContractError",
    "ExternalDatasetManifestV1",
    "ExternalDatasetSelectionV1",
    "ExternalLabelMappingV1",
    "ExternalLicenseAcknowledgementV1",
    "ExternalLicenseV1",
    "ExternalMediaRecordV1",
    "ExternalResourceRefV1",
    "LicensedDatasetSourceV1",
    "external_acquisition_plan_digest",
    "external_dataset_manifest_digest",
    "external_dataset_selection_digest",
    "licensed_dataset_source_digest",
    "validate_external_acquisition_plan",
    "validate_external_dataset_manifest",
    "validate_external_dataset_selection",
    "validate_licensed_dataset_source",
]
