"""Strict portable contracts for derived landmark feature representations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Final, Literal, Self, cast

from pydantic import BaseModel, Field, ValidationError, model_validator

from signlab.contracts.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    canonical_sha256,
    parse_json_object,
)
from signlab.contracts.core import (
    DottedId,
    NonNegativeSafeInteger,
    PositiveSafeInteger,
    SafeInteger,
    SemanticVersion,
    StableId,
    StrictContractModel,
    contract_config,
)
from signlab.contracts.governance import RecordingId
from signlab.contracts.taxonomy import Sha256Digest

FEATURE_QUANTIZATION_SCALE: Final = 1_000_000
HAND_LANDMARK_ORDER: Final = tuple(range(21))
FEATURE_HAND_SLOTS: Final = ("hand_0", "hand_1")

FeatureRepresentation = Literal["hand_local", "body_relative", "combined"]
FeaturePartition = Literal["train"]


class FeatureContractError(ValueError):
    """Raised when feature contracts are invalid or mutually incompatible."""


class HandLocalRuleV1(StrictContractModel):
    """Registered hand-shape normalization using only hand-world coordinates."""

    coordinate_space: Literal["hand_world_xyz"]
    landmark_order: tuple[NonNegativeSafeInteger, ...] = Field(min_length=21, max_length=21)
    center: Literal["wrist_landmark_0"]
    scale: Literal["wrist_to_middle_mcp_landmark_9_euclidean"]
    source_mirror_rule: Literal["undo_world_x_when_source_mirrored"]
    anatomical_canonicalization: Literal[
        "swap_vendor_label_when_not_mirrored_then_reflect_left_hand_x"
    ]
    zero_scale_rule: Literal["mask_hand_features"]

    @model_validator(mode="after")
    def _require_registered_landmarks(self) -> Self:
        if self.landmark_order != HAND_LANDMARK_ORDER:
            raise ValueError("hand-local landmarks must use the registered MediaPipe order")
        return self


class BodyRelativeRuleV1(StrictContractModel):
    """Registered image-space wrist and palm trajectory normalization."""

    coordinate_space: Literal["image_xy"]
    trajectory_points: tuple[Literal["wrist"], Literal["palm_centroid"]]
    palm_landmarks: tuple[
        Literal[0],
        Literal[5],
        Literal[9],
        Literal[17],
    ]
    center: Literal["shoulder_midpoint"]
    scale: Literal["shoulder_width_xy_euclidean"]
    source_mirror_rule: Literal["undo_image_x_when_source_mirrored"]
    missing_anchor_rule: Literal["mask_body_keep_hand_local"]
    zero_scale_rule: Literal["mask_body_features"]

    @model_validator(mode="after")
    def _require_registered_body_trajectory(self) -> Self:
        if self.trajectory_points != ("wrist", "palm_centroid"):
            raise ValueError("body trajectory points must be wrist then palm centroid")
        if self.palm_landmarks != (0, 5, 9, 17):
            raise ValueError("palm centroid must use the registered four landmarks")
        return self


class TemporalFeatureRuleV1(StrictContractModel):
    """Elapsed-time resampling rules inherited from landmark quality evidence."""

    clock: Literal["relative_timestamp_us"]
    grid_rule: Literal["nominal_elapsed_time_append_final/1"]
    target_rate_numerator: Literal[30]
    target_rate_denominator: Literal[1]
    interpolation: Literal["quality_report_approved_linear_coordinates_only"]
    extrapolation_allowed: Literal[False]
    forward_fill_allowed: Literal[False]
    derivative_rule: Literal["backward_elapsed_time_finite_difference/1"]
    derivative_application_order: Literal["resample_then_derive_then_select_then_pad/1"]


class PaddingFeatureRuleV1(StrictContractModel):
    """Fixed-shape selection and neutral right-padding rules."""

    target_frame_count: Annotated[PositiveSafeInteger, Field(ge=2, le=4096)]
    long_sequence_rule: Literal["uniform_endpoint_preserving_index_selection/1"]
    padding_side: Literal["right"]
    padding_value_q: Literal[0]
    padding_mask_rule: Literal["all_feature_masks_false"]
    padding_timestamp_rule: Literal["continue_nominal_grid"]


class OptionalFeatureRuleV1(StrictContractModel):
    """Small registered set of optional kinematic and geometric features."""

    include_velocity: bool
    include_acceleration: bool
    include_joint_angles: bool
    include_tip_distances: bool
    joint_angle_rule: Literal["five_registered_pip_flexion_angles_radians/1"]
    tip_distance_rule: Literal["five_registered_wrist_to_fingertip_distances/1"]

    @model_validator(mode="after")
    def _require_velocity_before_acceleration(self) -> Self:
        if self.include_acceleration and not self.include_velocity:
            raise ValueError("acceleration requires velocity")
        return self


class LearnedStatisticsRuleV1(StrictContractModel):
    """Train-only masked feature-standardization policy."""

    mode: Literal["none", "train_only_masked_zscore/1"]
    partition_evidence: Literal["explicit_train_membership_required"]
    masked_value_rule: Literal["exclude_from_fit"]
    zero_count_rule: Literal["mean_zero_scale_one"]
    zero_variance_rule: Literal["scale_one"]


_ANGLE_NAMES: Final = ("thumb", "index", "middle", "ring", "pinky")
_TIP_NAMES: Final = ("thumb", "index", "middle", "ring", "pinky")


def registered_feature_names(
    representation: FeatureRepresentation,
    optional: OptionalFeatureRuleV1,
) -> tuple[str, ...]:
    """Return the only legal channel order for one resolved feature plan."""

    positions: list[str] = []
    if representation in {"hand_local", "combined"}:
        positions.extend(
            f"{slot}.local.landmark_{landmark:02d}.{axis}"
            for slot in FEATURE_HAND_SLOTS
            for landmark in HAND_LANDMARK_ORDER
            for axis in ("x", "y", "z")
        )
    if representation in {"body_relative", "combined"}:
        positions.extend(
            f"{slot}.body.{point}.{axis}"
            for slot in FEATURE_HAND_SLOTS
            for point in ("wrist", "palm")
            for axis in ("x", "y")
        )
    names = list(positions)
    if optional.include_joint_angles:
        names.extend(
            f"{slot}.geometry.{finger}.angle"
            for slot in FEATURE_HAND_SLOTS
            for finger in _ANGLE_NAMES
        )
    if optional.include_tip_distances:
        names.extend(
            f"{slot}.geometry.{finger}.tip_distance"
            for slot in FEATURE_HAND_SLOTS
            for finger in _TIP_NAMES
        )
    if optional.include_velocity:
        names.extend(f"{name}.velocity" for name in positions)
    if optional.include_acceleration:
        names.extend(f"{name}.acceleration" for name in positions)
    return tuple(names)


class LandmarkFeaturePlanV1(StrictContractModel):
    """Exact cross-runtime definition of one derived landmark representation."""

    model_config = contract_config("landmark-feature-plan-1.schema.json")

    schema_version: Literal["landmark-feature-plan/1"]
    plan_id: StableId
    version: SemanticVersion
    representation: FeatureRepresentation
    compatible_runtimes: tuple[Literal["python", "typescript"], ...] = Field(min_length=2)
    hand_slots: tuple[Literal["hand_0"], Literal["hand_1"]]
    handedness_source: Literal["mediapipe_vendor_report_corrected_by_source_mirror_state"]
    swap_rule: Literal["preserve_slots_never_repair"]
    hand_local: HandLocalRuleV1
    body_relative: BodyRelativeRuleV1
    temporal: TemporalFeatureRuleV1
    padding: PaddingFeatureRuleV1
    optional: OptionalFeatureRuleV1
    learned_statistics: LearnedStatisticsRuleV1
    quantization_scale: Literal[1000000]
    quantization_rule: Literal["round_half_away_from_zero/1"]
    interchange_values: Literal["signed_integer_divided_by_quantization_scale"]
    feature_order: tuple[DottedId, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_registered_plan(self) -> Self:
        if self.compatible_runtimes != ("python", "typescript"):
            raise ValueError("feature runtimes must be python then typescript")
        if self.hand_slots != FEATURE_HAND_SLOTS:
            raise ValueError("feature hand slots must remain hand_0 then hand_1")
        expected = registered_feature_names(self.representation, self.optional)
        if self.feature_order != expected:
            raise ValueError("feature order does not match the registered plan settings")
        if len(self.feature_order) != len(set(self.feature_order)):
            raise ValueError("feature names must be unique")
        return self


class PortableFeatureSequenceV1(StrictContractModel):
    """Fixed-shape quantized features plus explicit provenance and masks."""

    model_config = contract_config("portable-feature-sequence-1.schema.json")

    schema_version: Literal["portable-feature-sequence/1"]
    source_recording_id: RecordingId
    source_media_sha256: Sha256Digest
    source_landmarks_sha256: Sha256Digest
    extraction_config_sha256: Sha256Digest
    quality_policy_sha256: Sha256Digest
    quality_report_sha256: Sha256Digest
    feature_plan_sha256: Sha256Digest
    statistics_sha256: Sha256Digest | None
    feature_names: tuple[DottedId, ...] = Field(min_length=1)
    quantization_scale: Literal[1000000]
    source_grid_frame_count: PositiveSafeInteger
    selected_source_indices: tuple[NonNegativeSafeInteger, ...] = Field(min_length=1)
    timestamps_us: tuple[NonNegativeSafeInteger, ...] = Field(min_length=2)
    values_q: tuple[tuple[SafeInteger, ...], ...] = Field(min_length=2)
    valid_mask: tuple[tuple[bool, ...], ...] = Field(min_length=2)
    observed_mask: tuple[tuple[bool, ...], ...] = Field(min_length=2)
    interpolated_mask: tuple[tuple[bool, ...], ...] = Field(min_length=2)
    hand_present_mask: tuple[tuple[bool, bool], ...] = Field(min_length=2)
    body_available_mask: tuple[bool, ...] = Field(min_length=2)
    padding_mask: tuple[bool, ...] = Field(min_length=2)
    sequence_sha256: Sha256Digest

    @model_validator(mode="after")
    def _require_fixed_shape_masks_and_identity(self) -> Self:
        frame_count = len(self.timestamps_us)
        feature_count = len(self.feature_names)
        frame_fields = (
            self.values_q,
            self.valid_mask,
            self.observed_mask,
            self.interpolated_mask,
            self.hand_present_mask,
            self.body_available_mask,
            self.padding_mask,
        )
        if any(len(field) != frame_count for field in frame_fields):
            raise ValueError("feature frame arrays must have one fixed length")
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("feature names must be unique")
        for row in (*self.values_q, *self.valid_mask, *self.observed_mask, *self.interpolated_mask):
            if len(row) != feature_count:
                raise ValueError("every feature row must match feature_names")
        if self.timestamps_us[0] != 0 or any(
            right <= left
            for left, right in zip(self.timestamps_us, self.timestamps_us[1:], strict=False)
        ):
            raise ValueError("feature timestamps must increase strictly from zero")
        selected_count = len(self.selected_source_indices)
        if selected_count > frame_count or selected_count > self.source_grid_frame_count:
            raise ValueError("selected feature frames exceed source or output shape")
        if self.selected_source_indices != tuple(sorted(set(self.selected_source_indices))):
            raise ValueError("source selection indexes must be unique and increasing")
        if self.selected_source_indices[0] != 0 or (
            self.source_grid_frame_count > 1
            and self.selected_source_indices[-1] != self.source_grid_frame_count - 1
        ):
            raise ValueError("feature selection must preserve both elapsed-time endpoints")
        if self.padding_mask != (False,) * selected_count + (True,) * (
            frame_count - selected_count
        ):
            raise ValueError("feature padding must be a contiguous right suffix")
        hand_feature_indexes = {
            slot: tuple(
                index
                for index, feature_name in enumerate(self.feature_names)
                if feature_name.startswith(f"{slot}.")
            )
            for slot in FEATURE_HAND_SLOTS
        }
        body_feature_indexes = tuple(
            index
            for index, feature_name in enumerate(self.feature_names)
            if ".body." in feature_name
        )
        derivative_features = tuple(
            feature_name.endswith((".velocity", ".acceleration"))
            for feature_name in self.feature_names
        )
        for frame_index in range(frame_count):
            padded = self.padding_mask[frame_index]
            for feature_index in range(feature_count):
                value = self.values_q[frame_index][feature_index]
                valid = self.valid_mask[frame_index][feature_index]
                observed = self.observed_mask[frame_index][feature_index]
                interpolated = self.interpolated_mask[frame_index][feature_index]
                if observed and interpolated:
                    raise ValueError("a feature cannot be observed and interpolated")
                if (observed or interpolated) and not valid:
                    raise ValueError("feature evidence requires a valid value")
                if valid and derivative_features[feature_index] and (observed or interpolated):
                    raise ValueError("derivative features cannot claim source-value evidence")
                if (
                    valid
                    and not derivative_features[feature_index]
                    and not (observed or interpolated)
                ):
                    raise ValueError("valid non-derivative features require source-value evidence")
                if (not valid and value != 0) or (padded and (valid or value != 0)):
                    raise ValueError("masked and padded features must use neutral storage")
            for slot_index, slot in enumerate(FEATURE_HAND_SLOTS):
                if not self.hand_present_mask[frame_index][slot_index] and any(
                    self.valid_mask[frame_index][feature_index]
                    for feature_index in hand_feature_indexes[slot]
                ):
                    raise ValueError("valid hand features require a sampled hand observation")
            if not self.body_available_mask[frame_index] and any(
                self.valid_mask[frame_index][feature_index]
                for feature_index in body_feature_indexes
            ):
                raise ValueError("valid body-relative features require body anchors")
            if padded and (
                any(self.hand_present_mask[frame_index]) or self.body_available_mask[frame_index]
            ):
                raise ValueError("padding cannot claim source observations")
        if self.sequence_sha256 != portable_feature_sequence_digest(self):
            raise ValueError("feature sequence digest does not match canonical content")
        return self


class TrainingFeatureSequenceV1(StrictContractModel):
    """One split-bound training input to dataset statistics."""

    schema_version: Literal["training-feature-sequence/1"]
    partition: FeaturePartition
    split_manifest_sha256: Sha256Digest
    sequence: PortableFeatureSequenceV1


class FeatureStatisticsV1(StrictContractModel):
    """Masked z-score statistics fitted only from identified training features."""

    model_config = contract_config("feature-statistics-1.schema.json")

    schema_version: Literal["feature-statistics/1"]
    statistics_id: StableId
    version: SemanticVersion
    feature_plan_sha256: Sha256Digest
    split_manifest_sha256: Sha256Digest
    feature_names: tuple[DottedId, ...] = Field(min_length=1)
    quantization_scale: Literal[1000000]
    training_sequence_sha256: tuple[Sha256Digest, ...] = Field(min_length=1)
    observation_count: tuple[NonNegativeSafeInteger, ...] = Field(min_length=1)
    mean_q: tuple[SafeInteger, ...] = Field(min_length=1)
    standard_deviation_q: tuple[PositiveSafeInteger, ...] = Field(min_length=1)
    statistics_sha256: Sha256Digest

    @model_validator(mode="after")
    def _require_canonical_statistics(self) -> Self:
        width = len(self.feature_names)
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("statistics feature names must be unique")
        if any(
            len(values) != width
            for values in (self.observation_count, self.mean_q, self.standard_deviation_q)
        ):
            raise ValueError("statistics vectors must match feature_names")
        if self.training_sequence_sha256 != tuple(sorted(set(self.training_sequence_sha256))):
            raise ValueError("training feature identities must be unique and sorted")
        for count, mean, scale in zip(
            self.observation_count,
            self.mean_q,
            self.standard_deviation_q,
            strict=True,
        ):
            if count == 0 and (mean != 0 or scale != self.quantization_scale):
                raise ValueError("empty feature statistics must use mean zero and scale one")
        if self.statistics_sha256 != feature_statistics_digest(self):
            raise ValueError("feature statistics digest does not match canonical content")
        return self


class FeatureCacheKeyV1(StrictContractModel):
    """Storage-independent identity of one fully resolved feature derivation."""

    model_config = contract_config("feature-cache-key-1.schema.json")

    schema_version: Literal["feature-cache-key/1"]
    source_recording_id: RecordingId
    source_media_sha256: Sha256Digest
    source_landmarks_sha256: Sha256Digest
    extraction_config_sha256: Sha256Digest
    quality_policy_sha256: Sha256Digest
    quality_report_sha256: Sha256Digest
    feature_plan_sha256: Sha256Digest
    statistics_sha256: Sha256Digest | None
    cache_key_sha256: Sha256Digest

    @model_validator(mode="after")
    def _require_cache_identity(self) -> Self:
        if self.cache_key_sha256 != feature_cache_key_digest(self):
            raise ValueError("feature cache digest does not match its bound inputs")
        return self


FeatureInput = str | bytes | bytearray | Mapping[str, object]


def _validate_model[ModelT: BaseModel](
    document: FeatureInput,
    model: type[ModelT],
    label: str,
) -> ModelT:
    try:
        if isinstance(document, Mapping):
            payload = cast(Mapping[str, object], parse_json_object(document))
        else:
            payload = parse_json_object(document)
        return model.model_validate_json(canonical_json_bytes(payload), strict=True)
    except (CanonicalizationError, ValidationError) as error:
        raise FeatureContractError(f"invalid {label}") from error


def landmark_feature_plan_digest(
    document: LandmarkFeaturePlanV1 | Mapping[str, object],
) -> str:
    """Return the stable identity of one fully resolved landmark feature plan."""

    try:
        checked = (
            LandmarkFeaturePlanV1.model_validate(document, strict=True)
            if isinstance(document, LandmarkFeaturePlanV1)
            else validate_landmark_feature_plan(document)
        )
    except ValidationError as error:
        raise FeatureContractError("invalid landmark feature plan") from error
    return canonical_sha256(checked, domain=checked.schema_version)


def portable_feature_sequence_digest(
    document: PortableFeatureSequenceV1 | Mapping[str, object],
) -> str:
    """Hash feature content while excluding its self-referential digest."""

    payload = (
        cast(dict[str, object], document.model_dump(mode="json", round_trip=True))
        if isinstance(document, BaseModel)
        else dict(document)
    )
    payload.pop("sequence_sha256", None)
    return canonical_sha256(payload, domain="portable-feature-sequence/1")


def feature_statistics_digest(
    document: FeatureStatisticsV1 | Mapping[str, object],
) -> str:
    """Hash fitted statistics while excluding their self-referential digest."""

    payload = (
        cast(dict[str, object], document.model_dump(mode="json", round_trip=True))
        if isinstance(document, BaseModel)
        else dict(document)
    )
    payload.pop("statistics_sha256", None)
    return canonical_sha256(payload, domain="feature-statistics/1")


def feature_cache_key_digest(
    document: FeatureCacheKeyV1 | Mapping[str, object],
) -> str:
    """Hash cache inputs while excluding their self-referential digest."""

    payload = (
        cast(dict[str, object], document.model_dump(mode="json", round_trip=True))
        if isinstance(document, BaseModel)
        else dict(document)
    )
    payload.pop("cache_key_sha256", None)
    return canonical_sha256(payload, domain="feature-cache-key/1")


def validate_landmark_feature_plan(document: FeatureInput) -> LandmarkFeaturePlanV1:
    """Strictly validate a landmark feature plan without implicit migration."""

    return _validate_model(document, LandmarkFeaturePlanV1, "landmark feature plan")


def validate_portable_feature_sequence(document: FeatureInput) -> PortableFeatureSequenceV1:
    """Strictly validate one derived portable feature sequence."""

    return _validate_model(document, PortableFeatureSequenceV1, "portable feature sequence")


def validate_feature_statistics(document: FeatureInput) -> FeatureStatisticsV1:
    """Strictly validate train-only fitted feature statistics."""

    return _validate_model(document, FeatureStatisticsV1, "feature statistics")


def validate_feature_cache_key(document: FeatureInput) -> FeatureCacheKeyV1:
    """Strictly validate one feature cache identity."""

    return _validate_model(document, FeatureCacheKeyV1, "feature cache key")


__all__ = [
    "FEATURE_HAND_SLOTS",
    "FEATURE_QUANTIZATION_SCALE",
    "HAND_LANDMARK_ORDER",
    "BodyRelativeRuleV1",
    "FeatureCacheKeyV1",
    "FeatureContractError",
    "FeatureInput",
    "FeaturePartition",
    "FeatureRepresentation",
    "FeatureStatisticsV1",
    "HandLocalRuleV1",
    "LandmarkFeaturePlanV1",
    "LearnedStatisticsRuleV1",
    "OptionalFeatureRuleV1",
    "PaddingFeatureRuleV1",
    "PortableFeatureSequenceV1",
    "TemporalFeatureRuleV1",
    "TrainingFeatureSequenceV1",
    "feature_cache_key_digest",
    "feature_statistics_digest",
    "landmark_feature_plan_digest",
    "portable_feature_sequence_digest",
    "registered_feature_names",
    "validate_feature_cache_key",
    "validate_feature_statistics",
    "validate_landmark_feature_plan",
    "validate_portable_feature_sequence",
]
