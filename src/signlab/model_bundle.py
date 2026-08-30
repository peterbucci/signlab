"""Strict v1 manifest and byte-level validator for browser candidate bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal, Self, cast

from pydantic import Field, ValidationError, model_validator

from signlab.candidate_events import CandidateEventConfigV1, candidate_event_config_digest
from signlab.contracts.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    canonical_sha256,
    parse_json_object,
)
from signlab.contracts.core import (
    ArtifactRefV1,
    GitCommit,
    SemanticVersion,
    StableId,
    StrictContractModel,
    WorkspaceRelativeLocatorV1,
    contract_config,
)
from signlab.contracts.extraction import (
    mediapipe_extraction_config_digest,
    validate_mediapipe_extraction_config,
)
from signlab.contracts.features import (
    landmark_feature_plan_digest,
    validate_landmark_feature_plan,
)
from signlab.contracts.quality import (
    landmark_quality_policy_digest,
    validate_landmark_quality_policy,
)
from signlab.contracts.taxonomy import EXPECTED_CLASS_IDS, Sha256Digest
from signlab.datasets.storage import DatasetStorageError, verify_artifact_references

_REQUIRED_ASSETS: Final = {
    "decision_policy": ("decision-policy.json", "application/json"),
    "feature_plan": ("feature-plan.json", "application/json"),
    "golden_smoke": ("golden/smoke.json", "application/json"),
    "landmarker": ("landmarker.json", "application/json"),
    "model": ("model.onnx", "application/onnx"),
    "model_card": ("model-card.md", "text/markdown"),
    "quality_policy": ("quality-policy.json", "application/json"),
    "segmenter": ("segmenter.json", "application/json"),
}
_REQUIRED_LICENSES: Final = (
    ("mediapipe", "Apache-2.0", "redistributable"),
    ("popsign_source_data", "CC-BY-4.0", "redistributable_with_attribution"),
    ("signlab_code", "MIT", "redistributable"),
    ("trained_model", "NOASSERTION", "local_evaluation_only"),
)


class ModelBundleError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class DecisionPolicyIdentitiesV1(StrictContractModel):
    configuration_sha256: Sha256Digest
    corpus_sha256: Sha256Digest
    derivative_set_sha256: Sha256Digest
    input_feature_plan_sha256: Sha256Digest
    model_sha256: Sha256Digest
    source_feature_plan_sha256: Sha256Digest
    split_sha256: Sha256Digest
    taxonomy_sha256: Sha256Digest


class DecisionPolicyV1(StrictContractModel):
    format: Literal["signlab-decision-policy/1"]
    evidence_kind: Literal["constructed_transition_calibration_conformance"]
    metric_claim: Literal["development_mechanics_only"]
    status: Literal["selected"]
    test_status: Literal["sealed_not_loaded"]
    class_map: dict[str, str]
    decision_precedence: tuple[str, ...]
    temperature: dict[str, str | int]
    abstention: dict[str, str | int | bool]
    identities: DecisionPolicyIdentitiesV1

    @model_validator(mode="after")
    def _require_exact_runtime_semantics(self) -> Self:
        if (
            self.class_map != {str(index): label for index, label in enumerate(EXPECTED_CLASS_IDS)}
            or self.decision_precedence
            != (
                "no_candidate_to_inactive",
                "invalid_policy_or_probabilities_to_abstain",
                "below_threshold_to_abstain",
                "accepted_argmax_to_target_or_other",
            )
            or self.temperature
            != {
                "method": "softmax_log_probability_scalar_temperature/1",
                "temperature_milli": 50,
            }
            or self.abstention
            != {
                "inclusive": True,
                "objective": "maximize_target_coverage_zero_observed_accepted_errors/1",
                "threshold_percent": 0,
            }
            or not isinstance(self.abstention.get("inclusive"), bool)
            or isinstance(self.abstention.get("threshold_percent"), bool)
        ):
            raise ValueError("decision policy runtime semantics are unsupported")
        return self


class CandidateLineageV1(StrictContractModel):
    candidate_id: StableId
    candidate_version: SemanticVersion
    nomination_scope: Literal["portable_export_only"]
    champion_status: Literal["none_blocked"]
    metric_claim: Literal["development_only"]
    test_status: Literal["sealed_not_loaded"]
    source_run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    source_commit: GitCommit
    nomination_report_sha256: Sha256Digest
    dossier_sha256: Sha256Digest
    research_checkpoint_sha256: Sha256Digest
    configuration_sha256: Sha256Digest
    corpus_sha256: Sha256Digest
    derivative_set_sha256: Sha256Digest
    external_dataset_sha256: Sha256Digest
    split_sha256: Sha256Digest
    input_feature_plan_sha256: Sha256Digest
    source_feature_plan_sha256: Sha256Digest
    taxonomy_sha256: Sha256Digest


class ComponentIdentitiesV1(StrictContractModel):
    landmarker_sha256: Sha256Digest
    quality_policy_sha256: Sha256Digest
    feature_plan_sha256: Sha256Digest
    segmenter_sha256: Sha256Digest
    decision_policy_sha256: Sha256Digest


class OnnxContractV1(StrictContractModel):
    format: Literal["onnx"]
    opset: Literal[18]
    input_name: Literal["input"]
    input_shape: tuple[Literal[1], Literal[64], Literal[126]]
    input_dtype: Literal["float32"]
    input_semantics: Literal["hand_local_feature_sequence"]
    output_name: Literal["probabilities"]
    output_shape: tuple[Literal[1], Literal[6]]
    output_dtype: Literal["float32"]
    output_semantics: Literal["uncalibrated_class_probabilities"]


class LicenseRecordV1(StrictContractModel):
    scope: Literal["mediapipe", "popsign_source_data", "signlab_code", "trained_model"]
    spdx: Literal["Apache-2.0", "CC-BY-4.0", "MIT", "NOASSERTION"]
    distribution: Literal[
        "redistributable",
        "redistributable_with_attribution",
        "local_evaluation_only",
    ]


class BrowserModelBundleManifestV1(StrictContractModel):
    model_config = contract_config("browser-model-bundle-manifest-1.schema.json")

    format: Literal["browser-model-bundle/1"]
    bundle_id: StableId
    version: SemanticVersion
    candidate: CandidateLineageV1
    components: ComponentIdentitiesV1
    onnx: OnnxContractV1
    labels: tuple[str, ...] = Field(min_length=6, max_length=6)
    licenses: tuple[LicenseRecordV1, ...] = Field(min_length=4, max_length=4)
    assets: tuple[ArtifactRefV1, ...] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def _require_canonical_inventory_and_licenses(self) -> Self:
        if self.labels != EXPECTED_CLASS_IDS:
            raise ValueError("bundle labels must use the immutable taxonomy order")
        if not all(isinstance(asset.locator, WorkspaceRelativeLocatorV1) for asset in self.assets):
            raise ValueError("bundle assets must use workspace-relative paths")
        actual_assets = tuple(
            (
                asset.role,
                asset.artifact_id,
                cast(WorkspaceRelativeLocatorV1, asset.locator).path,
                asset.media_type,
                asset.size_bytes > 0,
            )
            for asset in self.assets
        )
        expected_assets = tuple(
            (role, role, path, media_type, True)
            for role, (path, media_type) in _REQUIRED_ASSETS.items()
        )
        if actual_assets != expected_assets:
            raise ValueError("bundle asset inventory is unsupported")
        actual_licenses = tuple(
            (record.scope, record.spdx, record.distribution) for record in self.licenses
        )
        if actual_licenses != _REQUIRED_LICENSES:
            raise ValueError("bundle licenses must be exact and sorted")
        return self


def _canonical_model[ModelT: StrictContractModel](
    raw: bytes, model: type[ModelT], code: str, expected_format: str | None = None
) -> ModelT:
    try:
        payload = parse_json_object(raw)
        if expected_format is not None and payload.get("format") != expected_format:
            raise ModelBundleError("unsupported_manifest_version")
        checked = model.model_validate_json(canonical_json_bytes(payload), strict=True)
        if raw != canonical_json_bytes(checked) + b"\n":
            raise ModelBundleError(f"noncanonical_{code}")
        return checked
    except ModelBundleError:
        raise
    except (CanonicalizationError, ValidationError, TypeError, ValueError) as error:
        raise ModelBundleError(f"invalid_{code}") from error


def load_browser_bundle_manifest(raw: bytes) -> BrowserModelBundleManifestV1:
    return _canonical_model(raw, BrowserModelBundleManifestV1, "manifest", "browser-model-bundle/1")


def decision_policy_digest(policy: DecisionPolicyV1) -> str:
    return canonical_sha256(policy, domain=policy.format)


def browser_bundle_digest(manifest: BrowserModelBundleManifestV1) -> str:
    return canonical_sha256(manifest, domain=manifest.format)


def _read_assets(root: Path, manifest: BrowserModelBundleManifestV1) -> dict[str, bytes]:
    try:
        expected = {
            "manifest.json",
            *(cast(WorkspaceRelativeLocatorV1, asset.locator).path for asset in manifest.assets),
        }
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        if actual != expected:
            raise ModelBundleError("bundle_inventory_mismatch")
        verify_artifact_references(manifest.assets, root)
        return {
            asset.role: root.joinpath(
                *cast(WorkspaceRelativeLocatorV1, asset.locator).path.split("/")
            ).read_bytes()
            for asset in manifest.assets
            if asset.media_type == "application/json"
        }
    except ModelBundleError:
        raise
    except (DatasetStorageError, OSError, RuntimeError, ValueError) as error:
        raise ModelBundleError("bundle_asset_bytes_invalid") from error


def _validate_components(manifest: BrowserModelBundleManifestV1, assets: dict[str, bytes]) -> None:
    try:
        landmarker = validate_mediapipe_extraction_config(assets["landmarker"])
        quality = validate_landmark_quality_policy(assets["quality_policy"])
        feature = validate_landmark_feature_plan(assets["feature_plan"])
        segmenter = _canonical_model(assets["segmenter"], CandidateEventConfigV1, "segmenter")
        policy = _canonical_model(assets["decision_policy"], DecisionPolicyV1, "decision_policy")
        golden = parse_json_object(assets["golden_smoke"])
        if assets["golden_smoke"] != canonical_json_bytes(golden) + b"\n":
            raise ModelBundleError("noncanonical_golden_smoke")
    except ModelBundleError:
        raise
    except (CanonicalizationError, KeyError, TypeError, ValidationError, ValueError) as error:
        raise ModelBundleError("invalid_bundle_component") from error

    expected = manifest.components
    component_matches = (
        mediapipe_extraction_config_digest(landmarker) == expected.landmarker_sha256,
        landmark_quality_policy_digest(quality) == expected.quality_policy_sha256,
        landmark_feature_plan_digest(feature) == expected.feature_plan_sha256,
        candidate_event_config_digest(segmenter) == expected.segmenter_sha256,
        decision_policy_digest(policy) == expected.decision_policy_sha256,
        segmenter.quality_policy_sha256 == expected.quality_policy_sha256,
        feature.representation == "hand_local",
        feature.padding.target_frame_count == manifest.onnx.input_shape[1],
        len(feature.feature_order) == manifest.onnx.input_shape[2],
        policy.identities.model_sha256 == manifest.candidate.research_checkpoint_sha256,
        policy.identities.configuration_sha256 == manifest.candidate.configuration_sha256,
        policy.identities.corpus_sha256 == manifest.candidate.corpus_sha256,
        policy.identities.derivative_set_sha256 == manifest.candidate.derivative_set_sha256,
        policy.identities.split_sha256 == manifest.candidate.split_sha256,
        policy.identities.input_feature_plan_sha256
        == manifest.candidate.input_feature_plan_sha256
        == expected.feature_plan_sha256,
        policy.identities.source_feature_plan_sha256
        == manifest.candidate.source_feature_plan_sha256,
        policy.identities.taxonomy_sha256 == manifest.candidate.taxonomy_sha256,
    )
    if not all(component_matches):
        raise ModelBundleError("bundle_component_identity_mismatch")


def validate_browser_bundle(bundle_root: str | Path) -> tuple[BrowserModelBundleManifestV1, str]:
    """Validate one complete directory and return only portable identity data."""

    try:
        root = Path(bundle_root).resolve(strict=True)
        manifest_path = root / "manifest.json"
        if not root.is_dir() or manifest_path.is_symlink():
            raise OSError
        manifest_raw = manifest_path.read_bytes()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ModelBundleError("bundle_unavailable") from error
    manifest = load_browser_bundle_manifest(manifest_raw)
    assets = _read_assets(root, manifest)
    _validate_components(manifest, assets)
    return manifest, browser_bundle_digest(manifest)


def browser_bundle_json_schema() -> dict[str, object]:
    schema = cast(dict[str, object], BrowserModelBundleManifestV1.model_json_schema())
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$comment"] = (
        "Structure-only interoperability schema. SignLab validation remains authoritative "
        "for canonical bytes, hashes, inventory, and cross-component identities."
    )
    return schema
