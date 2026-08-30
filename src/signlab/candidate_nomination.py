"""Fail-closed nomination of one development checkpoint for portable export."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Annotated, Final, Literal, cast

from pydantic import Field, ValidationError, model_validator

from signlab.contracts.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    canonical_sha256,
    parse_json_object,
)
from signlab.contracts.core import (
    GitCommit,
    PositiveSafeInteger,
    SemanticVersion,
    StableId,
    StrictContractModel,
    WorkspacePath,
)
from signlab.contracts.taxonomy import Sha256Digest

_DOSSIER_DOMAIN: Final = "signlab-development-candidate-dossier/1"
_DOSSIER_SHA: Final = "sha256:7b6a18b91689fb2ba9765f30f6d6efd87e53e720070108a86436e67a5580ca5e"


class CandidateNominationError(ValueError):
    """A stable, path-free candidate-nomination failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class EvidenceFile(StrictContractModel):
    path: WorkspacePath
    sha256: Sha256Digest


class CandidateEvidence(StrictContractModel):
    provider_registry: EvidenceFile
    split_report: EvidenceFile
    ablation_report: EvidenceFile
    calibration_config: EvidenceFile
    calibration_report: EvidenceFile
    decision_policy: EvidenceFile
    continuous_replay_report: EvidenceFile
    dataset_card: EvidenceFile
    model_card: EvidenceFile


class CandidateIdentities(StrictContractModel):
    configuration_sha256: Sha256Digest
    corpus_sha256: Sha256Digest
    external_dataset_sha256: Sha256Digest
    derivative_set_sha256: Sha256Digest
    input_feature_plan_sha256: Sha256Digest
    source_feature_plan_sha256: Sha256Digest
    split_sha256: Sha256Digest
    taxonomy_sha256: Sha256Digest
    decision_policy_sha256: Sha256Digest
    research_model_sha256: Sha256Digest


class DevelopmentCandidateDossier(StrictContractModel):
    format: Literal["signlab-development-candidate-dossier/1"]
    candidate_id: StableId
    version: SemanticVersion
    nomination_scope: Literal["portable_export_only"]
    architecture: StableId
    representation: StableId
    input_frames: PositiveSafeInteger
    input_width: PositiveSafeInteger
    labels: Annotated[tuple[str, ...], Field(min_length=2)]
    parameter_count: PositiveSafeInteger
    checkpoint_size_bytes: PositiveSafeInteger
    source_run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    source_commit: GitCommit
    source_dirty: Literal[False]
    split_strategy: StableId
    test_status: Literal["sealed_not_loaded"]
    identities: CandidateIdentities
    evidence: CandidateEvidence

    @model_validator(mode="after")
    def _require_fixed_labels(self) -> DevelopmentCandidateDossier:
        if self.labels != ("hello", "no", "please", "thank_you", "yes", "other"):
            raise ValueError("candidate label order is unsupported")
        return self


def _sha256(raw: bytes) -> str:
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def load_candidate_dossier(path: str | Path) -> DevelopmentCandidateDossier:
    try:
        raw = Path(path).read_bytes()
        payload = parse_json_object(raw)
        dossier = DevelopmentCandidateDossier.model_validate_json(
            canonical_json_bytes(payload), strict=True
        )
        if raw != canonical_json_bytes(dossier) + b"\n":
            raise CandidateNominationError("noncanonical_dossier")
        return dossier
    except CandidateNominationError:
        raise
    except ValidationError as error:
        missing_hash = any(
            item["type"] == "missing" and str(item["loc"][-1]).endswith("sha256")
            for item in error.errors()
        )
        raise CandidateNominationError(
            "missing_hash" if missing_hash else "invalid_dossier"
        ) from error
    except (CanonicalizationError, OSError, TypeError, ValueError) as error:
        raise CandidateNominationError("invalid_dossier") from error


def _gate(name: str, status: Literal["pass", "fail"], reason: str) -> dict[str, str]:
    return {"gate": name, "reason": reason, "status": status}


def _read_evidence(
    repository_root: Path, name: str, reference: EvidenceFile
) -> tuple[bytes | None, dict[str, str]]:
    try:
        root = repository_root.resolve(strict=True)
        path = root.joinpath(*PurePosixPath(reference.path).parts).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise OSError
        raw = path.read_bytes()
    except FileNotFoundError:
        return None, _gate(f"{name}_current", "fail", f"missing_{name}")
    except OSError:
        return None, _gate(f"{name}_current", "fail", f"unreadable_{name}")
    if _sha256(raw) != reference.sha256:
        return None, _gate(f"{name}_current", "fail", f"stale_{name}")
    return raw, _gate(f"{name}_current", "pass", f"{name}_sha256_verified")


def _canonical_object(raw: bytes | None) -> dict[str, object] | None:
    if raw is None:
        return None
    try:
        payload = parse_json_object(raw)
        if raw != canonical_json_bytes(payload) + b"\n":
            return None
        return cast(dict[str, object], payload)
    except (CanonicalizationError, TypeError, ValueError):
        return None


def _checkpoint_gate(
    dossier: DevelopmentCandidateDossier, checkpoint_path: str | Path
) -> tuple[dict[str, str], str]:
    try:
        raw = Path(checkpoint_path).read_bytes()
    except OSError:
        gate = _gate("checkpoint_identity", "fail", "nomination_blocked_missing_artifact")
        return gate, "unavailable"
    if len(raw) != dossier.checkpoint_size_bytes or _sha256(raw) != (
        dossier.identities.research_model_sha256
    ):
        gate = _gate("checkpoint_identity", "fail", "checkpoint_identity_mismatch")
        return gate, "identity_failed"
    gate = _gate("checkpoint_identity", "pass", "checkpoint_sha256_and_size_verified")
    return gate, "local_hash_verified_not_published"


def _provenance_gate(
    dossier: DevelopmentCandidateDossier,
    calibration: dict[str, object] | None,
    policy: dict[str, object] | None,
) -> dict[str, str]:
    if calibration is None or policy is None:
        return _gate("provenance", "fail", "invalid_provenance_evidence")
    identities = policy.get("identities")
    expected = dossier.identities
    matches = isinstance(identities, dict) and all(
        (
            canonical_sha256(dossier, domain=_DOSSIER_DOMAIN) == _DOSSIER_SHA,
            calibration.get("model") == dossier.architecture,
            calibration.get("input_feature_plan_id") == "hand_local_64_frames",
            calibration.get("input_frames") == dossier.input_frames,
            calibration.get("input_width") == dossier.input_width,
            calibration.get("labels") == list(dossier.labels),
            calibration.get("corpus_sha256") == expected.corpus_sha256,
            calibration.get("external_dataset_sha256") == expected.external_dataset_sha256,
            calibration.get("split_sha256") == expected.split_sha256,
            calibration.get("input_feature_plan_sha256") == expected.input_feature_plan_sha256,
            calibration.get("feature_plan_sha256") == expected.source_feature_plan_sha256,
            calibration.get("taxonomy_sha256") == expected.taxonomy_sha256,
            identities.get("configuration_sha256") == expected.configuration_sha256,
            identities.get("derivative_set_sha256") == expected.derivative_set_sha256,
            identities.get("model_sha256") == expected.research_model_sha256,
            identities.get("split_sha256") == expected.split_sha256,
            identities.get("input_feature_plan_sha256") == expected.input_feature_plan_sha256,
            identities.get("source_feature_plan_sha256") == expected.source_feature_plan_sha256,
            identities.get("taxonomy_sha256") == expected.taxonomy_sha256,
            dossier.evidence.decision_policy.sha256 == expected.decision_policy_sha256,
        )
    )
    return _gate(
        "provenance",
        "pass" if matches else "fail",
        "research_lineage_consistent" if matches else "provenance_identity_mismatch",
    )


def _boundary_gates(
    dossier: DevelopmentCandidateDossier,
    policy: dict[str, object] | None,
    replay: dict[str, object] | None,
) -> tuple[dict[str, str], dict[str, str]]:
    sealed = (
        policy is not None
        and replay is not None
        and policy.get("test_status") == "sealed_not_loaded"
        and replay.get("test_status") == "sealed_not_loaded"
        and dossier.test_status == "sealed_not_loaded"
    )
    boundaries = (
        policy is not None
        and replay is not None
        and policy.get("evidence_kind") == "constructed_transition_calibration_conformance"
        and policy.get("metric_claim") == "development_mechanics_only"
        and replay.get("evidence_kind") == "constructed_continuous_replay_scoring_conformance"
        and replay.get("metric_claim") == "none"
        and replay.get("natural_session_status") == "unavailable"
        and replay.get("live_path_status") == "unavailable_pending_browser_runtime"
        and replay.get("model_bundle_status") == "unavailable_pending_contract"
    )
    return (
        _gate(
            "sealed_test_boundary",
            "pass" if sealed else "fail",
            "test_partition_sealed" if sealed else "test_partition_not_sealed",
        ),
        _gate(
            "constructed_claim_boundaries",
            "pass" if boundaries else "fail",
            "constructed_evidence_limited" if boundaries else "claim_boundary_mismatch",
        ),
    )


_RELEASE_GATES: Final = (
    ("unseen_signer_release_performance", "locked_test_workflow_pending"),
    ("per_class_release_floors", "locked_test_workflow_pending"),
    ("natural_other_calibration", "natural_other_evidence_pending_issue_18"),
    ("natural_continuous_false_activations", "natural_session_evidence_pending_issue_18"),
    ("end_to_end_runtime_latency", "runtime_budget_pending_issue_51"),
    ("portable_bundle_size", "candidate_bundle_pending_issue_35"),
    ("native_onnx_parity", "native_onnx_parity_pending_issue_37"),
    ("typescript_web_parity", "cross_runtime_parity_pending_issue_39"),
)


def build_candidate_nomination_report(
    dossier: DevelopmentCandidateDossier,
    *,
    repository_root: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, object]:
    evidence_bytes: dict[str, bytes | None] = {}
    development_gates: list[dict[str, str]] = []
    for name in CandidateEvidence.model_fields:
        raw, gate = _read_evidence(Path(repository_root), name, getattr(dossier.evidence, name))
        evidence_bytes[name] = raw
        development_gates.append(gate)

    architecture_ok = dossier.architecture == "tcn"
    representation_ok = dossier.representation == "hand_local"
    split_ok = dossier.split_strategy == "official_signer_disjoint_source_split_v1"
    development_gates[:0] = [
        _gate(
            "supported_architecture",
            "pass" if architecture_ok else "fail",
            "tcn_supported" if architecture_ok else "unsupported_architecture",
        ),
        _gate(
            "supported_representation",
            "pass" if representation_ok else "fail",
            "hand_local_supported" if representation_ok else "unsupported_representation",
        ),
        _gate(
            "split_policy",
            "pass" if split_ok else "fail",
            "signer_disjoint_split_declared" if split_ok else "invalid_split",
        ),
    ]
    checkpoint_gate, artifact_status = _checkpoint_gate(dossier, checkpoint_path)
    development_gates.append(checkpoint_gate)
    calibration = _canonical_object(evidence_bytes["calibration_config"])
    policy = _canonical_object(evidence_bytes["decision_policy"])
    replay = _canonical_object(evidence_bytes["continuous_replay_report"])
    development_gates.append(_provenance_gate(dossier, calibration, policy))
    development_gates.extend(_boundary_gates(dossier, policy, replay))
    nominated = all(gate["status"] == "pass" for gate in development_gates)

    return {
        "format": "signlab-development-candidate-nomination-report/1",
        "nomination_scope": dossier.nomination_scope,
        "candidate_status": (
            "nominated_for_portable_export" if nominated else "nomination_blocked"
        ),
        "champion_status": "none_blocked",
        "metric_claim": "development_only",
        "test_status": dossier.test_status,
        "model_artifact_status": artifact_status,
        "candidate": {
            "candidate_id": dossier.candidate_id,
            "version": dossier.version,
            "architecture": dossier.architecture,
            "representation": dossier.representation,
            "input_shape": [dossier.input_frames, dossier.input_width],
            "labels": list(dossier.labels),
            "parameter_count": dossier.parameter_count,
            "checkpoint_size_bytes": dossier.checkpoint_size_bytes,
            "source_run_id": dossier.source_run_id,
            "source_commit": dossier.source_commit,
        },
        "identities": {
            "dossier_sha256": canonical_sha256(dossier, domain=_DOSSIER_DOMAIN),
            **dossier.identities.model_dump(mode="json"),
            "dataset_card_sha256": dossier.evidence.dataset_card.sha256,
            "model_card_sha256": dossier.evidence.model_card.sha256,
        },
        "development_gates": development_gates,
        "release_gates": [
            {"gate": name, "reason": reason, "status": "blocked"} for name, reason in _RELEASE_GATES
        ],
        "limits": [
            "Grouped #30 metrics support the design choice, not this exact "
            "checkpoint's release performance.",
            "Calibration and other behavior use constructed #31 fragments, not "
            "natural OOV evidence.",
            "Continuous counts and latency in #33 prove scorer mechanics, not candidate behavior.",
            "Nomination permits portable export and evaluation only; no champion exists.",
        ],
    }


def run_candidate_nomination(
    dossier_path: str | Path,
    repository_root: str | Path,
    checkpoint_path: str | Path,
    report_path: str | Path,
) -> dict[str, object]:
    dossier = load_candidate_dossier(dossier_path)
    report = build_candidate_nomination_report(
        dossier,
        repository_root=repository_root,
        checkpoint_path=checkpoint_path,
    )
    try:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(canonical_json_bytes(report) + b"\n")
    except (CanonicalizationError, OSError, TypeError, ValueError) as error:
        raise CandidateNominationError("report_exists_or_unwritable") from error
    return report
