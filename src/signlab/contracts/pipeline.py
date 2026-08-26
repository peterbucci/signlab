"""Authoritative v1 identities and compatibility rules for the SignLab pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Any, Final, Literal, Self, cast

from pydantic import BaseModel, Field, StringConstraints, ValidationError, model_validator

from signlab.contracts.canonical import (
    CanonicalizationError,
    canonical_json_bytes,
    canonical_sha256,
    parse_json_object,
)
from signlab.contracts.core import (
    ArtifactRefV1,
    ComponentSpecV1,
    ContractKind,
    ContractRefV1,
    FiniteFloat,
    GitCommit,
    NonNegativeSafeInteger,
    ParameterV1,
    PortableLocatorV1,
    SafeInteger,
    SchemaName,
    SemanticVersion,
    StableId,
    StrictContractModel,
    UtcTimestamp,
    contract_config,
    same_artifact_reference,
    same_contract_reference,
)
from signlab.contracts.governance import DocumentRef, ParticipantId
from signlab.contracts.taxonomy import EXPECTED_CLASS_IDS, Sha256Digest, TaxonomyRef

SampleId = Annotated[str, StringConstraints(pattern=r"^sample_[0-9a-f]{32}$")]
SessionId = Annotated[str, StringConstraints(pattern=r"^session_[0-9a-f]{32}$")]
RecordingId = Annotated[str, StringConstraints(pattern=r"^recording_[0-9a-f]{32}$")]
PythonVersion = Annotated[str, StringConstraints(pattern=r"^3\.12\.[0-9]+$")]
LabelId = Literal["hello", "no", "please", "thank_you", "yes", "other"]
PartitionName = Literal["train", "validation", "test"]

_EXPECTED_PARTITIONS: Final = ("train", "validation", "test")


class PipelineContractError(ValueError):
    """Raised when a portable pipeline contract is invalid or incompatible."""


class ContractVersionError(PipelineContractError):
    """Raised before model validation when a reader cannot handle a schema version."""


class DatasetSampleIdentityV1(StrictContractModel):
    """Minimal membership and leakage-group identity for one immutable sample."""

    sample_id: SampleId
    participant_id: ParticipantId
    session_id: SessionId
    source_recording_id: RecordingId
    label_id: LabelId
    artifact: ArtifactRefV1

    @model_validator(mode="after")
    def _bind_artifact_to_sample(self) -> Self:
        if self.artifact.artifact_id != self.sample_id or self.artifact.role != "sample_data":
            raise ValueError("sample artifact identity and role must bind to the sample")
        return self


class DatasetContentV1(StrictContractModel):
    """Storage-independent semantic content used to calculate a stable data identity."""

    schema_version: Literal["dataset-content/1"]
    taxonomy: TaxonomyRef
    governance_policy: DocumentRef
    lineage_inventory_sha256: Sha256Digest
    sample_schema_version: SchemaName
    samples: tuple[DatasetSampleIdentityV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_canonical_unique_samples(self) -> Self:
        if self.governance_policy.document_type != "governance_policy":
            raise ValueError("dataset content must bind the registered governance policy")
        sample_ids = tuple(sample.sample_id for sample in self.samples)
        if sample_ids != tuple(sorted(set(sample_ids))):
            raise ValueError("dataset samples must have unique IDs in sorted order")
        artifact_digests = tuple(sample.artifact.sha256 for sample in self.samples)
        if len(artifact_digests) != len(set(artifact_digests)):
            raise ValueError("dataset samples must not duplicate artifact content")
        return self


def _dataset_semantic_payload(content: DatasetContentV1) -> dict[str, object]:
    return {
        "schema_version": content.schema_version,
        "taxonomy": content.taxonomy.model_dump(mode="json", round_trip=True),
        "governance_policy": content.governance_policy.model_dump(mode="json", round_trip=True),
        "lineage_inventory_sha256": content.lineage_inventory_sha256,
        "sample_schema_version": content.sample_schema_version,
        "samples": [
            {
                "sample_id": sample.sample_id,
                "participant_id": sample.participant_id,
                "session_id": sample.session_id,
                "source_recording_id": sample.source_recording_id,
                "label_id": sample.label_id,
                "artifact": {
                    "artifact_id": sample.artifact.artifact_id,
                    "role": sample.artifact.role,
                    "media_type": sample.artifact.media_type,
                    "sha256": sample.artifact.sha256,
                    "size_bytes": sample.artifact.size_bytes,
                },
            }
            for sample in content.samples
        ],
    }


def dataset_content_digest(content: DatasetContentV1) -> str:
    """Hash logical sample content independently of machine or storage location."""

    try:
        return canonical_sha256(
            _dataset_semantic_payload(content),
            domain=content.schema_version,
        )
    except CanonicalizationError as error:
        raise PipelineContractError("dataset content cannot be canonicalized") from error


class DatasetManifestV1(StrictContractModel):
    """Portable dataset envelope; detailed participant tables arrive in Story #15."""

    model_config = contract_config("dataset-manifest-1.schema.json")

    schema_version: Literal["dataset-manifest/1"]
    dataset_id: StableId
    version: SemanticVersion
    content: DatasetContentV1
    data_sha256: Sha256Digest

    @model_validator(mode="after")
    def _verify_data_identity(self) -> Self:
        if self.data_sha256 != dataset_content_digest(self.content):
            raise ValueError("data_sha256 does not match canonical storage-independent content")
        return self


class SplitPartitionV1(StrictContractModel):
    """Exact membership and grouping closure for one immutable partition."""

    name: PartitionName
    sample_ids: tuple[SampleId, ...] = Field(min_length=1)
    participant_ids: tuple[ParticipantId, ...] = Field(min_length=1)
    session_ids: tuple[SessionId, ...] = Field(min_length=1)
    source_recording_ids: tuple[RecordingId, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_sorted_unique_membership(self) -> Self:
        for field_name in (
            "sample_ids",
            "participant_ids",
            "session_ids",
            "source_recording_ids",
        ):
            values = getattr(self, field_name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field_name} must be unique and sorted")
        return self


class SplitManifestV1(StrictContractModel):
    """Leakage-resistant participant-and-session grouped split membership."""

    model_config = contract_config("split-manifest-1.schema.json")

    schema_version: Literal["split-manifest/1"]
    split_id: StableId
    version: SemanticVersion
    dataset: ContractRefV1
    dataset_data_sha256: Sha256Digest
    strategy: Literal["participant-and-session-grouped"]
    random_seed: SafeInteger
    partitions: tuple[SplitPartitionV1, SplitPartitionV1, SplitPartitionV1]

    @model_validator(mode="after")
    def _require_disjoint_canonical_partitions(self) -> Self:
        if self.dataset.kind != "dataset":
            raise ValueError("split manifest must reference a dataset contract")
        names = tuple(partition.name for partition in self.partitions)
        if names != _EXPECTED_PARTITIONS:
            raise ValueError("split partitions must be ordered train, validation, test")
        for field_name in (
            "sample_ids",
            "participant_ids",
            "session_ids",
            "source_recording_ids",
        ):
            flattened = [
                value for partition in self.partitions for value in getattr(partition, field_name)
            ]
            if len(flattened) != len(set(flattened)):
                raise ValueError(f"{field_name} must not cross split partitions")
        return self


class PreprocessingStepV1(StrictContractModel):
    """One ordered, versioned operation with explicit feature-schema compatibility."""

    index: NonNegativeSafeInteger
    operation_id: StableId
    implementation_version: SemanticVersion
    input_schema_version: SchemaName
    output_schema_version: SchemaName
    parameters: tuple[ParameterV1, ...]

    @model_validator(mode="after")
    def _require_canonical_parameters(self) -> Self:
        names = tuple(parameter.name for parameter in self.parameters)
        if names != tuple(sorted(set(names))):
            raise ValueError("preprocessing parameters must have unique names in sorted order")
        return self


class PreprocessingPlanV1(StrictContractModel):
    """Ordered preprocessing definition shared by training and inference consumers."""

    model_config = contract_config("preprocessing-plan-1.schema.json")

    schema_version: Literal["preprocessing-plan/1"]
    plan_id: StableId
    version: SemanticVersion
    taxonomy: TaxonomyRef
    input_schema_version: SchemaName
    output_schema_version: SchemaName
    compatible_runtimes: tuple[Literal["python", "typescript"], ...] = Field(min_length=1)
    steps: tuple[PreprocessingStepV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_connected_step_chain(self) -> Self:
        if self.compatible_runtimes != tuple(sorted(set(self.compatible_runtimes))):
            raise ValueError("compatible runtimes must be unique and sorted")
        if tuple(step.index for step in self.steps) != tuple(range(len(self.steps))):
            raise ValueError("preprocessing step indices must be contiguous from zero")
        if self.steps[0].input_schema_version != self.input_schema_version:
            raise ValueError("first preprocessing step must accept the plan input schema")
        if self.steps[-1].output_schema_version != self.output_schema_version:
            raise ValueError("last preprocessing step must produce the plan output schema")
        for left, right in zip(self.steps, self.steps[1:], strict=False):
            if left.output_schema_version != right.input_schema_version:
                raise ValueError("adjacent preprocessing step schemas are incompatible")
        return self


class ResolvedConfigurationV1(StrictContractModel):
    """Fully expanded experiment input with immutable upstream identities and defaults."""

    model_config = contract_config("resolved-configuration-1.schema.json")

    schema_version: Literal["resolved-configuration/1"]
    config_id: StableId
    version: SemanticVersion
    taxonomy: TaxonomyRef
    dataset: ContractRefV1
    dataset_data_sha256: Sha256Digest
    split: ContractRefV1
    preprocessing: ContractRefV1
    random_seed: SafeInteger
    deterministic_algorithms: Literal[True]
    model: ComponentSpecV1
    optimizer: ComponentSpecV1
    trainer: ComponentSpecV1
    evaluator: ComponentSpecV1

    @model_validator(mode="after")
    def _require_exact_component_roles(self) -> Self:
        roles = (
            self.model.role,
            self.optimizer.role,
            self.trainer.role,
            self.evaluator.role,
        )
        if roles != ("model", "optimizer", "trainer", "evaluator"):
            raise ValueError("resolved components do not match their pipeline roles")
        if (
            self.dataset.kind != "dataset"
            or self.split.kind != "split"
            or self.preprocessing.kind != "preprocessing"
        ):
            raise ValueError("resolved configuration contains an incompatible contract reference")
        return self


class SourceIdentityV1(StrictContractModel):
    """Sanitized, immutable code identity required for reproducible runs."""

    repository: Literal["https://github.com/peterbucci/signlab"]
    git_commit: GitCommit
    lockfile_sha256: Sha256Digest
    working_tree_clean: Literal[True]


class RuntimeIdentityV1(StrictContractModel):
    """Non-identifying runtime facts that affect reproducibility."""

    signlab_version: SemanticVersion
    python_version: PythonVersion
    os_family: Literal["linux", "windows", "macos"]
    accelerator: Literal["cpu", "cuda", "directml", "mps", "other"]
    deterministic_algorithms: Literal[True]


class MetricRecordV1(StrictContractModel):
    """One finite, registered aggregate without per-participant data."""

    name: StableId
    partition: PartitionName
    value: FiniteFloat
    unit: StableId


class FailureRecordV1(StrictContractModel):
    """Structured failure metadata that deliberately excludes messages and tracebacks."""

    code: StableId
    stage: StableId
    retryable: bool


class RunRecordV1(StrictContractModel):
    """Immutable terminal experiment record with complete upstream provenance."""

    model_config = contract_config("run-record-1.schema.json")

    schema_version: Literal["run-record/1"]
    run_id: StableId
    version: SemanticVersion
    status: Literal["succeeded", "failed", "cancelled"]
    started_at: UtcTimestamp
    finished_at: UtcTimestamp
    source: SourceIdentityV1
    runtime: RuntimeIdentityV1
    resolved_configuration: ContractRefV1
    dataset: ContractRefV1
    dataset_data_sha256: Sha256Digest
    split: ContractRefV1
    preprocessing: ContractRefV1
    metrics: tuple[MetricRecordV1, ...]
    outputs: tuple[ArtifactRefV1, ...]
    failure: FailureRecordV1 | None

    @model_validator(mode="after")
    def _require_terminal_consistent_record(self) -> Self:
        if _parse_utc(self.finished_at) < _parse_utc(self.started_at):
            raise ValueError("run finish time must not precede start time")
        if self.status == "succeeded":
            if self.failure is not None or not self.outputs:
                raise ValueError("successful runs require outputs and no failure record")
        elif self.failure is None:
            raise ValueError("failed or cancelled runs require a structured failure record")
        if (
            self.resolved_configuration.kind != "resolved_configuration"
            or self.dataset.kind != "dataset"
            or self.split.kind != "split"
            or self.preprocessing.kind != "preprocessing"
        ):
            raise ValueError("run record contains an incompatible contract reference")
        metric_keys = tuple((metric.partition, metric.name) for metric in self.metrics)
        if metric_keys != tuple(sorted(set(metric_keys))):
            raise ValueError("run metrics must have unique partition/name keys in sorted order")
        output_keys = tuple((output.role, output.artifact_id) for output in self.outputs)
        if output_keys != tuple(sorted(set(output_keys))):
            raise ValueError("run outputs must have unique role/ID keys in sorted order")
        return self


class ModelManifestV1(StrictContractModel):
    """Research-model identity; the browser bundle contract remains a later story."""

    model_config = contract_config("model-manifest-1.schema.json")

    schema_version: Literal["model-manifest/1"]
    model_id: StableId
    version: SemanticVersion
    model_format: StableId
    format_version: SemanticVersion
    taxonomy: TaxonomyRef
    label_order: tuple[LabelId, ...]
    input_schema_version: SchemaName
    output_schema_version: SchemaName
    training_run: ContractRefV1
    resolved_configuration: ContractRefV1
    dataset: ContractRefV1
    dataset_data_sha256: Sha256Digest
    split: ContractRefV1
    preprocessing: ContractRefV1
    artifact: ArtifactRefV1

    @model_validator(mode="after")
    def _require_model_identity_chain(self) -> Self:
        if self.label_order != EXPECTED_CLASS_IDS:
            raise ValueError("model label order must match the immutable taxonomy order")
        if (
            self.training_run.kind != "run"
            or self.resolved_configuration.kind != "resolved_configuration"
            or self.dataset.kind != "dataset"
            or self.split.kind != "split"
            or self.preprocessing.kind != "preprocessing"
        ):
            raise ValueError("model manifest contains an incompatible contract reference")
        if self.artifact.role != "model":
            raise ValueError("model manifest artifact must use the model role")
        return self


type CoreContract = (
    DatasetManifestV1
    | SplitManifestV1
    | PreprocessingPlanV1
    | ResolvedConfigurationV1
    | RunRecordV1
    | ModelManifestV1
)
type ContractInput = CoreContract | str | bytes | bytearray | Mapping[str, object]

CORE_CONTRACT_MODELS: Final[dict[str, type[BaseModel]]] = {
    "dataset-manifest/1": DatasetManifestV1,
    "split-manifest/1": SplitManifestV1,
    "preprocessing-plan/1": PreprocessingPlanV1,
    "resolved-configuration/1": ResolvedConfigurationV1,
    "run-record/1": RunRecordV1,
    "model-manifest/1": ModelManifestV1,
}
CORE_CONTRACT_SCHEMA_FILENAMES: Final[dict[str, str]] = {
    "dataset-manifest/1": "dataset-manifest-1.schema.json",
    "split-manifest/1": "split-manifest-1.schema.json",
    "preprocessing-plan/1": "preprocessing-plan-1.schema.json",
    "resolved-configuration/1": "resolved-configuration-1.schema.json",
    "run-record/1": "run-record-1.schema.json",
    "model-manifest/1": "model-manifest-1.schema.json",
}


def _parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def _validation_message(error: ValidationError) -> str:
    details: list[str] = []
    for item in error.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in item["loc"]) or "document"
        details.append(f"{location}: {item['msg']}")
    return "; ".join(details)


def _document_object(document: ContractInput) -> dict[str, Any]:
    if isinstance(document, BaseModel):
        document = cast(dict[str, object], document.model_dump(mode="json", round_trip=True))
    try:
        return parse_json_object(document)
    except CanonicalizationError as error:
        raise PipelineContractError(
            "contract input is not a valid interoperable JSON object"
        ) from error


def _unsupported_version_message(expected: str | None = None) -> str:
    supported = (
        expected
        if expected is not None and expected in CORE_CONTRACT_MODELS
        else ", ".join(sorted(CORE_CONTRACT_MODELS))
    )
    return (
        f"unsupported or missing contract schema version; supported: {supported}; "
        "writers must emit a supported version, and retained documents must follow "
        "docs/contracts.md#compatibility-and-migration"
    )


def validate_contract(
    document: ContractInput,
    *,
    expected_schema_version: str | None = None,
) -> CoreContract:
    """Dispatch by exact schema version, never silently coercing or migrating input."""

    payload = _document_object(document)
    schema_version = payload.get("schema_version")
    if (
        not isinstance(schema_version, str)
        or schema_version not in CORE_CONTRACT_MODELS
        or (expected_schema_version is not None and schema_version != expected_schema_version)
    ):
        raise ContractVersionError(_unsupported_version_message(expected_schema_version))
    model = CORE_CONTRACT_MODELS[schema_version]
    try:
        checked = model.model_validate_json(canonical_json_bytes(payload), strict=True)
    except CanonicalizationError as error:
        raise PipelineContractError("contract input cannot be canonicalized") from error
    except ValidationError as error:
        raise PipelineContractError(
            f"invalid {schema_version} contract: {_validation_message(error)}"
        ) from error
    return cast(CoreContract, checked)


def _validate_as(
    document: ContractInput,
    schema_version: str,
    model: type[CoreContract],
) -> CoreContract:
    checked = validate_contract(document, expected_schema_version=schema_version)
    if not isinstance(checked, model):
        raise PipelineContractError("contract reader returned an incompatible model")
    return checked


def validate_dataset_manifest(document: ContractInput) -> DatasetManifestV1:
    return cast(
        DatasetManifestV1,
        _validate_as(document, "dataset-manifest/1", DatasetManifestV1),
    )


def validate_split_manifest(document: ContractInput) -> SplitManifestV1:
    return cast(SplitManifestV1, _validate_as(document, "split-manifest/1", SplitManifestV1))


def validate_preprocessing_plan(document: ContractInput) -> PreprocessingPlanV1:
    return cast(
        PreprocessingPlanV1,
        _validate_as(document, "preprocessing-plan/1", PreprocessingPlanV1),
    )


def validate_resolved_configuration(document: ContractInput) -> ResolvedConfigurationV1:
    return cast(
        ResolvedConfigurationV1,
        _validate_as(document, "resolved-configuration/1", ResolvedConfigurationV1),
    )


def validate_run_record(document: ContractInput) -> RunRecordV1:
    return cast(RunRecordV1, _validate_as(document, "run-record/1", RunRecordV1))


def validate_model_manifest(document: ContractInput) -> ModelManifestV1:
    return cast(
        ModelManifestV1,
        _validate_as(document, "model-manifest/1", ModelManifestV1),
    )


def contract_digest(document: ContractInput) -> str:
    """Return a domain-separated RFC 8785 identity for one validated contract."""

    checked = validate_contract(document)
    try:
        return canonical_sha256(checked, domain=checked.schema_version)
    except CanonicalizationError as error:
        raise PipelineContractError("validated contract cannot be canonicalized") from error


def dataset_manifest_digest(document: ContractInput) -> str:
    return contract_digest(validate_dataset_manifest(document))


def split_manifest_digest(document: ContractInput) -> str:
    return contract_digest(validate_split_manifest(document))


def preprocessing_plan_digest(document: ContractInput) -> str:
    return contract_digest(validate_preprocessing_plan(document))


def resolved_configuration_digest(document: ContractInput) -> str:
    return contract_digest(validate_resolved_configuration(document))


def run_record_digest(document: ContractInput) -> str:
    return contract_digest(validate_run_record(document))


def model_manifest_digest(document: ContractInput) -> str:
    return contract_digest(validate_model_manifest(document))


def _contract_identity(document: CoreContract) -> tuple[ContractKind, StableId, SemanticVersion]:
    if isinstance(document, DatasetManifestV1):
        return "dataset", document.dataset_id, document.version
    if isinstance(document, SplitManifestV1):
        return "split", document.split_id, document.version
    if isinstance(document, PreprocessingPlanV1):
        return "preprocessing", document.plan_id, document.version
    if isinstance(document, ResolvedConfigurationV1):
        return "resolved_configuration", document.config_id, document.version
    if isinstance(document, RunRecordV1):
        return "run", document.run_id, document.version
    return "model", document.model_id, document.version


def contract_reference(document: ContractInput, locator: PortableLocatorV1) -> ContractRefV1:
    """Create a portable, content-bound reference to a validated core contract."""

    checked = validate_contract(document)
    kind, contract_id, version = _contract_identity(checked)
    return ContractRefV1(
        schema_version="contract-reference/1",
        kind=kind,
        contract_schema_version=checked.schema_version,
        contract_id=contract_id,
        contract_version=version,
        canonicalization="rfc8785/1",
        sha256=contract_digest(checked),
        locator=locator,
    )


def _assert_reference(
    reference: ContractRefV1,
    document: CoreContract,
    *,
    label: str,
) -> None:
    expected = contract_reference(document, reference.locator)
    if not same_contract_reference(reference, expected):
        raise PipelineContractError(f"{label} does not match the referenced contract identity")


def assert_split_compatible(dataset: DatasetManifestV1, split: SplitManifestV1) -> None:
    """Prove exact sample coverage and participant/session/recording isolation."""

    dataset = validate_dataset_manifest(dataset)
    split = validate_split_manifest(split)
    _assert_reference(split.dataset, dataset, label="split dataset reference")
    if split.dataset_data_sha256 != dataset.data_sha256:
        raise PipelineContractError("split data identity does not match the dataset content")

    samples = {sample.sample_id: sample for sample in dataset.content.samples}
    assigned = {sample_id for partition in split.partitions for sample_id in partition.sample_ids}
    if assigned != set(samples):
        raise PipelineContractError("split partitions must cover every dataset sample exactly once")

    for partition in split.partitions:
        members = tuple(samples[sample_id] for sample_id in partition.sample_ids)
        expected_groups = {
            "participant_ids": tuple(sorted({sample.participant_id for sample in members})),
            "session_ids": tuple(sorted({sample.session_id for sample in members})),
            "source_recording_ids": tuple(
                sorted({sample.source_recording_id for sample in members})
            ),
        }
        for field_name, expected in expected_groups.items():
            if getattr(partition, field_name) != expected:
                raise PipelineContractError(
                    f"split partition {field_name} does not close over its sample membership"
                )


def assert_resolved_configuration_compatible(
    dataset: DatasetManifestV1,
    split: SplitManifestV1,
    preprocessing: PreprocessingPlanV1,
    configuration: ResolvedConfigurationV1,
) -> None:
    """Bind one resolved configuration to exact compatible upstream contracts."""

    dataset = validate_dataset_manifest(dataset)
    split = validate_split_manifest(split)
    preprocessing = validate_preprocessing_plan(preprocessing)
    configuration = validate_resolved_configuration(configuration)
    assert_split_compatible(dataset, split)
    _assert_reference(configuration.dataset, dataset, label="configuration dataset reference")
    _assert_reference(configuration.split, split, label="configuration split reference")
    _assert_reference(
        configuration.preprocessing,
        preprocessing,
        label="configuration preprocessing reference",
    )
    if configuration.dataset_data_sha256 != dataset.data_sha256:
        raise PipelineContractError("configuration data identity does not match the dataset")
    if not (configuration.taxonomy == dataset.content.taxonomy == preprocessing.taxonomy):
        raise PipelineContractError("configuration taxonomy does not match its inputs")
    if preprocessing.input_schema_version != dataset.content.sample_schema_version:
        raise PipelineContractError("preprocessing input schema does not match dataset samples")


def assert_run_compatible(
    dataset: DatasetManifestV1,
    split: SplitManifestV1,
    preprocessing: PreprocessingPlanV1,
    configuration: ResolvedConfigurationV1,
    run: RunRecordV1,
) -> None:
    """Bind one terminal run to the exact configuration and upstream chain."""

    run = validate_run_record(run)
    assert_resolved_configuration_compatible(dataset, split, preprocessing, configuration)
    _assert_reference(
        run.resolved_configuration, configuration, label="run configuration reference"
    )
    _assert_reference(run.dataset, dataset, label="run dataset reference")
    _assert_reference(run.split, split, label="run split reference")
    _assert_reference(run.preprocessing, preprocessing, label="run preprocessing reference")
    if run.dataset_data_sha256 != dataset.data_sha256:
        raise PipelineContractError("run data identity does not match the dataset")


def assert_model_compatible(
    dataset: DatasetManifestV1,
    split: SplitManifestV1,
    preprocessing: PreprocessingPlanV1,
    configuration: ResolvedConfigurationV1,
    run: RunRecordV1,
    model: ModelManifestV1,
) -> None:
    """Prove an entire dataset-to-model provenance and compatibility chain."""

    model = validate_model_manifest(model)
    run = validate_run_record(run)
    assert_run_compatible(dataset, split, preprocessing, configuration, run)
    if run.status != "succeeded":
        raise PipelineContractError("a model manifest requires a successful training run")
    _assert_reference(model.training_run, run, label="model run reference")
    _assert_reference(
        model.resolved_configuration,
        configuration,
        label="model configuration reference",
    )
    _assert_reference(model.dataset, dataset, label="model dataset reference")
    _assert_reference(model.split, split, label="model split reference")
    _assert_reference(
        model.preprocessing,
        preprocessing,
        label="model preprocessing reference",
    )
    if model.dataset_data_sha256 != dataset.data_sha256:
        raise PipelineContractError("model data identity does not match the dataset")
    if model.taxonomy != dataset.content.taxonomy or model.taxonomy != preprocessing.taxonomy:
        raise PipelineContractError("model taxonomy does not match its provenance")
    if model.input_schema_version != preprocessing.output_schema_version:
        raise PipelineContractError("model input schema does not match preprocessing output")
    if not any(same_artifact_reference(model.artifact, output) for output in run.outputs):
        raise PipelineContractError("model artifact is not an output of its training run")


def canonical_contract_json(document: ContractInput) -> bytes:
    """Validate and render one contract as canonical RFC 8785 JSON bytes."""

    checked = validate_contract(document)
    try:
        return canonical_json_bytes(checked)
    except CanonicalizationError as error:
        raise PipelineContractError("validated contract cannot be canonicalized") from error


def contract_json_object(document: ContractInput) -> dict[str, Any]:
    """Return a validated JSON object for deterministic human-readable rendering."""

    checked = validate_contract(document)
    return cast(dict[str, Any], json.loads(checked.model_dump_json(round_trip=True)))


__all__ = [
    "CORE_CONTRACT_MODELS",
    "CORE_CONTRACT_SCHEMA_FILENAMES",
    "ContractVersionError",
    "DatasetContentV1",
    "DatasetManifestV1",
    "DatasetSampleIdentityV1",
    "FailureRecordV1",
    "MetricRecordV1",
    "ModelManifestV1",
    "PipelineContractError",
    "PreprocessingPlanV1",
    "PreprocessingStepV1",
    "ResolvedConfigurationV1",
    "RunRecordV1",
    "RuntimeIdentityV1",
    "SourceIdentityV1",
    "SplitManifestV1",
    "SplitPartitionV1",
    "assert_model_compatible",
    "assert_resolved_configuration_compatible",
    "assert_run_compatible",
    "assert_split_compatible",
    "canonical_contract_json",
    "contract_digest",
    "contract_json_object",
    "contract_reference",
    "dataset_content_digest",
    "dataset_manifest_digest",
    "model_manifest_digest",
    "preprocessing_plan_digest",
    "resolved_configuration_digest",
    "run_record_digest",
    "split_manifest_digest",
    "validate_contract",
    "validate_dataset_manifest",
    "validate_model_manifest",
    "validate_preprocessing_plan",
    "validate_resolved_configuration",
    "validate_run_record",
    "validate_split_manifest",
]
