"""Deterministic schemas and synthetic examples for the core pipeline contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from importlib.resources import files
from importlib.resources.abc import Traversable
from typing import Final, Literal, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import BaseModel

from signlab.contracts.core import (
    ArtifactRefV1,
    ArtifactUriLocatorV1,
    ComponentSpecV1,
    ContractKind,
    ParameterV1,
    WorkspaceRelativeLocatorV1,
)
from signlab.contracts.pipeline import (
    CORE_CONTRACT_MODELS,
    CORE_CONTRACT_SCHEMA_FILENAMES,
    DatasetContentV1,
    DatasetManifestV1,
    DatasetSampleIdentityV1,
    MetricRecordV1,
    ModelManifestV1,
    PipelineContractError,
    PreprocessingPlanV1,
    PreprocessingStepV1,
    ResolvedConfigurationV1,
    RunRecordV1,
    RuntimeIdentityV1,
    SourceIdentityV1,
    SplitManifestV1,
    SplitPartitionV1,
    assert_model_compatible,
    contract_digest,
    contract_reference,
    dataset_content_digest,
    validate_contract,
)
from signlab.contracts.taxonomy import EXPECTED_CLASS_IDS, load_builtin_taxonomy, taxonomy_reference
from signlab.governance.resources import build_example_inventory, build_governance_policy

RESOURCE_PACKAGE: Final = "signlab.resources.contracts"
_SCHEMA_BASE: Final = "https://signlab.dev/schemas/"
_SCHEMA_BOUNDARY_COMMENT: Final = (
    "This JSON Schema enforces portable structure and locally expressible invariants. "
    "SignLab Pydantic and application compatibility validation remains authoritative "
    "for canonical digests, dataset membership and leakage closure, preprocessing schema "
    "chains, cross-contract reference reconciliation, run provenance, and model-output "
    "binding."
)

CONTRACT_KINDS: Final[tuple[ContractKind, ...]] = (
    "dataset",
    "split",
    "preprocessing",
    "resolved_configuration",
    "run",
    "model",
)
CONTRACT_EXAMPLE_FILENAMES: Final[dict[ContractKind, str]] = {
    "dataset": "dataset-manifest.example.json",
    "split": "split-manifest.example.json",
    "preprocessing": "preprocessing-plan.example.json",
    "resolved_configuration": "resolved-configuration.example.json",
    "run": "run-record.example.json",
    "model": "model-manifest.example.json",
}
# These published examples are the immutable Story #13 v1 compatibility corpus.
# They deliberately do not follow CURRENT_CONTRACT_SCHEMAS when a writer advances.
PUBLISHED_EXAMPLE_SCHEMA_VERSIONS: Final[dict[ContractKind, str]] = {
    "dataset": "dataset-manifest/1",
    "split": "split-manifest/1",
    "preprocessing": "preprocessing-plan/1",
    "resolved_configuration": "resolved-configuration/1",
    "run": "run-record/1",
    "model": "model-manifest/1",
}
CONTRACT_SCHEMA_MODELS: Final[dict[str, type[BaseModel]]] = {
    CORE_CONTRACT_SCHEMA_FILENAMES[schema_version]: model
    for schema_version, model in CORE_CONTRACT_MODELS.items()
}
GENERATED_RESOURCE_NAMES: Final = {
    *(f"schemas/{filename}" for filename in CONTRACT_SCHEMA_MODELS),
    *(f"examples/{filename}" for filename in CONTRACT_EXAMPLE_FILENAMES.values()),
}

# These are immutable semantic identities of the six reviewed public examples. They
# intentionally differ from byte hashes of the pretty-printed resource files.
PUBLISHED_EXAMPLE_CONTRACT_DIGESTS: Final[dict[ContractKind, str]] = {
    "dataset": "sha256:ed55707743ddf6ca144124ee671994ceb8eb7cea06c8f7d97ec10203d5dd8717",
    "split": "sha256:6fea000166d7eddc699f2bf03fbf85b2b88d945a68ff98e236345279255b84a5",
    "preprocessing": ("sha256:5fe847d97c77686900ccc8442d081c2ef428c5609ccf27fd70daf0360e559cbe"),
    "resolved_configuration": (
        "sha256:c7691a990279743d53189a8e7fa9685ad64aba0b04a174cc9b8b59f41fdf3e9b"
    ),
    "run": "sha256:2aa346eddc0ef19f3bd19dbc7bc0017858641b04fad525706b60fd4e17678aa6",
    "model": "sha256:aec88f012bbe936d0ae6c53d73c3cd19c46e94b7545207929075fd7b89887525",
}

type ContractExample = (
    DatasetManifestV1
    | SplitManifestV1
    | PreprocessingPlanV1
    | ResolvedConfigurationV1
    | RunRecordV1
    | ModelManifestV1
)
type ContractExampleChain = tuple[
    DatasetManifestV1,
    SplitManifestV1,
    PreprocessingPlanV1,
    ResolvedConfigurationV1,
    RunRecordV1,
    ModelManifestV1,
]
type LabelId = Literal["hello", "no", "please", "thank_you", "yes", "other"]
type PartitionName = Literal["train", "validation", "test"]
type ComponentRole = Literal["model", "optimizer", "trainer", "evaluator"]


class ContractResourceError(PipelineContractError):
    """Raised when packaged core-contract resources are incomplete or stale."""


def _synthetic_sha256(label: str) -> str:
    payload = f"SignLab synthetic pipeline fixture: {label}\n".encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _artifact_locator(category: str, artifact_id: str) -> ArtifactUriLocatorV1:
    return ArtifactUriLocatorV1(
        kind="artifact_uri",
        uri=f"signlab://synthetic/{category}/{artifact_id}",
    )


def _contract_locator(kind: ContractKind) -> WorkspaceRelativeLocatorV1:
    return WorkspaceRelativeLocatorV1(
        kind="workspace_relative",
        path=f"src/signlab/resources/contracts/examples/{CONTRACT_EXAMPLE_FILENAMES[kind]}",
    )


def _build_dataset() -> DatasetManifestV1:
    taxonomy = taxonomy_reference(load_builtin_taxonomy())
    policy = build_governance_policy()
    inventory = build_example_inventory()
    labels: tuple[LabelId, ...] = (
        "hello",
        "no",
        "please",
        "thank_you",
        "yes",
        "other",
        "other",
        "hello",
    )
    samples: list[DatasetSampleIdentityV1] = []
    for index, label in enumerate(labels, start=1):
        sample_id = f"sample_{index:032x}"
        samples.append(
            DatasetSampleIdentityV1(
                sample_id=sample_id,
                participant_id=f"participant_{index:032x}",
                session_id=f"session_{index:032x}",
                source_recording_id=f"recording_{index:032x}",
                label_id=label,
                artifact=ArtifactRefV1(
                    schema_version="artifact-reference/1",
                    artifact_id=sample_id,
                    role="sample_data",
                    media_type="application/vnd.signlab.landmarks+json",
                    sha256=_synthetic_sha256(f"landmark sample {index}"),
                    size_bytes=4096 + index,
                    locator=_artifact_locator("samples", sample_id),
                ),
            )
        )
    content = DatasetContentV1(
        schema_version="dataset-content/1",
        taxonomy=taxonomy,
        governance_policy=policy.policy_document,
        lineage_inventory_sha256=inventory.inventory_sha256,
        sample_schema_version="landmark-sequence/1",
        samples=tuple(samples),
    )
    return DatasetManifestV1(
        schema_version="dataset-manifest/1",
        dataset_id="synthetic_gesture_dataset",
        version="1.0.0",
        content=content,
        data_sha256=dataset_content_digest(content),
    )


def _partition(
    name: PartitionName,
    samples: tuple[DatasetSampleIdentityV1, ...],
) -> SplitPartitionV1:
    return SplitPartitionV1(
        name=name,
        sample_ids=tuple(sample.sample_id for sample in samples),
        participant_ids=tuple(sample.participant_id for sample in samples),
        session_ids=tuple(sample.session_id for sample in samples),
        source_recording_ids=tuple(sample.source_recording_id for sample in samples),
    )


def _build_split(dataset: DatasetManifestV1) -> SplitManifestV1:
    samples = dataset.content.samples
    return SplitManifestV1(
        schema_version="split-manifest/1",
        split_id="synthetic_grouped_split",
        version="1.0.0",
        dataset=contract_reference(dataset, _contract_locator("dataset")),
        dataset_data_sha256=dataset.data_sha256,
        strategy="participant-and-session-grouped",
        random_seed=20260826,
        partitions=(
            _partition("train", samples[:6]),
            _partition("validation", samples[6:7]),
            _partition("test", samples[7:]),
        ),
    )


def _build_preprocessing() -> PreprocessingPlanV1:
    return PreprocessingPlanV1(
        schema_version="preprocessing-plan/1",
        plan_id="synthetic_landmark_preprocessing",
        version="1.0.0",
        taxonomy=taxonomy_reference(load_builtin_taxonomy()),
        input_schema_version="landmark-sequence/1",
        output_schema_version="feature-window/1",
        compatible_runtimes=("python",),
        steps=(
            PreprocessingStepV1(
                index=0,
                operation_id="normalize_coordinates",
                implementation_version="1.0.0",
                input_schema_version="landmark-sequence/1",
                output_schema_version="normalized-landmarks/1",
                parameters=(
                    ParameterV1(name="center_landmark", value="wrist"),
                    ParameterV1(name="scale_landmark", value="middle_mcp"),
                ),
            ),
            PreprocessingStepV1(
                index=1,
                operation_id="resample_sequence",
                implementation_version="1.0.0",
                input_schema_version="normalized-landmarks/1",
                output_schema_version="feature-window/1",
                parameters=(
                    ParameterV1(name="padding_mode", value="repeat_edge"),
                    ParameterV1(name="sequence_length", value=30),
                ),
            ),
        ),
    )


def _component(
    role: ComponentRole,
    implementation_id: str,
    parameters: tuple[ParameterV1, ...],
) -> ComponentSpecV1:
    return ComponentSpecV1(
        schema_version="component-spec/1",
        role=role,
        implementation_id=implementation_id,
        implementation_version="1.0.0",
        parameters=parameters,
    )


def _build_configuration(
    dataset: DatasetManifestV1,
    split: SplitManifestV1,
    preprocessing: PreprocessingPlanV1,
) -> ResolvedConfigurationV1:
    return ResolvedConfigurationV1(
        schema_version="resolved-configuration/1",
        config_id="synthetic_resolved_experiment",
        version="1.0.0",
        taxonomy=dataset.content.taxonomy,
        dataset=contract_reference(dataset, _contract_locator("dataset")),
        dataset_data_sha256=dataset.data_sha256,
        split=contract_reference(split, _contract_locator("split")),
        preprocessing=contract_reference(
            preprocessing,
            _contract_locator("preprocessing"),
        ),
        random_seed=20260826,
        deterministic_algorithms=True,
        model=_component(
            "model",
            "gru_classifier",
            (
                ParameterV1(name="dropout", value=0.2),
                ParameterV1(name="hidden_size", value=64),
                ParameterV1(name="layers", value=2),
            ),
        ),
        optimizer=_component(
            "optimizer",
            "adamw",
            (
                ParameterV1(name="learning_rate", value=0.001),
                ParameterV1(name="weight_decay", value=0.0001),
            ),
        ),
        trainer=_component(
            "trainer",
            "sequence_trainer",
            (
                ParameterV1(name="batch_size", value=32),
                ParameterV1(name="epochs", value=25),
            ),
        ),
        evaluator=_component(
            "evaluator",
            "classification_metrics",
            (
                ParameterV1(name="abstention_enabled", value=True),
                ParameterV1(name="primary_metric", value="macro_f1"),
            ),
        ),
    )


def _model_artifact() -> ArtifactRefV1:
    return ArtifactRefV1(
        schema_version="artifact-reference/1",
        artifact_id="synthetic_model_onnx",
        role="model",
        media_type="application/onnx",
        sha256=_synthetic_sha256("trained ONNX model"),
        size_bytes=65536,
        locator=_artifact_locator("models", "synthetic_model_onnx"),
    )


def _build_run(
    dataset: DatasetManifestV1,
    split: SplitManifestV1,
    preprocessing: PreprocessingPlanV1,
    configuration: ResolvedConfigurationV1,
) -> RunRecordV1:
    return RunRecordV1(
        schema_version="run-record/1",
        run_id="synthetic_training_run",
        version="1.0.0",
        status="succeeded",
        started_at="2026-08-26T15:00:00Z",
        finished_at="2026-08-26T15:05:00Z",
        source=SourceIdentityV1(
            repository="https://github.com/peterbucci/signlab",
            git_commit="0000000000000000000000000000000000000001",
            lockfile_sha256=_synthetic_sha256("uv lockfile"),
            working_tree_clean=True,
        ),
        runtime=RuntimeIdentityV1(
            signlab_version="0.1.0",
            python_version="3.12.14",
            os_family="linux",
            accelerator="cpu",
            deterministic_algorithms=True,
        ),
        resolved_configuration=contract_reference(
            configuration,
            _contract_locator("resolved_configuration"),
        ),
        dataset=contract_reference(dataset, _contract_locator("dataset")),
        dataset_data_sha256=dataset.data_sha256,
        split=contract_reference(split, _contract_locator("split")),
        preprocessing=contract_reference(
            preprocessing,
            _contract_locator("preprocessing"),
        ),
        metrics=(
            MetricRecordV1(name="accuracy", partition="test", value=0.75, unit="ratio"),
            MetricRecordV1(name="macro_f1", partition="test", value=0.70, unit="ratio"),
            MetricRecordV1(
                name="accuracy",
                partition="validation",
                value=0.80,
                unit="ratio",
            ),
            MetricRecordV1(
                name="macro_f1",
                partition="validation",
                value=0.76,
                unit="ratio",
            ),
        ),
        outputs=(_model_artifact(),),
        failure=None,
    )


def _build_model(
    dataset: DatasetManifestV1,
    split: SplitManifestV1,
    preprocessing: PreprocessingPlanV1,
    configuration: ResolvedConfigurationV1,
    run: RunRecordV1,
) -> ModelManifestV1:
    return ModelManifestV1(
        schema_version="model-manifest/1",
        model_id="synthetic_gesture_model",
        version="1.0.0",
        model_format="onnx",
        format_version="1.16.0",
        taxonomy=dataset.content.taxonomy,
        label_order=EXPECTED_CLASS_IDS,
        input_schema_version=preprocessing.output_schema_version,
        output_schema_version="gesture-scores/1",
        training_run=contract_reference(run, _contract_locator("run")),
        resolved_configuration=contract_reference(
            configuration,
            _contract_locator("resolved_configuration"),
        ),
        dataset=contract_reference(dataset, _contract_locator("dataset")),
        dataset_data_sha256=dataset.data_sha256,
        split=contract_reference(split, _contract_locator("split")),
        preprocessing=contract_reference(
            preprocessing,
            _contract_locator("preprocessing"),
        ),
        artifact=run.outputs[0],
    )


def build_example_contract_chain() -> ContractExampleChain:
    """Build and prove the six synthetic contracts in dependency order."""

    dataset = _build_dataset()
    split = _build_split(dataset)
    preprocessing = _build_preprocessing()
    configuration = _build_configuration(dataset, split, preprocessing)
    run = _build_run(dataset, split, preprocessing, configuration)
    model = _build_model(dataset, split, preprocessing, configuration, run)
    assert_model_compatible(dataset, split, preprocessing, configuration, run, model)
    return dataset, split, preprocessing, configuration, run, model


def _examples_by_kind(chain: ContractExampleChain) -> dict[ContractKind, ContractExample]:
    dataset, split, preprocessing, configuration, run, model = chain
    return {
        "dataset": dataset,
        "split": split,
        "preprocessing": preprocessing,
        "resolved_configuration": configuration,
        "run": run,
        "model": model,
    }


def _schema_nodes(value: object) -> Iterator[dict[str, object]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _schema_nodes(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _schema_nodes(nested)


def _strip_nested_schema_ids(value: object, *, root: bool = False) -> None:
    if isinstance(value, dict):
        if not root:
            value.pop("$id", None)
        for nested in value.values():
            _strip_nested_schema_ids(nested)
    elif isinstance(value, list):
        for nested in value:
            _strip_nested_schema_ids(nested)


_UNIQUE_ARRAY_FIELDS: Final = {
    "compatible_runtimes",
    "label_order",
    "metrics",
    "outputs",
    "parameters",
    "participant_ids",
    "sample_ids",
    "samples",
    "session_ids",
    "source_recording_ids",
    "steps",
}


def _harden_unique_arrays(schema: dict[str, object]) -> None:
    for node in _schema_nodes(schema):
        properties = node.get("properties")
        if not isinstance(properties, dict):
            continue
        for field_name, field_schema in properties.items():
            if (
                field_name in _UNIQUE_ARRAY_FIELDS
                and isinstance(field_schema, dict)
                and field_schema.get("type") == "array"
            ):
                field_schema["uniqueItems"] = True


def _properties(schema: dict[str, object]) -> dict[str, object]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise ContractResourceError("generated contract schema has no object properties")
    return properties


def _append_nested_const(
    schema: dict[str, object],
    field_name: str,
    nested_name: str,
    value: object,
) -> None:
    all_of = schema.setdefault("allOf", [])
    if not isinstance(all_of, list):
        raise ContractResourceError("generated contract schema has an invalid allOf")
    all_of.append(
        {
            "properties": {
                field_name: {
                    "properties": {nested_name: {"const": value}},
                }
            }
        }
    )


def _harden_generated_schema(schema_version: str, schema: dict[str, object]) -> None:
    _harden_unique_arrays(schema)
    if schema_version == "dataset-manifest/1":
        definitions = schema.get("$defs")
        if not isinstance(definitions, dict):
            raise ContractResourceError("dataset schema is missing definitions")
        sample = definitions.get("DatasetSampleIdentityV1")
        if not isinstance(sample, dict):
            raise ContractResourceError("dataset schema is missing sample identity")
        _append_nested_const(sample, "artifact", "role", "sample_data")
    elif schema_version == "split-manifest/1":
        partitions = _properties(schema).get("partitions")
        if not isinstance(partitions, dict):
            raise ContractResourceError("split schema is missing partitions")
        prefix_items = partitions.get("prefixItems")
        if not isinstance(prefix_items, list) or len(prefix_items) != 3:
            raise ContractResourceError("split schema has an invalid partition tuple")
        for index, name in enumerate(("train", "validation", "test")):
            prefix_items[index] = {
                "allOf": [
                    prefix_items[index],
                    {"properties": {"name": {"const": name}}, "required": ["name"]},
                ]
            }
        _append_nested_const(schema, "dataset", "kind", "dataset")
    elif schema_version == "preprocessing-plan/1":
        runtimes = _properties(schema).get("compatible_runtimes")
        if not isinstance(runtimes, dict):
            raise ContractResourceError("preprocessing schema is missing runtimes")
        runtimes["oneOf"] = [
            {"const": ["python"]},
            {"const": ["python", "typescript"]},
            {"const": ["typescript"]},
        ]
    elif schema_version == "resolved-configuration/1":
        for field_name, role in (
            ("model", "model"),
            ("optimizer", "optimizer"),
            ("trainer", "trainer"),
            ("evaluator", "evaluator"),
        ):
            _append_nested_const(schema, field_name, "role", role)
        for field_name, kind in (
            ("dataset", "dataset"),
            ("split", "split"),
            ("preprocessing", "preprocessing"),
        ):
            _append_nested_const(schema, field_name, "kind", kind)
    elif schema_version == "run-record/1":
        for field_name, kind in (
            ("resolved_configuration", "resolved_configuration"),
            ("dataset", "dataset"),
            ("split", "split"),
            ("preprocessing", "preprocessing"),
        ):
            _append_nested_const(schema, field_name, "kind", kind)
        all_of = schema.setdefault("allOf", [])
        if not isinstance(all_of, list):
            raise ContractResourceError("run schema has an invalid allOf")
        all_of.append(
            {
                "if": {
                    "properties": {"status": {"const": "succeeded"}},
                    "required": ["status"],
                },
                "then": {
                    "properties": {
                        "failure": {"type": "null"},
                        "outputs": {"minItems": 1},
                    }
                },
                "else": {"properties": {"failure": {"not": {"type": "null"}}}},
            }
        )
    elif schema_version == "model-manifest/1":
        label_order = _properties(schema).get("label_order")
        if not isinstance(label_order, dict):
            raise ContractResourceError("model schema is missing label order")
        label_order["const"] = list(EXPECTED_CLASS_IDS)
        _append_nested_const(schema, "artifact", "role", "model")
        for field_name, kind in (
            ("training_run", "run"),
            ("resolved_configuration", "resolved_configuration"),
            ("dataset", "dataset"),
            ("split", "split"),
            ("preprocessing", "preprocessing"),
        ):
            _append_nested_const(schema, field_name, "kind", kind)


def generated_contract_schemas() -> dict[str, dict[str, object]]:
    """Generate the six standalone Draft 2020-12 pipeline schemas."""

    generated: dict[str, dict[str, object]] = {}
    for schema_version, filename in CORE_CONTRACT_SCHEMA_FILENAMES.items():
        model = CORE_CONTRACT_MODELS[schema_version]
        schema = model.model_json_schema(mode="validation")
        _strip_nested_schema_ids(schema, root=True)
        schema["$id"] = f"{_SCHEMA_BASE}{filename}"
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$comment"] = _SCHEMA_BOUNDARY_COMMENT
        _harden_generated_schema(schema_version, schema)
        Draft202012Validator.check_schema(schema)
        generated[filename] = schema
    if set(generated) != set(CONTRACT_SCHEMA_MODELS):
        raise ContractResourceError("generated contract schema registry is incomplete")
    return generated


def render_json_document(value: BaseModel | Mapping[str, object]) -> str:
    """Render stable, reviewable UTF-8 JSON with one trailing newline."""

    payload: object
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json", round_trip=True)
    else:
        payload = dict(value)
    try:
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
    except (TypeError, ValueError) as error:
        raise ContractResourceError("contract resource is not JSON-serializable") from error


def generated_contract_resource_texts() -> dict[str, str]:
    """Render the exact schema and synthetic-example package inventory."""

    examples = _examples_by_kind(build_example_contract_chain())
    rendered = {
        f"examples/{CONTRACT_EXAMPLE_FILENAMES[kind]}": render_json_document(examples[kind])
        for kind in CONTRACT_KINDS
    }
    rendered.update(
        {
            f"schemas/{filename}": render_json_document(schema)
            for filename, schema in generated_contract_schemas().items()
        }
    )
    if set(rendered) != GENERATED_RESOURCE_NAMES:
        raise ContractResourceError("generated contract resource registry is incomplete")
    return rendered


def _resource_bytes(relative_name: str) -> bytes:
    try:
        return files(RESOURCE_PACKAGE).joinpath(*relative_name.split("/")).read_bytes()
    except OSError as error:
        raise ContractResourceError("a packaged contract resource is missing") from error


def _packaged_resource_inventory() -> set[str]:
    root = files(RESOURCE_PACKAGE)
    inventory: set[str] = set()

    def visit(directory: Traversable, prefix: str = "") -> None:
        for child in directory.iterdir():
            name = child.name
            relative = f"{prefix}/{name}" if prefix else name
            if child.is_dir():
                if name != "__pycache__":
                    visit(child, relative)
            elif name != "__init__.py" and not name.endswith(".pyc"):
                inventory.add(relative)

    visit(root)
    return inventory


def load_packaged_contract_example(kind: ContractKind) -> ContractExample:
    """Load and authoritatively validate one packaged synthetic example by kind."""

    try:
        relative_name = f"examples/{CONTRACT_EXAMPLE_FILENAMES[kind]}"
        expected_schema_version = PUBLISHED_EXAMPLE_SCHEMA_VERSIONS[kind]
    except KeyError as error:
        raise ContractResourceError("unsupported packaged contract example kind") from error
    return cast(
        ContractExample,
        validate_contract(
            _resource_bytes(relative_name),
            expected_schema_version=expected_schema_version,
        ),
    )


def validate_packaged_contract_resources() -> None:
    """Check exact inventory, drift, schemas, examples, identities, and lineage."""

    try:
        if _packaged_resource_inventory() != GENERATED_RESOURCE_NAMES:
            raise ContractResourceError("packaged contract resource inventory is not exact")
        generated = generated_contract_resource_texts()
        for name, expected_text in generated.items():
            if _resource_bytes(name).decode("utf-8") != expected_text:
                raise ContractResourceError(
                    "packaged contract resource drift detected; regenerate resources"
                )

        schemas = generated_contract_schemas()
        loaded = {kind: load_packaged_contract_example(kind) for kind in CONTRACT_KINDS}
        for kind in CONTRACT_KINDS:
            schema_filename = CORE_CONTRACT_SCHEMA_FILENAMES[loaded[kind].schema_version]
            Draft202012Validator(schemas[schema_filename]).validate(
                loaded[kind].model_dump(mode="json", round_trip=True)
            )
            if contract_digest(loaded[kind]) != PUBLISHED_EXAMPLE_CONTRACT_DIGESTS[kind]:
                raise ContractResourceError("a published contract example identity changed")

        dataset = cast(DatasetManifestV1, loaded["dataset"])
        split = cast(SplitManifestV1, loaded["split"])
        preprocessing = cast(PreprocessingPlanV1, loaded["preprocessing"])
        configuration = cast(ResolvedConfigurationV1, loaded["resolved_configuration"])
        run = cast(RunRecordV1, loaded["run"])
        model = cast(ModelManifestV1, loaded["model"])
        assert_model_compatible(dataset, split, preprocessing, configuration, run, model)
    except ContractResourceError:
        raise
    except (OSError, ValueError, SchemaError, JsonSchemaValidationError) as error:
        raise ContractResourceError(
            "packaged contract resources are missing, invalid, or inconsistent"
        ) from error


__all__ = [
    "CONTRACT_EXAMPLE_FILENAMES",
    "CONTRACT_KINDS",
    "CONTRACT_SCHEMA_MODELS",
    "GENERATED_RESOURCE_NAMES",
    "PUBLISHED_EXAMPLE_CONTRACT_DIGESTS",
    "PUBLISHED_EXAMPLE_SCHEMA_VERSIONS",
    "ContractResourceError",
    "build_example_contract_chain",
    "generated_contract_resource_texts",
    "generated_contract_schemas",
    "load_packaged_contract_example",
    "render_json_document",
    "validate_packaged_contract_resources",
]
