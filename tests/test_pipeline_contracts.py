"""Adversarial acceptance tests for the versioned pipeline contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, cast

import pytest
from pydantic import BaseModel, ValidationError

import signlab.contracts.core as core_contracts
from signlab.contracts.canonical import canonical_json_bytes, canonical_sha256
from signlab.contracts.core import (
    ArtifactRefV1,
    ArtifactUriLocatorV1,
    ComponentSpecV1,
    ContractRefV1,
    ParameterV1,
    WorkspaceRelativeLocatorV1,
)
from signlab.contracts.pipeline import (
    ContractInput,
    ContractVersionError,
    DatasetContentV1,
    DatasetManifestV1,
    ModelManifestV1,
    PipelineContractError,
    PreprocessingPlanV1,
    ResolvedConfigurationV1,
    RunRecordV1,
    SplitManifestV1,
    assert_model_compatible,
    assert_resolved_configuration_compatible,
    assert_run_compatible,
    assert_split_compatible,
    canonical_contract_json,
    contract_digest,
    contract_json_object,
    contract_reference,
    dataset_content_digest,
    dataset_manifest_digest,
    model_manifest_digest,
    preprocessing_plan_digest,
    resolved_configuration_digest,
    run_record_digest,
    split_manifest_digest,
    validate_contract,
    validate_dataset_manifest,
    validate_model_manifest,
    validate_preprocessing_plan,
    validate_resolved_configuration,
    validate_run_record,
    validate_split_manifest,
)
from signlab.contracts.taxonomy import load_builtin_taxonomy, taxonomy_reference
from signlab.governance.resources import build_example_inventory, build_governance_policy

PipelineModel = (
    DatasetManifestV1
    | SplitManifestV1
    | PreprocessingPlanV1
    | ResolvedConfigurationV1
    | RunRecordV1
    | ModelManifestV1
)
Validator = Callable[[ContractInput], PipelineModel]
Digester = Callable[[ContractInput], str]
ComponentRole = Literal["model", "optimizer", "trainer", "evaluator"]

_LABELS = ("hello", "no", "please", "thank_you", "yes", "other")
_ZERO_SHA = "sha256:" + "0" * 64


@dataclass(frozen=True)
class ContractChain:
    dataset: DatasetManifestV1
    split: SplitManifestV1
    preprocessing: PreprocessingPlanV1
    configuration: ResolvedConfigurationV1
    run: RunRecordV1
    model: ModelManifestV1


def _sha(number: int) -> str:
    return f"sha256:{number:064x}"


def _stable_id(prefix: str, number: int) -> str:
    return f"{prefix}_{number:032x}"


def _workspace(path: str) -> WorkspaceRelativeLocatorV1:
    return WorkspaceRelativeLocatorV1(kind="workspace_relative", path=path)


def _artifact_uri(uri: str) -> ArtifactUriLocatorV1:
    return ArtifactUriLocatorV1(kind="artifact_uri", uri=uri)


def _artifact(
    *,
    artifact_id: str,
    role: str,
    digest_number: int,
    path: str,
    media_type: str = "application/octet-stream",
) -> ArtifactRefV1:
    return ArtifactRefV1(
        schema_version="artifact-reference/1",
        artifact_id=artifact_id,
        role=role,
        media_type=media_type,
        sha256=_sha(digest_number),
        size_bytes=1000 + digest_number,
        locator=_workspace(path),
    )


def _contract_locator(kind: str, contract_id: str) -> WorkspaceRelativeLocatorV1:
    return _workspace(f"contracts/{kind}/{contract_id}.json")


def _dump(model: PipelineModel) -> dict[str, Any]:
    return deepcopy(model.model_dump(mode="json"))


def _json_document(document: dict[str, object]) -> dict[str, Any]:
    """Convert builder-only nested models and tuples to plain JSON values."""

    def convert(value: object) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json", round_trip=True)
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    return cast(dict[str, Any], convert(document))


def _build_dataset() -> DatasetManifestV1:
    taxonomy = taxonomy_reference(load_builtin_taxonomy())
    governance = build_governance_policy().policy_document
    inventory = build_example_inventory()
    samples: list[dict[str, object]] = []
    for number, label in enumerate(("hello", "no", "please", "yes"), start=1):
        sample_id = _stable_id("sample", number)
        samples.append(
            {
                "sample_id": sample_id,
                "participant_id": _stable_id("participant", number),
                "session_id": _stable_id("session", number),
                "source_recording_id": _stable_id("recording", number),
                "label_id": label,
                "artifact": _artifact(
                    artifact_id=sample_id,
                    role="sample_data",
                    digest_number=number,
                    path=f"data/samples/{sample_id}.parquet",
                    media_type="application/x-parquet",
                ),
            }
        )
    content = DatasetContentV1.model_validate(
        {
            "schema_version": "dataset-content/1",
            "taxonomy": taxonomy,
            "governance_policy": governance,
            "lineage_inventory_sha256": inventory.inventory_sha256,
            "sample_schema_version": "sample-landmarks/1",
            "samples": tuple(samples),
        },
    )
    return validate_dataset_manifest(
        _json_document(
            {
                "schema_version": "dataset-manifest/1",
                "dataset_id": "dataset_core",
                "version": "1.0.0",
                "content": content,
                "data_sha256": dataset_content_digest(content),
            }
        )
    )


def _partition(
    name: str,
    sample_numbers: tuple[int, ...],
) -> dict[str, object]:
    return {
        "name": name,
        "sample_ids": tuple(_stable_id("sample", number) for number in sample_numbers),
        "participant_ids": tuple(_stable_id("participant", number) for number in sample_numbers),
        "session_ids": tuple(_stable_id("session", number) for number in sample_numbers),
        "source_recording_ids": tuple(_stable_id("recording", number) for number in sample_numbers),
    }


def _build_split(dataset: DatasetManifestV1) -> SplitManifestV1:
    return validate_split_manifest(
        _json_document(
            {
                "schema_version": "split-manifest/1",
                "split_id": "split_primary",
                "version": "1.0.0",
                "dataset": contract_reference(
                    dataset,
                    locator=_contract_locator("dataset", dataset.dataset_id),
                ),
                "dataset_data_sha256": dataset.data_sha256,
                "strategy": "participant-and-session-grouped",
                "random_seed": 1729,
                "partitions": (
                    _partition("train", (1, 2)),
                    _partition("validation", (3,)),
                    _partition("test", (4,)),
                ),
            }
        )
    )


def _build_preprocessing() -> PreprocessingPlanV1:
    return validate_preprocessing_plan(
        _json_document(
            {
                "schema_version": "preprocessing-plan/1",
                "plan_id": "preprocessing_primary",
                "version": "1.0.0",
                "taxonomy": taxonomy_reference(load_builtin_taxonomy()),
                "input_schema_version": "sample-landmarks/1",
                "output_schema_version": "feature-vector/1",
                "compatible_runtimes": ("python", "typescript"),
                "steps": (
                    {
                        "index": 0,
                        "operation_id": "normalize_landmarks",
                        "implementation_version": "1.0.0",
                        "input_schema_version": "sample-landmarks/1",
                        "output_schema_version": "normalized-landmarks/1",
                        "parameters": (
                            {"name": "mirror.enabled", "value": False},
                            {"name": "normalization.epsilon", "value": 0.000001},
                        ),
                    },
                    {
                        "index": 1,
                        "operation_id": "window_landmarks",
                        "implementation_version": "1.0.0",
                        "input_schema_version": "normalized-landmarks/1",
                        "output_schema_version": "feature-vector/1",
                        "parameters": ({"name": "window.frames", "value": 24},),
                    },
                ),
            }
        )
    )


def _component(role: ComponentRole, implementation_id: str) -> ComponentSpecV1:
    parameter_by_role: dict[str, ParameterV1] = {
        "model": ParameterV1(name="hidden.size", value=128),
        "optimizer": ParameterV1(name="learning.rate", value=0.001),
        "trainer": ParameterV1(name="epochs", value=20),
        "evaluator": ParameterV1(name="calibration.enabled", value=True),
    }
    return ComponentSpecV1(
        schema_version="component-spec/1",
        role=role,
        implementation_id=implementation_id,
        implementation_version="1.0.0",
        parameters=(parameter_by_role[role],),
    )


def _build_configuration(
    dataset: DatasetManifestV1,
    split: SplitManifestV1,
    preprocessing: PreprocessingPlanV1,
) -> ResolvedConfigurationV1:
    return validate_resolved_configuration(
        _json_document(
            {
                "schema_version": "resolved-configuration/1",
                "config_id": "config_baseline",
                "version": "1.0.0",
                "taxonomy": taxonomy_reference(load_builtin_taxonomy()),
                "dataset": contract_reference(
                    dataset,
                    locator=_contract_locator("dataset", dataset.dataset_id),
                ),
                "dataset_data_sha256": dataset.data_sha256,
                "split": contract_reference(
                    split,
                    locator=_contract_locator("split", split.split_id),
                ),
                "preprocessing": contract_reference(
                    preprocessing,
                    locator=_contract_locator("preprocessing", preprocessing.plan_id),
                ),
                "random_seed": split.random_seed,
                "deterministic_algorithms": True,
                "model": _component("model", "temporal_mlp"),
                "optimizer": _component("optimizer", "adamw"),
                "trainer": _component("trainer", "supervised_trainer"),
                "evaluator": _component("evaluator", "classification_evaluator"),
            }
        )
    )


def _model_artifact() -> ArtifactRefV1:
    return _artifact(
        artifact_id="model_weights",
        role="model",
        digest_number=100,
        path="artifacts/models/model_weights.onnx",
        media_type="application/onnx",
    )


def _build_run(
    dataset: DatasetManifestV1,
    split: SplitManifestV1,
    preprocessing: PreprocessingPlanV1,
    configuration: ResolvedConfigurationV1,
    model_artifact: ArtifactRefV1,
) -> RunRecordV1:
    return validate_run_record(
        _json_document(
            {
                "schema_version": "run-record/1",
                "run_id": "run_baseline",
                "version": "1.0.0",
                "status": "succeeded",
                "started_at": "2026-08-26T12:00:00Z",
                "finished_at": "2026-08-26T12:05:00Z",
                "source": {
                    "repository": "https://github.com/peterbucci/signlab",
                    "git_commit": "a" * 40,
                    "lockfile_sha256": _sha(101),
                    "working_tree_clean": True,
                },
                "runtime": {
                    "signlab_version": "0.1.0",
                    "python_version": "3.12.7",
                    "os_family": "linux",
                    "accelerator": "cpu",
                    "deterministic_algorithms": True,
                },
                "resolved_configuration": contract_reference(
                    configuration,
                    locator=_contract_locator("configuration", configuration.config_id),
                ),
                "dataset": contract_reference(
                    dataset,
                    locator=_contract_locator("dataset", dataset.dataset_id),
                ),
                "dataset_data_sha256": dataset.data_sha256,
                "split": contract_reference(
                    split,
                    locator=_contract_locator("split", split.split_id),
                ),
                "preprocessing": contract_reference(
                    preprocessing,
                    locator=_contract_locator("preprocessing", preprocessing.plan_id),
                ),
                "metrics": (
                    {
                        "name": "accuracy",
                        "partition": "test",
                        "value": 0.91,
                        "unit": "ratio",
                    },
                    {
                        "name": "loss",
                        "partition": "train",
                        "value": 0.08,
                        "unit": "cross_entropy",
                    },
                    {
                        "name": "accuracy",
                        "partition": "validation",
                        "value": 0.89,
                        "unit": "ratio",
                    },
                ),
                "outputs": (model_artifact,),
                "failure": None,
            }
        )
    )


def _build_model(
    dataset: DatasetManifestV1,
    split: SplitManifestV1,
    preprocessing: PreprocessingPlanV1,
    configuration: ResolvedConfigurationV1,
    run: RunRecordV1,
    model_artifact: ArtifactRefV1,
) -> ModelManifestV1:
    relocated_artifact = model_artifact.model_copy(
        update={
            "locator": _artifact_uri("signlab://registry/models/model_weights"),
        }
    )
    return validate_model_manifest(
        _json_document(
            {
                "schema_version": "model-manifest/1",
                "model_id": "model_baseline",
                "version": "1.0.0",
                "model_format": "onnx",
                "format_version": "1.17.0",
                "taxonomy": taxonomy_reference(load_builtin_taxonomy()),
                "label_order": _LABELS,
                "input_schema_version": preprocessing.output_schema_version,
                "output_schema_version": "gesture-logits/1",
                "training_run": contract_reference(
                    run,
                    locator=_contract_locator("run", run.run_id),
                ),
                "resolved_configuration": contract_reference(
                    configuration,
                    locator=_contract_locator("configuration", configuration.config_id),
                ),
                "dataset": contract_reference(
                    dataset,
                    locator=_contract_locator("dataset", dataset.dataset_id),
                ),
                "dataset_data_sha256": dataset.data_sha256,
                "split": contract_reference(
                    split,
                    locator=_contract_locator("split", split.split_id),
                ),
                "preprocessing": contract_reference(
                    preprocessing,
                    locator=_contract_locator("preprocessing", preprocessing.plan_id),
                ),
                "artifact": relocated_artifact,
            }
        )
    )


@pytest.fixture
def chain() -> ContractChain:
    dataset = _build_dataset()
    split = _build_split(dataset)
    preprocessing = _build_preprocessing()
    configuration = _build_configuration(dataset, split, preprocessing)
    model_artifact = _model_artifact()
    run = _build_run(dataset, split, preprocessing, configuration, model_artifact)
    model = _build_model(
        dataset,
        split,
        preprocessing,
        configuration,
        run,
        model_artifact,
    )
    return ContractChain(dataset, split, preprocessing, configuration, run, model)


def _cases(chain: ContractChain) -> tuple[tuple[PipelineModel, Validator, Digester], ...]:
    return (
        (chain.dataset, validate_dataset_manifest, dataset_manifest_digest),
        (chain.split, validate_split_manifest, split_manifest_digest),
        (
            chain.preprocessing,
            validate_preprocessing_plan,
            preprocessing_plan_digest,
        ),
        (
            chain.configuration,
            validate_resolved_configuration,
            resolved_configuration_digest,
        ),
        (chain.run, validate_run_record, run_record_digest),
        (chain.model, validate_model_manifest, model_manifest_digest),
    )


def test_six_contracts_round_trip_with_stable_domain_separated_digests(
    chain: ContractChain,
) -> None:
    digests: list[str] = []
    for model, validator, digester in _cases(chain):
        canonical = canonical_contract_json(model)
        assert validator(canonical) == model
        assert validator(contract_json_object(model)) == model
        assert digester(model) == digester(canonical)
        assert digester(model) == contract_digest(model)
        assert digester(model) != canonical_sha256(
            contract_json_object(model),
            domain="unrelated-contract/1",
        )
        digests.append(digester(model))

    assert len(set(digests)) == 6


def _leaf_mutations(value: object) -> Iterator[object]:
    """Yield documents with exactly one serialized leaf changed."""

    def mutated_scalar(scalar: object) -> object:
        if scalar is None:
            return True
        if isinstance(scalar, bool):
            return not scalar
        if isinstance(scalar, int):
            return scalar + 1
        if isinstance(scalar, float):
            return scalar + 0.125
        if isinstance(scalar, str):
            return f"{scalar}x"
        raise AssertionError(f"unexpected JSON scalar: {type(scalar)!r}")

    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                for replacement in _leaf_mutations(child):
                    changed_mapping = deepcopy(value)
                    changed_mapping[key] = replacement
                    yield changed_mapping
            else:
                changed_mapping = deepcopy(value)
                changed_mapping[key] = mutated_scalar(child)
                yield changed_mapping
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, (dict, list)):
                for replacement in _leaf_mutations(child):
                    changed_list = deepcopy(value)
                    changed_list[index] = replacement
                    yield changed_list
            else:
                changed_list = deepcopy(value)
                changed_list[index] = mutated_scalar(child)
                yield changed_list


def test_every_serialized_leaf_participates_in_each_contract_digest(
    chain: ContractChain,
) -> None:
    for model, _, digester in _cases(chain):
        document = contract_json_object(model)
        expected = digester(model)
        mutations = tuple(_leaf_mutations(document))
        assert mutations
        for changed in mutations:
            assert isinstance(changed, dict)
            assert canonical_sha256(changed, domain=model.schema_version) != expected


def test_dataset_data_digest_is_independent_of_storage_locator(
    chain: ContractChain,
) -> None:
    document = _dump(chain.dataset)
    for sample in cast(list[dict[str, Any]], document["content"]["samples"]):
        sample_id = cast(str, sample["sample_id"])
        sample["artifact"]["locator"] = {
            "kind": "artifact_uri",
            "uri": f"signlab://registry/samples/{sample_id}",
        }
    relocated = validate_dataset_manifest(document)

    assert relocated.data_sha256 == chain.dataset.data_sha256
    assert dataset_content_digest(relocated.content) == dataset_content_digest(
        chain.dataset.content
    )
    assert dataset_manifest_digest(relocated) != dataset_manifest_digest(chain.dataset)


@pytest.mark.parametrize(
    "locator",
    [
        {"kind": "workspace_relative", "path": "data/samples/item.json"},
        {"kind": "artifact_uri", "uri": "signlab://registry/models/model_weights"},
    ],
)
def test_portable_locators_accept_both_supported_forms(locator: dict[str, str]) -> None:
    artifact = ArtifactRefV1.model_validate(
        {
            "schema_version": "artifact-reference/1",
            "artifact_id": "portable_artifact",
            "role": "model",
            "media_type": "application/octet-stream",
            "sha256": _sha(1),
            "size_bytes": 1,
            "locator": locator,
        },
        strict=True,
    )
    assert artifact.locator.kind == locator["kind"]


@pytest.mark.parametrize(
    ("locator_model", "value"),
    [
        (
            WorkspaceRelativeLocatorV1,
            f"{chr(67)}:{chr(47)}Users{chr(47)}name{chr(47)}file.json",
        ),
        (WorkspaceRelativeLocatorV1, f"{chr(67)}:relative/file.json"),
        (WorkspaceRelativeLocatorV1, chr(47) * 2 + "server/share/file.json"),
        (WorkspaceRelativeLocatorV1, chr(92) * 2 + r"server\share\file.json"),
        (WorkspaceRelativeLocatorV1, chr(47) + "var/tmp/file.json"),
        (WorkspaceRelativeLocatorV1, "../secret.json"),
        (WorkspaceRelativeLocatorV1, "safe/../secret.json"),
        (WorkspaceRelativeLocatorV1, r"safe\secret.json"),
        (WorkspaceRelativeLocatorV1, "safe/%2e%2e/secret.json"),
        (WorkspaceRelativeLocatorV1, "safe/CON.txt"),
        (WorkspaceRelativeLocatorV1, "safe/nul"),
        (ArtifactUriLocatorV1, "signlab://user@registry/models/item"),
        (ArtifactUriLocatorV1, "signlab://registry:443/models/item"),
        (ArtifactUriLocatorV1, "signlab://registry/models/item?download=1"),
        (ArtifactUriLocatorV1, "signlab://registry/models/item#fragment"),
        (ArtifactUriLocatorV1, "signlab://registry/models/%2e%2e/secret"),
        (ArtifactUriLocatorV1, r"signlab://registry/models\item"),
        (ArtifactUriLocatorV1, "signlab://registry/models/../secret"),
    ],
)
def test_portable_locators_reject_host_specific_or_ambiguous_inputs(
    locator_model: type[WorkspaceRelativeLocatorV1] | type[ArtifactUriLocatorV1],
    value: str,
) -> None:
    field = "path" if locator_model is WorkspaceRelativeLocatorV1 else "uri"
    kind = "workspace_relative" if field == "path" else "artifact_uri"
    with pytest.raises(ValidationError):
        locator_model.model_validate({"kind": kind, field: value}, strict=True)


def test_full_contract_chain_is_compatible(chain: ContractChain) -> None:
    assert_split_compatible(chain.dataset, chain.split)
    assert_resolved_configuration_compatible(
        chain.dataset,
        chain.split,
        chain.preprocessing,
        chain.configuration,
    )
    assert_run_compatible(
        chain.dataset,
        chain.split,
        chain.preprocessing,
        chain.configuration,
        chain.run,
    )
    assert_model_compatible(
        chain.dataset,
        chain.split,
        chain.preprocessing,
        chain.configuration,
        chain.run,
        chain.model,
    )


def test_dataset_membership_must_be_sorted_unique_and_content_addressed(
    chain: ContractChain,
) -> None:
    reversed_samples = _dump(chain.dataset)
    reversed_samples["content"]["samples"].reverse()
    with pytest.raises(PipelineContractError, match="sorted"):
        validate_dataset_manifest(reversed_samples)

    duplicate_digest = _dump(chain.dataset)
    first_digest = duplicate_digest["content"]["samples"][0]["artifact"]["sha256"]
    duplicate_digest["content"]["samples"][1]["artifact"]["sha256"] = first_digest
    with pytest.raises(PipelineContractError, match="duplicate artifact content"):
        validate_dataset_manifest(duplicate_digest)

    wrong_content_hash = _dump(chain.dataset)
    wrong_content_hash["data_sha256"] = _ZERO_SHA
    with pytest.raises(PipelineContractError, match="data_sha256"):
        validate_dataset_manifest(wrong_content_hash)


def test_membership_and_component_collections_must_be_sorted_and_unique(
    chain: ContractChain,
) -> None:
    duplicate_member = _dump(chain.split)
    duplicate_member["partitions"][0]["sample_ids"].append(
        duplicate_member["partitions"][0]["sample_ids"][0]
    )
    with pytest.raises(PipelineContractError, match="sorted"):
        validate_split_manifest(duplicate_member)

    unsorted_member = _dump(chain.split)
    unsorted_member["partitions"][0]["sample_ids"].reverse()
    with pytest.raises(PipelineContractError, match="sorted"):
        validate_split_manifest(unsorted_member)

    duplicate_parameter = _dump(chain.configuration)
    existing = deepcopy(duplicate_parameter["model"]["parameters"][0])
    duplicate_parameter["model"]["parameters"].append(existing)
    with pytest.raises(PipelineContractError, match="unique names"):
        validate_resolved_configuration(duplicate_parameter)

    unsorted_parameter = _dump(chain.configuration)
    unsorted_parameter["model"]["parameters"] = [
        {"name": "zeta", "value": 1},
        {"name": "alpha", "value": 2},
    ]
    with pytest.raises(PipelineContractError, match="sorted order"):
        validate_resolved_configuration(unsorted_parameter)


def test_split_requires_exact_dataset_coverage(chain: ContractChain) -> None:
    document = _dump(chain.split)
    for field in (
        "sample_ids",
        "participant_ids",
        "session_ids",
        "source_recording_ids",
    ):
        document["partitions"][0][field].pop()
    incomplete = validate_split_manifest(document)

    with pytest.raises(PipelineContractError, match="exactly once"):
        assert_split_compatible(chain.dataset, incomplete)


def test_split_rejects_unknown_samples_and_incorrect_group_closure(
    chain: ContractChain,
) -> None:
    unknown_document = _dump(chain.split)
    unknown_document["partitions"][0]["sample_ids"][1] = _stable_id("sample", 99)
    unknown = validate_split_manifest(unknown_document)
    with pytest.raises(PipelineContractError, match="exactly once"):
        assert_split_compatible(chain.dataset, unknown)

    closure_document = _dump(chain.split)
    closure_document["partitions"][0]["participant_ids"] = [
        _stable_id("participant", 91),
        _stable_id("participant", 92),
    ]
    wrong_closure = validate_split_manifest(closure_document)
    with pytest.raises(PipelineContractError, match="close over"):
        assert_split_compatible(chain.dataset, wrong_closure)


def test_split_rejects_group_leakage_across_partitions(chain: ContractChain) -> None:
    for field in ("participant_ids", "session_ids", "source_recording_ids"):
        document = _dump(chain.split)
        document["partitions"][1][field] = [document["partitions"][0][field][0]]
        with pytest.raises(PipelineContractError, match="must not cross split partitions"):
            validate_split_manifest(document)


def test_split_binds_complete_dataset_reference_and_data_identity(
    chain: ContractChain,
) -> None:
    for field, replacement in (
        ("contract_id", "dataset_other"),
        ("contract_version", "1.0.1"),
        ("sha256", _ZERO_SHA),
    ):
        document = _dump(chain.split)
        document["dataset"][field] = replacement
        split = validate_split_manifest(document)
        with pytest.raises(PipelineContractError, match="dataset reference"):
            assert_split_compatible(chain.dataset, split)

    document = _dump(chain.split)
    document["dataset_data_sha256"] = _ZERO_SHA
    split = validate_split_manifest(document)
    with pytest.raises(PipelineContractError, match="data identity"):
        assert_split_compatible(chain.dataset, split)

    relocated_document = _dump(chain.split)
    relocated_document["dataset"]["locator"] = {
        "kind": "artifact_uri",
        "uri": "signlab://registry/contracts/dataset_core",
    }
    assert_split_compatible(chain.dataset, validate_split_manifest(relocated_document))


def test_preprocessing_requires_ordered_compatible_schema_chain(
    chain: ContractChain,
) -> None:
    invalid_documents: list[dict[str, Any]] = []

    reversed_steps = _dump(chain.preprocessing)
    reversed_steps["steps"].reverse()
    invalid_documents.append(reversed_steps)

    gap = _dump(chain.preprocessing)
    gap["steps"][1]["index"] = 2
    invalid_documents.append(gap)

    wrong_first_input = _dump(chain.preprocessing)
    wrong_first_input["steps"][0]["input_schema_version"] = "other-input/1"
    invalid_documents.append(wrong_first_input)

    broken_adjacency = _dump(chain.preprocessing)
    broken_adjacency["steps"][0]["output_schema_version"] = "other-middle/1"
    invalid_documents.append(broken_adjacency)

    wrong_last_output = _dump(chain.preprocessing)
    wrong_last_output["steps"][1]["output_schema_version"] = "other-output/1"
    invalid_documents.append(wrong_last_output)

    runtime_order = _dump(chain.preprocessing)
    runtime_order["compatible_runtimes"].reverse()
    invalid_documents.append(runtime_order)

    duplicate_runtime = _dump(chain.preprocessing)
    duplicate_runtime["compatible_runtimes"].append("typescript")
    invalid_documents.append(duplicate_runtime)

    unsorted_parameters = _dump(chain.preprocessing)
    unsorted_parameters["steps"][0]["parameters"] = [
        {"name": "zeta", "value": 1},
        {"name": "alpha", "value": 2},
    ]
    invalid_documents.append(unsorted_parameters)

    for document in invalid_documents:
        with pytest.raises(PipelineContractError):
            validate_preprocessing_plan(document)


def test_configuration_binds_all_reference_tuple_fields_and_taxonomy(
    chain: ContractChain,
) -> None:
    for reference_name in ("dataset", "split", "preprocessing"):
        for field, replacement in (
            ("contract_id", f"{reference_name}_other"),
            ("contract_version", "1.0.1"),
            ("sha256", _ZERO_SHA),
        ):
            document = _dump(chain.configuration)
            document[reference_name][field] = replacement
            configuration = validate_resolved_configuration(document)
            with pytest.raises(PipelineContractError, match="reference"):
                assert_resolved_configuration_compatible(
                    chain.dataset,
                    chain.split,
                    chain.preprocessing,
                    configuration,
                )

    document = _dump(chain.configuration)
    document["dataset_data_sha256"] = _ZERO_SHA
    configuration = validate_resolved_configuration(document)
    with pytest.raises(PipelineContractError, match="data identity"):
        assert_resolved_configuration_compatible(
            chain.dataset,
            chain.split,
            chain.preprocessing,
            configuration,
        )

    forged_taxonomy = chain.configuration.taxonomy.model_copy(update={"version": "9.9.9"})
    forged_configuration = chain.configuration.model_copy(update={"taxonomy": forged_taxonomy})
    with pytest.raises(PipelineContractError):
        assert_resolved_configuration_compatible(
            chain.dataset,
            chain.split,
            chain.preprocessing,
            forged_configuration,
        )


def test_configuration_requires_preprocessing_input_compatible_with_dataset(
    chain: ContractChain,
) -> None:
    preprocessing_document = _dump(chain.preprocessing)
    preprocessing_document["input_schema_version"] = "different-samples/1"
    preprocessing_document["steps"][0]["input_schema_version"] = "different-samples/1"
    preprocessing = validate_preprocessing_plan(preprocessing_document)

    configuration_document = _dump(chain.configuration)
    configuration_document["preprocessing"] = contract_reference(
        preprocessing,
        locator=_contract_locator("preprocessing", preprocessing.plan_id),
    ).model_dump(mode="json")
    configuration = validate_resolved_configuration(configuration_document)

    with pytest.raises(PipelineContractError, match="input schema"):
        assert_resolved_configuration_compatible(
            chain.dataset,
            chain.split,
            preprocessing,
            configuration,
        )


def test_run_terminal_state_invariants(chain: ContractChain) -> None:
    invalid_documents: list[dict[str, Any]] = []

    success_without_output = _dump(chain.run)
    success_without_output["outputs"] = []
    invalid_documents.append(success_without_output)

    success_with_failure = _dump(chain.run)
    success_with_failure["failure"] = {
        "code": "training_failed",
        "stage": "trainer",
        "retryable": False,
    }
    invalid_documents.append(success_with_failure)

    failed_without_failure = _dump(chain.run)
    failed_without_failure["status"] = "failed"
    invalid_documents.append(failed_without_failure)

    cancelled_without_failure = _dump(chain.run)
    cancelled_without_failure["status"] = "cancelled"
    invalid_documents.append(cancelled_without_failure)

    reversed_times = _dump(chain.run)
    reversed_times["finished_at"] = "2026-08-26T11:59:59Z"
    invalid_documents.append(reversed_times)

    for document in invalid_documents:
        with pytest.raises(PipelineContractError):
            validate_run_record(document)

    for status in ("failed", "cancelled"):
        terminal = _dump(chain.run)
        terminal["status"] = status
        terminal["outputs"] = []
        terminal["failure"] = {
            "code": f"{status}_run",
            "stage": "trainer",
            "retryable": status == "failed",
        }
        assert validate_run_record(terminal).status == status


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_run_metrics_must_be_finite(
    chain: ContractChain,
    non_finite: float,
) -> None:
    document = _dump(chain.run)
    document["metrics"][0]["value"] = non_finite
    with pytest.raises(PipelineContractError, match="interoperable JSON object"):
        validate_run_record(document)


def test_run_metrics_and_outputs_must_be_sorted_and_unique(
    chain: ContractChain,
) -> None:
    reversed_metrics = _dump(chain.run)
    reversed_metrics["metrics"].reverse()
    with pytest.raises(PipelineContractError, match=r"metrics.*sorted"):
        validate_run_record(reversed_metrics)

    duplicate_metric = _dump(chain.run)
    duplicate_metric["metrics"].append(deepcopy(duplicate_metric["metrics"][0]))
    with pytest.raises(PipelineContractError, match=r"metrics.*unique"):
        validate_run_record(duplicate_metric)

    duplicate_output = _dump(chain.run)
    duplicate_output["outputs"].append(deepcopy(duplicate_output["outputs"][0]))
    with pytest.raises(PipelineContractError, match=r"outputs.*unique"):
        validate_run_record(duplicate_output)

    unsorted_output = _dump(chain.run)
    unsorted_output["outputs"].append(
        _artifact(
            artifact_id="checkpoint_final",
            role="checkpoint",
            digest_number=102,
            path="artifacts/checkpoints/final.bin",
        ).model_dump(mode="json")
    )
    with pytest.raises(PipelineContractError, match=r"outputs.*sorted"):
        validate_run_record(unsorted_output)


def test_failure_records_are_structured_and_do_not_echo_sensitive_values(
    chain: ContractChain,
) -> None:
    sensitive_path = (
        f"{chr(67)}:{chr(47)}Users{chr(47)}private-person{chr(47)}secret-experiment{chr(47)}raw.csv"
    )
    document = _dump(chain.run)
    document["status"] = "failed"
    document["outputs"] = []
    document["failure"] = {
        "code": "training_failed",
        "stage": "trainer",
        "retryable": True,
        "message": f"could not read {sensitive_path}",
        "traceback": f"File {sensitive_path}, line 1",
    }

    with pytest.raises(PipelineContractError) as captured:
        validate_run_record(document)
    assert sensitive_path not in str(captured.value)
    assert "private-person" not in str(captured.value)


def test_run_binds_complete_upstream_chain(chain: ContractChain) -> None:
    for reference_name in (
        "resolved_configuration",
        "dataset",
        "split",
        "preprocessing",
    ):
        for field, replacement in (
            ("contract_id", f"{reference_name}_other"),
            ("contract_version", "1.0.1"),
            ("sha256", _ZERO_SHA),
        ):
            document = _dump(chain.run)
            document[reference_name][field] = replacement
            run = validate_run_record(document)
            with pytest.raises(PipelineContractError, match="reference"):
                assert_run_compatible(
                    chain.dataset,
                    chain.split,
                    chain.preprocessing,
                    chain.configuration,
                    run,
                )

    document = _dump(chain.run)
    document["dataset_data_sha256"] = _ZERO_SHA
    run = validate_run_record(document)
    with pytest.raises(PipelineContractError, match="data identity"):
        assert_run_compatible(
            chain.dataset,
            chain.split,
            chain.preprocessing,
            chain.configuration,
            run,
        )


def test_model_binds_run_chain_input_schema_and_output_artifact(
    chain: ContractChain,
) -> None:
    for reference_name in (
        "training_run",
        "resolved_configuration",
        "dataset",
        "split",
        "preprocessing",
    ):
        for field, replacement in (
            ("contract_id", f"{reference_name}_other"),
            ("contract_version", "1.0.1"),
            ("sha256", _ZERO_SHA),
        ):
            document = _dump(chain.model)
            document[reference_name][field] = replacement
            model = validate_model_manifest(document)
            with pytest.raises(PipelineContractError, match="reference"):
                assert_model_compatible(
                    chain.dataset,
                    chain.split,
                    chain.preprocessing,
                    chain.configuration,
                    chain.run,
                    model,
                )

    wrong_data = _dump(chain.model)
    wrong_data["dataset_data_sha256"] = _ZERO_SHA
    model = validate_model_manifest(wrong_data)
    with pytest.raises(PipelineContractError, match="data identity"):
        assert_model_compatible(
            chain.dataset,
            chain.split,
            chain.preprocessing,
            chain.configuration,
            chain.run,
            model,
        )

    artifact_mutations: tuple[tuple[str, object], ...] = (
        ("artifact_id", "model_weights_other"),
        ("media_type", "application/octet-stream"),
        ("sha256", _sha(999)),
        ("size_bytes", 9999),
    )
    for artifact_field, artifact_replacement in artifact_mutations:
        wrong_artifact = _dump(chain.model)
        wrong_artifact["artifact"][artifact_field] = artifact_replacement
        model = validate_model_manifest(wrong_artifact)
        with pytest.raises(PipelineContractError, match="output of its training run"):
            assert_model_compatible(
                chain.dataset,
                chain.split,
                chain.preprocessing,
                chain.configuration,
                chain.run,
                model,
            )

    wrong_input = _dump(chain.model)
    wrong_input["input_schema_version"] = "different-features/1"
    model = validate_model_manifest(wrong_input)
    with pytest.raises(PipelineContractError, match="input schema"):
        assert_model_compatible(
            chain.dataset,
            chain.split,
            chain.preprocessing,
            chain.configuration,
            chain.run,
            model,
        )


def test_model_requires_successful_training_run(chain: ContractChain) -> None:
    run_document = _dump(chain.run)
    run_document["status"] = "failed"
    run_document["failure"] = {
        "code": "training_failed",
        "stage": "trainer",
        "retryable": True,
    }
    failed_run = validate_run_record(run_document)

    model_document = _dump(chain.model)
    model_document["training_run"] = contract_reference(
        failed_run,
        locator=_contract_locator("run", failed_run.run_id),
    ).model_dump(mode="json")
    model = validate_model_manifest(model_document)

    with pytest.raises(PipelineContractError, match="successful training run"):
        assert_model_compatible(
            chain.dataset,
            chain.split,
            chain.preprocessing,
            chain.configuration,
            failed_run,
            model,
        )


def test_model_output_schema_is_required_and_hashed(chain: ContractChain) -> None:
    missing = _dump(chain.model)
    del missing["output_schema_version"]
    with pytest.raises(PipelineContractError):
        validate_model_manifest(missing)

    changed = _dump(chain.model)
    changed["output_schema_version"] = "gesture-probabilities/1"
    changed_model = validate_model_manifest(changed)
    assert model_manifest_digest(changed_model) != model_manifest_digest(chain.model)


def test_dataset_semantic_content_fields_change_the_data_digest(
    chain: ContractChain,
) -> None:
    original = dataset_content_digest(chain.dataset.content)
    mutations: list[dict[str, Any]] = []

    lineage = deepcopy(chain.dataset.content.model_dump(mode="json"))
    lineage["lineage_inventory_sha256"] = _sha(201)
    mutations.append(lineage)

    sample_schema = deepcopy(chain.dataset.content.model_dump(mode="json"))
    sample_schema["sample_schema_version"] = "sample-landmarks/2"
    mutations.append(sample_schema)

    participant = deepcopy(chain.dataset.content.model_dump(mode="json"))
    participant["samples"][0]["participant_id"] = _stable_id("participant", 101)
    mutations.append(participant)

    session = deepcopy(chain.dataset.content.model_dump(mode="json"))
    session["samples"][0]["session_id"] = _stable_id("session", 101)
    mutations.append(session)

    recording = deepcopy(chain.dataset.content.model_dump(mode="json"))
    recording["samples"][0]["source_recording_id"] = _stable_id("recording", 101)
    mutations.append(recording)

    label = deepcopy(chain.dataset.content.model_dump(mode="json"))
    label["samples"][0]["label_id"] = "other"
    mutations.append(label)

    sample_identity = deepcopy(chain.dataset.content.model_dump(mode="json"))
    changed_sample_id = _stable_id("sample", 99)
    sample_identity["samples"][-1]["sample_id"] = changed_sample_id
    sample_identity["samples"][-1]["artifact"]["artifact_id"] = changed_sample_id
    mutations.append(sample_identity)

    artifact_digest = deepcopy(chain.dataset.content.model_dump(mode="json"))
    artifact_digest["samples"][0]["artifact"]["sha256"] = _sha(202)
    mutations.append(artifact_digest)

    artifact_size = deepcopy(chain.dataset.content.model_dump(mode="json"))
    artifact_size["samples"][0]["artifact"]["size_bytes"] += 1
    mutations.append(artifact_size)

    artifact_media = deepcopy(chain.dataset.content.model_dump(mode="json"))
    artifact_media["samples"][0]["artifact"]["media_type"] = "application/json"
    mutations.append(artifact_media)

    for document in mutations:
        content = DatasetContentV1.model_validate_json(
            canonical_json_bytes(document),
            strict=True,
        )
        assert dataset_content_digest(content) != original


def test_top_level_schema_versions_fail_closed_with_migration_guidance(
    chain: ContractChain,
) -> None:
    cases = _cases(chain)
    known_versions = tuple(model.schema_version for model, _, _ in cases)
    for index, (model, validator, _) in enumerate(cases):
        base = _dump(model)
        expected = model.schema_version
        future = f"{expected.rsplit('/', maxsplit=1)[0]}/2"

        missing = deepcopy(base)
        del missing["schema_version"]
        wrong = deepcopy(base)
        wrong["schema_version"] = known_versions[(index + 1) % len(known_versions)]
        newer = deepcopy(base)
        newer["schema_version"] = future

        for document in (missing, wrong, newer):
            with pytest.raises(ContractVersionError) as captured:
                validator(document)
            message = str(captured.value)
            assert f"supported: {expected}" in message
            assert "writers must emit a supported version" in message
            assert "docs/contracts.md#compatibility-and-migration" in message


def test_nested_schema_versions_are_exact_and_never_coerced(
    chain: ContractChain,
) -> None:
    for supplied in (None, "artifact-reference/0", "artifact-reference/2"):
        document = _dump(chain.dataset)
        artifact = document["content"]["samples"][0]["artifact"]
        if supplied is None:
            del artifact["schema_version"]
        else:
            artifact["schema_version"] = supplied
        with pytest.raises(PipelineContractError):
            validate_dataset_manifest(document)

    component_document = _dump(chain.configuration)
    component_document["model"]["schema_version"] = "component-spec/2"
    with pytest.raises(PipelineContractError):
        validate_resolved_configuration(component_document)

    reference_document = _dump(chain.run)
    reference_document["dataset"]["schema_version"] = "contract-reference/2"
    with pytest.raises(PipelineContractError):
        validate_run_record(reference_document)


def test_duplicate_json_object_keys_are_rejected_for_all_six_contracts(
    chain: ContractChain,
) -> None:
    for model, validator, _ in _cases(chain):
        canonical = canonical_contract_json(model).decode("utf-8")
        duplicate = f'{{"schema_version":"{model.schema_version}",' + canonical[1:]
        with pytest.raises(PipelineContractError, match="interoperable JSON object"):
            validator(duplicate)


def test_unsafe_model_copies_are_recursively_revalidated(chain: ContractChain) -> None:
    for model, validator, _ in _cases(chain):
        future = f"{model.schema_version.rsplit('/', maxsplit=1)[0]}/2"
        forged = model.model_copy(update={"schema_version": future})
        with pytest.raises(ContractVersionError):
            validator(forged)

    sample = chain.dataset.content.samples[0]
    forged_artifact = sample.artifact.model_copy(update={"role": "model"})
    forged_sample = sample.model_copy(update={"artifact": forged_artifact})
    forged_content = chain.dataset.content.model_copy(
        update={"samples": (forged_sample, *chain.dataset.content.samples[1:])}
    )
    forged_dataset = chain.dataset.model_copy(update={"content": forged_content})
    with pytest.raises(PipelineContractError):
        validate_dataset_manifest(forged_dataset)

    metric = chain.run.metrics[0].model_copy(update={"value": float("nan")})
    forged_run = chain.run.model_copy(update={"metrics": (metric, *chain.run.metrics[1:])})
    with pytest.raises(PipelineContractError):
        validate_run_record(forged_run)

    forged_reference = chain.model.dataset.model_copy(update={"kind": "split"})
    forged_model = chain.model.model_copy(update={"dataset": forged_reference})
    with pytest.raises(PipelineContractError):
        validate_model_manifest(forged_model)


def test_contract_instances_are_frozen_and_reject_extra_fields(
    chain: ContractChain,
) -> None:
    with pytest.raises(ValidationError):
        chain.dataset.version = "1.0.1"

    document = _dump(chain.model)
    document["developer_machine"] = f"{chr(67)}:{chr(47)}Users{chr(47)}private-person"
    with pytest.raises(PipelineContractError) as captured:
        validate_model_manifest(document)
    assert "private-person" not in str(captured.value)


def test_generic_reader_rejects_unknown_versions_without_guessing(
    chain: ContractChain,
) -> None:
    document = _dump(chain.dataset)
    document["schema_version"] = "dataset-manifest/99"
    with pytest.raises(ContractVersionError) as captured:
        validate_contract(document)
    message = str(captured.value)
    assert "dataset-manifest/1" in message
    assert "model-manifest/1" in message
    assert "compatibility-and-migration" in message


def test_unknown_expected_version_reports_only_actual_supported_readers(
    chain: ContractChain,
) -> None:
    with pytest.raises(ContractVersionError) as captured:
        validate_contract(
            chain.dataset,
            expected_schema_version="dataset-manifest/99",
        )

    message = str(captured.value)
    assert "dataset-manifest/1" in message
    assert "supported: dataset-manifest/99" not in message


def test_retained_v1_references_are_independent_of_the_current_writer(
    chain: ContractChain,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = chain.split.dataset.model_dump(mode="json", round_trip=True)
    monkeypatch.setitem(
        core_contracts.CURRENT_CONTRACT_SCHEMAS,
        "dataset",
        "dataset-manifest/2",
    )

    retained = ContractRefV1.model_validate_json(
        canonical_json_bytes(payload),
        strict=True,
    )

    assert retained.contract_schema_version == "dataset-manifest/1"
    assert core_contracts.SUPPORTED_CONTRACT_REFERENCE_SCHEMAS["dataset"] == frozenset(
        {"dataset-manifest/1"}
    )


def test_every_current_writer_and_v1_reader_is_retained() -> None:
    expected_v1: dict[core_contracts.ContractKind, str] = {
        "dataset": "dataset-manifest/1",
        "split": "split-manifest/1",
        "preprocessing": "preprocessing-plan/1",
        "resolved_configuration": "resolved-configuration/1",
        "run": "run-record/1",
        "model": "model-manifest/1",
    }

    for kind, v1_schema in expected_v1.items():
        retained = core_contracts.SUPPORTED_CONTRACT_REFERENCE_SCHEMAS[kind]
        assert v1_schema in retained
        assert core_contracts.CURRENT_CONTRACT_SCHEMAS[kind] in retained
