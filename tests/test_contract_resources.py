from __future__ import annotations

import json
import re
from collections.abc import Iterator
from importlib.resources import files
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from scripts.generate_contract_resources import write_resources

from signlab.contracts.core import ContractKind, same_artifact_reference, same_contract_reference
from signlab.contracts.pipeline import (
    CORE_CONTRACT_SCHEMA_FILENAMES,
    DatasetManifestV1,
    ModelManifestV1,
    PipelineContractError,
    PreprocessingPlanV1,
    ResolvedConfigurationV1,
    RunRecordV1,
    SplitManifestV1,
    assert_model_compatible,
    assert_split_compatible,
    contract_digest,
    contract_reference,
    validate_contract,
    validate_dataset_manifest,
    validate_split_manifest,
)
from signlab.contracts.resources import (
    CONTRACT_EXAMPLE_FILENAMES,
    CONTRACT_KINDS,
    CONTRACT_SCHEMA_MODELS,
    GENERATED_RESOURCE_NAMES,
    PUBLISHED_EXAMPLE_CONTRACT_DIGESTS,
    ContractResourceError,
    build_example_contract_chain,
    generated_contract_resource_texts,
    generated_contract_schemas,
    load_packaged_contract_example,
    render_json_document,
    validate_packaged_contract_resources,
)

EXPECTED_EXAMPLE_DIGESTS: dict[ContractKind, str] = {
    "dataset": "sha256:ed55707743ddf6ca144124ee671994ceb8eb7cea06c8f7d97ec10203d5dd8717",
    "split": "sha256:6fea000166d7eddc699f2bf03fbf85b2b88d945a68ff98e236345279255b84a5",
    "preprocessing": ("sha256:5fe847d97c77686900ccc8442d081c2ef428c5609ccf27fd70daf0360e559cbe"),
    "resolved_configuration": (
        "sha256:c7691a990279743d53189a8e7fa9685ad64aba0b04a174cc9b8b59f41fdf3e9b"
    ),
    "run": "sha256:2aa346eddc0ef19f3bd19dbc7bc0017858641b04fad525706b60fd4e17678aa6",
    "model": "sha256:aec88f012bbe936d0ae6c53d73c3cd19c46e94b7545207929075fd7b89887525",
}


def _schema_nodes(value: object) -> Iterator[dict[str, object]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _schema_nodes(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _schema_nodes(nested)


def _chain_by_kind() -> dict[
    ContractKind,
    DatasetManifestV1
    | SplitManifestV1
    | PreprocessingPlanV1
    | ResolvedConfigurationV1
    | RunRecordV1
    | ModelManifestV1,
]:
    dataset, split, preprocessing, configuration, run, model = build_example_contract_chain()
    return {
        "dataset": dataset,
        "split": split,
        "preprocessing": preprocessing,
        "resolved_configuration": configuration,
        "run": run,
        "model": model,
    }


def _packaged_inventory() -> set[str]:
    root = files("signlab.resources.contracts")
    inventory: set[str] = set()
    for directory_name in ("examples", "schemas"):
        directory = root.joinpath(directory_name)
        inventory.update(
            f"{directory_name}/{child.name}" for child in directory.iterdir() if child.is_file()
        )
    return inventory


def test_exactly_seven_standalone_draft_202012_schemas_are_generated_and_packaged() -> None:
    schemas = generated_contract_schemas()

    assert len(schemas) == 7
    assert set(schemas) == set(CONTRACT_SCHEMA_MODELS)
    for filename, schema in schemas.items():
        Draft202012Validator.check_schema(schema)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["$id"] == f"https://signlab.dev/schemas/{filename}"
        assert schema["$comment"]
        assert "application compatibility validation" in str(schema["$comment"])
        assert [node["$id"] for node in _schema_nodes(schema) if "$id" in node] == [schema["$id"]]
        assert all(
            not {"ge", "gt", "le", "lt"}.intersection(node) for node in _schema_nodes(schema)
        )
        packaged = json.loads(
            files("signlab.resources.contracts")
            .joinpath("schemas", filename)
            .read_text(encoding="utf-8")
        )
        assert packaged == schema


def test_six_examples_round_trip_through_schema_and_authoritative_pydantic_validation() -> None:
    schemas = generated_contract_schemas()
    examples = _chain_by_kind()

    assert set(examples) == set(CONTRACT_KINDS)
    for kind, example in examples.items():
        schema_filename = CORE_CONTRACT_SCHEMA_FILENAMES[example.schema_version]
        payload = example.model_dump(mode="json", round_trip=True)
        Draft202012Validator(schemas[schema_filename]).validate(payload)
        assert validate_contract(render_json_document(example)) == example
        assert load_packaged_contract_example(kind) == example


def test_json_schemas_reject_expressible_version_path_membership_and_label_bypasses() -> None:
    schemas = generated_contract_schemas()
    dataset, split, _, _, _, model = build_example_contract_chain()

    wrong_version = dataset.model_dump(mode="json", round_trip=True)
    wrong_version["schema_version"] = "dataset-manifest/2"
    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(schemas["dataset-manifest-1.schema.json"]).validate(wrong_version)

    absolute_path = dataset.model_dump(mode="json", round_trip=True)
    absolute_path["content"]["samples"][0]["artifact"]["locator"] = {
        "kind": "workspace_relative",
        "path": f"{chr(67)}:{chr(47)}private{chr(47)}sample.json",
    }
    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(schemas["dataset-manifest-1.schema.json"]).validate(absolute_path)

    duplicate_member = split.model_dump(mode="json", round_trip=True)
    duplicate_member["partitions"][0]["sample_ids"].append(
        duplicate_member["partitions"][0]["sample_ids"][0]
    )
    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(schemas["split-manifest-1.schema.json"]).validate(duplicate_member)

    wrong_labels = model.model_dump(mode="json", round_trip=True)
    wrong_labels["label_order"] = list(reversed(wrong_labels["label_order"]))
    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(schemas["model-manifest-1.schema.json"]).validate(wrong_labels)

    negative_size = dataset.model_dump(mode="json", round_trip=True)
    negative_size["content"]["samples"][0]["artifact"]["size_bytes"] = -1
    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(schemas["dataset-manifest-1.schema.json"]).validate(negative_size)

    negative_index = _chain_by_kind()["preprocessing"].model_dump(mode="json", round_trip=True)
    negative_index["steps"][0]["index"] = -1
    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(schemas["preprocessing-plan-1.schema.json"]).validate(negative_index)


def test_schema_boundary_defers_digest_and_cross_document_identity_to_application_code() -> None:
    schemas = generated_contract_schemas()
    dataset, split, _, _, _, _ = build_example_contract_chain()

    tampered_dataset = dataset.model_dump(mode="json", round_trip=True)
    tampered_dataset["data_sha256"] = "sha256:" + "f" * 64
    Draft202012Validator(schemas["dataset-manifest-1.schema.json"]).validate(tampered_dataset)
    with pytest.raises(PipelineContractError, match="data_sha256"):
        validate_dataset_manifest(json.dumps(tampered_dataset))

    replayed_split = split.model_dump(mode="json", round_trip=True)
    replayed_split["dataset"]["sha256"] = "sha256:" + "f" * 64
    Draft202012Validator(schemas["split-manifest-1.schema.json"]).validate(replayed_split)
    structurally_valid = validate_split_manifest(json.dumps(replayed_split))
    with pytest.raises(PipelineContractError, match="reference"):
        assert_split_compatible(dataset, structurally_valid)


def test_generated_resource_writer_is_complete_pretty_and_byte_stable(tmp_path: Path) -> None:
    expected = generated_contract_resource_texts()

    assert len(expected) == 13
    assert set(expected) == GENERATED_RESOURCE_NAMES
    first_written = write_resources(tmp_path)
    first_bytes = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes() for path in first_written
    }
    second_written = write_resources(tmp_path)

    assert first_written == second_written
    assert set(first_bytes) == GENERATED_RESOURCE_NAMES
    for relative_name, expected_text in expected.items():
        actual = tmp_path.joinpath(*relative_name.split("/")).read_bytes()
        assert actual == first_bytes[relative_name] == expected_text.encode()
        assert actual.endswith(b"\n")
        assert b"\r\n" not in actual
        assert json.loads(actual)


def test_packaged_resource_inventory_is_exact_and_has_no_drift() -> None:
    assert _packaged_inventory() == GENERATED_RESOURCE_NAMES
    assert len(_packaged_inventory()) == 13
    validate_packaged_contract_resources()


def test_end_to_end_chain_reconciles_every_reference_and_model_output() -> None:
    dataset, split, preprocessing, configuration, run, model = build_example_contract_chain()

    assert_model_compatible(dataset, split, preprocessing, configuration, run, model)
    references = (
        (split.dataset, dataset),
        (configuration.dataset, dataset),
        (configuration.split, split),
        (configuration.preprocessing, preprocessing),
        (run.resolved_configuration, configuration),
        (run.dataset, dataset),
        (run.split, split),
        (run.preprocessing, preprocessing),
        (model.training_run, run),
        (model.resolved_configuration, configuration),
        (model.dataset, dataset),
        (model.split, split),
        (model.preprocessing, preprocessing),
    )
    for reference, upstream in references:
        assert same_contract_reference(
            reference,
            contract_reference(upstream, reference.locator),
        )
    assert run.status == "succeeded"
    assert any(same_artifact_reference(model.artifact, output) for output in run.outputs)
    assert preprocessing.compatible_runtimes == ("python",)


@pytest.mark.golden
def test_every_public_example_has_an_immutable_semantic_digest() -> None:
    examples = _chain_by_kind()

    assert PUBLISHED_EXAMPLE_CONTRACT_DIGESTS == EXPECTED_EXAMPLE_DIGESTS
    assert {
        kind: contract_digest(example) for kind, example in examples.items()
    } == EXPECTED_EXAMPLE_DIGESTS
    assert {
        kind: contract_digest(load_packaged_contract_example(kind)) for kind in CONTRACT_KINDS
    } == EXPECTED_EXAMPLE_DIGESTS


def test_public_examples_are_clearly_synthetic_and_contain_no_pii_or_machine_paths() -> None:
    dataset, split, preprocessing, configuration, run, model = build_example_contract_chain()
    rendered = "\n".join(
        text
        for name, text in generated_contract_resource_texts().items()
        if name.startswith("examples/")
    )

    top_level_ids = (
        dataset.dataset_id,
        split.split_id,
        preprocessing.plan_id,
        configuration.config_id,
        run.run_id,
        model.model_id,
    )
    assert all("synthetic" in identifier for identifier in top_level_ids)
    machine_path_pattern = r"(?i)(?:(?<![a-z])[a-z]:" + r"[\\/]|/" + r"(?:home|users)/)"
    assert not re.search(machine_path_pattern, rendered)
    for forbidden in (
        "sb128",
        "@example.",
        "participant_name",
        "email_address",
        "phone_number",
        "signed_by",
        "identity_vault",
        "file:" + chr(47) * 2,
        "localhost",
    ):
        assert forbidden.casefold() not in rendered.casefold()

    assert all(
        re.fullmatch(r"participant_[0-9a-f]{32}", sample.participant_id)
        for sample in dataset.content.samples
    )


def test_load_packaged_example_fails_closed_for_an_unknown_kind() -> None:
    with pytest.raises(ContractResourceError, match="unsupported"):
        load_packaged_contract_example("unknown")  # type: ignore[arg-type]


def test_example_filename_registry_matches_distribution_policy() -> None:
    assert CONTRACT_EXAMPLE_FILENAMES == {
        "dataset": "dataset-manifest.example.json",
        "split": "split-manifest.example.json",
        "preprocessing": "preprocessing-plan.example.json",
        "resolved_configuration": "resolved-configuration.example.json",
        "run": "run-record.example.json",
        "model": "model-manifest.example.json",
    }
