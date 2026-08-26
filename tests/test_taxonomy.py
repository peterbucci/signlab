from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError

import signlab.contracts.taxonomy as taxonomy_contract
from signlab.contracts.taxonomy import (
    APPROVED_PRODUCT_CLAIM,
    BUILTIN_TAXONOMY_DIGEST,
    CONSUMER_BINDING_MODELS,
    EXPECTED_CLASS_IDS,
    EXPECTED_CONSUMERS,
    EXPECTED_OTHER_KINDS,
    EXPECTED_TARGET_IDS,
    TaxonomyContractError,
    TaxonomyRef,
    canonical_json_bytes,
    generated_json_schemas,
    load_builtin_taxonomy,
    normalize_legacy_label,
    taxonomy_digest,
    taxonomy_reference,
    validate_packaged_taxonomy_resources,
    validate_taxonomy,
    validate_training_label_counts,
    validate_training_label_map,
    validate_training_taxonomy_binding,
)

EXPECTED_DIGEST = "sha256:" + "c0f6cbddfe43e3a6eb3de01dbbbbc1ceebcb83d50cc197999776f58e3d9ce20d"
BINDING_VERSIONS = {
    "collection-taxonomy-binding-1.schema.json": "collection-taxonomy-binding/1",
    "annotation-taxonomy-binding-1.schema.json": "annotation-taxonomy-binding/1",
    "training-taxonomy-binding-1.schema.json": "training-taxonomy-binding/1",
    "evaluation-taxonomy-binding-1.schema.json": "evaluation-taxonomy-binding/1",
    "bundle-taxonomy-binding-1.schema.json": "bundle-taxonomy-binding/1",
    "public-copy-taxonomy-binding-1.schema.json": "public-copy-taxonomy-binding/1",
}


def _document() -> dict[str, Any]:
    return load_builtin_taxonomy().model_dump(mode="json", round_trip=True)


def _training_map() -> dict[str, int]:
    return {label: index for index, label in enumerate(EXPECTED_CLASS_IDS)}


def _training_counts() -> dict[str, int]:
    return dict.fromkeys(EXPECTED_CLASS_IDS, 3)


def _training_binding_document() -> dict[str, Any]:
    return {
        "schema_version": "training-taxonomy-binding/1",
        "taxonomy": taxonomy_reference(load_builtin_taxonomy()).model_dump(mode="json"),
        "label_map": _training_map(),
        "label_counts": _training_counts(),
    }


def test_builtin_taxonomy_has_stable_identity_and_output_order() -> None:
    taxonomy = load_builtin_taxonomy()

    assert taxonomy.taxonomy_id == "signlab-five"
    assert taxonomy.version == "1.0.0"
    assert tuple(label.id for label in taxonomy.labels) == EXPECTED_CLASS_IDS
    assert tuple(label.index for label in taxonomy.labels) == tuple(range(6))
    assert BUILTIN_TAXONOMY_DIGEST == EXPECTED_DIGEST
    assert taxonomy_digest(taxonomy) == EXPECTED_DIGEST
    assert taxonomy_reference(taxonomy).model_dump(mode="json") == {
        "schema_version": "taxonomy-reference/1",
        "id": "signlab-five",
        "version": "1.0.0",
        "sha256": EXPECTED_DIGEST,
    }


def test_every_target_has_all_operational_example_categories() -> None:
    taxonomy = load_builtin_taxonomy()

    targets = tuple(label for label in taxonomy.labels if label.role == "target")
    assert tuple(label.id for label in targets) == EXPECTED_TARGET_IDS
    for target in targets:
        assert target.examples.positive
        assert target.examples.negative
        assert target.examples.ambiguous
        assert target.examples.transition


def test_runtime_layers_annotation_states_and_edge_rules_are_explicit() -> None:
    taxonomy = load_builtin_taxonomy()

    assert {concept.id: concept.role for concept in taxonomy.concepts} == {
        "inactive": "detector_state",
        "other": "learned_class",
        "abstain": "decision_outcome",
    }
    assert tuple(item.id for item in taxonomy.annotation_dispositions) == (
        "ambiguous",
        "ignore",
    )
    assert taxonomy.decision_precedence == ("inactive", "target", "other", "abstain")
    assert taxonomy.other_kinds == EXPECTED_OTHER_KINDS
    assert all(
        getattr(taxonomy.rules, field)
        for field in (
            "handedness",
            "second_hand",
            "partial_gesture",
            "out_of_vocabulary",
            "poor_segmentation",
            "transitions",
            "ignore_regions",
            "ambiguous_annotations",
        )
    )


def test_default_claim_is_bounded_and_has_no_named_language_attribution() -> None:
    claim = load_builtin_taxonomy().claim

    assert claim.profile == "predefined_gestures"
    assert claim.statement == APPROVED_PRODUCT_CLAIM
    assert "five predefined hand gestures" in claim.statement
    assert "named_language_evidence" not in claim.model_dump(mode="json")


def test_v1_rejects_every_named_language_claim_until_human_review() -> None:
    document = _document()
    document["claim"] = {
        "profile": "validated_named_language",
        "statement": "An unsupported named-language claim.",
        "named_language_evidence": {
            "reviewer_role": "self-asserted fixture",
            "evidence_digest": "sha256:" + "a" * 64,
        },
    }

    with pytest.raises(TaxonomyContractError, match="predefined_gestures"):
        validate_taxonomy(document)


def test_predefined_claim_cannot_be_silently_reworded_or_carry_evidence() -> None:
    document = _document()
    claim = cast(dict[str, Any], document["claim"])
    claim["statement"] = "A broader unsupported claim."

    with pytest.raises(TaxonomyContractError, match="approved statement"):
        validate_taxonomy(document)

    document = _document()
    claim = cast(dict[str, Any], document["claim"])
    claim["named_language_evidence"] = {"reviewer_role": "fixture"}
    with pytest.raises(TaxonomyContractError, match="Extra inputs are not permitted"):
        validate_taxonomy(document)


def test_taxonomy_models_and_json_schemas_are_closed() -> None:
    taxonomy = load_builtin_taxonomy()
    with pytest.raises(ValidationError, match="frozen"):
        taxonomy.version = "2.0.0"

    document = _document()
    document["unexpected"] = True
    with pytest.raises(TaxonomyContractError, match="Extra inputs are not permitted"):
        validate_taxonomy(document)

    schema = generated_json_schemas()["gesture-taxonomy-1.schema.json"]
    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(schema).validate(document)


def test_public_validator_rechecks_unsafe_model_copies() -> None:
    taxonomy = load_builtin_taxonomy()
    unsafe = taxonomy.model_copy(update={"labels": taxonomy.labels[:-1]})

    with pytest.raises(TaxonomyContractError, match="classifier labels must be ordered"):
        validate_taxonomy(unsafe)
    with pytest.raises(TaxonomyContractError, match="classifier labels must be ordered"):
        taxonomy_reference(unsafe)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda document: document["labels"].pop(), "classifier labels must be ordered"),
        (
            lambda document: document["labels"][0].update({"index": 1}),
            "classifier indices must be contiguous",
        ),
        (
            lambda document: document["concepts"][0].update({"role": "learned_class"}),
            "distinct layer ownership",
        ),
        (
            lambda document: document["legacy_import"].update({"quarantined": []}),
            "at least 1 item",
        ),
        (
            lambda document: document["consumer_references"].pop(),
            "consumer references must be ordered",
        ),
        (
            lambda document: document["consumer_references"][0].update(
                {
                    "schema_id": (
                        "https://signlab.dev/schemas/annotation-taxonomy-binding-1.schema.json"
                    )
                }
            ),
            "published binding schema",
        ),
    ],
)
def test_cross_field_invariants_reject_semantic_drift(
    mutation: Any,
    message: str,
) -> None:
    document = _document()
    mutation(document)

    with pytest.raises(TaxonomyContractError, match=message):
        validate_taxonomy(document)


def test_generated_schemas_are_valid_and_match_packaged_documents() -> None:
    generated = generated_json_schemas()
    schema_root = files("signlab.resources.schemas")

    assert set(generated) == {
        "gesture-taxonomy-1.schema.json",
        "taxonomy-reference-1.schema.json",
        *BINDING_VERSIONS,
    }
    for filename, schema in generated.items():
        Draft202012Validator.check_schema(schema)
        packaged = json.loads(schema_root.joinpath(filename).read_text(encoding="utf-8"))
        assert packaged == schema


def test_generated_schemas_validate_taxonomy_reference_and_all_six_bindings() -> None:
    taxonomy = load_builtin_taxonomy()
    reference = taxonomy_reference(taxonomy)
    schemas = generated_json_schemas()

    Draft202012Validator(schemas["gesture-taxonomy-1.schema.json"]).validate(
        taxonomy.model_dump(mode="json")
    )
    Draft202012Validator(schemas["taxonomy-reference-1.schema.json"]).validate(
        reference.model_dump(mode="json")
    )
    for filename, model in CONSUMER_BINDING_MODELS.items():
        fields: dict[str, Any] = {
            "schema_version": BINDING_VERSIONS[filename],
            "taxonomy": reference,
        }
        if filename == "training-taxonomy-binding-1.schema.json":
            fields.update(label_map=_training_map(), label_counts=_training_counts())
        binding = model(
            **fields,
        )
        payload = binding.model_dump(mode="json")
        Draft202012Validator(schemas[filename]).validate(payload)
        assert payload["taxonomy"] == reference.model_dump(mode="json")

    assert tuple(rule.consumer for rule in taxonomy.consumer_references) == EXPECTED_CONSUMERS


def test_published_reference_and_every_binding_reject_unknown_identity() -> None:
    reference = taxonomy_reference(load_builtin_taxonomy()).model_dump(mode="json")
    unknown = {**reference, "id": "unreviewed", "version": "9.9.9"}

    with pytest.raises(ValidationError, match="unsupported published taxonomy reference"):
        TaxonomyRef.model_validate(unknown, strict=True)

    collision = {**reference, "sha256": "sha256:" + "0" * 64}
    with pytest.raises(ValidationError, match="does not match published"):
        TaxonomyRef.model_validate(collision, strict=True)

    schemas = generated_json_schemas()
    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(schemas["taxonomy-reference-1.schema.json"]).validate(unknown)
    for filename in BINDING_VERSIONS:
        payload: dict[str, Any] = {
            "schema_version": BINDING_VERSIONS[filename],
            "taxonomy": unknown,
        }
        if filename == "training-taxonomy-binding-1.schema.json":
            payload.update(label_map=_training_map(), label_counts=_training_counts())
        with pytest.raises(JsonSchemaValidationError):
            Draft202012Validator(schemas[filename]).validate(payload)
    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(schemas["taxonomy-reference-1.schema.json"]).validate(collision)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document["labels"].pop(),
        lambda document: document["claim"].update({"statement": "An unsupported claim."}),
        lambda document: document["concepts"][0].update({"role": "learned_class"}),
        lambda document: document["legacy_import"]["aliases"][0].update({"target": "other"}),
    ],
)
def test_portable_schema_rejects_every_reviewed_semantic_invariant(mutation: Any) -> None:
    document = _document()
    mutation(document)
    schema = generated_json_schemas()["gesture-taxonomy-1.schema.json"]

    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(schema).validate(document)


def test_published_version_cannot_be_reused_for_changed_content() -> None:
    document = _document()
    document["labels"][0]["description"] = "A semantically changed definition."

    with pytest.raises(TaxonomyContractError, match="is immutable"):
        validate_taxonomy(document)

    document = _document()
    document["version"] = "1.0.1"
    with pytest.raises(TaxonomyContractError, match=r"unsupported taxonomy.*publish a new"):
        validate_taxonomy(document)


def test_python_and_json_schema_share_numeric_and_whitespace_acceptance() -> None:
    taxonomy_document = _document()
    taxonomy_document["labels"][0]["index"] = 0.0
    taxonomy_schema = generated_json_schemas()["gesture-taxonomy-1.schema.json"]

    validated = validate_taxonomy(taxonomy_document)
    assert validated.labels[0].index == 0
    Draft202012Validator(taxonomy_schema).validate(taxonomy_document)

    padded = _document()
    padded["labels"][0]["description"] = f" {padded['labels'][0]['description']}"
    with pytest.raises(TaxonomyContractError, match="String should match pattern"):
        validate_taxonomy(padded)
    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(taxonomy_schema).validate(padded)

    training_document = _training_binding_document()
    training_document["label_map"]["hello"] = 0.0
    training_document["label_counts"]["other"] = 2.0
    training_schema = generated_json_schemas()["training-taxonomy-binding-1.schema.json"]
    training = validate_training_taxonomy_binding(training_document)
    assert training.label_map.hello == 0
    assert training.label_counts.other == 2
    Draft202012Validator(training_schema).validate(training_document)

    for invalid_integer in (False, "0"):
        invalid = _training_binding_document()
        invalid["label_map"]["hello"] = invalid_integer
        with pytest.raises(TaxonomyContractError):
            validate_training_taxonomy_binding(invalid)
        with pytest.raises(JsonSchemaValidationError):
            Draft202012Validator(training_schema).validate(invalid)


def test_generated_schemas_keep_every_registered_release_backward_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    version_one = load_builtin_taxonomy()
    version_two_unchecked = version_one.model_copy(update={"version": "1.0.1"})
    version_two_digest = taxonomy_digest(version_two_unchecked)
    monkeypatch.setattr(
        taxonomy_contract,
        "PUBLISHED_TAXONOMY_DIGESTS",
        {
            ("signlab-five", "1.0.0"): BUILTIN_TAXONOMY_DIGEST,
            ("signlab-five", "1.0.1"): version_two_digest,
        },
    )
    version_two = validate_taxonomy(version_two_unchecked)
    monkeypatch.setattr(
        taxonomy_contract,
        "load_published_taxonomies",
        lambda: (version_one, version_two),
    )

    schemas = generated_json_schemas()
    assert len(schemas["gesture-taxonomy-1.schema.json"]["oneOf"]) == 2
    assert len(schemas["taxonomy-reference-1.schema.json"]["oneOf"]) == 2
    for taxonomy in (version_one, version_two):
        reference = taxonomy_reference(taxonomy)
        Draft202012Validator(schemas["gesture-taxonomy-1.schema.json"]).validate(
            taxonomy.model_dump(mode="json")
        )
        Draft202012Validator(schemas["taxonomy-reference-1.schema.json"]).validate(
            reference.model_dump(mode="json")
        )
        Draft202012Validator(schemas["collection-taxonomy-binding-1.schema.json"]).validate(
            {
                "schema_version": "collection-taxonomy-binding/1",
                "taxonomy": reference.model_dump(mode="json"),
            }
        )


def test_canonical_serialization_is_deterministic_and_rejects_non_json_values() -> None:
    taxonomy = load_builtin_taxonomy()
    first = canonical_json_bytes(taxonomy)
    second = canonical_json_bytes(validate_taxonomy(first))

    assert first == second
    assert first == canonical_json_bytes(json.loads(first))
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json_bytes({"invalid": float("nan")})
    with pytest.raises(TaxonomyContractError, match="not JSON-serializable"):
        validate_taxonomy({"invalid": Path("not-json")})


def test_legacy_normalization_is_explicit_and_nothing_stays_quarantined() -> None:
    assert normalize_legacy_label("thank you") == "thank_you"
    assert normalize_legacy_label("thank_you") == "thank_you"

    with pytest.raises(TaxonomyContractError, match="must be reannotated"):
        normalize_legacy_label("nothing")
    with pytest.raises(TaxonomyContractError, match="no implicit normalization"):
        normalize_legacy_label("Thank You")


def test_training_label_map_requires_exact_six_output_order() -> None:
    assert validate_training_label_map(_training_map()) == EXPECTED_CLASS_IDS

    target_only = {label: index for index, label in enumerate(EXPECTED_TARGET_IDS)}
    with pytest.raises(TaxonomyContractError, match="required learned negative class 'other'"):
        validate_training_label_map(target_only)

    reordered = _training_map()
    reordered["hello"], reordered["no"] = reordered["no"], reordered["hello"]
    with pytest.raises(TaxonomyContractError, match="indices must match"):
        validate_training_label_map(reordered)

    invalid_type = _training_map()
    invalid_type["hello"] = cast(int, False)
    with pytest.raises(TaxonomyContractError, match="indices must be integers"):
        validate_training_label_map(invalid_type)


@pytest.mark.parametrize(
    "forbidden_label",
    ["inactive", "abstain", "ambiguous", "ignore", "nothing", "thank you"],
)
def test_non_classifier_tokens_are_rejected_from_training(forbidden_label: str) -> None:
    label_map = _training_map()
    label_map[forbidden_label] = 6

    with pytest.raises(TaxonomyContractError, match="unexpected"):
        validate_training_label_map(label_map)


def test_training_counts_require_observed_targets_and_other_examples() -> None:
    validate_training_label_counts(_training_counts())

    without_other = _training_counts()
    del without_other["other"]
    with pytest.raises(TaxonomyContractError, match="required learned negative class 'other'"):
        validate_training_label_counts(without_other)

    empty_other = _training_counts()
    empty_other["other"] = 0
    with pytest.raises(TaxonomyContractError, match="zero training examples"):
        validate_training_label_counts(empty_other)

    empty_target = _training_counts()
    empty_target["hello"] = 0
    with pytest.raises(TaxonomyContractError, match="target classes have zero"):
        validate_training_label_counts(empty_target)

    invalid = _training_counts()
    invalid["hello"] = -1
    with pytest.raises(TaxonomyContractError, match="non-negative integers"):
        validate_training_label_counts(invalid)


def test_training_binding_requires_exact_labels_and_positive_observed_counts() -> None:
    binding = validate_training_taxonomy_binding(_training_binding_document())
    assert binding.label_map.other == 5
    assert binding.label_counts.other == 3

    without_other = _training_binding_document()
    del without_other["label_map"]["other"]
    with pytest.raises(TaxonomyContractError, match=r"label_map\.other: Field required"):
        validate_training_taxonomy_binding(without_other)

    empty_other = _training_binding_document()
    empty_other["label_counts"]["other"] = 0
    with pytest.raises(TaxonomyContractError, match=r"label_counts\.other"):
        validate_training_taxonomy_binding(empty_other)

    schema = generated_json_schemas()["training-taxonomy-binding-1.schema.json"]
    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(schema).validate(without_other)
    with pytest.raises(JsonSchemaValidationError):
        Draft202012Validator(schema).validate(empty_other)


def test_every_packaged_schema_is_readable_valid_and_current() -> None:
    validate_packaged_taxonomy_resources()


def test_docs_publish_the_same_claim_and_do_not_call_it_five_class() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    public_copy = "\n".join(
        (repository_root / path).read_text(encoding="utf-8")
        for path in ("README.md", "docs/PROJECT_CHARTER.md", "docs/gesture-taxonomy.md")
    )

    normalized_claim = " ".join(APPROVED_PRODUCT_CLAIM.split())
    for path in ("README.md", "docs/PROJECT_CHARTER.md", "docs/gesture-taxonomy.md"):
        content = (repository_root / path).read_text(encoding="utf-8").replace("> ", "")
        assert normalized_claim in " ".join(content.split())
    assert "five-class" not in public_copy.casefold()
