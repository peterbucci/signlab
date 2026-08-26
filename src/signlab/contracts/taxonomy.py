"""Authoritative taxonomy models, validation, and stable identity helpers."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from importlib.resources import files
from typing import Annotated, Any, Final, Literal, Self

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

SCHEMA_BASE: Final = "https://signlab.dev/schemas/"
APPROVED_PRODUCT_CLAIM: Final = (
    "SignLab is a research prototype for recognizing isolated performances of five "
    "predefined hand gestures within continuous webcam video. It separates non-target "
    "events (`other`) from no active event (`inactive`) and uncertain decisions "
    "(`abstain`); no sign-language or translation capability is claimed."
)
EXPECTED_TARGET_IDS: Final = ("hello", "no", "please", "thank_you", "yes")
EXPECTED_CLASS_IDS: Final = (*EXPECTED_TARGET_IDS, "other")
EXPECTED_CONSUMERS: Final = (
    "collection",
    "annotation",
    "training",
    "evaluation",
    "bundle",
    "public_copy",
)
EXPECTED_CONSUMER_SCHEMA_IDS: Final = {
    consumer: f"{SCHEMA_BASE}{consumer.replace('_', '-')}-taxonomy-binding-1.schema.json"
    for consumer in EXPECTED_CONSUMERS
}
EXPECTED_OTHER_KINDS: Final = (
    "partial_target",
    "transition_fragment",
    "oov_gesture",
    "incidental_activity",
    "two_hand_non_target",
)
PUBLISHED_TAXONOMY_DIGESTS: Final = {
    (
        "signlab-five",
        "1.0.0",
    ): "sha256:c0f6cbddfe43e3a6eb3de01dbbbbc1ceebcb83d50cc197999776f58e3d9ce20d",
}
PUBLISHED_TAXONOMY_RESOURCES: Final = {
    ("signlab-five", "1.0.0"): "signlab-five-1.0.0.json",
}
CURRENT_TAXONOMY: Final = ("signlab-five", "1.0.0")
BUILTIN_TAXONOMY_DIGEST: Final = PUBLISHED_TAXONOMY_DIGESTS[CURRENT_TAXONOMY]

NonEmptyText = Annotated[
    str,
    StringConstraints(min_length=1, pattern=r"^\S(?:.*\S)?$"),
]
StableId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]*$")]
SemanticVersion = Annotated[
    str,
    StringConstraints(pattern=r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"),
]
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
ConsumerName = Literal[
    "collection",
    "annotation",
    "training",
    "evaluation",
    "bundle",
    "public_copy",
]


def _normalize_json_integer(value: object) -> object:
    """Accept JSON Schema integral numbers while retaining strict non-coercion."""
    if isinstance(value, bool):
        raise ValueError("boolean values are not integers")
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


JsonNonNegativeInteger = Annotated[
    int,
    Field(ge=0),
    BeforeValidator(_normalize_json_integer),
]
JsonPositiveInteger = Annotated[
    int,
    Field(gt=0),
    BeforeValidator(_normalize_json_integer),
]
JsonZero = Annotated[Literal[0], BeforeValidator(_normalize_json_integer)]
JsonOne = Annotated[Literal[1], BeforeValidator(_normalize_json_integer)]
JsonTwo = Annotated[Literal[2], BeforeValidator(_normalize_json_integer)]
JsonThree = Annotated[Literal[3], BeforeValidator(_normalize_json_integer)]
JsonFour = Annotated[Literal[4], BeforeValidator(_normalize_json_integer)]
JsonFive = Annotated[Literal[5], BeforeValidator(_normalize_json_integer)]


class TaxonomyContractError(ValueError):
    """Raised when a taxonomy or one of its references violates the public contract."""


def _contract_config(schema_name: str) -> ConfigDict:
    return ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        json_schema_extra={"$id": f"{SCHEMA_BASE}{schema_name}"},
    )


class StrictFrozenModel(BaseModel):
    """Closed, immutable base for governance and taxonomy contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class GestureExamples(StrictFrozenModel):
    """Observable inclusion and boundary examples for one class."""

    positive: tuple[NonEmptyText, ...] = Field(min_length=1)
    negative: tuple[NonEmptyText, ...] = Field(min_length=1)
    ambiguous: tuple[NonEmptyText, ...] = Field(min_length=1)
    transition: tuple[NonEmptyText, ...] = Field(min_length=1)


class GestureDefinition(StrictFrozenModel):
    """One learned classifier output and its operational definition."""

    index: JsonNonNegativeInteger
    id: StableId
    display_name: NonEmptyText
    role: Literal["target", "learned_negative"]
    description: NonEmptyText
    examples: GestureExamples


class RuntimeConcept(StrictFrozenModel):
    """A state or outcome that must not be conflated with other pipeline layers."""

    id: Literal["inactive", "other", "abstain"]
    role: Literal["detector_state", "learned_class", "decision_outcome"]
    description: NonEmptyText


class AnnotationDisposition(StrictFrozenModel):
    """A non-class annotation disposition and its handling rule."""

    id: Literal["ambiguous", "ignore"]
    description: NonEmptyText


class TaxonomyRules(StrictFrozenModel):
    """Cross-cutting decisions for edge cases and event boundaries."""

    handedness: NonEmptyText
    second_hand: NonEmptyText
    partial_gesture: NonEmptyText
    out_of_vocabulary: NonEmptyText
    poor_segmentation: NonEmptyText
    transitions: NonEmptyText
    ignore_regions: NonEmptyText
    ambiguous_annotations: NonEmptyText


class LegacyAlias(StrictFrozenModel):
    """One explicit, reviewable legacy spelling migration."""

    source: NonEmptyText
    target: StableId


class LegacyImportPolicy(StrictFrozenModel):
    """Explicit migration behavior for historical labels."""

    aliases: tuple[LegacyAlias, ...]
    quarantined: tuple[NonEmptyText, ...] = Field(min_length=1)


class ProductClaim(StrictFrozenModel):
    """The only public claim authorized for the current reviewed taxonomy."""

    profile: Literal["predefined_gestures"]
    statement: NonEmptyText

    @model_validator(mode="after")
    def _enforce_claim_boundary(self) -> Self:
        if self.statement != APPROVED_PRODUCT_CLAIM:
            raise ValueError("the predefined-gesture claim must use the approved statement")
        return self


class ConsumerReferenceRule(StrictFrozenModel):
    """A domain contract that must embed an immutable taxonomy reference."""

    consumer: ConsumerName
    schema_id: Annotated[str, StringConstraints(pattern=r"^https://signlab\.dev/schemas/.+$")]
    required: Literal[True]


class GestureTaxonomy(StrictFrozenModel):
    """The authoritative, versioned definition of SignLab's learned vocabulary."""

    model_config = _contract_config("gesture-taxonomy-1.schema.json")

    schema_version: Literal["gesture-taxonomy/1"]
    taxonomy_id: Literal["signlab-five"]
    version: SemanticVersion
    claim: ProductClaim
    labels: tuple[GestureDefinition, ...]
    concepts: tuple[RuntimeConcept, ...]
    annotation_dispositions: tuple[AnnotationDisposition, ...]
    decision_precedence: tuple[Literal["inactive", "target", "other", "abstain"], ...]
    other_kinds: tuple[StableId, ...]
    rules: TaxonomyRules
    legacy_import: LegacyImportPolicy
    consumer_references: tuple[ConsumerReferenceRule, ...]

    @model_validator(mode="after")
    def _enforce_signlab_five_contract(self) -> Self:
        label_ids = tuple(label.id for label in self.labels)
        label_indices = tuple(label.index for label in self.labels)
        label_roles = tuple(label.role for label in self.labels)
        if label_ids != EXPECTED_CLASS_IDS:
            raise ValueError(f"classifier labels must be ordered exactly as {EXPECTED_CLASS_IDS}")
        if label_indices != tuple(range(len(EXPECTED_CLASS_IDS))):
            raise ValueError("classifier indices must be contiguous and ordered from zero")
        if label_roles != (*("target" for _ in EXPECTED_TARGET_IDS), "learned_negative"):
            raise ValueError("five targets must be followed by the learned negative class 'other'")

        concepts = {concept.id: concept.role for concept in self.concepts}
        expected_concepts = {
            "inactive": "detector_state",
            "other": "learned_class",
            "abstain": "decision_outcome",
        }
        if concepts != expected_concepts or len(self.concepts) != len(expected_concepts):
            raise ValueError("inactive, other, and abstain must retain distinct layer ownership")

        dispositions = tuple(disposition.id for disposition in self.annotation_dispositions)
        if dispositions != ("ambiguous", "ignore"):
            raise ValueError("annotation dispositions must be ordered as ambiguous, ignore")
        if self.decision_precedence != ("inactive", "target", "other", "abstain"):
            raise ValueError("decision precedence must be inactive, target, other, abstain")
        if self.other_kinds != EXPECTED_OTHER_KINDS:
            raise ValueError(f"other kinds must be ordered exactly as {EXPECTED_OTHER_KINDS}")

        aliases = {(alias.source, alias.target) for alias in self.legacy_import.aliases}
        if aliases != {("thank you", "thank_you")} or len(self.legacy_import.aliases) != 1:
            raise ValueError("the only v1 legacy alias must map 'thank you' to 'thank_you'")
        if self.legacy_import.quarantined != ("nothing",):
            raise ValueError("legacy label 'nothing' must remain quarantined")

        consumers = tuple(rule.consumer for rule in self.consumer_references)
        if consumers != EXPECTED_CONSUMERS:
            raise ValueError(f"consumer references must be ordered exactly as {EXPECTED_CONSUMERS}")
        schema_ids = {rule.consumer: rule.schema_id for rule in self.consumer_references}
        if schema_ids != EXPECTED_CONSUMER_SCHEMA_IDS:
            raise ValueError("each taxonomy consumer must use its published binding schema")
        return self


class TaxonomyRef(StrictFrozenModel):
    """Portable identity embedded in downstream manifests and bundles."""

    model_config = _contract_config("taxonomy-reference-1.schema.json")

    schema_version: Literal["taxonomy-reference/1"]
    id: StableId
    version: SemanticVersion
    sha256: Sha256Digest

    @model_validator(mode="after")
    def _resolve_published_identity(self) -> Self:
        expected_digest = PUBLISHED_TAXONOMY_DIGESTS.get((self.id, self.version))
        if expected_digest is None:
            raise ValueError(f"unsupported published taxonomy reference {self.id}@{self.version}")
        if self.sha256 != expected_digest:
            raise ValueError(
                f"taxonomy reference digest does not match published {self.id}@{self.version}"
            )
        return self


class TrainingLabelMap(StrictFrozenModel):
    """The immutable output order required before a training run can start."""

    hello: JsonZero
    no: JsonOne
    please: JsonTwo
    thank_you: JsonThree
    yes: JsonFour
    other: JsonFive


class TrainingLabelCounts(StrictFrozenModel):
    """Observed training counts, including a nonempty learned negative class."""

    hello: JsonPositiveInteger
    no: JsonPositiveInteger
    please: JsonPositiveInteger
    thank_you: JsonPositiveInteger
    yes: JsonPositiveInteger
    other: JsonPositiveInteger


class CollectionTaxonomyBinding(StrictFrozenModel):
    model_config = _contract_config("collection-taxonomy-binding-1.schema.json")
    schema_version: Literal["collection-taxonomy-binding/1"]
    taxonomy: TaxonomyRef


class AnnotationTaxonomyBinding(StrictFrozenModel):
    model_config = _contract_config("annotation-taxonomy-binding-1.schema.json")
    schema_version: Literal["annotation-taxonomy-binding/1"]
    taxonomy: TaxonomyRef


class TrainingTaxonomyBinding(StrictFrozenModel):
    model_config = _contract_config("training-taxonomy-binding-1.schema.json")
    schema_version: Literal["training-taxonomy-binding/1"]
    taxonomy: TaxonomyRef
    label_map: TrainingLabelMap
    label_counts: TrainingLabelCounts


class EvaluationTaxonomyBinding(StrictFrozenModel):
    model_config = _contract_config("evaluation-taxonomy-binding-1.schema.json")
    schema_version: Literal["evaluation-taxonomy-binding/1"]
    taxonomy: TaxonomyRef


class BundleTaxonomyBinding(StrictFrozenModel):
    model_config = _contract_config("bundle-taxonomy-binding-1.schema.json")
    schema_version: Literal["bundle-taxonomy-binding/1"]
    taxonomy: TaxonomyRef


class PublicCopyTaxonomyBinding(StrictFrozenModel):
    model_config = _contract_config("public-copy-taxonomy-binding-1.schema.json")
    schema_version: Literal["public-copy-taxonomy-binding/1"]
    taxonomy: TaxonomyRef


CONSUMER_BINDING_MODELS: Final[dict[str, type[BaseModel]]] = {
    "collection-taxonomy-binding-1.schema.json": CollectionTaxonomyBinding,
    "annotation-taxonomy-binding-1.schema.json": AnnotationTaxonomyBinding,
    "training-taxonomy-binding-1.schema.json": TrainingTaxonomyBinding,
    "evaluation-taxonomy-binding-1.schema.json": EvaluationTaxonomyBinding,
    "bundle-taxonomy-binding-1.schema.json": BundleTaxonomyBinding,
    "public-copy-taxonomy-binding-1.schema.json": PublicCopyTaxonomyBinding,
}

type JsonObject = dict[str, Any]
type TaxonomyInput = GestureTaxonomy | str | bytes | bytearray | Mapping[str, object]
type TrainingBindingInput = TrainingTaxonomyBinding | str | bytes | bytearray | Mapping[str, object]


def _validation_message(error: ValidationError) -> str:
    details: list[str] = []
    for item in error.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in item["loc"]) or "document"
        details.append(f"{location}: {item['msg']}")
    return "; ".join(details)


def validate_taxonomy(document: TaxonomyInput) -> GestureTaxonomy:
    """Validate JSON-compatible input without permitting Python-side coercion."""
    if isinstance(document, GestureTaxonomy):
        document = document.model_dump_json(round_trip=True)
    if isinstance(document, Mapping):
        try:
            document = json.dumps(document, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise TaxonomyContractError("taxonomy is not JSON-serializable") from error
    try:
        taxonomy = GestureTaxonomy.model_validate_json(document, strict=True)
    except ValidationError as error:
        raise TaxonomyContractError(
            f"invalid gesture taxonomy: {_validation_message(error)}"
        ) from error
    identity = (taxonomy.taxonomy_id, taxonomy.version)
    expected_digest = PUBLISHED_TAXONOMY_DIGESTS.get(identity)
    if expected_digest is None:
        supported = ", ".join(
            f"{taxonomy_id}@{version}"
            for taxonomy_id, version in sorted(PUBLISHED_TAXONOMY_DIGESTS)
        )
        raise TaxonomyContractError(
            f"unsupported taxonomy {taxonomy.taxonomy_id}@{taxonomy.version}; "
            f"supported releases: {supported}; publish a new immutable artifact to migrate"
        )
    if taxonomy_digest(taxonomy) != expected_digest:
        raise TaxonomyContractError(
            f"published taxonomy {taxonomy.taxonomy_id}@{taxonomy.version} is immutable; "
            "semantic changes require a new version and registered digest"
        )
    return taxonomy


def validate_training_taxonomy_binding(
    document: TrainingBindingInput,
) -> TrainingTaxonomyBinding:
    """Validate the taxonomy, exact label map, and observed counts for training."""
    if isinstance(document, TrainingTaxonomyBinding):
        document = document.model_dump_json(round_trip=True)
    if isinstance(document, Mapping):
        try:
            document = json.dumps(document, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise TaxonomyContractError(
                "training taxonomy input is not JSON-serializable"
            ) from error
    try:
        return TrainingTaxonomyBinding.model_validate_json(document, strict=True)
    except ValidationError as error:
        raise TaxonomyContractError(
            f"invalid training taxonomy input: {_validation_message(error)}"
        ) from error


def canonical_json_bytes(value: BaseModel | Mapping[str, object]) -> bytes:
    """Serialize validated public data into SignLab's deterministic JSON form."""
    payload: object
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json", round_trip=True)
    else:
        payload = dict(value)
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def taxonomy_digest(taxonomy: GestureTaxonomy) -> str:
    """Return the content identity of a validated taxonomy."""
    return f"sha256:{hashlib.sha256(canonical_json_bytes(taxonomy)).hexdigest()}"


def taxonomy_reference(taxonomy: GestureTaxonomy) -> TaxonomyRef:
    """Build the immutable reference required by every taxonomy consumer."""
    taxonomy = validate_taxonomy(taxonomy)
    return TaxonomyRef(
        schema_version="taxonomy-reference/1",
        id=taxonomy.taxonomy_id,
        version=taxonomy.version,
        sha256=taxonomy_digest(taxonomy),
    )


def load_builtin_taxonomy() -> GestureTaxonomy:
    """Load and integrity-check the taxonomy shipped inside the Python package."""
    resource_name = PUBLISHED_TAXONOMY_RESOURCES[CURRENT_TAXONOMY]
    document = files("signlab.resources.taxonomies").joinpath(resource_name).read_bytes()
    taxonomy = validate_taxonomy(document)
    actual_digest = taxonomy_digest(taxonomy)
    if (
        taxonomy.taxonomy_id,
        taxonomy.version,
    ) != CURRENT_TAXONOMY or actual_digest != BUILTIN_TAXONOMY_DIGEST:
        raise TaxonomyContractError(
            "built-in taxonomy content does not match its immutable published digest"
        )
    return taxonomy


def load_published_taxonomies() -> tuple[GestureTaxonomy, ...]:
    """Load every registered release so portable schemas remain backward compatible."""
    if set(PUBLISHED_TAXONOMY_RESOURCES) != set(PUBLISHED_TAXONOMY_DIGESTS):
        raise TaxonomyContractError("published taxonomy resource and digest registries disagree")
    loaded: list[GestureTaxonomy] = []
    resource_root = files("signlab.resources.taxonomies")
    for identity, resource_name in sorted(PUBLISHED_TAXONOMY_RESOURCES.items()):
        try:
            taxonomy = validate_taxonomy(resource_root.joinpath(resource_name).read_bytes())
        except OSError as error:
            raise TaxonomyContractError("a published taxonomy resource is missing") from error
        if (taxonomy.taxonomy_id, taxonomy.version) != identity:
            raise TaxonomyContractError("a published taxonomy resource has the wrong identity")
        loaded.append(taxonomy)
    return tuple(loaded)


def generated_json_schemas() -> dict[str, JsonObject]:
    """Generate every committed taxonomy schema from its authoritative model."""
    models: dict[str, type[BaseModel]] = {
        "gesture-taxonomy-1.schema.json": GestureTaxonomy,
        "taxonomy-reference-1.schema.json": TaxonomyRef,
        **CONSUMER_BINDING_MODELS,
    }
    generated: dict[str, JsonObject] = {}
    for filename, model in models.items():
        schema = model.model_json_schema(mode="validation")
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        generated[filename] = schema
    taxonomies = load_published_taxonomies()
    taxonomy_documents = [
        taxonomy.model_dump(mode="json", round_trip=True) for taxonomy in taxonomies
    ]
    reference_documents = [
        taxonomy_reference(taxonomy).model_dump(mode="json", round_trip=True)
        for taxonomy in taxonomies
    ]
    generated["gesture-taxonomy-1.schema.json"]["oneOf"] = [
        {"const": document} for document in taxonomy_documents
    ]
    generated["taxonomy-reference-1.schema.json"]["oneOf"] = [
        {"const": document} for document in reference_documents
    ]
    for filename in CONSUMER_BINDING_MODELS:
        taxonomy_definition = generated[filename]["$defs"]["TaxonomyRef"]
        taxonomy_definition["oneOf"] = [{"const": document} for document in reference_documents]
    return generated


def validate_packaged_taxonomy_resources() -> None:
    """Check that every packaged schema is valid and matches generated output."""
    expected = generated_json_schemas()
    schema_root = files("signlab.resources.schemas")
    try:
        for filename, generated in expected.items():
            packaged = json.loads(schema_root.joinpath(filename).read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(packaged)
            if packaged != generated:
                raise TaxonomyContractError(f"packaged schema drift detected for {filename}")
    except (OSError, json.JSONDecodeError, SchemaError) as error:
        raise TaxonomyContractError("packaged taxonomy schemas are missing or invalid") from error


def normalize_legacy_label(label: str) -> str:
    """Apply only the legacy migrations explicitly approved by the taxonomy."""
    taxonomy = load_builtin_taxonomy()
    if label in EXPECTED_CLASS_IDS:
        return label
    aliases = {alias.source: alias.target for alias in taxonomy.legacy_import.aliases}
    if label in aliases:
        return aliases[label]
    if label in taxonomy.legacy_import.quarantined:
        raise TaxonomyContractError(
            f"legacy label {label!r} is quarantined and must be reannotated; it is not 'other'"
        )
    raise TaxonomyContractError(f"unknown legacy label {label!r}; no implicit normalization exists")


def validate_training_label_map(label_map: Mapping[str, int]) -> tuple[str, ...]:
    """Reject training outputs that drift from the immutable classifier order."""
    invalid_values = sorted(label for label, index in label_map.items() if type(index) is not int)
    if invalid_values:
        raise TaxonomyContractError(f"label indices must be integers: {invalid_values}")
    if "other" not in label_map:
        raise TaxonomyContractError(
            "training label map is missing required learned negative class 'other'"
        )
    expected = {label: index for index, label in enumerate(EXPECTED_CLASS_IDS)}
    missing = sorted(set(expected) - set(label_map))
    unexpected = sorted(set(label_map) - set(expected))
    if missing or unexpected:
        raise TaxonomyContractError(
            "training label map does not match taxonomy; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if dict(label_map) != expected:
        raise TaxonomyContractError(f"training label indices must match {expected}")
    return EXPECTED_CLASS_IDS


def validate_training_label_counts(label_counts: Mapping[str, int]) -> None:
    """Require observed examples for every target and the learned negative class."""
    if "other" not in label_counts:
        raise TaxonomyContractError(
            "training data is missing required learned negative class 'other'"
        )
    missing = sorted(set(EXPECTED_CLASS_IDS) - set(label_counts))
    unexpected = sorted(set(label_counts) - set(EXPECTED_CLASS_IDS))
    if missing or unexpected:
        raise TaxonomyContractError(
            "training label counts do not match taxonomy; "
            f"missing={missing}, unexpected={unexpected}"
        )
    invalid = sorted(
        label for label, count in label_counts.items() if type(count) is not int or count < 0
    )
    if invalid:
        raise TaxonomyContractError(f"label counts must be non-negative integers: {invalid}")
    if label_counts["other"] == 0:
        raise TaxonomyContractError("learned negative class 'other' has zero training examples")
    empty_targets = sorted(label for label in EXPECTED_TARGET_IDS if label_counts[label] == 0)
    if empty_targets:
        raise TaxonomyContractError(f"target classes have zero training examples: {empty_targets}")
