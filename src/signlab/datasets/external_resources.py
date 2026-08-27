"""Reviewed registry, label selection, and schemas for licensed external data."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from importlib.resources import files
from importlib.resources.abc import Traversable
from types import MappingProxyType
from typing import Final

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel

from signlab.contracts.external_dataset import (
    ExternalAcquisitionPlanV1,
    ExternalDatasetManifestV1,
    ExternalDatasetSelectionV1,
    ExternalLabelMappingV1,
    ExternalLicenseV1,
    ExternalResourceRefV1,
    ExternalTargetLabel,
    LicensedDatasetSourceV1,
    SourceLabel,
    external_dataset_selection_digest,
    licensed_dataset_source_digest,
    validate_external_dataset_selection,
    validate_licensed_dataset_source,
)
from signlab.contracts.taxonomy import load_builtin_taxonomy, taxonomy_reference

RESOURCE_PACKAGE: Final = "signlab.resources.external_datasets"
POPSIGN_SOURCE_FILENAME: Final = "popsign-asl-1.0.0.json"
POPSIGN_SELECTION_FILENAME: Final = "signlab-five-popsign-1.0.0.json"
POPSIGN_SOURCE_IDENTITY: Final = ("popsign-asl", "1.0.0")
POPSIGN_SELECTION_IDENTITY: Final = ("signlab-five-popsign", "1.0.0")

EXTERNAL_SCHEMA_MODELS: Final[Mapping[str, type[BaseModel]]] = MappingProxyType(
    {
        "external-acquisition-plan-1.schema.json": ExternalAcquisitionPlanV1,
        "external-dataset-manifest-1.schema.json": ExternalDatasetManifestV1,
        "external-dataset-selection-1.schema.json": ExternalDatasetSelectionV1,
        "licensed-dataset-source-1.schema.json": LicensedDatasetSourceV1,
    }
)
GENERATED_EXTERNAL_DATASET_RESOURCE_NAMES: Final = frozenset(
    {
        f"registry/{POPSIGN_SOURCE_FILENAME}",
        f"selections/{POPSIGN_SELECTION_FILENAME}",
        *(f"schemas/{filename}" for filename in EXTERNAL_SCHEMA_MODELS),
    }
)

# Frozen after initial review. These hash the exact pretty-printed package bytes.
PUBLISHED_EXTERNAL_DATASET_RESOURCE_DIGESTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "registry/popsign-asl-1.0.0.json": (
            "sha256:d6f3ce8a87f5940560c16ff60dfa883ec767f964077878dc114746b323c08683"
        ),
        "schemas/external-acquisition-plan-1.schema.json": (
            "sha256:8f7d0e568e387469b37599bfb4ebe4d816c568f97d0f21e05c8a998042f73a8b"
        ),
        "schemas/external-dataset-manifest-1.schema.json": (
            "sha256:b1433d29e2db5bec8f3d68b64fd7358b17044276a8cc4419b62dab41c2c5869e"
        ),
        "schemas/external-dataset-selection-1.schema.json": (
            "sha256:facb01c0460d6b5241137cc605d3898f378cb80910aea01ac0f63f11987dbd7e"
        ),
        "schemas/licensed-dataset-source-1.schema.json": (
            "sha256:a669f954202fcef671610f61760487820a1a9efa41ee1a142b9afa75b7b538f3"
        ),
        "selections/signlab-five-popsign-1.0.0.json": (
            "sha256:01326455ceeb5f26071bada3a6fd72b07fdc4e9be97fd1c1b34e265c3f655a1b"
        ),
    }
)
PUBLISHED_POPSIGN_SOURCE_SEMANTIC_DIGEST: Final = (
    "sha256:48f47f05a64d519abc0d1b3d089810000a0f9378f2fc9702d8ee365c2f78bce9"
)
PUBLISHED_POPSIGN_SELECTION_SEMANTIC_DIGEST: Final = (
    "sha256:39496395df08fa66c0b6a77d39425fcca1259f8075566bdd4c15fde39b93fc0e"
)

_SCHEMA_BASE: Final = "https://signlab.dev/schemas/"
_SCHEMA_COMMENT: Final = (
    "This JSON Schema enforces portable structure. SignLab's strict external-data "
    "validators remain authoritative for canonical hashes, immutable resource "
    "bindings, label mappings, archive ordering, signer split isolation, license "
    "boundaries, and artifact-byte verification."
)


class ExternalDatasetResourceError(ValueError):
    """Raised when packaged external-data resources are missing, stale, or invalid."""


def build_popsign_source() -> LicensedDatasetSourceV1:
    """Build the reviewed PopSign ASL v1.0 source record."""

    return LicensedDatasetSourceV1(
        schema_version="licensed-dataset-source/1",
        source_id="popsign-asl",
        version="1.0.0",
        title="PopSign ASL v1.0",
        publishers=(
            "Deaf Professional Arts Network",
            "Georgia Institute of Technology",
        ),
        dataset_url=("https://signdata.cc.gatech.edu/view/datasets/popsign_v1_0/index.html"),
        download_guide_url=(
            "https://signdata.cc.gatech.edu/view/guides/downloading_popsign/index.html"
        ),
        download_id="popsign_v1_0",
        download_url_template=(
            "https://signdata.cc.gatech.edu/data/{download_id}/{category}/{split}/"
            "{source_label}.tar"
        ),
        license=ExternalLicenseV1(
            schema_version="external-license/1",
            license_id="CC-BY-4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            attribution_text=(
                "PopSign ASL v1.0, Georgia Institute of Technology and Deaf "
                "Professional Arts Network, licensed under CC BY 4.0."
            ),
            attribution_required=True,
            change_notice_required=True,
            redistribution_permitted=True,
        ),
        categories=("game", "non-game"),
        splits=("train", "val", "test"),
        total_videos=200_686,
        total_signs=250,
        total_signers=47,
        contains_identifiable_human_video=True,
        provider_reports_participant_consent=True,
        signlab_participant_consent_applicable=False,
        publisher_checksums_available=False,
        website_preview_media_permitted=False,
        suitable_uses=(
            "isolated_sign_recognition",
            "mobile_isolated_sign_recognition",
        ),
        unsuitable_uses=(
            "continuous_sign_recognition",
            "sign_language_translation",
        ),
    )


def build_signlab_five_popsign_selection() -> ExternalDatasetSelectionV1:
    """Build the reviewed five-target PopSign selection without a language claim."""

    mapping_pairs: tuple[tuple[ExternalTargetLabel, SourceLabel], ...] = (
        ("hello", "hello"),
        ("no", "no"),
        ("please", "please"),
        ("thank_you", "thankyou"),
        ("yes", "yes"),
    )
    mappings = tuple(
        ExternalLabelMappingV1(
            schema_version="external-label-mapping/1",
            source_label=source_label,
            target_label_id=target_label,
            review_status="reviewed_gloss_alignment",
            language_equivalence_claimed=False,
        )
        for target_label, source_label in mapping_pairs
    )
    return ExternalDatasetSelectionV1(
        schema_version="external-dataset-selection/1",
        selection_id="signlab-five-popsign",
        version="1.0.0",
        source_id="popsign-asl",
        source_version="1.0.0",
        taxonomy=taxonomy_reference(load_builtin_taxonomy()),
        category="game",
        splits=("train", "val", "test"),
        mappings=mappings,
        learned_negative_included=False,
        claim_scope="signlab_predefined_gestures_only",
    )


def external_resource_reference(
    resource: LicensedDatasetSourceV1 | ExternalDatasetSelectionV1,
) -> ExternalResourceRefV1:
    """Create a digest-bound reference for one reviewed package resource."""

    if isinstance(resource, LicensedDatasetSourceV1):
        return ExternalResourceRefV1(
            schema_version="external-resource-reference/1",
            resource_kind="source",
            resource_id=resource.source_id,
            version=resource.version,
            sha256=licensed_dataset_source_digest(resource),
        )
    return ExternalResourceRefV1(
        schema_version="external-resource-reference/1",
        resource_kind="selection",
        resource_id=resource.selection_id,
        version=resource.version,
        sha256=external_dataset_selection_digest(resource),
    )


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


def _harden_arrays(schema: dict[str, object]) -> None:
    for node in _schema_nodes(schema):
        properties = node.get("properties")
        if not isinstance(properties, dict):
            continue
        for field_name in (
            "archives",
            "categories",
            "mappings",
            "media",
            "publishers",
            "suitable_uses",
            "unsuitable_uses",
        ):
            field_schema = properties.get(field_name)
            if isinstance(field_schema, dict) and field_schema.get("type") == "array":
                field_schema["uniqueItems"] = True


def generated_external_dataset_schemas() -> dict[str, dict[str, object]]:
    """Return standalone Draft 2020-12 schemas for each public root contract."""

    generated: dict[str, dict[str, object]] = {}
    for filename, model in EXTERNAL_SCHEMA_MODELS.items():
        schema = model.model_json_schema(mode="validation")
        _strip_nested_schema_ids(schema, root=True)
        schema["$id"] = f"{_SCHEMA_BASE}{filename}"
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$comment"] = _SCHEMA_COMMENT
        _harden_arrays(schema)
        Draft202012Validator.check_schema(schema)
        generated[filename] = schema
    return generated


def render_external_dataset_json(value: BaseModel | Mapping[str, object]) -> str:
    """Render one reviewable external-data resource with deterministic bytes."""

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
        raise ExternalDatasetResourceError(
            "external dataset resource is not JSON-serializable"
        ) from error


def generated_external_dataset_resource_texts() -> dict[str, str]:
    """Render the complete, exact package resource inventory."""

    rendered = {
        f"registry/{POPSIGN_SOURCE_FILENAME}": render_external_dataset_json(build_popsign_source()),
        f"selections/{POPSIGN_SELECTION_FILENAME}": render_external_dataset_json(
            build_signlab_five_popsign_selection()
        ),
        **{
            f"schemas/{filename}": render_external_dataset_json(schema)
            for filename, schema in generated_external_dataset_schemas().items()
        },
    }
    if set(rendered) != GENERATED_EXTERNAL_DATASET_RESOURCE_NAMES:
        raise ExternalDatasetResourceError("external dataset resource registry is incomplete")
    return rendered


def _resource_bytes(relative_name: str) -> bytes:
    try:
        return files(RESOURCE_PACKAGE).joinpath(*relative_name.split("/")).read_bytes()
    except OSError as error:
        raise ExternalDatasetResourceError("an external dataset resource is missing") from error


def _packaged_inventory() -> set[str]:
    root = files(RESOURCE_PACKAGE)
    inventory: set[str] = set()

    def visit(directory: Traversable, prefix: str = "") -> None:
        for child in directory.iterdir():
            relative = f"{prefix}/{child.name}" if prefix else child.name
            if child.is_dir():
                if child.name != "__pycache__":
                    visit(child, relative)
            elif child.name != "__init__.py" and not child.name.endswith(".pyc"):
                inventory.add(relative)

    visit(root)
    return inventory


def load_popsign_source() -> LicensedDatasetSourceV1:
    """Load and verify the immutable packaged PopSign source record."""

    resource_name = f"registry/{POPSIGN_SOURCE_FILENAME}"
    try:
        checked = validate_licensed_dataset_source(_resource_bytes(resource_name))
    except (OSError, TypeError, ValueError) as error:
        raise ExternalDatasetResourceError("the PopSign source is invalid") from error
    if (checked.source_id, checked.version) != POPSIGN_SOURCE_IDENTITY:
        raise ExternalDatasetResourceError("the PopSign source has the wrong identity")
    if licensed_dataset_source_digest(checked) != PUBLISHED_POPSIGN_SOURCE_SEMANTIC_DIGEST:
        raise ExternalDatasetResourceError("the PopSign source changed in place")
    return checked


def load_signlab_five_popsign_selection() -> ExternalDatasetSelectionV1:
    """Load and verify the immutable packaged five-target PopSign selection."""

    resource_name = f"selections/{POPSIGN_SELECTION_FILENAME}"
    try:
        checked = validate_external_dataset_selection(_resource_bytes(resource_name))
    except (OSError, TypeError, ValueError) as error:
        raise ExternalDatasetResourceError("the PopSign selection is invalid") from error
    if (checked.selection_id, checked.version) != POPSIGN_SELECTION_IDENTITY:
        raise ExternalDatasetResourceError("the PopSign selection has the wrong identity")
    if external_dataset_selection_digest(checked) != PUBLISHED_POPSIGN_SELECTION_SEMANTIC_DIGEST:
        raise ExternalDatasetResourceError("the PopSign selection changed in place")
    return checked


def validate_packaged_external_dataset_resources() -> None:
    """Check exact inventory, generated bytes, schemas, and publication hashes."""

    try:
        if _packaged_inventory() != GENERATED_EXTERNAL_DATASET_RESOURCE_NAMES:
            raise ExternalDatasetResourceError(
                "packaged external dataset resource inventory is not exact"
            )
        if set(PUBLISHED_EXTERNAL_DATASET_RESOURCE_DIGESTS) != (
            GENERATED_EXTERNAL_DATASET_RESOURCE_NAMES
        ):
            raise ExternalDatasetResourceError(
                "external dataset publication baselines are incomplete"
            )
        generated = generated_external_dataset_resource_texts()
        for relative_name, expected in generated.items():
            captured = _resource_bytes(relative_name)
            if captured != expected.encode("utf-8"):
                raise ExternalDatasetResourceError("packaged external dataset resource drift")
            digest = f"sha256:{hashlib.sha256(captured).hexdigest()}"
            if digest != PUBLISHED_EXTERNAL_DATASET_RESOURCE_DIGESTS[relative_name]:
                raise ExternalDatasetResourceError(
                    "a published external dataset resource changed in place"
                )
        load_popsign_source()
        selection = load_signlab_five_popsign_selection()
        if (
            selection.source_id,
            selection.source_version,
        ) != POPSIGN_SOURCE_IDENTITY:
            raise ExternalDatasetResourceError("the PopSign selection references another source")
    except ExternalDatasetResourceError:
        raise
    except (OSError, TypeError, ValueError, SchemaError) as error:
        raise ExternalDatasetResourceError(
            "packaged external dataset resources are invalid"
        ) from error


__all__ = [
    "EXTERNAL_SCHEMA_MODELS",
    "GENERATED_EXTERNAL_DATASET_RESOURCE_NAMES",
    "POPSIGN_SELECTION_FILENAME",
    "POPSIGN_SELECTION_IDENTITY",
    "POPSIGN_SOURCE_FILENAME",
    "POPSIGN_SOURCE_IDENTITY",
    "PUBLISHED_EXTERNAL_DATASET_RESOURCE_DIGESTS",
    "PUBLISHED_POPSIGN_SELECTION_SEMANTIC_DIGEST",
    "PUBLISHED_POPSIGN_SOURCE_SEMANTIC_DIGEST",
    "ExternalDatasetResourceError",
    "build_popsign_source",
    "build_signlab_five_popsign_selection",
    "external_resource_reference",
    "generated_external_dataset_resource_texts",
    "generated_external_dataset_schemas",
    "load_popsign_source",
    "load_signlab_five_popsign_selection",
    "render_external_dataset_json",
    "validate_packaged_external_dataset_resources",
]
