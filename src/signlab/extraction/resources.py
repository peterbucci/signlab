"""Generated public resources for version-pinned landmark extraction."""

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
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import BaseModel

from signlab.contracts.extraction import (
    BODY_ANCHOR_NAMES,
    LandmarkExtractionManifestV1,
    LandmarkFramesTableV1,
    MediaPipeExtractionConfigV1,
    MediaPipeTaskAssetV1,
    mediapipe_extraction_config_digest,
    validate_mediapipe_extraction_config,
)
from signlab.extraction.parquet import landmark_parquet_schema_snapshot

RESOURCE_PACKAGE: Final = "signlab.resources.extraction"
_SCHEMA_BASE: Final = "https://signlab.dev/schemas/"
_SCHEMA_COMMENT: Final = (
    "This JSON Schema enforces portable extraction structure. SignLab's strict "
    "extraction validators remain authoritative for canonical hashes, timeline "
    "recurrence, masks, counts, model registration, lineage, and artifact binding."
)

DEFAULT_CONFIG_FILENAME: Final = "mediapipe-extraction-config-1.default.json"
LANDMARK_ARROW_SCHEMA_FILENAME: Final = "landmark-frames-table-1.arrow-schema.json"
MODEL_LOCK_FILENAME: Final = "mediapipe-tasks-1.0.1.lock.json"

EXTRACTION_SCHEMA_MODELS: Final[Mapping[str, type[BaseModel]]] = MappingProxyType(
    {
        "landmark-extraction-manifest-1.schema.json": LandmarkExtractionManifestV1,
        "landmark-frames-table-1.schema.json": LandmarkFramesTableV1,
        "mediapipe-extraction-config-1.schema.json": MediaPipeExtractionConfigV1,
    }
)
GENERATED_EXTRACTION_RESOURCE_NAMES: Final = frozenset(
    {
        *(f"schemas/{filename}" for filename in EXTRACTION_SCHEMA_MODELS),
        f"config/{DEFAULT_CONFIG_FILENAME}",
        f"arrow/{LANDMARK_ARROW_SCHEMA_FILENAME}",
        f"models/{MODEL_LOCK_FILENAME}",
    }
)

# Frozen after first publication. These values hash exact pretty-printed UTF-8
# resource bytes, independently of their generators and semantic identities.
PUBLISHED_EXTRACTION_RESOURCE_DIGESTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "arrow/landmark-frames-table-1.arrow-schema.json": (
            "sha256:a3d54a29f58f07805459ecf6c87f598c2133a183de7e4841c70111fce851ce57"
        ),
        "config/mediapipe-extraction-config-1.default.json": (
            "sha256:1b35c18a7ef026f4b29da17ca30c1054327c5e45c83296aef83dbfa359978431"
        ),
        "models/mediapipe-tasks-1.0.1.lock.json": (
            "sha256:83866bc13895fa999c1ec6dc551be7feb3a372d81abaa7d5f8b9468e45e033b0"
        ),
        "schemas/landmark-extraction-manifest-1.schema.json": (
            "sha256:68ce21dd70c8a643fcd554b5fcf81d9457b9d15bf5b9463057a7549fe64bbbc1"
        ),
        "schemas/landmark-frames-table-1.schema.json": (
            "sha256:9211896d49f51774776d38f4cc15bd5973044e5f8ea8a4e3478ff914fef19fae"
        ),
        "schemas/mediapipe-extraction-config-1.schema.json": (
            "sha256:148da938ee440d539bb5f5af596aac56963f6c76755e5ecdc09010cbed4f3982"
        ),
    }
)
PUBLISHED_DEFAULT_CONFIG_SEMANTIC_DIGEST: Final = (
    "sha256:7343cd8bb724313b4063a3ebd5d7f7470a78b00f2eeda275a15e5f9b2e66e94c"
)


class ExtractionResourceError(ValueError):
    """Raised when packaged extraction resources are missing, stale, or invalid."""


def _hand_task_asset() -> MediaPipeTaskAssetV1:
    return MediaPipeTaskAssetV1(
        schema_version="mediapipe-task-asset/1",
        task_kind="hand_landmarker",
        model_id="mediapipe-hand-landmarker-full",
        model_revision="1.0.0",
        filename="hand_landmarker.task",
        sha256="sha256:fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
        size_bytes=7_819_105,
        compatible_runtimes=("browser", "python"),
    )


def _pose_task_asset() -> MediaPipeTaskAssetV1:
    return MediaPipeTaskAssetV1(
        schema_version="mediapipe-task-asset/1",
        task_kind="pose_landmarker",
        model_id="mediapipe-pose-landmarker-lite",
        model_revision="1.0.0",
        filename="pose_landmarker_lite.task",
        sha256="sha256:59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
        size_bytes=5_777_746,
        compatible_runtimes=("browser", "python"),
    )


def build_default_extraction_config() -> MediaPipeExtractionConfigV1:
    """Build the one reviewable CPU/VIDEO extraction configuration."""

    return MediaPipeExtractionConfigV1(
        schema_version="mediapipe-extraction-config/1",
        config_id="mediapipe_tasks_video",
        version="1.0.0",
        python_package="mediapipe",
        python_package_version="1.0.1",
        browser_package="@mediapipe/tasks-vision",
        browser_package_version="1.0.1",
        decoder_package="av",
        decoder_package_version="18.1.0",
        delegate="CPU",
        running_mode="VIDEO",
        num_hands=2,
        num_poses=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_hand_tracking_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_pose_tracking_confidence=0.5,
        body_anchors=BODY_ANCHOR_NAMES,
        timestamp_rule="source_pts_relative_us_then_strict_monotonic_floor_ms/1",
        tracking_algorithm="deterministic_wrist_mcp_centroid_minimum_cost",
        tracking_algorithm_version="1.0.0",
        max_spatial_cost=0.25,
        handedness_disagreement_penalty=0.05,
        ambiguity_margin=1e-9,
        hand_task_asset=_hand_task_asset(),
        pose_task_asset=_pose_task_asset(),
    )


def _task_lock(
    asset: MediaPipeTaskAssetV1,
    *,
    source_url: str,
    model_card_url: str,
) -> dict[str, object]:
    return {
        "compatible_runtimes": list(asset.compatible_runtimes),
        "filename": asset.filename,
        "model_card_url": model_card_url,
        "model_id": asset.model_id,
        "revision": asset.model_revision,
        "sha256": asset.sha256,
        "size_bytes": asset.size_bytes,
        "source_url": source_url,
        "task_kind": asset.task_kind,
    }


def build_mediapipe_model_lock() -> dict[str, object]:
    """Build provenance for model bytes shared by Python and browser runtimes."""

    config = build_default_extraction_config()
    return {
        "browser_package": {
            "name": config.browser_package,
            "version": config.browser_package_version,
        },
        "license": "Apache-2.0",
        "python_package": {
            "name": config.python_package,
            "version": config.python_package_version,
        },
        "schema_version": "mediapipe-task-model-lock/1",
        "tasks": [
            _task_lock(
                config.hand_task_asset,
                source_url=(
                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                    "hand_landmarker/float16/1/hand_landmarker.task?generation=1682480004222387"
                ),
                model_card_url=(
                    "https://storage.googleapis.com/mediapipe-assets/"
                    "Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf"
                ),
            ),
            _task_lock(
                config.pose_task_asset,
                source_url=(
                    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task?"
                    "generation=1682624736756847"
                ),
                model_card_url=(
                    "https://storage.googleapis.com/mediapipe-assets/"
                    "Model%20Card%20BlazePose%20GHUM%203D.pdf"
                ),
            ),
        ],
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


def _harden_unique_arrays(schema: dict[str, object]) -> None:
    for node in _schema_nodes(schema):
        properties = node.get("properties")
        if not isinstance(properties, dict):
            continue
        for field_name in ("compatible_runtimes", "parent_artifact_ids", "sequences"):
            field_schema = properties.get(field_name)
            if isinstance(field_schema, dict) and field_schema.get("type") == "array":
                field_schema["uniqueItems"] = True


def generated_extraction_schemas() -> dict[str, dict[str, object]]:
    """Return standalone Draft 2020-12 schemas for extraction handoffs."""

    generated: dict[str, dict[str, object]] = {}
    for filename, model in EXTRACTION_SCHEMA_MODELS.items():
        schema = model.model_json_schema(mode="validation")
        _strip_nested_schema_ids(schema, root=True)
        schema["$id"] = f"{_SCHEMA_BASE}{filename}"
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$comment"] = _SCHEMA_COMMENT
        _harden_unique_arrays(schema)
        Draft202012Validator.check_schema(schema)
        generated[filename] = schema
    if set(generated) != set(EXTRACTION_SCHEMA_MODELS):
        raise ExtractionResourceError("generated extraction schema registry is incomplete")
    return generated


def render_extraction_json(value: BaseModel | Mapping[str, object]) -> str:
    """Render deterministic reviewable JSON with one trailing newline."""

    payload: object = (
        value.model_dump(mode="json", round_trip=True)
        if isinstance(value, BaseModel)
        else dict(value)
    )
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
        raise ExtractionResourceError("extraction resource is not JSON-serializable") from error


def generated_extraction_resource_texts() -> dict[str, str]:
    """Render the exact schema, config, model-lock, and Arrow inventory."""

    rendered = {
        f"schemas/{filename}": render_extraction_json(schema)
        for filename, schema in generated_extraction_schemas().items()
    }
    rendered[f"config/{DEFAULT_CONFIG_FILENAME}"] = render_extraction_json(
        build_default_extraction_config()
    )
    rendered[f"arrow/{LANDMARK_ARROW_SCHEMA_FILENAME}"] = render_extraction_json(
        landmark_parquet_schema_snapshot()
    )
    rendered[f"models/{MODEL_LOCK_FILENAME}"] = render_extraction_json(build_mediapipe_model_lock())
    if set(rendered) != GENERATED_EXTRACTION_RESOURCE_NAMES:
        raise ExtractionResourceError("generated extraction resource registry is incomplete")
    return rendered


def _resource_bytes(relative_name: str) -> bytes:
    try:
        return files(RESOURCE_PACKAGE).joinpath(*relative_name.split("/")).read_bytes()
    except OSError as error:
        raise ExtractionResourceError("a packaged extraction resource is missing") from error


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


def load_packaged_default_extraction_config() -> MediaPipeExtractionConfigV1:
    """Load and strictly validate the packaged default configuration."""

    try:
        return validate_mediapipe_extraction_config(
            _resource_bytes(f"config/{DEFAULT_CONFIG_FILENAME}")
        )
    except (TypeError, ValueError) as error:
        raise ExtractionResourceError("packaged extraction config is invalid") from error


def validate_packaged_extraction_resources() -> None:
    """Check exact inventory, generated bytes, frozen hashes, and cross-bindings."""

    try:
        if _packaged_inventory() != GENERATED_EXTRACTION_RESOURCE_NAMES:
            raise ExtractionResourceError("packaged extraction resource inventory is not exact")
        if set(PUBLISHED_EXTRACTION_RESOURCE_DIGESTS) != GENERATED_EXTRACTION_RESOURCE_NAMES:
            raise ExtractionResourceError("published extraction resource baselines are incomplete")

        generated = generated_extraction_resource_texts()
        for relative_name, expected_text in generated.items():
            captured = _resource_bytes(relative_name)
            if captured != expected_text.encode("utf-8"):
                raise ExtractionResourceError("packaged extraction resource drift detected")
            digest = f"sha256:{hashlib.sha256(captured).hexdigest()}"
            if digest != PUBLISHED_EXTRACTION_RESOURCE_DIGESTS[relative_name]:
                raise ExtractionResourceError("a published extraction resource changed in place")

        config = load_packaged_default_extraction_config()
        config_payload = config.model_dump(mode="json", round_trip=True)
        config_schema = generated_extraction_schemas()["mediapipe-extraction-config-1.schema.json"]
        Draft202012Validator(config_schema).validate(config_payload)
        if mediapipe_extraction_config_digest(config) != PUBLISHED_DEFAULT_CONFIG_SEMANTIC_DIGEST:
            raise ExtractionResourceError("the published extraction config identity changed")

        packaged_snapshot = json.loads(_resource_bytes(f"arrow/{LANDMARK_ARROW_SCHEMA_FILENAME}"))
        if packaged_snapshot != landmark_parquet_schema_snapshot():
            raise ExtractionResourceError("the packaged landmark Arrow schema changed")
        packaged_lock = json.loads(_resource_bytes(f"models/{MODEL_LOCK_FILENAME}"))
        if packaged_lock != build_mediapipe_model_lock():
            raise ExtractionResourceError("the packaged MediaPipe model lock changed")
    except ExtractionResourceError:
        raise
    except (
        JsonSchemaValidationError,
        OSError,
        SchemaError,
        TypeError,
        ValueError,
    ) as error:
        raise ExtractionResourceError("packaged extraction resources are invalid") from error


__all__ = [
    "DEFAULT_CONFIG_FILENAME",
    "EXTRACTION_SCHEMA_MODELS",
    "GENERATED_EXTRACTION_RESOURCE_NAMES",
    "LANDMARK_ARROW_SCHEMA_FILENAME",
    "MODEL_LOCK_FILENAME",
    "PUBLISHED_DEFAULT_CONFIG_SEMANTIC_DIGEST",
    "PUBLISHED_EXTRACTION_RESOURCE_DIGESTS",
    "ExtractionResourceError",
    "build_default_extraction_config",
    "build_mediapipe_model_lock",
    "generated_extraction_resource_texts",
    "generated_extraction_schemas",
    "load_packaged_default_extraction_config",
    "render_extraction_json",
    "validate_packaged_extraction_resources",
]
