"""Generated schemas and synthetic examples for table-backed datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from types import MappingProxyType
from typing import Final, Literal, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import BaseModel

from signlab.contracts.core import ArtifactRefV1, WorkspaceRelativeLocatorV1
from signlab.contracts.dataset import (
    DATASET_TABLE_SCHEMA_VERSIONS,
    DATASET_TABLE_WRAPPER_MODELS,
    AnnotationRowV1,
    AnnotationsTableV1,
    ClipRowV1,
    ClipsTableV1,
    DatasetContentV2,
    DatasetContractError,
    DatasetManifestV2,
    DatasetSampleIdentityV1,
    DatasetTable,
    DatasetTableInput,
    DatasetTableRefV1,
    DatasetTableSetV1,
    DerivedArtifactRowV1,
    DerivedArtifactsTableV1,
    LabelId,
    MediaIntervalV1,
    ParticipantRowV1,
    ParticipantsTableV1,
    RecordingRowV1,
    RecordingsTableV1,
    SessionRowV1,
    SessionsTableV1,
    TableName,
    dataset_content_digest,
    dataset_table_digest,
    validate_dataset_table,
)
from signlab.contracts.pipeline import (
    PipelineContractError,
    contract_digest,
    validate_dataset_manifest_v2,
)
from signlab.contracts.taxonomy import load_builtin_taxonomy, taxonomy_reference
from signlab.datasets.parquet import parquet_schema_snapshot
from signlab.datasets.validation import DatasetValidationError, validate_dataset_manifest_tables
from signlab.governance.resources import (
    build_example_inventory,
    build_example_recording_grant,
    build_governance_policy,
)

RESOURCE_PACKAGE: Final = "signlab.resources.datasets"
_SCHEMA_BASE: Final = "https://signlab.dev/schemas/"
_SCHEMA_BOUNDARY_COMMENT: Final = (
    "This JSON Schema enforces portable structure and locally expressible invariants. "
    "SignLab's strict table and dataset validators remain authoritative for canonical "
    "ordering, semantic hashes, foreign keys, intervals, consent binding, lineage, "
    "sample projection, split inheritance, and artifact-byte verification."
)

DATASET_TABLE_SCHEMA_FILENAMES: Final[dict[TableName, str]] = {
    table_name: f"{schema_version.replace('/', '-')}.schema.json"
    for table_name, schema_version in DATASET_TABLE_SCHEMA_VERSIONS.items()
}
DATASET_TABLE_EXAMPLE_FILENAMES: Final[dict[TableName, str]] = {
    table_name: f"{schema_version.replace('/', '-')}.example.json"
    for table_name, schema_version in DATASET_TABLE_SCHEMA_VERSIONS.items()
}
DATASET_ARROW_SCHEMA_FILENAMES: Final[dict[TableName, str]] = {
    table_name: f"{schema_version.replace('/', '-')}.arrow-schema.json"
    for table_name, schema_version in DATASET_TABLE_SCHEMA_VERSIONS.items()
}
DATASET_MANIFEST_EXAMPLE_FILENAME: Final = "dataset-manifest-2.example.json"

GENERATED_DATASET_RESOURCE_NAMES: Final = frozenset(
    {
        *(f"schemas/{name}" for name in DATASET_TABLE_SCHEMA_FILENAMES.values()),
        *(f"examples/{name}" for name in DATASET_TABLE_EXAMPLE_FILENAMES.values()),
        *(f"arrow/{name}" for name in DATASET_ARROW_SCHEMA_FILENAMES.values()),
        f"examples/{DATASET_MANIFEST_EXAMPLE_FILENAME}",
    }
)

_PARTICIPANT_ID: Final = "participant_00000000000000000000000000000001"
_SESSION_ID: Final = "session_00000000000000000000000000000001"
_DEVICE_ID: Final = "device_00000000000000000000000000000001"
_RECORDING_ID: Final = "recording_00000000000000000000000000000031"
_CLIP_ID: Final = "clip_00000000000000000000000000000001"
_ANNOTATION_ID: Final = "annotation_00000000000000000000000000000001"
_SPLIT_ID: Final = "synthetic_grouped_split"
_DERIVED_IDS: Final = tuple(f"derived_artifact_{index:032x}" for index in range(1, 4))
_SAMPLE_IDS: Final = tuple(f"sample_{index:032x}" for index in range(1, 4))

# These are semantic identities, not hashes of pretty-printed files or Parquet
# physical bytes.
PUBLISHED_DATASET_TABLE_DIGESTS: Final[dict[TableName, str]] = {
    "participants": "sha256:eb47911ecc4c60d30110210b73dcb265279fb69284fa31a25a0560a2f6a3a227",
    "sessions": "sha256:0a77852161a088baded1248c7c71945aa43569b6ef01771bb649af7602e5730a",
    "recordings": "sha256:5c79b0629d997de6bdaab75f78e2be94ed02bb1dc9e9cf3219db3d1641aca364",
    "clips": "sha256:819f4bb5f4a915deac1d344e06cc56805f787701c2d8662c486da08e46461448",
    "annotations": "sha256:dcd6919628522a36d34f0d2a2aefbc7a2c64ca5340ce7e1e173029b67dd2577b",
    "derived_artifacts": (
        "sha256:b65cc0cc4b779437780487946244ee50691c448783d7a0eac4f7d8ddad343f65"
    ),
}
# These hash the exact generated UTF-8 JSON resources. They are deliberately
# independent of the generators so an in-place change to a published /1 schema
# fails until a new schema version and a newly reviewed baseline are introduced.
PUBLISHED_DATASET_JSON_SCHEMA_DIGESTS: Final[dict[TableName, str]] = {
    "participants": "sha256:a60b6134b82556b9141fb75e8286ed03d892bb7862320f3975b89575dcf922d6",
    "sessions": "sha256:5c9866d6068b849a848b5eddaf4c8912528a746eb126481a7ac13356ed94ea1e",
    "recordings": "sha256:b5865d70df8e38d1b6c39ce6196337b5daf8227381899a9fd421f65abfb01361",
    "clips": "sha256:70088cb7626cf7bc3990cf95542376fa02869fc25d12825fbc9f3c36c03bc564",
    "annotations": "sha256:b37744a2d1b3c648e7388d5960db78390e50b8a059e5ecb3c8e0e41c949a80d3",
    "derived_artifacts": (
        "sha256:92c3f89d6c66f9ff855a3e3dfa7daff0c38d6cd372cf47026a47f15df21d3477"
    ),
}
PUBLISHED_DATASET_ARROW_SCHEMA_DIGESTS: Final[dict[TableName, str]] = {
    "participants": "sha256:2d2e18bb5b6e309b3815f52a5ca3b51f01d0742488cd035a0717bb486667cfd5",
    "sessions": "sha256:70490a2a40fdd6ccf733a39419843d541789d29b2fb9e3362b4b8d58be10650a",
    "recordings": "sha256:437949b228b3ef263e0e0726a9731fa64356ce77859115864aa24addc746ac5f",
    "clips": "sha256:4b6a07e17a3d0636bd89c75731895a9a5e355532eba5a77bcfeb6d73bece007b",
    "annotations": "sha256:f53b1a3f1676dc36a74b7152bb7aaebfbc2e56dd1cfc7d0e4416a05fefd501c7",
    "derived_artifacts": (
        "sha256:143674be79cc861a29c40733e680357ad47f97e1b2ccf3a689a0feecbac116ec"
    ),
}
PUBLISHED_DATASET_MANIFEST_DIGEST: Final = (
    "sha256:69b705dddf972e8cf4ecb5692fe7aca779f136e3e0d258a374f79156b5d2d9a0"
)


class DatasetResourceError(DatasetContractError):
    """Raised when public dataset resources are incomplete, stale, or invalid."""


@dataclass(frozen=True, slots=True)
class DatasetResourceBundle:
    """One coherent, identity-free manifest and its six logical tables."""

    manifest: DatasetManifestV2
    tables: Mapping[TableName, DatasetTable]


def _synthetic_sha256(label: str) -> str:
    payload = f"SignLab synthetic dataset fixture: {label}\n".encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _locator(path: str) -> WorkspaceRelativeLocatorV1:
    return WorkspaceRelativeLocatorV1(kind="workspace_relative", path=path)


def _content_addressed_path(sha256: str, artifact_id: str) -> str:
    digest = sha256.removeprefix("sha256:")
    return f"objects/sha256/p-{digest[:2]}/sha256-{digest}/{artifact_id}"


def _artifact(
    artifact_id: str,
    *,
    role: str,
    media_type: str,
    label: str,
    size_bytes: int,
    path: str | None = None,
) -> ArtifactRefV1:
    sha256 = _synthetic_sha256(label)
    return ArtifactRefV1(
        schema_version="artifact-reference/1",
        artifact_id=artifact_id,
        role=role,
        media_type=media_type,
        sha256=sha256,
        size_bytes=size_bytes,
        locator=_locator(path or _content_addressed_path(sha256, artifact_id)),
    )


def _sample_artifact(index: int) -> ArtifactRefV1:
    sample_id = _SAMPLE_IDS[index - 1]
    return _artifact(
        sample_id,
        role="sample_data",
        media_type="application/vnd.signlab.feature-window+json",
        label=f"feature window sample {index}",
        size_bytes=2048 + index,
    )


def build_example_dataset_tables() -> dict[TableName, DatasetTable]:
    """Build six canonical tables with materialized clip and chained lineage."""

    participants = ParticipantsTableV1(
        schema_version="participants-table/1",
        rows=(ParticipantRowV1(participant_id=_PARTICIPANT_ID, handedness="right"),),
    )
    sessions = SessionsTableV1(
        schema_version="sessions-table/1",
        rows=(
            SessionRowV1(
                session_id=_SESSION_ID,
                participant_id=_PARTICIPANT_ID,
                device_id=_DEVICE_ID,
                started_at="2026-08-26T12:00:00Z",
                finished_at="2026-08-26T12:30:00Z",
                capture_mode="continuous",
                capture_software_version="1.0.0",
                camera_facing="front",
                frame_width_px=1920,
                frame_height_px=1080,
                frame_rate_numerator=30_000,
                frame_rate_denominator=1001,
                rotation_degrees=0,
                mirror_state="mirrored",
            ),
        ),
    )
    recording_media = _artifact(
        _RECORDING_ID,
        role="raw_recording",
        media_type="video/mp4",
        label="raw recording 31",
        size_bytes=4_194_304,
    )
    recordings = RecordingsTableV1(
        schema_version="recordings-table/1",
        rows=(
            RecordingRowV1(
                recording_id=_RECORDING_ID,
                participant_id=_PARTICIPANT_ID,
                session_id=_SESSION_ID,
                device_id=_DEVICE_ID,
                captured_at="2026-08-26T12:10:00Z",
                duration_us=5_000_000,
                handedness="right",
                mirror_state="mirrored",
                rotation_degrees=0,
                audio_present=False,
                media=recording_media,
                consent_grant=build_example_recording_grant(),
            ),
        ),
    )
    clip_artifact = _artifact(
        _CLIP_ID,
        role="clip_media",
        media_type="video/mp4",
        label="materialized clip 1",
        size_bytes=1_048_576,
    )
    clips = ClipsTableV1(
        schema_version="clips-table/1",
        rows=(
            ClipRowV1(
                clip_id=_CLIP_ID,
                participant_id=_PARTICIPANT_ID,
                session_id=_SESSION_ID,
                source_recording_id=_RECORDING_ID,
                interval=MediaIntervalV1(
                    schema_version="media-interval/1",
                    start_us=100_000,
                    end_us=4_000_000,
                ),
                handedness="right",
                mirror_state="mirrored",
                artifact=clip_artifact,
            ),
        ),
    )
    annotations = AnnotationsTableV1(
        schema_version="annotations-table/1",
        rows=(
            AnnotationRowV1(
                annotation_id=_ANNOTATION_ID,
                participant_id=_PARTICIPANT_ID,
                session_id=_SESSION_ID,
                source_recording_id=_RECORDING_ID,
                clip_id=_CLIP_ID,
                interval=MediaIntervalV1(
                    schema_version="media-interval/1",
                    start_us=250_000,
                    end_us=3_750_000,
                ),
                disposition="class_label",
                label_id="hello",
                other_kind=None,
                reason_code=None,
                review_status="reviewed",
                eligible_for_training=True,
            ),
        ),
    )

    sample_artifacts = tuple(_sample_artifact(index) for index in range(1, 4))
    derivation_kinds: tuple[Literal["crop", "augmentation", "window"], ...] = (
        "crop",
        "augmentation",
        "window",
    )
    operation_ids = ("crop_annotation", "augment_landmarks", "window_landmarks")
    parent_ids = (
        (clip_artifact.artifact_id,),
        (sample_artifacts[0].artifact_id,),
        (sample_artifacts[1].artifact_id,),
    )
    derived_rows = tuple(
        DerivedArtifactRowV1(
            derived_artifact_id=_DERIVED_IDS[index],
            derivation_kind=derivation_kinds[index],
            parent_artifact_ids=parent_ids[index],
            participant_id=_PARTICIPANT_ID,
            session_id=_SESSION_ID,
            source_recording_id=_RECORDING_ID,
            clip_id=_CLIP_ID,
            annotation_id=_ANNOTATION_ID,
            sample_id=_SAMPLE_IDS[index],
            label_id="hello",
            split_id=_SPLIT_ID,
            partition="train",
            handedness="right",
            mirror_state="mirrored",
            operation_id=operation_ids[index],
            operation_version="1.0.0",
            artifact=sample_artifacts[index],
        )
        for index in range(3)
    )
    derived_artifacts = DerivedArtifactsTableV1(
        schema_version="derived-artifacts-table/1",
        rows=derived_rows,
    )
    return {
        "participants": participants,
        "sessions": sessions,
        "recordings": recordings,
        "clips": clips,
        "annotations": annotations,
        "derived_artifacts": derived_artifacts,
    }


def _table_reference(
    table_name: TableName,
    table: DatasetTable,
    index: int,
) -> DatasetTableRefV1:
    return DatasetTableRefV1(
        schema_version="dataset-table-reference/1",
        table_name=table_name,
        table_schema_version=DATASET_TABLE_SCHEMA_VERSIONS[table_name],
        row_count=len(table.rows),
        content_sha256=dataset_table_digest(table),
        artifact=_artifact(
            f"synthetic_{table_name}_table",
            role="dataset_table",
            media_type="application/vnd.apache.parquet",
            label=f"example Parquet bytes for {table_name}",
            size_bytes=8192 + index,
            path=f"tables/{table_name}.parquet",
        ),
    )


def _build_example_dataset_manifest(
    tables: Mapping[TableName, DatasetTable],
) -> DatasetManifestV2:
    references = {
        table_name: _table_reference(table_name, tables[table_name], index)
        for index, table_name in enumerate(DATASET_TABLE_SCHEMA_VERSIONS, start=1)
    }
    derived = cast(DerivedArtifactsTableV1, tables["derived_artifacts"])
    samples = tuple(
        DatasetSampleIdentityV1(
            sample_id=cast(str, row.sample_id),
            participant_id=row.participant_id,
            session_id=row.session_id,
            source_recording_id=row.source_recording_id,
            label_id=cast(LabelId, row.label_id),
            artifact=row.artifact,
        )
        for row in derived.rows
    )
    content = DatasetContentV2(
        schema_version="dataset-content/2",
        taxonomy=taxonomy_reference(load_builtin_taxonomy()),
        governance_policy=build_governance_policy().policy_document,
        lineage_inventory_sha256=build_example_inventory().inventory_sha256,
        sample_schema_version="feature-window/1",
        tables=DatasetTableSetV1(
            schema_version="dataset-table-set/1",
            participants=references["participants"],
            sessions=references["sessions"],
            recordings=references["recordings"],
            clips=references["clips"],
            annotations=references["annotations"],
            derived_artifacts=references["derived_artifacts"],
        ),
        samples=samples,
    )
    return DatasetManifestV2(
        schema_version="dataset-manifest/2",
        dataset_id="synthetic_table_dataset",
        version="2.0.0",
        content=content,
        data_sha256=dataset_content_digest(content),
    )


def build_example_dataset_bundle() -> DatasetResourceBundle:
    """Build and semantically prove the complete synthetic dataset example."""

    tables = build_example_dataset_tables()
    manifest = _build_example_dataset_manifest(tables)
    validate_dataset_manifest_tables(
        manifest,
        cast(Mapping[str, DatasetTableInput], tables),
    )
    return DatasetResourceBundle(
        manifest=manifest,
        tables=MappingProxyType(tables),
    )


def build_example_dataset_manifest() -> DatasetManifestV2:
    """Build the public table-backed dataset-manifest/2 example."""

    return build_example_dataset_bundle().manifest


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
        for field_name in ("parent_artifact_ids", "rows"):
            field_schema = properties.get(field_name)
            if isinstance(field_schema, dict) and field_schema.get("type") == "array":
                field_schema["uniqueItems"] = True


def generated_dataset_schemas() -> dict[str, dict[str, object]]:
    """Generate the six standalone table-wrapper Draft 2020-12 schemas."""

    generated: dict[str, dict[str, object]] = {}
    for table_name, filename in DATASET_TABLE_SCHEMA_FILENAMES.items():
        model = DATASET_TABLE_WRAPPER_MODELS[table_name]
        schema = model.model_json_schema(mode="validation")
        _strip_nested_schema_ids(schema, root=True)
        schema["$id"] = f"{_SCHEMA_BASE}{filename}"
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$comment"] = _SCHEMA_BOUNDARY_COMMENT
        _harden_unique_arrays(schema)
        Draft202012Validator.check_schema(schema)
        generated[filename] = schema
    if set(generated) != set(DATASET_TABLE_SCHEMA_FILENAMES.values()):
        raise DatasetResourceError("generated dataset schema registry is incomplete")
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
        raise DatasetResourceError("dataset resource is not JSON-serializable") from error


def _rendered_resource_digest(value: BaseModel | Mapping[str, object]) -> str:
    rendered = render_json_document(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(rendered).hexdigest()}"


def generated_dataset_resource_texts() -> dict[str, str]:
    """Render the exact table schema, example, and Arrow snapshot inventory."""

    bundle = build_example_dataset_bundle()
    rendered = {
        f"examples/{DATASET_MANIFEST_EXAMPLE_FILENAME}": render_json_document(bundle.manifest)
    }
    rendered.update(
        {
            f"schemas/{filename}": render_json_document(schema)
            for filename, schema in generated_dataset_schemas().items()
        }
    )
    for table_name, table in bundle.tables.items():
        rendered[f"examples/{DATASET_TABLE_EXAMPLE_FILENAMES[table_name]}"] = render_json_document(
            table
        )
        rendered[f"arrow/{DATASET_ARROW_SCHEMA_FILENAMES[table_name]}"] = render_json_document(
            parquet_schema_snapshot(DATASET_TABLE_SCHEMA_VERSIONS[table_name])
        )
    if set(rendered) != GENERATED_DATASET_RESOURCE_NAMES:
        raise DatasetResourceError("generated dataset resource registry is incomplete")
    return rendered


def _resource_bytes(relative_name: str) -> bytes:
    try:
        return files(RESOURCE_PACKAGE).joinpath(*relative_name.split("/")).read_bytes()
    except OSError as error:
        raise DatasetResourceError("a packaged dataset resource is missing") from error


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


def load_packaged_dataset_table(table_name: TableName) -> DatasetTable:
    """Load and strictly validate one packaged logical table example."""

    try:
        filename = DATASET_TABLE_EXAMPLE_FILENAMES[table_name]
    except KeyError as error:
        raise DatasetResourceError("unsupported packaged dataset table") from error
    try:
        table = validate_dataset_table(_resource_bytes(f"examples/{filename}"))
    except (TypeError, ValueError) as error:
        raise DatasetResourceError("packaged dataset table is invalid") from error
    if table.schema_version != DATASET_TABLE_SCHEMA_VERSIONS[table_name]:
        raise DatasetResourceError("packaged dataset table has the wrong schema")
    return table


def load_packaged_dataset_manifest() -> DatasetManifestV2:
    """Load and strictly validate the packaged dataset-manifest/2 example."""

    try:
        return validate_dataset_manifest_v2(
            _resource_bytes(f"examples/{DATASET_MANIFEST_EXAMPLE_FILENAME}")
        )
    except (PipelineContractError, TypeError, ValueError) as error:
        raise DatasetResourceError("packaged dataset manifest is invalid") from error


def validate_packaged_dataset_resources() -> None:
    """Check exact inventory, byte drift, schemas, identities, and table semantics."""

    try:
        if _packaged_resource_inventory() != GENERATED_DATASET_RESOURCE_NAMES:
            raise DatasetResourceError("packaged dataset resource inventory is not exact")
        generated = generated_dataset_resource_texts()
        for name, expected_text in generated.items():
            if _resource_bytes(name).decode("utf-8") != expected_text:
                raise DatasetResourceError(
                    "packaged dataset resource drift detected; regenerate resources"
                )

        schemas = generated_dataset_schemas()
        if not (
            set(PUBLISHED_DATASET_JSON_SCHEMA_DIGESTS)
            == set(PUBLISHED_DATASET_ARROW_SCHEMA_DIGESTS)
            == set(DATASET_TABLE_SCHEMA_VERSIONS)
        ):
            raise DatasetResourceError("published dataset schema baselines are incomplete")
        tables = {
            table_name: load_packaged_dataset_table(table_name)
            for table_name in DATASET_TABLE_SCHEMA_VERSIONS
        }
        for table_name, table in tables.items():
            schema = schemas[DATASET_TABLE_SCHEMA_FILENAMES[table_name]]
            if (
                _rendered_resource_digest(schema)
                != PUBLISHED_DATASET_JSON_SCHEMA_DIGESTS[table_name]
            ):
                raise DatasetResourceError(
                    "a published dataset JSON Schema changed without a new version"
                )
            Draft202012Validator(schema).validate(table.model_dump(mode="json", round_trip=True))
            if dataset_table_digest(table) != PUBLISHED_DATASET_TABLE_DIGESTS[table_name]:
                raise DatasetResourceError("a published dataset table identity changed")
            snapshot_name = f"arrow/{DATASET_ARROW_SCHEMA_FILENAMES[table_name]}"
            packaged_snapshot = json.loads(_resource_bytes(snapshot_name))
            generated_snapshot = parquet_schema_snapshot(DATASET_TABLE_SCHEMA_VERSIONS[table_name])
            if packaged_snapshot != generated_snapshot:
                raise DatasetResourceError("a packaged Arrow schema snapshot changed")
            if (
                _rendered_resource_digest(generated_snapshot)
                != PUBLISHED_DATASET_ARROW_SCHEMA_DIGESTS[table_name]
            ):
                raise DatasetResourceError("a published Arrow schema changed without a new version")

        manifest = load_packaged_dataset_manifest()
        validate_dataset_manifest_tables(
            manifest,
            cast(Mapping[str, DatasetTableInput], tables),
        )
        if contract_digest(manifest) != PUBLISHED_DATASET_MANIFEST_DIGEST:
            raise DatasetResourceError("the published dataset manifest identity changed")
    except DatasetResourceError:
        raise
    except (
        DatasetValidationError,
        OSError,
        TypeError,
        ValueError,
        SchemaError,
        JsonSchemaValidationError,
    ) as error:
        raise DatasetResourceError(
            "packaged dataset resources are missing, invalid, or inconsistent"
        ) from error


__all__ = [
    "DATASET_ARROW_SCHEMA_FILENAMES",
    "DATASET_MANIFEST_EXAMPLE_FILENAME",
    "DATASET_TABLE_EXAMPLE_FILENAMES",
    "DATASET_TABLE_SCHEMA_FILENAMES",
    "GENERATED_DATASET_RESOURCE_NAMES",
    "PUBLISHED_DATASET_ARROW_SCHEMA_DIGESTS",
    "PUBLISHED_DATASET_JSON_SCHEMA_DIGESTS",
    "PUBLISHED_DATASET_MANIFEST_DIGEST",
    "PUBLISHED_DATASET_TABLE_DIGESTS",
    "DatasetResourceBundle",
    "DatasetResourceError",
    "build_example_dataset_bundle",
    "build_example_dataset_manifest",
    "build_example_dataset_tables",
    "generated_dataset_resource_texts",
    "generated_dataset_schemas",
    "load_packaged_dataset_manifest",
    "load_packaged_dataset_table",
    "render_json_document",
    "validate_packaged_dataset_resources",
]
