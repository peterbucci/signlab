"""Deterministic public-fixture stages and the authoritative DVC stage registry."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

from signlab.contracts.canonical import canonical_json_bytes, parse_json_object

type StageName = Literal["ingest", "validate", "extract", "quality", "split", "feature"]
type JsonObject = dict[str, object]
type DirectoryIdentity = tuple[int, int]
type FileIdentity = tuple[int, int, int, int]

FIXTURE_PROFILE: Final = "public-synthetic-reproducibility"
FIXTURE_IMPLEMENTATION: Final = "fixture-only/1"
SOURCE_PATH: Final = "tests/fixtures/public/dvc/source.json"
CONFIG_PATH: Final = "configs/pipeline/synthetic-dvc.json"
MAX_INPUT_BYTES: Final = 1_048_576
_REPARSE_POINT: Final = 0x400
_SHARED_DEPENDENCIES: Final = (
    CONFIG_PATH,
    ".python-version",
    "pyproject.toml",
    "src/signlab",
    "uv.lock",
)
_RECEIPT_KEYS: Final = {
    "fixture_only",
    "implementation",
    "payload",
    "profile",
    "schema_version",
    "stage",
    "upstream_sha256",
}


class ReproductionStageError(ValueError):
    """Raised when the synthetic reproducibility proof cannot run safely."""


@dataclass(frozen=True)
class StageSpec:
    """One immutable edge in the public synthetic DVC graph."""

    name: StageName
    predecessor: StageName | None
    input_path: str
    output_path: str

    @property
    def command(self) -> str:
        return f"python -m signlab.cli data run-reproduction-stage {self.name}"

    @property
    def dependencies(self) -> tuple[str, ...]:
        try:
            stage_index = next(
                index for index, candidate in enumerate(STAGE_REGISTRY) if candidate is self
            )
        except StopIteration as error:
            raise ReproductionStageError("stage registry lineage is invalid") from error
        # Receipt validation walks the complete chain back to the source. Declare
        # every file it reads so DVC never caches a stage with an undeclared input.
        lineage_inputs = tuple(
            candidate.input_path for candidate in STAGE_REGISTRY[: stage_index + 1]
        )
        return (*_SHARED_DEPENDENCIES, *lineage_inputs)


STAGE_REGISTRY: Final = (
    StageSpec(
        name="ingest",
        predecessor=None,
        input_path=SOURCE_PATH,
        output_path="data/raw/dvc-public-fixture/ingest.json",
    ),
    StageSpec(
        name="validate",
        predecessor="ingest",
        input_path="data/raw/dvc-public-fixture/ingest.json",
        output_path="data/interim/dvc-public-fixture/validate.json",
    ),
    StageSpec(
        name="extract",
        predecessor="validate",
        input_path="data/interim/dvc-public-fixture/validate.json",
        output_path="data/interim/dvc-public-fixture/extract.json",
    ),
    StageSpec(
        name="quality",
        predecessor="extract",
        input_path="data/interim/dvc-public-fixture/extract.json",
        output_path="data/interim/dvc-public-fixture/quality.json",
    ),
    StageSpec(
        name="split",
        predecessor="quality",
        input_path="data/interim/dvc-public-fixture/quality.json",
        output_path="data/processed/dvc-public-fixture/split.json",
    ),
    StageSpec(
        name="feature",
        predecessor="split",
        input_path="data/processed/dvc-public-fixture/split.json",
        output_path="data/processed/dvc-public-fixture/feature.json",
    ),
)
STAGE_NAMES: Final = tuple(spec.name for spec in STAGE_REGISTRY)
_STAGES_BY_NAME: Final = {spec.name: spec for spec in STAGE_REGISTRY}


def _is_reparse_point(details: os.stat_result) -> bool:
    attributes = getattr(details, "st_file_attributes", 0)
    return bool(attributes & _REPARSE_POINT)


def _assert_safe_existing_path(path: Path, *, regular_file: bool) -> os.stat_result:
    try:
        details = path.lstat()
    except OSError as error:
        raise ReproductionStageError("required fixture input is unavailable") from error
    if stat.S_ISLNK(details.st_mode) or _is_reparse_point(details):
        raise ReproductionStageError("reproduction paths must not use links or reparse points")
    expected = stat.S_ISREG(details.st_mode) if regular_file else stat.S_ISDIR(details.st_mode)
    if not expected or (regular_file and details.st_nlink != 1):
        raise ReproductionStageError("reproduction paths must use regular files and directories")
    return details


def _directory_identity(path: Path) -> DirectoryIdentity:
    details = _assert_safe_existing_path(path, regular_file=False)
    return details.st_dev, details.st_ino


def _file_identity(details: os.stat_result) -> FileIdentity:
    return (
        details.st_dev,
        details.st_ino,
        details.st_size,
        details.st_mtime_ns,
    )


def _require_directory_identity(path: Path, expected: DirectoryIdentity) -> None:
    if _directory_identity(path) != expected:
        raise ReproductionStageError("reproduction output parent changed during publication")


def _resolve_workspace_path(root: Path, relative_path: str, *, must_exist: bool) -> Path:
    if not root.is_absolute():
        raise ReproductionStageError("repository root must be absolute")
    _assert_safe_existing_path(root, regular_file=False)
    candidate = root.joinpath(*relative_path.split("/"))
    try:
        resolved_root = root.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=must_exist)
    except OSError as error:
        raise ReproductionStageError("reproduction path could not be resolved") from error
    if not resolved_candidate.is_relative_to(resolved_root):
        raise ReproductionStageError("reproduction path escapes the repository")
    return candidate


def _read_json(root: Path, relative_path: str) -> tuple[JsonObject, bytes]:
    path = _resolve_workspace_path(root, relative_path, must_exist=True)
    parent_identity = _directory_identity(path.parent)
    before = _assert_safe_existing_path(path, regular_file=True)
    try:
        if before.st_size > MAX_INPUT_BYTES:
            raise ReproductionStageError("reproduction input exceeds the public-fixture limit")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            remaining = MAX_INPUT_BYTES + 1
            chunks: list[bytes] = []
            while remaining:
                chunk = os.read(descriptor, min(remaining, 65_536))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        document_bytes = b"".join(chunks)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _file_identity(before) != _file_identity(opened)
            or _file_identity(opened) != _file_identity(after)
            or len(document_bytes) != opened.st_size
            or _directory_identity(path.parent) != parent_identity
        ):
            raise ReproductionStageError("reproduction input changed while being read")
        if len(document_bytes) > MAX_INPUT_BYTES:
            raise ReproductionStageError("reproduction input exceeds the public-fixture limit")
        document = parse_json_object(document_bytes)
    except ReproductionStageError:
        raise
    except OSError as error:
        raise ReproductionStageError("reproduction input could not be read") from error
    except ValueError as error:
        raise ReproductionStageError("reproduction input is not strict JSON") from error
    return cast(JsonObject, document), document_bytes


def _ensure_output_parent(root: Path, relative_path: str) -> tuple[Path, DirectoryIdentity]:
    path = _resolve_workspace_path(root, relative_path, must_exist=False)
    root_identity = _directory_identity(root)
    relative_parent = path.parent.relative_to(root)
    current = root
    for segment in relative_parent.parts:
        current /= segment
        if current.exists():
            _assert_safe_existing_path(current, regular_file=False)
        else:
            try:
                current.mkdir()
            except OSError as error:
                raise ReproductionStageError(
                    "reproduction output directory could not be created"
                ) from error
    if path.exists() or path.is_symlink():
        _assert_safe_existing_path(path, regular_file=True)
    _require_directory_identity(root, root_identity)
    return path, _directory_identity(path.parent)


def _atomic_write_json(root: Path, relative_path: str, document: Mapping[str, object]) -> None:
    output, parent_identity = _ensure_output_parent(root, relative_path)
    payload = canonical_json_bytes(document) + b"\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        _require_directory_identity(output.parent, parent_identity)
        _assert_safe_existing_path(temporary_path, regular_file=True)
        if output.exists() or output.is_symlink():
            _assert_safe_existing_path(output, regular_file=True)
        os.replace(temporary_path, output)
        _require_directory_identity(output.parent, parent_identity)
        _assert_safe_existing_path(output, regular_file=True)
    except OSError as error:
        raise ReproductionStageError("reproduction output could not be published") from error
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)


def _require_exact_keys(document: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(document) != expected:
        raise ReproductionStageError(f"{label} does not match the fixture contract")


def _load_config(root: Path) -> JsonObject:
    config, _ = _read_json(root, CONFIG_PATH)
    _require_exact_keys(
        config,
        {"fixture_only", "minimum_valid_vectors", "profile", "random_seed", "schema_version"},
        "reproduction configuration",
    )
    if (
        config["schema_version"] != "synthetic-dvc-profile/1"
        or config["profile"] != FIXTURE_PROFILE
        or config["fixture_only"] is not True
        or type(config["random_seed"]) is not int
        or type(config["minimum_valid_vectors"]) is not int
        or config["minimum_valid_vectors"] < 1
    ):
        raise ReproductionStageError("reproduction configuration is invalid")
    return config


def _source_records(document: Mapping[str, object]) -> list[JsonObject]:
    _require_exact_keys(document, {"fixture_only", "records", "schema_version"}, "source fixture")
    if (
        document["schema_version"] != "synthetic-recording-source/1"
        or document["fixture_only"] is not True
    ):
        raise ReproductionStageError("source fixture is not explicitly synthetic")
    records = document["records"]
    if not isinstance(records, list) or not records:
        raise ReproductionStageError("source fixture records are invalid")
    checked: list[JsonObject] = []
    for record in records:
        if not isinstance(record, dict):
            raise ReproductionStageError("source fixture records are invalid")
        _require_exact_keys(
            record,
            {"observations", "recording_id", "session_group", "signer_group"},
            "source record",
        )
        if not all(
            isinstance(record[key], str) and cast(str, record[key]).startswith("synthetic_")
            for key in ("recording_id", "session_group", "signer_group")
        ):
            raise ReproductionStageError("source fixture identifiers must be synthetic")
        observations = record["observations"]
        if (
            not isinstance(observations, list)
            or not observations
            or any(
                not isinstance(observation, list)
                or len(observation) != 2
                or any(type(value) is not int for value in observation)
                for observation in observations
            )
        ):
            raise ReproductionStageError("source fixture observations are invalid")
        checked.append(cast(JsonObject, record))
    recording_ids = [cast(str, record["recording_id"]) for record in checked]
    if recording_ids != sorted(set(recording_ids)):
        raise ReproductionStageError("source fixture recording IDs must be sorted and unique")
    signer_groups = {cast(str, record["signer_group"]) for record in checked}
    session_groups = {cast(str, record["session_group"]) for record in checked}
    if len(signer_groups) < 3 or len(session_groups) < 3:
        raise ReproductionStageError("source fixture requires three independent groups")
    return checked


def _stage_document(stage: StageName, upstream_bytes: bytes, payload: JsonObject) -> JsonObject:
    return {
        "fixture_only": True,
        "implementation": FIXTURE_IMPLEMENTATION,
        "payload": payload,
        "profile": FIXTURE_PROFILE,
        "schema_version": "synthetic-dvc-stage/1",
        "stage": stage,
        "upstream_sha256": f"sha256:{hashlib.sha256(upstream_bytes).hexdigest()}",
    }


def _predecessor_spec(spec: StageSpec) -> StageSpec | None:
    if spec.predecessor is None:
        if spec.name != "ingest" or spec.input_path != SOURCE_PATH:
            raise ReproductionStageError("stage registry lineage is invalid")
        return None
    try:
        predecessor = _STAGES_BY_NAME[spec.predecessor]
    except KeyError as error:
        raise ReproductionStageError("stage registry lineage is invalid") from error
    if predecessor.output_path != spec.input_path:
        raise ReproductionStageError("stage registry lineage is invalid")
    return predecessor


def _validate_stage_receipt(
    root: Path,
    spec: StageSpec,
    document: JsonObject,
    *,
    visited: frozenset[StageName] = frozenset(),
) -> None:
    if spec.name in visited:
        raise ReproductionStageError("stage registry lineage is invalid")
    _require_exact_keys(document, _RECEIPT_KEYS, "upstream stage receipt")
    if (
        document["schema_version"] != "synthetic-dvc-stage/1"
        or document["profile"] != FIXTURE_PROFILE
        or document["fixture_only"] is not True
        or document["implementation"] != FIXTURE_IMPLEMENTATION
        or document["stage"] != spec.name
        or not isinstance(document["payload"], dict)
    ):
        raise ReproductionStageError("upstream stage receipt is invalid")

    predecessor = _predecessor_spec(spec)
    upstream_document, upstream_bytes = _read_json(root, spec.input_path)
    expected_sha256 = f"sha256:{hashlib.sha256(upstream_bytes).hexdigest()}"
    if document["upstream_sha256"] != expected_sha256:
        raise ReproductionStageError("upstream stage receipt is invalid")
    if predecessor is None:
        _source_records(upstream_document)
        return
    _validate_stage_receipt(
        root,
        predecessor,
        upstream_document,
        visited=visited | {spec.name},
    )


def _load_stage_input(root: Path, spec: StageSpec) -> tuple[JsonObject, bytes]:
    document, document_bytes = _read_json(root, spec.input_path)
    predecessor = _predecessor_spec(spec)
    if predecessor is None:
        _source_records(document)
        return document, document_bytes
    _validate_stage_receipt(root, predecessor, document)
    return document, document_bytes


def _ingest(document: JsonObject, _config: JsonObject) -> JsonObject:
    records = _source_records(document)
    return {
        "record_count": len(records),
        "records": records,
        "storage_claim": "public synthetic JSON bytes verified at read time",
    }


def _validate(document: JsonObject, _config: JsonObject) -> JsonObject:
    payload = cast(JsonObject, document["payload"])
    _require_exact_keys(payload, {"record_count", "records", "storage_claim"}, "ingest payload")
    records = _source_records(
        {
            "fixture_only": True,
            "records": payload["records"],
            "schema_version": "synthetic-recording-source/1",
        }
    )
    return {
        "checks": ["explicitly_synthetic", "group_ids_unique", "recording_ids_unique"],
        "record_count": len(records),
        "records": records,
    }


def _extract(document: JsonObject, _config: JsonObject) -> JsonObject:
    payload = cast(JsonObject, document["payload"])
    records = cast(list[JsonObject], payload.get("records"))
    extracted: list[JsonObject] = []
    for record in records:
        observations = cast(list[list[int]], record["observations"])
        vectors = [[x, y, x - y] for x, y in observations]
        extracted.append(
            {
                "recording_id": record["recording_id"],
                "session_group": record["session_group"],
                "signer_group": record["signer_group"],
                "synthetic_vectors": vectors,
            }
        )
    return {
        "extractor": "synthetic-vector-transform/1",
        "production_landmarks_computed": False,
        "records": extracted,
    }


def _quality(document: JsonObject, config: JsonObject) -> JsonObject:
    payload = cast(JsonObject, document["payload"])
    records = cast(list[JsonObject], payload.get("records"))
    minimum = cast(int, config["minimum_valid_vectors"])
    checked: list[JsonObject] = []
    for record in records:
        vectors = cast(list[list[int]], record["synthetic_vectors"])
        checked.append(
            {**record, "accepted": len(vectors) >= minimum, "valid_vectors": len(vectors)}
        )
    if not all(cast(bool, record["accepted"]) for record in checked):
        raise ReproductionStageError("synthetic quality fixture unexpectedly failed")
    return {
        "minimum_valid_vectors": minimum,
        "production_quality_evaluated": False,
        "records": checked,
    }


def _split(document: JsonObject, config: JsonObject) -> JsonObject:
    payload = cast(JsonObject, document["payload"])
    records = cast(list[JsonObject], payload.get("records"))
    seed = cast(int, config["random_seed"])
    signer_groups = sorted({cast(str, record["signer_group"]) for record in records})
    ranked_groups = sorted(
        signer_groups,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(),
    )
    partition_names = ("train", "validation", "test")
    assignments = {
        group: partition_names[min(index, 2)] for index, group in enumerate(ranked_groups)
    }
    split_records = [
        {**record, "partition": assignments[cast(str, record["signer_group"])]}
        for record in records
    ]
    return {
        "group_leakage_detected": False,
        "production_split_computed": False,
        "records": split_records,
        "strategy": "synthetic-signer-grouped/1",
    }


def _feature(document: JsonObject, _config: JsonObject) -> JsonObject:
    payload = cast(JsonObject, document["payload"])
    records = cast(list[JsonObject], payload.get("records"))
    features: list[JsonObject] = []
    for record in records:
        vectors = cast(list[list[int]], record["synthetic_vectors"])
        columns = tuple(zip(*vectors, strict=True))
        features.append(
            {
                "feature_count": len(vectors),
                "feature_max": [max(column) for column in columns],
                "feature_min": [min(column) for column in columns],
                "feature_sum": [sum(column) for column in columns],
                "partition": record["partition"],
                "recording_id": record["recording_id"],
            }
        )
    return {
        "features": features,
        "production_features_computed": False,
        "representation": "synthetic-integer-summary/1",
    }


_TRANSFORMS: Final[dict[StageName, Callable[[JsonObject, JsonObject], JsonObject]]] = {
    "ingest": _ingest,
    "validate": _validate,
    "extract": _extract,
    "quality": _quality,
    "split": _split,
    "feature": _feature,
}


def run_reproduction_stage(stage: StageName, repository_root: Path) -> Path:
    """Run one deterministic public-fixture stage and atomically publish its receipt."""

    try:
        spec = _STAGES_BY_NAME[stage]
    except KeyError as error:
        raise ReproductionStageError("unknown reproduction stage") from error
    root = repository_root
    if not root.is_absolute():
        raise ReproductionStageError("repository root must be absolute")
    config = _load_config(root)
    upstream, upstream_bytes = _load_stage_input(root, spec)
    try:
        payload = _TRANSFORMS[stage](upstream, config)
    except ReproductionStageError:
        raise
    except (KeyError, OverflowError, TypeError, ValueError) as error:
        raise ReproductionStageError("upstream synthetic stage receipt is invalid") from error
    output_document = _stage_document(stage, upstream_bytes, payload)
    _atomic_write_json(root, spec.output_path, output_document)
    return root.joinpath(*spec.output_path.split("/"))


def render_dvc_pipeline() -> str:
    """Render the root DVC graph from the typed registry without YAML features."""

    lines = ["stages:"]
    for spec in STAGE_REGISTRY:
        lines.extend((f"  {spec.name}:", f"    cmd: {spec.command}", "    deps:"))
        lines.extend(f"      - {dependency}" for dependency in spec.dependencies)
        lines.extend(("    outs:", f"      - {spec.output_path}"))
    return "\n".join(lines) + "\n"


__all__ = [
    "CONFIG_PATH",
    "FIXTURE_IMPLEMENTATION",
    "FIXTURE_PROFILE",
    "SOURCE_PATH",
    "STAGE_NAMES",
    "STAGE_REGISTRY",
    "ReproductionStageError",
    "StageName",
    "StageSpec",
    "render_dvc_pipeline",
    "run_reproduction_stage",
]
