"""Typed DVC graph and deliberately small synthetic smoke-test stages."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final, Literal, cast

import yaml

from signlab.contracts.canonical import canonical_json_bytes, parse_json_object

type StageName = Literal["ingest", "validate", "extract", "quality", "split", "feature"]
type JsonObject = dict[str, object]

FIXTURE_PROFILE: Final = "public-synthetic-reproducibility"
FIXTURE_IMPLEMENTATION: Final = "fixture-smoke/1"
SOURCE_PATH: Final = "tests/fixtures/public/dvc/source.json"
CONFIG_PATH: Final = "configs/pipeline/synthetic-dvc.json"
IMPLEMENTATION_PATH: Final = "src/signlab/reproducibility/stages.py"


class ReproductionStageError(ValueError):
    """Raised when a synthetic pipeline stage cannot produce a valid receipt."""


@dataclass(frozen=True, slots=True)
class StageSpec:
    """One edge in the single SignLab data-stage graph."""

    name: StageName
    predecessor: StageName | None
    input_path: str
    output_path: str

    @property
    def command(self) -> str:
        return f"python -m signlab.cli data run-reproduction-stage {self.name}"

    @property
    def dependencies(self) -> tuple[str, ...]:
        return (CONFIG_PATH, IMPLEMENTATION_PATH, self.input_path)


STAGE_REGISTRY: Final = (
    StageSpec("ingest", None, SOURCE_PATH, "data/raw/dvc-public-fixture/ingest.json"),
    StageSpec(
        "validate",
        "ingest",
        "data/raw/dvc-public-fixture/ingest.json",
        "data/interim/dvc-public-fixture/validate.json",
    ),
    StageSpec(
        "extract",
        "validate",
        "data/interim/dvc-public-fixture/validate.json",
        "data/interim/dvc-public-fixture/extract.json",
    ),
    StageSpec(
        "quality",
        "extract",
        "data/interim/dvc-public-fixture/extract.json",
        "data/interim/dvc-public-fixture/quality.json",
    ),
    StageSpec(
        "split",
        "quality",
        "data/interim/dvc-public-fixture/quality.json",
        "data/processed/dvc-public-fixture/split.json",
    ),
    StageSpec(
        "feature",
        "split",
        "data/processed/dvc-public-fixture/split.json",
        "data/processed/dvc-public-fixture/feature.json",
    ),
)
STAGE_NAMES: Final = tuple(spec.name for spec in STAGE_REGISTRY)
_STAGES_BY_NAME: Final = {spec.name: spec for spec in STAGE_REGISTRY}


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _read_json(path: Path) -> tuple[JsonObject, bytes]:
    try:
        payload = path.read_bytes()
        return cast(JsonObject, parse_json_object(payload)), payload
    except (OSError, TypeError, ValueError) as error:
        raise ReproductionStageError("synthetic fixture input is invalid") from error


def _load_config(repository_root: Path) -> tuple[JsonObject, bytes]:
    config, payload = _read_json(repository_root / CONFIG_PATH)
    if (
        config.get("fixture_only") is not True
        or config.get("profile") != FIXTURE_PROFILE
        or config.get("schema_version") != "synthetic-dvc-profile/1"
    ):
        raise ReproductionStageError("synthetic fixture configuration is invalid")
    return config, payload


def _validate_stage_input(spec: StageSpec, document: JsonObject) -> None:
    if spec.predecessor is None:
        if (
            document.get("fixture_only") is not True
            or document.get("schema_version") != "synthetic-recording-source/1"
            or not isinstance(document.get("records"), list)
        ):
            raise ReproductionStageError("synthetic source fixture is invalid")
        return
    if (
        document.get("fixture_only") is not True
        or document.get("implementation") != FIXTURE_IMPLEMENTATION
        or document.get("profile") != FIXTURE_PROFILE
        or document.get("schema_version") != "synthetic-dvc-stage/1"
        or document.get("stage") != spec.predecessor
    ):
        raise ReproductionStageError("synthetic stage lineage is invalid")


def _write_receipt(path: Path, receipt: JsonObject) -> None:
    payload = canonical_json_bytes(receipt) + b"\n"
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
        temporary_path.replace(path)
    except OSError as error:
        raise ReproductionStageError("synthetic stage output could not be written") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def run_reproduction_stage(stage: StageName, repository_root: Path) -> Path:
    """Run a deterministic receipt stage used only to prove DVC wiring."""

    try:
        spec = _STAGES_BY_NAME[stage]
    except KeyError as error:
        raise ReproductionStageError("unknown reproduction stage") from error
    root = repository_root.resolve()
    _, config_bytes = _load_config(root)
    input_document, input_bytes = _read_json(root / spec.input_path)
    _validate_stage_input(spec, input_document)
    output = root / spec.output_path
    _write_receipt(
        output,
        {
            "config_sha256": _sha256(config_bytes),
            "fixture_only": True,
            "implementation": FIXTURE_IMPLEMENTATION,
            "input_sha256": _sha256(input_bytes),
            "profile": FIXTURE_PROFILE,
            "schema_version": "synthetic-dvc-stage/1",
            "stage": stage,
        },
    )
    return output


def render_dvc_pipeline() -> str:
    """Render ``dvc.yaml`` from the sole stage registry."""

    document = {
        "stages": {
            spec.name: {
                "cmd": spec.command,
                "deps": list(spec.dependencies),
                "outs": [spec.output_path],
            }
            for spec in STAGE_REGISTRY
        }
    }
    return yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=100)


__all__ = [
    "CONFIG_PATH",
    "FIXTURE_IMPLEMENTATION",
    "FIXTURE_PROFILE",
    "IMPLEMENTATION_PATH",
    "SOURCE_PATH",
    "STAGE_NAMES",
    "STAGE_REGISTRY",
    "ReproductionStageError",
    "StageName",
    "StageSpec",
    "render_dvc_pipeline",
    "run_reproduction_stage",
]
