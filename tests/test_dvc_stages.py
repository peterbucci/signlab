from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast

import pytest
import yaml

from signlab.contracts.canonical import canonical_json_bytes, parse_json_object
from signlab.reproducibility.stages import (
    CONFIG_PATH,
    FIXTURE_IMPLEMENTATION,
    FIXTURE_PROFILE,
    IMPLEMENTATION_PATH,
    SOURCE_PATH,
    STAGE_NAMES,
    STAGE_REGISTRY,
    ReproductionStageError,
    StageName,
    render_dvc_pipeline,
    run_reproduction_stage,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STAGES = ("ingest", "validate", "extract", "quality", "split", "feature")


def _copy_fixture_workspace(target: Path) -> Path:
    for relative_path in (CONFIG_PATH, SOURCE_PATH):
        source = REPOSITORY_ROOT.joinpath(*relative_path.split("/"))
        destination = target.joinpath(*relative_path.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    return target.resolve()


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def test_registry_is_the_single_six_stage_graph_with_immediate_dependencies() -> None:
    assert STAGE_NAMES == EXPECTED_STAGES
    assert tuple(spec.predecessor for spec in STAGE_REGISTRY) == (
        None,
        "ingest",
        "validate",
        "extract",
        "quality",
        "split",
    )

    for index, spec in enumerate(STAGE_REGISTRY):
        assert spec.command == f"python -m signlab.cli data run-reproduction-stage {spec.name}"
        assert spec.dependencies == (CONFIG_PATH, IMPLEMENTATION_PATH, spec.input_path)
        if index == 0:
            assert spec.input_path == SOURCE_PATH
        else:
            assert spec.input_path == STAGE_REGISTRY[index - 1].output_path

    rendered = yaml.safe_load(render_dvc_pipeline())
    assert rendered == {
        "stages": {
            spec.name: {
                "cmd": spec.command,
                "deps": list(spec.dependencies),
                "outs": [spec.output_path],
            }
            for spec in STAGE_REGISTRY
        }
    }


def test_fixture_chain_is_deterministic_and_binds_each_immediate_input(tmp_path: Path) -> None:
    root = _copy_fixture_workspace(tmp_path)
    config_bytes = root.joinpath(*CONFIG_PATH.split("/")).read_bytes()
    upstream_bytes = root.joinpath(*SOURCE_PATH.split("/")).read_bytes()
    first_outputs: dict[str, bytes] = {}

    for spec in STAGE_REGISTRY:
        output = run_reproduction_stage(spec.name, root)
        payload = output.read_bytes()
        receipt = parse_json_object(payload)
        assert receipt == {
            "config_sha256": _sha256(config_bytes),
            "fixture_only": True,
            "implementation": FIXTURE_IMPLEMENTATION,
            "input_sha256": _sha256(upstream_bytes),
            "profile": FIXTURE_PROFILE,
            "schema_version": "synthetic-dvc-stage/1",
            "stage": spec.name,
        }
        first_outputs[spec.name] = payload
        upstream_bytes = payload

    for spec in STAGE_REGISTRY:
        output = run_reproduction_stage(spec.name, root)
        assert output.read_bytes() == first_outputs[spec.name]


def test_runner_rejects_unknown_stage_invalid_config_and_broken_lineage(tmp_path: Path) -> None:
    root = _copy_fixture_workspace(tmp_path)

    with pytest.raises(ReproductionStageError, match="unknown reproduction stage"):
        run_reproduction_stage(cast(StageName, "publish"), root)

    config_path = root.joinpath(*CONFIG_PATH.split("/"))
    original_config = config_path.read_bytes()
    config_path.write_text('{"fixture_only":false}\n', encoding="utf-8")
    with pytest.raises(ReproductionStageError, match="configuration is invalid"):
        run_reproduction_stage("ingest", root)

    config_path.write_bytes(original_config)
    ingest_output = run_reproduction_stage("ingest", root)
    receipt = parse_json_object(ingest_output.read_bytes())
    receipt["stage"] = "feature"
    ingest_output.write_bytes(canonical_json_bytes(receipt) + b"\n")
    with pytest.raises(ReproductionStageError, match="lineage is invalid"):
        run_reproduction_stage("validate", root)
