from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import cast

import pytest

from signlab.contracts.canonical import canonical_json_bytes, parse_json_object
from signlab.reproducibility import stages as stage_services
from signlab.reproducibility.stages import (
    CONFIG_PATH,
    SOURCE_PATH,
    STAGE_NAMES,
    STAGE_REGISTRY,
    ReproductionStageError,
    render_dvc_pipeline,
    run_reproduction_stage,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _copy_fixture_workspace(target: Path) -> Path:
    for relative_path in (CONFIG_PATH, SOURCE_PATH):
        source = REPOSITORY_ROOT.joinpath(*relative_path.split("/"))
        destination = target.joinpath(*relative_path.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return target.resolve()


def test_registry_is_the_exact_leakage_safe_six_stage_graph() -> None:
    assert STAGE_NAMES == ("ingest", "validate", "extract", "quality", "split", "feature")
    assert tuple(spec.predecessor for spec in STAGE_REGISTRY) == (
        None,
        "ingest",
        "validate",
        "extract",
        "quality",
        "split",
    )
    assert len({spec.output_path.casefold() for spec in STAGE_REGISTRY}) == len(STAGE_REGISTRY)
    expected_shared_dependencies = (
        CONFIG_PATH,
        ".python-version",
        "pyproject.toml",
        "src/signlab",
        "uv.lock",
    )
    for index, spec in enumerate(STAGE_REGISTRY):
        assert spec.command == f"python -m signlab.cli data run-reproduction-stage {spec.name}"
        assert not any(token in spec.command for token in ("&&", "||", ";", "|", ">", "<"))
        assert spec.dependencies == (
            *expected_shared_dependencies,
            *(candidate.input_path for candidate in STAGE_REGISTRY[: index + 1]),
        )
        assert "\\" not in spec.input_path
        assert "\\" not in spec.output_path
        assert spec.output_path.startswith(("data/raw/", "data/interim/", "data/processed/"))
        if index == 0:
            assert spec.input_path == SOURCE_PATH
        else:
            assert spec.input_path == STAGE_REGISTRY[index - 1].output_path


def test_root_dvc_yaml_is_generated_from_the_registry() -> None:
    assert (REPOSITORY_ROOT / "dvc.yaml").read_text(encoding="utf-8") == render_dvc_pipeline()


def test_public_fixture_chain_is_deterministic_and_truthfully_labeled(tmp_path: Path) -> None:
    repository = _copy_fixture_workspace(tmp_path)

    first_outputs = [run_reproduction_stage(stage, repository) for stage in STAGE_NAMES]
    first_bytes = [output.read_bytes() for output in first_outputs]
    second_outputs = [run_reproduction_stage(stage, repository) for stage in STAGE_NAMES]

    assert [output.read_bytes() for output in second_outputs] == first_bytes
    for expected_stage, output in zip(STAGE_NAMES, second_outputs, strict=True):
        document = parse_json_object(output.read_bytes())
        assert document["stage"] == expected_stage
        assert document["fixture_only"] is True
        assert document["implementation"] == "fixture-only/1"
    final_document = parse_json_object(second_outputs[-1].read_bytes())
    final_payload = final_document["payload"]
    assert isinstance(final_payload, dict)
    assert final_payload["production_features_computed"] is False
    features = cast(list[dict[str, object]], final_payload["features"])
    assert {feature["partition"] for feature in features} == {
        "train",
        "validation",
        "test",
    }


def test_each_receipt_binds_the_exact_upstream_bytes(tmp_path: Path) -> None:
    repository = _copy_fixture_workspace(tmp_path)
    previous = repository.joinpath(*SOURCE_PATH.split("/"))

    for stage in STAGE_NAMES:
        output = run_reproduction_stage(stage, repository)
        document = parse_json_object(output.read_bytes())
        assert (
            document["upstream_sha256"]
            == f"sha256:{hashlib.sha256(previous.read_bytes()).hexdigest()}"
        )
        previous = output


def test_failed_rerun_does_not_replace_a_valid_output(tmp_path: Path) -> None:
    repository = _copy_fixture_workspace(tmp_path)
    output = run_reproduction_stage("ingest", repository)
    original = output.read_bytes()
    source = repository.joinpath(*SOURCE_PATH.split("/"))
    source.write_text('{"fixture_only":false}\n', encoding="utf-8")

    with pytest.raises(ReproductionStageError, match="fixture contract"):
        run_reproduction_stage("ingest", repository)

    assert output.read_bytes() == original


def test_duplicate_json_members_fail_closed_without_echoing_input(tmp_path: Path) -> None:
    repository = _copy_fixture_workspace(tmp_path)
    source = repository.joinpath(*SOURCE_PATH.split("/"))
    source.write_text(
        '{"schema_version":"synthetic-recording-source/1","fixture_only":true,'
        '"fixture_only":true,"records":[]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ReproductionStageError, match="strict JSON") as raised:
        run_reproduction_stage("ingest", repository)

    assert "records" not in str(raised.value)


def test_linked_source_fixture_is_rejected_when_supported(tmp_path: Path) -> None:
    repository = _copy_fixture_workspace(tmp_path)
    source = repository.joinpath(*SOURCE_PATH.split("/"))
    external = tmp_path / "external.json"
    shutil.copy2(source, external)
    source.unlink()
    try:
        source.symlink_to(external)
    except OSError:
        pytest.skip("file symlinks are unavailable for this account")

    with pytest.raises(ReproductionStageError, match="links or reparse"):
        run_reproduction_stage("ingest", repository)


def test_hardlinked_source_fixture_is_rejected_when_supported(tmp_path: Path) -> None:
    repository = _copy_fixture_workspace(tmp_path)
    source = repository.joinpath(*SOURCE_PATH.split("/"))
    external = tmp_path / "external-hardlink.json"
    shutil.copy2(source, external)
    source.unlink()
    try:
        os.link(external, source)
    except OSError:
        pytest.skip("hard links are unavailable for this account")

    with pytest.raises(ReproductionStageError, match="regular files and directories"):
        run_reproduction_stage("ingest", repository)


def test_malformed_upstream_receipt_has_stable_error_and_preserves_output(
    tmp_path: Path,
) -> None:
    repository = _copy_fixture_workspace(tmp_path)
    for stage in ("ingest", "validate", "extract"):
        output = run_reproduction_stage(stage, repository)
    original = output.read_bytes()
    validate_output = repository.joinpath(*STAGE_REGISTRY[1].output_path.split("/"))
    receipt = parse_json_object(validate_output.read_bytes())
    payload = cast(dict[str, object], receipt["payload"])
    payload["records"] = None
    validate_output.write_bytes(canonical_json_bytes(receipt) + b"\n")

    with pytest.raises(
        ReproductionStageError,
        match="upstream synthetic stage receipt is invalid",
    ):
        run_reproduction_stage("extract", repository)

    assert output.read_bytes() == original


def test_valid_immediate_hash_cannot_hide_a_corrupt_ancestor_receipt(tmp_path: Path) -> None:
    repository = _copy_fixture_workspace(tmp_path)
    outputs = {
        stage: run_reproduction_stage(stage, repository)
        for stage in ("ingest", "validate", "extract")
    }
    original_extract = outputs["extract"].read_bytes()

    ingest_receipt = parse_json_object(outputs["ingest"].read_bytes())
    ingest_receipt["upstream_sha256"] = "sha256:" + "0" * 64
    outputs["ingest"].write_bytes(canonical_json_bytes(ingest_receipt) + b"\n")

    validate_receipt = parse_json_object(outputs["validate"].read_bytes())
    validate_receipt["upstream_sha256"] = (
        f"sha256:{hashlib.sha256(outputs['ingest'].read_bytes()).hexdigest()}"
    )
    outputs["validate"].write_bytes(canonical_json_bytes(validate_receipt) + b"\n")

    with pytest.raises(ReproductionStageError, match="upstream stage receipt is invalid"):
        run_reproduction_stage("extract", repository)

    assert outputs["extract"].read_bytes() == original_extract


def test_atomic_writer_rejects_a_replaced_output_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _copy_fixture_workspace(tmp_path)
    original = stage_services._ensure_output_parent

    def replace_parent(
        root: Path,
        relative_path: str,
    ) -> tuple[Path, tuple[int, int]]:
        output, identity = original(root, relative_path)
        displaced = output.parent.with_name(f"{output.parent.name}-displaced")
        output.parent.rename(displaced)
        output.parent.mkdir()
        return output, identity

    monkeypatch.setattr(stage_services, "_ensure_output_parent", replace_parent)

    with pytest.raises(ReproductionStageError, match="output parent changed"):
        run_reproduction_stage("ingest", repository)

    output = repository.joinpath(*STAGE_REGISTRY[0].output_path.split("/"))
    assert not output.exists()
    assert tuple(output.parent.glob("*.tmp")) == ()
