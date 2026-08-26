from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from signlab import cli
from signlab.commands import taxonomy as taxonomy_command
from signlab.contracts.taxonomy import (
    BUILTIN_TAXONOMY_DIGEST,
    EXPECTED_CLASS_IDS,
    TaxonomyContractError,
    load_builtin_taxonomy,
    taxonomy_reference,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"NO_COLOR": "1"})


def test_validate_builtin_taxonomy_prints_only_portable_identity(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["taxonomy", "validate"])

    assert result.exit_code == 0
    assert "signlab-five@1.0.0" in result.output
    assert BUILTIN_TAXONOMY_DIGEST in result.output
    assert "Users" not in result.output


def test_validate_external_taxonomy_uses_a_generic_source_label(
    runner: CliRunner, tmp_path: Path
) -> None:
    path = tmp_path / "participant-name-must-not-echo.json"
    path.write_text(
        json.dumps(load_builtin_taxonomy().model_dump(mode="json")),
        encoding="utf-8",
    )

    result = runner.invoke(cli.app, ["taxonomy", "validate", str(path)])

    assert result.exit_code == 0
    assert "external" in result.output
    assert path.name not in result.output
    assert str(tmp_path) not in result.output


def test_validate_invalid_taxonomy_returns_a_redacted_failure(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{}", encoding="utf-8")

    result = runner.invoke(cli.app, ["taxonomy", "validate", str(path)])

    assert result.exit_code == 1
    assert "Taxonomy validation failed" in result.output
    assert str(tmp_path) not in result.output
    assert "schema_version" in result.output


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_unreadable_external_paths_never_echo_identifiers(
    runner: CliRunner,
    tmp_path: Path,
    kind: str,
) -> None:
    path = tmp_path / "participant-name-must-not-echo.json"
    if kind == "directory":
        path.mkdir()

    result = runner.invoke(cli.app, ["taxonomy", "validate", str(path)])

    assert result.exit_code == 1
    assert "file could not be read" in result.output
    assert path.name not in result.output
    assert str(tmp_path) not in result.output


def test_show_outputs_the_validated_packaged_document(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["taxonomy", "show"])

    assert result.exit_code == 0
    assert json.loads(result.output)["taxonomy_id"] == "signlab-five"


def test_validate_resources_reads_every_packaged_schema(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["taxonomy", "validate-resources"])

    assert result.exit_code == 0
    assert result.output.strip() == "Packaged taxonomy and schemas are valid."


@pytest.mark.parametrize("command", ["validate", "validate-resources", "show"])
def test_builtin_integrity_failure_returns_nonzero_without_traceback(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    monkeypatch.setattr(
        taxonomy_command,
        "load_builtin_taxonomy",
        lambda: (_ for _ in ()).throw(TaxonomyContractError("integrity check failed")),
    )

    result = runner.invoke(cli.app, ["taxonomy", command])

    assert result.exit_code == 1
    assert "integrity check failed" in result.output
    assert "Traceback" not in result.output


def test_train_entry_point_fails_before_work_when_other_is_missing(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    reference = taxonomy_reference(load_builtin_taxonomy()).model_dump(mode="json")
    document: dict[str, Any] = {
        "schema_version": "training-taxonomy-binding/1",
        "taxonomy": reference,
        "label_map": {label: index for index, label in enumerate(EXPECTED_CLASS_IDS)},
        "label_counts": dict.fromkeys(EXPECTED_CLASS_IDS, 2),
    }
    valid_path = tmp_path / "valid.json"
    valid_path.write_text(json.dumps(document), encoding="utf-8")

    valid = runner.invoke(cli.app, ["train", "validate-taxonomy", str(valid_path)])
    assert valid.exit_code == 0
    assert "learned negative 'other'" in valid.output

    del document["label_map"]["other"]
    invalid_path = tmp_path / "participant-name-must-not-echo.json"
    invalid_path.write_text(json.dumps(document), encoding="utf-8")
    invalid = runner.invoke(cli.app, ["train", "validate-taxonomy", str(invalid_path)])

    assert invalid.exit_code == 1
    assert "label_map.other: Field required" in invalid.output
    assert invalid_path.name not in invalid.output
    assert str(tmp_path) not in invalid.output


def test_train_entry_point_redacts_missing_input_path(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    path = tmp_path / "participant-name-must-not-echo.json"

    result = runner.invoke(cli.app, ["train", "validate-taxonomy", str(path)])

    assert result.exit_code == 1
    assert "input file could not be read" in result.output
    assert path.name not in result.output
