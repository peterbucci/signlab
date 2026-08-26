from __future__ import annotations

import configparser
import subprocess
import tomllib
from pathlib import Path
from typing import cast

import yaml

from signlab.reproducibility.stages import (
    CONFIG_PATH,
    IMPLEMENTATION_PATH,
    STAGE_NAMES,
    STAGE_REGISTRY,
    render_dvc_pipeline,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return REPOSITORY_ROOT.joinpath(*relative_path.split("/")).read_text(encoding="utf-8")


def test_dvc_and_s3_transport_are_pinned_in_the_reproducibility_group() -> None:
    project = tomllib.loads(_read("pyproject.toml"))

    assert project["dependency-groups"]["reproducibility"] == ["dvc[s3]==3.67.1"]
    assert project["tool"]["uv"]["default-groups"] == ["dev"]
    lock = _read("uv.lock")
    assert 'name = "dvc"\nversion = "3.67.1"' in lock
    assert 'name = "dvc-s3"\nversion = "3.3.0"' in lock


def test_public_dvc_config_is_offline_and_contains_no_remote_or_credentials() -> None:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(_read(".dvc/config"))

    assert {section: dict(parser[section]) for section in parser.sections()} == {
        "core": {"analytics": "false", "autostage": "false", "check_update": "false"},
        "cache": {"type": '"reflink,copy"'},
        "studio": {"offline": "true"},
        "exp": {"auto_push": "false"},
    }
    payload = _read(".dvc/config").casefold()
    assert "remote" not in {section.casefold() for section in parser.sections()}
    assert not any(word in payload for word in ("credential", "password", "secret", "token"))


def test_local_dvc_state_outputs_and_reproduction_reports_are_ignored() -> None:
    ignored_paths = (
        ".dvc/cache/example",
        ".dvc/config.local",
        ".dvc/tmp/example",
        "reports/reproduction/proof.json",
        *(spec.output_path for spec in STAGE_REGISTRY),
    )

    for relative_path in ignored_paths:
        result = subprocess.run(
            ("git", "check-ignore", "--quiet", "--no-index", relative_path),
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        assert result.returncode == 0, relative_path


def test_root_pipeline_is_generated_exactly_from_the_typed_registry() -> None:
    assert _read("dvc.yaml") == render_dvc_pipeline()

    pipeline = yaml.safe_load(_read("dvc.yaml"))
    stages = cast(dict[str, dict[str, object]], pipeline["stages"])
    assert tuple(stages) == STAGE_NAMES
    for spec in STAGE_REGISTRY:
        assert stages[spec.name] == {
            "cmd": spec.command,
            "deps": list(spec.dependencies),
            "outs": [spec.output_path],
        }
        assert spec.dependencies == (CONFIG_PATH, IMPLEMENTATION_PATH, spec.input_path)


def test_lockfile_matches_the_generated_public_fixture_graph() -> None:
    lock = yaml.safe_load(_read("dvc.lock"))

    assert lock["schema"] == "2.0"
    stages = cast(dict[str, dict[str, object]], lock["stages"])
    assert tuple(stages) == STAGE_NAMES
    for spec in STAGE_REGISTRY:
        stage = stages[spec.name]
        dependencies = cast(list[dict[str, object]], stage["deps"])
        outputs = cast(list[dict[str, object]], stage["outs"])
        assert stage["cmd"] == spec.command
        assert {entry["path"] for entry in dependencies} == set(spec.dependencies)
        assert [entry["path"] for entry in outputs] == [spec.output_path]


def test_public_repository_tracks_no_private_dvc_pointers_or_nested_graphs() -> None:
    result = subprocess.run(
        ("git", "ls-files", "--cached", "--others", "--exclude-standard"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    paths = tuple(line.strip().replace("\\", "/") for line in result.stdout.splitlines())

    assert not any(path.endswith(".dvc") for path in paths)
    assert [path for path in paths if path.endswith("dvc.yaml")] == ["dvc.yaml"]
    assert ".dvc/config.local" not in paths
