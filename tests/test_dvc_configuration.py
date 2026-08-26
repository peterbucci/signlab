from __future__ import annotations

import configparser
import subprocess
import tomllib
from pathlib import Path
from typing import cast

import yaml

from signlab.contracts.core import WorkspaceRelativeLocatorV1
from signlab.reproducibility.stages import STAGE_NAMES, STAGE_REGISTRY

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return REPOSITORY_ROOT.joinpath(*relative_path.split("/")).read_text(encoding="utf-8")


def test_dvc_and_s3_transport_are_exact_locked_research_dependencies() -> None:
    project = tomllib.loads(_read("pyproject.toml"))

    assert project["dependency-groups"]["reproducibility"] == ["dvc[s3]==3.67.1"]
    assert project["project"]["dependencies"].count("PyYAML==6.0.3") == 1
    assert project["tool"]["uv"]["default-groups"] == ["dev", "reproducibility"]
    lock = _read("uv.lock")
    assert 'name = "dvc"\nversion = "3.67.1"' in lock
    assert 'name = "dvc-s3"\nversion = "3.3.0"' in lock


def test_tracked_dvc_config_has_only_fail_closed_public_defaults() -> None:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(_read(".dvc/config"))

    assert {section: dict(parser[section]) for section in parser.sections()} == {
        "core": {"analytics": "false", "autostage": "false", "check_update": "false"},
        "cache": {"type": '"reflink,copy"'},
        "studio": {"offline": "true"},
        "exp": {"auto_push": "false"},
    }
    assert "remote" not in {section.casefold() for section in parser.sections()}


def test_local_dvc_state_and_all_pipeline_outputs_are_ignored() -> None:
    ignored_paths = (
        ".dvc/cache/example",
        ".dvc/config.local",
        ".dvc/tmp/example",
        *(spec.output_path for spec in STAGE_REGISTRY),
    )
    for relative_path in ignored_paths:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", relative_path],
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        assert result.returncode == 0, relative_path


def test_runtime_source_dependency_ignores_only_generated_python_caches() -> None:
    ignored_paths = (
        "src/signlab/__pycache__/cli.cpython-312.pyc",
        "src/signlab/reproducibility/__pycache__/stages.cpython-312.pyo",
    )
    for relative_path in ignored_paths:
        result = subprocess.run(
            ["dvc", "check-ignore", "--quiet", "--", relative_path],
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        assert result.returncode == 0, relative_path

    tracked_source = subprocess.run(
        ["dvc", "check-ignore", "--quiet", "--", "src/signlab/reproducibility/stages.py"],
        cwd=REPOSITORY_ROOT,
        check=False,
    )
    assert tracked_source.returncode == 1


def test_root_dvc_pipeline_is_the_exact_typed_registry_projection() -> None:
    pipeline = yaml.safe_load(_read("dvc.yaml"))
    assert isinstance(pipeline, dict)
    assert set(pipeline) == {"stages"}
    stages = cast(dict[str, dict[str, object]], pipeline["stages"])

    assert tuple(stages) == STAGE_NAMES
    for spec in STAGE_REGISTRY:
        stage = stages[spec.name]
        assert set(stage) == {"cmd", "deps", "outs"}
        assert stage["cmd"] == spec.command
        assert stage["deps"] == list(spec.dependencies)
        assert stage["outs"] == [spec.output_path]
        for path in (*spec.dependencies, spec.output_path):
            if path == ".python-version":
                assert path.count("/") == 0
            else:
                WorkspaceRelativeLocatorV1(kind="workspace_relative", path=path)


def test_lockfile_matches_the_exact_public_fixture_graph() -> None:
    lock = yaml.safe_load(_read("dvc.lock"))
    assert isinstance(lock, dict)
    assert set(lock) == {"schema", "stages"}
    assert lock["schema"] == "2.0"
    stages = cast(dict[str, dict[str, object]], lock["stages"])
    assert tuple(stages) == STAGE_NAMES

    for spec in STAGE_REGISTRY:
        stage = stages[spec.name]
        assert stage["cmd"] == spec.command
        dependencies = cast(list[dict[str, object]], stage["deps"])
        outputs = cast(list[dict[str, object]], stage["outs"])
        assert {dependency["path"] for dependency in dependencies} == set(spec.dependencies)
        assert [output["path"] for output in outputs] == [spec.output_path]
        assert all(entry["hash"] == "md5" for entry in (*dependencies, *outputs))
        for entry in (*dependencies, *outputs):
            native_hash = entry["md5"]
            assert isinstance(native_hash, str)
            is_directory = native_hash.endswith(".dir")
            digest = native_hash.removesuffix(".dir")
            assert len(digest) == 32
            assert all(character in "0123456789abcdef" for character in digest)
            assert is_directory is ("nfiles" in entry)
            if is_directory:
                file_count = entry["nfiles"]
                assert type(file_count) is int
                assert file_count > 0


def test_public_repository_tracks_no_dvc_data_pointers_or_nested_pipelines() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    paths = tuple(line.strip().replace("\\", "/") for line in result.stdout.splitlines())

    assert not any(path.endswith(".dvc") for path in paths)
    assert [path for path in paths if path.endswith("dvc.yaml")] == ["dvc.yaml"]
    assert ".dvc/config.local" not in paths
