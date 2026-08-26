from __future__ import annotations

import configparser
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from signlab.reproducibility import DVC_VERSION, remote


def _install_fake_dvc(
    monkeypatch: pytest.MonkeyPatch,
    repository: Path,
) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []
    local_config = repository / ".dvc" / "config.local"

    def run_dvc(root: Path, arguments: Sequence[str]) -> None:
        assert root == repository
        call = tuple(arguments)
        calls.append(call)
        parser = configparser.ConfigParser(interpolation=None)
        if local_config.exists():
            parser.read(local_config, encoding="utf-8")
        section = f'remote "{remote.DVC_REMOTE_NAME}"'
        if not parser.has_section(section):
            parser.add_section(section)
        if call[:2] == ("remote", "add"):
            parser.set(section, "url", call[-1])
        elif call[:2] == ("remote", "modify"):
            parser.set(section, call[-2], call[-1])
        else:  # pragma: no cover - catches accidental command expansion.
            raise AssertionError(call)
        with local_config.open("w", encoding="utf-8", newline="\n") as stream:
            parser.write(stream)

    monkeypatch.setattr(remote, "_run_dvc", run_dvc)
    return calls


def _prepare_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repository = tmp_path.resolve()
    (repository / ".dvc").mkdir()
    monkeypatch.setattr(remote, "version", lambda _package: DVC_VERSION)
    monkeypatch.setattr(remote, "_require_repository", lambda _root: repository)
    return repository


def test_configure_private_remote_writes_only_local_non_secret_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _prepare_repository(tmp_path, monkeypatch)
    calls = _install_fake_dvc(monkeypatch, repository)
    secret = "private-provider-secret"
    environment = {
        remote.DVC_REMOTE_URL_ENV: "s3://signlab-private/datasets",
        remote.DVC_ENDPOINT_URL_ENV: "https://objects.example.test/",
        remote.DVC_REGION_ENV: "us-east-1",
        "AWS_SECRET_ACCESS_KEY": secret,
    }

    result = remote.configure_private_dvc_remote(repository, environment)

    assert result == remote.DvcRemoteConfigurationResult(
        remote_name="private",
        endpoint_configured=True,
        region_configured=True,
    )
    assert calls == [
        (
            "remote",
            "add",
            "--local",
            "--force",
            "--default",
            "private",
            "s3://signlab-private/datasets",
        ),
        (
            "remote",
            "modify",
            "--local",
            "private",
            "endpointurl",
            "https://objects.example.test",
        ),
        ("remote", "modify", "--local", "private", "region", "us-east-1"),
    ]
    payload = (repository / ".dvc" / "config.local").read_text(encoding="utf-8")
    assert secret not in payload
    assert all(secret not in argument for call in calls for argument in call)


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {remote.DVC_REMOTE_URL_ENV: "s3://user:password@signlab-private/data"},
        {remote.DVC_REMOTE_URL_ENV: "s3://signlab-private/data?token=private"},
        {remote.DVC_REMOTE_URL_ENV: "https://signlab-private/data"},
        {
            remote.DVC_REMOTE_URL_ENV: "s3://signlab-private/data",
            remote.DVC_ENDPOINT_URL_ENV: "http://objects.example.test",
        },
        {
            remote.DVC_REMOTE_URL_ENV: "s3://signlab-private/data",
            remote.DVC_REGION_ENV: "US_EAST_1",
        },
    ],
)
def test_invalid_remote_settings_fail_without_echoing_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
) -> None:
    monkeypatch.setattr(remote, "version", lambda _package: DVC_VERSION)

    with pytest.raises(remote.DvcRemoteConfigurationError) as captured:
        remote.configure_private_dvc_remote(tmp_path, environment)

    assert str(captured.value) in {
        "private DVC remote URL is required",
        "private DVC remote configuration is invalid",
    }
    assert not any(value and value in str(captured.value) for value in environment.values())


@pytest.mark.parametrize("had_previous_config", [False, True])
def test_failed_configuration_rolls_back_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    had_previous_config: bool,
) -> None:
    repository = _prepare_repository(tmp_path, monkeypatch)
    local_config = repository / ".dvc" / "config.local"
    previous = b'[remote "existing"]\nurl = s3://existing-private/data\n'
    if had_previous_config:
        local_config.write_bytes(previous)

    def fail_after_partial_write(_root: Path, _arguments: Sequence[str]) -> None:
        local_config.write_text(
            '[remote "private"]\nurl = s3://private-sentinel/data\n',
            encoding="utf-8",
        )
        raise remote.DvcRemoteConfigurationError("private-sentinel")

    monkeypatch.setattr(remote, "_run_dvc", fail_after_partial_write)

    with pytest.raises(
        remote.DvcRemoteConfigurationError,
        match=r"^private DVC remote configuration failed$",
    ) as captured:
        remote.configure_private_dvc_remote(
            repository,
            {remote.DVC_REMOTE_URL_ENV: "s3://signlab-private/data"},
        )

    assert "sentinel" not in str(captured.value)
    if had_previous_config:
        assert local_config.read_bytes() == previous
    else:
        assert not local_config.exists()


def test_dvc_subprocess_uses_the_current_interpreter_and_redacts_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    secret = "private-command-output"

    def fail(
        command: Sequence[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        observed["command"] = tuple(command)
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 1, stdout=secret, stderr=secret)

    monkeypatch.setattr(subprocess, "run", fail)

    with pytest.raises(
        remote.DvcRemoteConfigurationError,
        match=r"^DVC configuration command failed$",
    ) as captured:
        remote._run_dvc(tmp_path, ("remote", "list"))

    assert observed["command"] == (
        sys.executable,
        "-I",
        "-m",
        "dvc",
        "remote",
        "list",
    )
    assert observed["cwd"] == tmp_path
    assert observed["capture_output"] is True
    assert secret not in str(captured.value)


def test_local_config_validator_rejects_credential_fields(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.local"
    secret = "private-sentinel"
    config.write_text(
        f'[remote "private"]\nurl = s3://signlab-private/data\naccess_key_id = {secret}\n',
        encoding="utf-8",
    )

    with pytest.raises(remote.DvcRemoteConfigurationError) as captured:
        remote._validate_local_config(
            config,
            "s3://signlab-private/data",
            endpoint=None,
            region=None,
        )

    assert str(captured.value) == "credentials must not be stored in DVC config"
    assert secret not in str(captured.value)
