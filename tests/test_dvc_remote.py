from __future__ import annotations

import os
import stat
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

from signlab.reproducibility import remote

_ISOLATED_DIRECTORY_VARIABLES = (
    "APPDATA",
    "DVC_GLOBAL_CONFIG_DIR",
    "DVC_SITE_CACHE_DIR",
    "DVC_SYSTEM_CONFIG_DIR",
    "HOME",
    "LOCALAPPDATA",
    "TEMP",
    "USERPROFILE",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
)


@dataclass(frozen=True, slots=True)
class _RecordedCall:
    arguments: tuple[str, ...]
    cwd: Path
    check: bool
    capture_output: bool
    text: bool
    shell: bool
    timeout: float
    environment: Mapping[str, str] | None


class _RecordingRunner:
    def __init__(
        self,
        repository: Path,
        *,
        failure_call: int | None = None,
        failure_stderr: str = "",
        generated_configuration: bytes | None = None,
        replace_with_directory: bool = False,
        failure_exception: BaseException | None = None,
        git_root: Path | None = None,
        dvc_root: Path | None = None,
        tracked_paths: frozenset[str] = frozenset(),
        unignored_paths: frozenset[str] = frozenset(),
    ) -> None:
        self.repository = repository.resolve()
        self.failure_call = failure_call
        self.failure_stderr = failure_stderr
        self.generated_configuration = generated_configuration
        self.replace_with_directory = replace_with_directory
        self.failure_exception = failure_exception
        self.git_root = git_root.resolve() if git_root is not None else self.repository
        self.dvc_root = dvc_root.resolve() if dvc_root is not None else self.repository
        self.tracked_paths = tracked_paths
        self.unignored_paths = unignored_paths
        self.calls: list[_RecordedCall] = []
        self.identity_calls: list[_RecordedCall] = []

    def __call__(
        self,
        arguments: list[str],
        *,
        cwd: Path,
        check: bool,
        capture_output: bool,
        text: bool,
        shell: bool,
        timeout: float,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if env is not None:
            isolated_directories = {Path(env[name]) for name in _ISOLATED_DIRECTORY_VARIABLES}
            assert all(path.is_dir() for path in isolated_directories)
            assert len({path.parent for path in isolated_directories}) == 1
            assert env["TEMP"] == env["TMP"] == env["TMPDIR"]
            assert env["HOME"] == env["USERPROFILE"]
        recorded = _RecordedCall(
            arguments=tuple(arguments),
            cwd=cwd,
            check=check,
            capture_output=capture_output,
            text=text,
            shell=shell,
            timeout=timeout,
            environment=env,
        )
        if arguments[0] == "git":
            self.identity_calls.append(recorded)
            if arguments[1:3] == ["rev-parse", "--show-toplevel"]:
                return subprocess.CompletedProcess(
                    arguments,
                    0,
                    stdout=f"{self.git_root}\n",
                    stderr="",
                )
            if arguments[1] == "check-ignore":
                returncode = 1 if arguments[-1] in self.unignored_paths else 0
                return subprocess.CompletedProcess(arguments, returncode, stdout="", stderr="")
            if arguments[1] == "ls-files":
                returncode = 0 if arguments[-1] in self.tracked_paths else 1
                return subprocess.CompletedProcess(arguments, returncode, stdout="", stderr="")
            raise AssertionError(arguments)
        if arguments == [sys.executable, "-I", "-m", "dvc", "root"]:
            self.identity_calls.append(recorded)
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout=f"{self.dvc_root}\n",
                stderr="",
            )

        self.calls.append(recorded)
        configuration = self.repository / ".dvc" / "config.local"
        if self.replace_with_directory:
            configuration.unlink(missing_ok=True)
            configuration.mkdir()
        else:
            if self.generated_configuration is not None:
                generated = self.generated_configuration
            else:
                remote_url = os.environ[remote.DVC_REMOTE_URL_ENV]
                values = {"url": remote_url}
                for call in self.calls:
                    if call.arguments[-2] in {"endpointurl", "region"}:
                        values[call.arguments[-2]] = call.arguments[-1]
                lines = ["[core]", "    remote = private", "['remote \"private\"']"]
                lines.extend(f"    {key} = {value}" for key, value in values.items())
                generated = ("\n".join(lines) + "\n").encode()
            configuration.write_bytes(generated)
        if self.failure_call == len(self.calls):
            if self.failure_exception is not None:
                raise self.failure_exception
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=arguments,
                stderr=self.failure_stderr,
            )
        return subprocess.CompletedProcess(arguments, 0, stdout="", stderr="")


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    subprocess.run(
        ["git", "init", "--quiet", str(repository)],
        check=True,
        capture_output=True,
    )
    (repository / ".dvc").mkdir()
    (repository / ".dvc" / "tmp").mkdir()
    (repository / ".dvc" / ".gitignore").write_text(
        "/config.local\n/tmp\n/cache\n",
        encoding="utf-8",
    )
    return repository


def _set_remote_environment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    remote_url: str = "s3://signlab-private/datasets",
    endpoint_url: str | None = None,
    region: str | None = None,
) -> None:
    monkeypatch.setenv(remote.DVC_REMOTE_URL_ENV, remote_url)
    if endpoint_url is None:
        monkeypatch.delenv(remote.DVC_ENDPOINT_URL_ENV, raising=False)
    else:
        monkeypatch.setenv(remote.DVC_ENDPOINT_URL_ENV, endpoint_url)
    if region is None:
        monkeypatch.delenv(remote.DVC_REGION_ENV, raising=False)
    else:
        monkeypatch.setenv(remote.DVC_REGION_ENV, region)


def _install_runner(
    monkeypatch: pytest.MonkeyPatch,
    runner: _RecordingRunner,
) -> None:
    monkeypatch.setattr("signlab.reproducibility.remote.subprocess.run", runner)


def test_remote_configuration_uses_exact_local_dvc_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    runner = _RecordingRunner(repository)
    _install_runner(monkeypatch, runner)
    _set_remote_environment(
        monkeypatch,
        endpoint_url="https://objects.example.test:9443",
        region="us-east-1",
    )

    result = remote.configure_private_dvc_remote(repository)

    assert result == remote.DvcRemoteConfigurationResult(
        remote_name="private",
        endpoint_configured=True,
        region_configured=True,
    )
    assert [call.arguments for call in runner.calls] == [
        (
            sys.executable,
            "-I",
            "-m",
            "dvc",
            "remote",
            "add",
            "--local",
            "--default",
            "--force",
            "private",
            "s3://signlab-private/datasets",
        ),
        (
            sys.executable,
            "-I",
            "-m",
            "dvc",
            "remote",
            "modify",
            "--local",
            "private",
            "endpointurl",
            "https://objects.example.test:9443",
        ),
        (
            sys.executable,
            "-I",
            "-m",
            "dvc",
            "remote",
            "modify",
            "--local",
            "private",
            "region",
            "us-east-1",
        ),
    ]
    assert all(
        call.cwd == repository.resolve()
        and call.check
        and call.capture_output
        and call.text
        and call.shell is False
        and call.timeout == 30.0
        for call in runner.calls
    )
    expected_identity_commands = {
        ("git", "rev-parse", "--show-toplevel"),
        (
            sys.executable,
            "-I",
            "-m",
            "dvc",
            "root",
        ),
        (
            "git",
            "check-ignore",
            "--quiet",
            "--no-index",
            "--",
            ".dvc/config.local",
        ),
        (
            "git",
            "ls-files",
            "--error-unmatch",
            "--",
            ".dvc/config.local",
        ),
    }
    observed_identity_commands = {call.arguments for call in runner.identity_calls}
    assert expected_identity_commands <= observed_identity_commands
    assert all(call.environment is not None for call in runner.identity_calls)
    assert all(call.environment is not None for call in runner.calls)


@pytest.mark.integration
def test_real_dvc_writes_only_the_expected_ignored_local_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "real-repository"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    environment = {**os.environ, "DVC_NO_ANALYTICS": "true"}
    subprocess.run(
        [sys.executable, "-I", "-m", "dvc", "init"],
        cwd=repository,
        check=True,
        capture_output=True,
        env=environment,
    )
    decoy_root = tmp_path / "decoy-root"
    (decoy_root / ".dvc").mkdir(parents=True)
    poisoned_global_config = tmp_path / "poisoned-global-config"
    poisoned_global_config.mkdir()
    (poisoned_global_config / "config").write_text("[invalid\n", encoding="utf-8")
    with monkeypatch.context() as poisoned:
        _set_remote_environment(
            poisoned,
            endpoint_url="https://objects.example.test:9443",
            region="us-east-1",
        )
        poisoned.setenv("AWS_SECRET_ACCESS_KEY", "never-persist-this-sentinel")
        poisoned.setenv("DVC_GLOBAL_CONFIG_DIR", str(poisoned_global_config))
        poisoned.setenv("DVC_ROOT", str(decoy_root))
        poisoned.setenv("GIT_DIR", str(tmp_path / "untrusted-git-directory"))
        poisoned.setenv("PYTHONHOME", str(tmp_path / "untrusted-python-home"))

        result = remote.configure_private_dvc_remote(repository)

    assert result.endpoint_configured is True
    assert result.region_configured is True
    assert not (decoy_root / ".dvc" / "config.local").exists()
    configuration = (repository / ".dvc" / "config.local").read_text(encoding="utf-8")
    assert "never-persist-this-sentinel" not in configuration
    assert set(configuration.splitlines()) == {
        "[core]",
        "    remote = private",
        "['remote \"private\"']",
        "    url = s3://signlab-private/datasets",
        "    endpointurl = https://objects.example.test:9443",
        "    region = us-east-1",
    }
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", "--", ".dvc/config.local"],
        cwd=repository,
        check=False,
    )
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", ".dvc/config.local"],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    assert ignored.returncode == 0
    assert tracked.returncode == 1
    assert not (repository / ".dvc" / "tmp" / "signlab-private-remote.lock").exists()


@pytest.mark.parametrize("reported_root", ["git", "dvc"])
def test_repository_path_must_equal_git_and_dvc_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reported_root: str,
) -> None:
    repository = _repository(tmp_path)
    different_root = tmp_path / "different-root"
    different_root.mkdir()
    runner = _RecordingRunner(
        repository,
        git_root=different_root if reported_root == "git" else None,
        dvc_root=different_root if reported_root == "dvc" else None,
    )
    _install_runner(monkeypatch, runner)
    _set_remote_environment(monkeypatch)

    with pytest.raises(remote.DvcRemoteConfigurationError) as raised:
        remote.configure_private_dvc_remote(repository)

    assert str(raised.value) == "repository root must be a real initialized Git and DVC repository"
    assert runner.calls == []
    assert not (repository / ".dvc" / "config.local").exists()


def test_git_must_report_local_configuration_as_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    runner = _RecordingRunner(
        repository,
        unignored_paths=frozenset({".dvc/config.local"}),
    )
    _install_runner(monkeypatch, runner)
    _set_remote_environment(monkeypatch)

    with pytest.raises(remote.DvcRemoteConfigurationError) as raised:
        remote.configure_private_dvc_remote(repository)

    assert str(raised.value) == (
        ".dvc/config.local must be an ignored credential-free regular file"
    )
    assert runner.calls == []


def test_git_tracked_local_configuration_is_rejected_while_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    configuration = repository / ".dvc" / "config.local"
    original = b"[core]\n    remote = previous\n"
    configuration.write_bytes(original)
    runner = _RecordingRunner(
        repository,
        tracked_paths=frozenset({".dvc/config.local"}),
    )
    _install_runner(monkeypatch, runner)
    _set_remote_environment(monkeypatch)

    with pytest.raises(remote.DvcRemoteConfigurationError) as raised:
        remote.configure_private_dvc_remote(repository)

    assert str(raised.value) == (
        ".dvc/config.local must be an ignored credential-free regular file"
    )
    assert configuration.read_bytes() == original
    assert runner.calls == []


def test_remote_configuration_requires_the_locked_dvc_distribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    runner = _RecordingRunner(repository)
    _install_runner(monkeypatch, runner)
    _set_remote_environment(monkeypatch)
    monkeypatch.setattr(remote, "distribution_version", lambda _name: "3.67.2")

    with pytest.raises(remote.DvcRemoteConfigurationError, match="configuration failed"):
        remote.configure_private_dvc_remote(repository)

    assert runner.calls == []
    assert not (repository / ".dvc" / "config.local").exists()


def test_remote_configuration_redacts_temporary_environment_creation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    runner = _RecordingRunner(repository)
    _install_runner(monkeypatch, runner)
    _set_remote_environment(monkeypatch)

    def fail_temporary_directory(*_args: object, **_kwargs: object) -> None:
        raise OSError("unit-test-sensitive-temporary-root")

    monkeypatch.setattr(
        "signlab.reproducibility.remote.tempfile.TemporaryDirectory",
        fail_temporary_directory,
    )

    with pytest.raises(remote.DvcRemoteConfigurationError) as raised:
        remote.configure_private_dvc_remote(repository)

    assert (
        str(raised.value)
        == "private DVC remote configuration failed; local configuration was restored"
    )
    assert raised.value.__cause__ is None
    assert runner.identity_calls == []
    assert runner.calls == []


def test_remote_configuration_redacts_temporary_environment_layout_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    runner = _RecordingRunner(repository)
    _install_runner(monkeypatch, runner)
    _set_remote_environment(monkeypatch)

    def fail_environment_layout(_root: Path) -> dict[str, str]:
        raise OSError("unit-test-sensitive-environment-layout")

    monkeypatch.setattr(
        remote,
        "_build_isolated_subprocess_environment",
        fail_environment_layout,
    )

    with pytest.raises(remote.DvcRemoteConfigurationError) as raised:
        remote.configure_private_dvc_remote(repository)

    assert (
        str(raised.value)
        == "private DVC remote configuration failed; local configuration was restored"
    )
    assert raised.value.__cause__ is None
    assert runner.identity_calls == []
    assert runner.calls == []


def test_optional_settings_are_omitted_instead_of_written_as_empty_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    runner = _RecordingRunner(repository)
    _install_runner(monkeypatch, runner)
    _set_remote_environment(monkeypatch)

    result = remote.configure_private_dvc_remote(repository)

    assert len(runner.calls) == 1
    assert result.endpoint_configured is False
    assert result.region_configured is False


@pytest.mark.parametrize(
    "remote_url",
    [
        "https://objects.example.test/bucket",
        "S3://signlab-private/data",
        "s3://",
        "s3://UPPERCASE/data",
        "s3://user@signlab-private/data",
        "s3://user:password@signlab-private/data",
        "s3://signlab-private:443/data",
        "s3://signlab-private:invalid/data",
        "s3://[signlab-private/data",
        "s3://signlab-private/data?signature=value",
        "s3://signlab-private/data#fragment",
        "s3://signlab-private//data",
        "s3://signlab-private/../data",
        "s3://signlab-private/%2e%2e/data",
        "s3://signlab-private/data\\child",
        " s3://signlab-private/data",
    ],
)
def test_remote_url_rejects_noncanonical_or_credential_bearing_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remote_url: str,
) -> None:
    repository = _repository(tmp_path)
    runner = _RecordingRunner(repository)
    _install_runner(monkeypatch, runner)
    _set_remote_environment(monkeypatch, remote_url=remote_url)

    with pytest.raises(remote.DvcRemoteConfigurationError) as raised:
        remote.configure_private_dvc_remote(repository)

    assert str(raised.value) == (f"{remote.DVC_REMOTE_URL_ENV} must be a credential-free s3:// URL")
    assert runner.calls == []


@pytest.mark.parametrize(
    "remote_url",
    [
        "s3://b",
        "s3://signlab-private",
        "s3://signlab-private/",
        "s3://signlab.private/datasets/version-1",
    ],
)
def test_remote_url_accepts_provider_neutral_s3_locations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remote_url: str,
) -> None:
    repository = _repository(tmp_path)
    runner = _RecordingRunner(repository)
    _install_runner(monkeypatch, runner)
    _set_remote_environment(monkeypatch, remote_url=remote_url)

    remote.configure_private_dvc_remote(repository)

    assert runner.calls[0].arguments[-1] == remote_url


@pytest.mark.parametrize("remote_url", ["", "   "])
def test_remote_url_is_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remote_url: str,
) -> None:
    repository = _repository(tmp_path)
    _set_remote_environment(monkeypatch, remote_url=remote_url)

    with pytest.raises(remote.DvcRemoteConfigurationError) as raised:
        remote.configure_private_dvc_remote(repository)

    assert str(raised.value) == f"{remote.DVC_REMOTE_URL_ENV} is required"


def test_remote_url_is_required_when_environment_variable_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    monkeypatch.delenv(remote.DVC_REMOTE_URL_ENV, raising=False)
    monkeypatch.delenv(remote.DVC_ENDPOINT_URL_ENV, raising=False)
    monkeypatch.delenv(remote.DVC_REGION_ENV, raising=False)

    with pytest.raises(remote.DvcRemoteConfigurationError) as raised:
        remote.configure_private_dvc_remote(repository)

    assert str(raised.value) == f"{remote.DVC_REMOTE_URL_ENV} is required"


@pytest.mark.parametrize(
    "endpoint_url",
    [
        "https://objects.example.test",
        "https://objects.example.test:9443/",
        "http://localhost:9000",
        "http://127.0.0.1:9000",
        "http://127.10.20.30:9000",
        "http://[::1]:9000",
    ],
)
def test_endpoint_accepts_https_and_loopback_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    endpoint_url: str,
) -> None:
    repository = _repository(tmp_path)
    runner = _RecordingRunner(repository)
    _install_runner(monkeypatch, runner)
    _set_remote_environment(monkeypatch, endpoint_url=endpoint_url)

    remote.configure_private_dvc_remote(repository)

    assert runner.calls[1].arguments[-1] == endpoint_url


@pytest.mark.parametrize(
    "endpoint_url",
    [
        "",
        "objects.example.test",
        "ftp://objects.example.test",
        "HTTP://localhost:9000",
        "http://objects.example.test:9000",
        "http://localhost.example.test:9000",
        "https://user@objects.example.test",
        "https://user:password@objects.example.test",
        "https://objects.example.test/path",
        "https://objects.example.test?token=value",
        "https://objects.example.test#fragment",
        "https://objects.example.test:0",
        "https://objects.example.test:99999",
        "https://objects.exämple.test",
    ],
)
def test_endpoint_rejects_non_origins_and_remote_plaintext_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    endpoint_url: str,
) -> None:
    repository = _repository(tmp_path)
    runner = _RecordingRunner(repository)
    _install_runner(monkeypatch, runner)
    _set_remote_environment(monkeypatch, endpoint_url=endpoint_url)

    with pytest.raises(remote.DvcRemoteConfigurationError) as raised:
        remote.configure_private_dvc_remote(repository)

    assert str(raised.value) == (
        f"{remote.DVC_ENDPOINT_URL_ENV} must be a credential-free HTTPS origin "
        "(HTTP is allowed only for localhost)"
    )
    assert runner.calls == []


@pytest.mark.parametrize("region", ["", "US-EAST-1", " us-east-1", "us_east_1", "-east"])
def test_region_rejects_noncanonical_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    region: str,
) -> None:
    repository = _repository(tmp_path)
    runner = _RecordingRunner(repository)
    _install_runner(monkeypatch, runner)
    _set_remote_environment(monkeypatch, region=region)

    with pytest.raises(remote.DvcRemoteConfigurationError) as raised:
        remote.configure_private_dvc_remote(repository)

    assert str(raised.value) == f"{remote.DVC_REGION_ENV} must be a canonical region name"
    assert runner.calls == []


@pytest.mark.parametrize("missing_entry", [".git", ".dvc"])
def test_repository_must_be_initialized_and_real(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_entry: str,
) -> None:
    repository = _repository(tmp_path)
    entry = repository / missing_entry
    entry.rename(repository / f"{missing_entry}.missing")
    _set_remote_environment(monkeypatch)

    with pytest.raises(remote.DvcRemoteConfigurationError) as raised:
        remote.configure_private_dvc_remote(repository)

    assert str(raised.value) == "repository root must be a real initialized Git and DVC repository"


@pytest.mark.parametrize(
    "ignore_content",
    [
        "/tmp\n/cache\n",
        "/config.local\n!/config.local\n",
        "/config.local\n!unrelated\n",
    ],
)
def test_local_configuration_must_be_ignored_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ignore_content: str,
) -> None:
    repository = _repository(tmp_path)
    (repository / ".dvc" / ".gitignore").write_text(ignore_content, encoding="utf-8")
    _set_remote_environment(monkeypatch)

    with pytest.raises(remote.DvcRemoteConfigurationError) as raised:
        remote.configure_private_dvc_remote(repository)

    assert str(raised.value) == (
        ".dvc/config.local must be an ignored credential-free regular file"
    )


def test_non_utf8_ignore_file_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    (repository / ".dvc" / ".gitignore").write_bytes(b"/config.local\n\xff")
    _set_remote_environment(monkeypatch)

    with pytest.raises(remote.DvcRemoteConfigurationError):
        remote.configure_private_dvc_remote(repository)


def test_non_utf8_local_configuration_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    (repository / ".dvc" / "config.local").write_bytes(b"[core]\n\xff")
    _set_remote_environment(monkeypatch)

    with pytest.raises(remote.DvcRemoteConfigurationError):
        remote.configure_private_dvc_remote(repository)


def test_repository_root_must_be_a_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_file = tmp_path / "repository-file"
    repository_file.write_text("not a repository\n", encoding="utf-8")
    _set_remote_environment(monkeypatch)

    with pytest.raises(remote.DvcRemoteConfigurationError) as raised:
        remote.configure_private_dvc_remote(repository_file)

    assert str(raised.value) == "repository root must be a real initialized Git and DVC repository"


def test_existing_local_configuration_must_be_a_regular_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    (repository / ".dvc" / "config.local").mkdir()
    _set_remote_environment(monkeypatch)

    with pytest.raises(remote.DvcRemoteConfigurationError) as raised:
        remote.configure_private_dvc_remote(repository)

    assert str(raised.value) == (
        ".dvc/config.local must be an ignored credential-free regular file"
    )


def test_existing_authentication_configuration_is_rejected_without_echo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    secret = "unit-test-private-value"
    (repository / ".dvc" / "config.local").write_text(
        "['remote \"private\"']\n    secret_access" + f"_key = {secret}\n",
        encoding="utf-8",
    )
    _set_remote_environment(monkeypatch)

    with pytest.raises(remote.DvcRemoteConfigurationError) as raised:
        remote.configure_private_dvc_remote(repository)

    assert secret not in str(raised.value)
    assert str(raised.value) == (
        ".dvc/config.local must be an ignored credential-free regular file"
    )


def _symlink_or_skip(link: Path, target: Path, *, target_is_directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symbolic links are unavailable: {type(error).__name__}")


def test_config_local_symlink_is_rejected_without_touching_its_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    target = tmp_path / "outside-configuration"
    target.write_text("outside\n", encoding="utf-8")
    _symlink_or_skip(repository / ".dvc" / "config.local", target)
    _set_remote_environment(monkeypatch)

    with pytest.raises(remote.DvcRemoteConfigurationError):
        remote.configure_private_dvc_remote(repository)

    assert target.read_text(encoding="utf-8") == "outside\n"


def test_config_local_hardlink_is_rejected_without_touching_its_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    target = tmp_path / "outside-configuration"
    original = b"[core]\n    remote = previous\n"
    target.write_bytes(original)
    configuration = repository / ".dvc" / "config.local"
    try:
        os.link(target, configuration)
    except OSError as error:
        pytest.skip(f"hardlinks are unavailable: {type(error).__name__}")
    runner = _RecordingRunner(repository)
    _install_runner(monkeypatch, runner)
    _set_remote_environment(monkeypatch)

    with pytest.raises(remote.DvcRemoteConfigurationError) as raised:
        remote.configure_private_dvc_remote(repository)

    assert str(raised.value) == (
        ".dvc/config.local must be an ignored credential-free regular file"
    )
    assert target.read_bytes() == original
    assert configuration.read_bytes() == original
    assert runner.calls == []


def test_repository_root_symlink_or_reparse_point_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    linked_root = tmp_path / "linked-repository"
    _symlink_or_skip(linked_root, repository, target_is_directory=True)
    _set_remote_environment(monkeypatch)

    with pytest.raises(remote.DvcRemoteConfigurationError) as raised:
        remote.configure_private_dvc_remote(linked_root)

    assert str(raised.value) == "repository root must be a real initialized Git and DVC repository"


def test_active_repository_lock_prevents_a_competing_configuration_and_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    configuration = repository / ".dvc" / "config.local"
    original = b"[core]\n    remote = winner\n"
    configuration.write_bytes(original)
    runner = _RecordingRunner(repository)
    _install_runner(monkeypatch, runner)
    _set_remote_environment(monkeypatch)
    _, dvc_directory, dvc_status = remote._validated_repository(repository)
    lock_path = dvc_directory / "tmp" / "signlab-private-remote.lock"

    with remote._configuration_lock(dvc_directory, dvc_status):
        assert lock_path.is_file()
        with pytest.raises(remote.DvcRemoteConfigurationError) as raised:
            remote.configure_private_dvc_remote(repository)

        assert str(raised.value) == "private DVC remote configuration lock is unavailable"
        assert configuration.read_bytes() == original
        assert runner.calls == []

    assert not lock_path.exists()


def test_subprocess_failure_restores_existing_configuration_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    configuration = repository / ".dvc" / "config.local"
    original = b"[core]\n    remote = previous\n"
    configuration.write_bytes(original)
    original_mode = stat.S_IMODE(configuration.stat().st_mode)
    runner = _RecordingRunner(repository, failure_call=2)
    _install_runner(monkeypatch, runner)
    _set_remote_environment(monkeypatch, endpoint_url="https://objects.example.test")

    with pytest.raises(remote.DvcRemoteConfigurationError) as raised:
        remote.configure_private_dvc_remote(repository)

    assert str(raised.value) == (
        "private DVC remote configuration failed; local configuration was restored"
    )
    assert configuration.read_bytes() == original
    assert stat.S_IMODE(configuration.stat().st_mode) == original_mode
    assert list((repository / ".dvc").glob(".config.local.restore-*")) == []


def test_subprocess_failure_removes_new_partial_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    runner = _RecordingRunner(repository, failure_call=1)
    _install_runner(monkeypatch, runner)
    _set_remote_environment(monkeypatch)

    with pytest.raises(remote.DvcRemoteConfigurationError) as raised:
        remote.configure_private_dvc_remote(repository)

    assert str(raised.value) == (
        "private DVC remote configuration failed; local configuration was restored"
    )
    assert not (repository / ".dvc" / "config.local").exists()


@pytest.mark.parametrize(
    ("failure", "expected_type"),
    [
        (KeyboardInterrupt(), KeyboardInterrupt),
        (SystemExit(17), SystemExit),
    ],
)
def test_process_exit_restores_configuration_and_releases_lock_before_propagating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    expected_type: type[BaseException],
) -> None:
    repository = _repository(tmp_path)
    configuration = repository / ".dvc" / "config.local"
    original = b"[core]\n    remote = previous\n"
    configuration.write_bytes(original)
    runner = _RecordingRunner(
        repository,
        failure_call=2,
        failure_exception=failure,
    )
    _install_runner(monkeypatch, runner)
    _set_remote_environment(monkeypatch, endpoint_url="https://objects.example.test")

    with pytest.raises(expected_type) as raised:
        remote.configure_private_dvc_remote(repository)

    if isinstance(failure, SystemExit):
        assert isinstance(raised.value, SystemExit)
        assert raised.value.code == 17
    assert configuration.read_bytes() == original
    assert not (repository / ".dvc" / "tmp" / "signlab-private-remote.lock").exists()
    assert len(runner.calls) == 2


def test_generated_authentication_material_is_rejected_and_rolled_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    secret = "unit-test-generated-secret-value"
    runner = _RecordingRunner(
        repository,
        generated_configuration=("secret_access" + f"_key = {secret}\n").encode(),
    )
    _install_runner(monkeypatch, runner)
    _set_remote_environment(monkeypatch)

    with pytest.raises(remote.DvcRemoteConfigurationError) as raised:
        remote.configure_private_dvc_remote(repository)

    assert secret not in str(raised.value)
    assert not (repository / ".dvc" / "config.local").exists()


def test_rollback_failure_has_a_stable_redacted_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _repository(tmp_path)
    runner = _RecordingRunner(
        repository,
        failure_call=1,
        failure_stderr="unit-test-sensitive-stderr",
        replace_with_directory=True,
    )
    _install_runner(monkeypatch, runner)
    _set_remote_environment(monkeypatch)

    with pytest.raises(remote.DvcRemoteConfigurationError) as raised:
        remote.configure_private_dvc_remote(repository)

    assert str(raised.value) == (
        "private DVC remote configuration failed and local configuration could not be restored"
    )
    assert "unit-test-sensitive-stderr" not in str(raised.value)


def test_credentials_and_process_overrides_are_not_forwarded_or_echoed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = _repository(tmp_path)
    access_id = "unit-test-access-id-value"
    secret = "unit-test-secret-value"
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", access_id)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret)
    monkeypatch.setenv("AWS_SESSION_TOKEN", "unit-test-session-token")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "unit-test-azure-secret")
    monkeypatch.setenv(
        "GOOGLE_APPLICATION_CREDENTIALS",
        "unit-test-google-credentials",
    )
    monkeypatch.setenv("DVC_GLOBAL_CONFIG_DIR", "unit-test-untrusted-dvc-global")
    monkeypatch.setenv("DVC_ROOT", "unit-test-untrusted-dvc-root")
    monkeypatch.setenv("DVC_STUDIO_TOKEN", "unit-test-dvc-studio-token")
    monkeypatch.setenv("GIT_DIR", "unit-test-untrusted-git-directory")
    monkeypatch.setenv("GIT_WORK_TREE", "unit-test-untrusted-work-tree")
    monkeypatch.setenv("HTTPS_PROXY", "https://unit-test-proxy.invalid")
    monkeypatch.setenv("PYTHONHOME", "unit-test-untrusted-python-home")
    monkeypatch.setenv("PYTHONPATH", "unit-test-untrusted-python-path")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "unit-test-untrusted-ca-bundle")
    runner = _RecordingRunner(
        repository,
        failure_call=1,
        failure_stderr=f"provider failed with {access_id} and {secret}",
    )
    _install_runner(monkeypatch, runner)
    _set_remote_environment(monkeypatch)

    with pytest.raises(remote.DvcRemoteConfigurationError) as raised:
        remote.configure_private_dvc_remote(repository)

    captured = capsys.readouterr()
    rendered_calls = repr(runner.calls)
    assert os.environ["AWS_ACCESS_KEY_ID"] == access_id
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == secret
    assert access_id not in rendered_calls
    assert secret not in rendered_calls
    assert all(
        call.environment is not None
        and access_id not in call.environment.values()
        and secret not in call.environment.values()
        and "GIT_DIR" not in call.environment
        and "PYTHONPATH" not in call.environment
        for call in runner.identity_calls
    )
    forbidden_names = {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AZURE_CLIENT_SECRET",
        "DVC_ROOT",
        "DVC_STUDIO_TOKEN",
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "HTTPS_PROXY",
        "PYTHONHOME",
        "PYTHONPATH",
        "REQUESTS_CA_BUNDLE",
        remote.DVC_ENDPOINT_URL_ENV,
        remote.DVC_REGION_ENV,
        remote.DVC_REMOTE_URL_ENV,
    }
    assert all(
        call.environment is not None
        and access_id not in call.environment.values()
        and secret not in call.environment.values()
        and forbidden_names.isdisjoint(call.environment)
        for call in runner.calls
    )
    all_calls = [*runner.identity_calls, *runner.calls]
    environments = [call.environment for call in all_calls]
    assert all(environment is environments[0] for environment in environments)
    environment = environments[0]
    assert environment is not None
    assert environment["CI"] == "1"
    assert environment["DVC_EXP_AUTO_PUSH"] == "false"
    assert environment["DVC_NO_ANALYTICS"] == "true"
    assert environment["DVC_STUDIO_OFFLINE"] == "true"
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert not Path(environment["HOME"]).parent.exists()
    assert access_id not in str(raised.value)
    assert secret not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert captured.out == ""
    assert captured.err == ""
