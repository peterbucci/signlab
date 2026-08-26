from __future__ import annotations

import subprocess
import sys

import pytest
from typer.testing import CliRunner

from signlab import __version__, cli
from signlab.commands import doctor


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"NO_COLOR": "1"})


def test_top_level_help_exposes_thin_command_groups(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "data",
        "train",
        "evaluate",
        "export",
        "doctor",
        "taxonomy",
        "governance",
    ):
        assert command in result.output


@pytest.mark.parametrize(
    "command",
    ["data", "train", "evaluate", "export", "doctor", "taxonomy", "governance"],
)
def test_command_group_help_has_no_pipeline_prerequisites(
    runner: CliRunner,
    command: str,
) -> None:
    result = runner.invoke(cli.app, [command, "--help"])

    assert result.exit_code == 0
    assert "Usage" in result.output


def test_cli_import_does_not_load_optional_ml_runtimes() -> None:
    probe = (
        "import sys; import signlab.cli; "
        "blocked = {'mediapipe', 'mlflow', 'onnxruntime', 'torch'} & set(sys.modules); "
        "raise SystemExit(','.join(sorted(blocked)) if blocked else 0)"
    )

    subprocess.run([sys.executable, "-c", probe], check=True)


def test_version_option_is_eager(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_invalid_command_has_a_stable_usage_error(runner: CliRunner) -> None:
    result = runner.invoke(cli.app, ["does-not-exist"])

    assert result.exit_code == 2
    assert result.output.strip() == (
        "Error: invalid command usage; run --help for accepted arguments."
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["governance", "validate-consent", "receipt.json", "participant_" + "f" * 32],
        [
            "governance",
            "readiness-check",
            f"{chr(67)}:{chr(92)}private{chr(92)}participant-name.json",
            "extra",
        ],
        ["governance", "not-a-command-private-sentinel"],
    ],
)
def test_usage_errors_do_not_echo_untrusted_tokens(
    runner: CliRunner,
    arguments: list[str],
) -> None:
    result = runner.invoke(cli.app, arguments)

    assert result.exit_code == 2
    assert result.output.strip() == (
        "Error: invalid command usage; run --help for accepted arguments."
    )
    assert "participant_" not in result.output
    assert "private" not in result.output.casefold()


def test_doctor_checks_are_deterministic_and_redacted() -> None:
    diagnostics = doctor.inspect_environment(
        python_version=(3, 12, 14),
        implementation="CPython",
        filesystem_encoding="utf-8",
    )

    assert all(diagnostic.passed for diagnostic in diagnostics)
    assert not any(
        "Users" in diagnostic.detail or "home" in diagnostic.detail for diagnostic in diagnostics
    )


def test_doctor_returns_nonzero_when_the_runtime_is_unsupported(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = (doctor.Diagnostic(name="Python", passed=False, detail="unsupported fixture"),)
    monkeypatch.setattr(doctor, "inspect_environment", lambda: failed)

    result = runner.invoke(cli.app, ["doctor", "check"])

    assert result.exit_code == 1
    assert "[error] Python" in result.output


def test_main_invokes_the_typer_application(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_app() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(cli, "app", fake_app)

    cli.main()

    assert calls == 1
