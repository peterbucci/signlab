from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from scripts import check_repository_hygiene as hygiene


def _rules(path: str, content: bytes) -> set[str]:
    return {violation.rule for violation in hygiene.inspect_tracked_file(path, content)}


def test_current_tracked_repository_is_public_safe() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    assert hygiene.check_repository(repository_root) == ()


@pytest.mark.parametrize(
    ("path", "content", "expected_rule"),
    [
        ("data/raw/private.txt", b"private", "private-root"),
        ("data/private/participant.txt", b"private", "private-root"),
        ("Data/raw/participant.txt", b"private", "private-root"),
        ("models/champion.onnx", b"model", "artifact-type"),
        (".env.local", b"TOKEN=fake", "secret-file"),
        (".ENV", b"TOKEN=fake", "secret-file"),
        ("checkpoint.joblib", b"model", "artifact-type"),
        ("checkpoint.safetensors", b"model", "artifact-type"),
        ("checkpoint.tflite", b"model", "artifact-type"),
        ("native.dll", b"binary", "artifact-type"),
        ("native.so", b"binary", "artifact-type"),
        ("large.txt", b"x" * (hygiene.MAX_TRACKED_BYTES + 1), "size"),
        ("windows.txt", b"C" + b":" + b"\\Users\\fixture\\data", "machine-path"),
        ("unix.txt", b"/" + b"home/fixture/data", "machine-path"),
        ("uri.txt", b"file" + b":///" + b"C" + b":/fixture/data", "machine-path"),
        ("secret.txt", b"AKIA" + b"A" * 16, "secret"),
        ("line-endings.txt", b"first\r\nsecond\r\n", "line-ending"),
    ],
    ids=[
        "private-root",
        "private-data",
        "mixed-case-private-data",
        "artifact-type",
        "secret-file",
        "mixed-case-env",
        "joblib-artifact",
        "safetensors-artifact",
        "tflite-artifact",
        "windows-native-library",
        "unix-native-library",
        "oversized-file",
        "windows-path",
        "unix-path",
        "file-uri",
        "secret-pattern",
        "crlf",
    ],
)
def test_seeded_repository_policy_failures_are_detected(
    path: str,
    content: bytes,
    expected_rule: str,
) -> None:
    assert expected_rule in _rules(path, content)


def test_small_reviewed_public_fixture_may_use_an_artifact_extension() -> None:
    assert _rules("tests/fixtures/public/replay.mp4", b"synthetic") == set()


def test_web_url_with_users_path_is_not_a_machine_path() -> None:
    assert _rules("README.md", b"https://github.com/users/example/projects/1") == set()


def test_violation_messages_never_echo_matched_content() -> None:
    secret = b"ghp_" + b"a" * 40

    violations = hygiene.inspect_tracked_file("config.txt", secret)

    assert violations
    assert all(secret.decode() not in violation.message for violation in violations)


def test_checkout_cleanliness_detects_untracked_files(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)

    assert hygiene.is_checkout_clean(tmp_path)

    (tmp_path / "new-file.txt").write_text("untracked\n", encoding="utf-8")

    assert not hygiene.is_checkout_clean(tmp_path)
