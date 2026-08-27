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
        ("hand_landmarker.task", b"model", "artifact-type"),
        ("download.tar", b"archive", "artifact-type"),
        ("download.tar.gz", b"archive", "artifact-type"),
        ("download.tgz", b"archive", "artifact-type"),
        ("download.zip", b"archive", "artifact-type"),
        ("checkpoint.tflite", b"model", "artifact-type"),
        ("native.dll", b"binary", "artifact-type"),
        ("native.so", b"binary", "artifact-type"),
        ("large.txt", b"x" * (hygiene.MAX_TRACKED_BYTES + 1), "size"),
        ("windows.txt", b"C" + b":" + b"\\Users\\fixture\\data", "machine-path"),
        ("unix.txt", b"/" + b"home/fixture/data", "machine-path"),
        ("uri.txt", b"file" + b":///" + b"C" + b":/fixture/data", "machine-path"),
        ("secret.txt", b"AKIA" + b"A" * 16, "secret"),
        (".dvc/config.local", b'[remote "private"]\nurl=s3://bucket\n', "dvc-local-config"),
        ("private-data.dvc", b"outs:\n- md5: deadbeef\n", "dvc-pointer"),
        (".dvc/config", b'[remote "private"]\nurl=s3://bucket\n', "dvc-remote"),
        (
            "dvc.yaml",
            b"stages" + b":\n  x" + b":\n    cmd" + b": x\n    deps" + b": [s3://bucket/key]\n",
            "dvc-remote",
        ),
        ("dvc.lock", b"access_key_id: fixture\n", "dvc-credential"),
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
        "mediapipe-task-artifact",
        "tar-archive",
        "compressed-tar-archive",
        "tgz-archive",
        "zip-archive",
        "tflite-artifact",
        "windows-native-library",
        "unix-native-library",
        "oversized-file",
        "windows-path",
        "unix-path",
        "file-uri",
        "secret-pattern",
        "dvc-local-config",
        "dvc-pointer",
        "dvc-config-remote",
        "dvc-stage-remote",
        "dvc-credential",
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


def test_dvc_managed_windows_newlines_are_normalized_by_git() -> None:
    for path in (".dvc/.gitignore", ".dvc/config", "dvc.lock"):
        assert "line-ending" not in _rules(path, b"generated\r\n")


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


def test_only_reviewed_extraction_model_lock_is_unignored() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    def is_ignored(relative_path: str) -> bool:
        completed = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", relative_path],
            cwd=repository_root,
            check=False,
        )
        assert completed.returncode in {0, 1}
        return completed.returncode == 0

    assert not is_ignored("src/signlab/resources/extraction/models/mediapipe-tasks-1.0.1.lock.json")
    assert is_ignored("src/signlab/resources/extraction/models/unreviewed.lock.json")
    assert is_ignored("src/signlab/resources/extraction/models/hand_landmarker.task")
