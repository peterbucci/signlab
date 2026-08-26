"""Fail when tracked files violate SignLab's public-repository boundary."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MAX_TRACKED_BYTES = 1_048_576
PUBLIC_FIXTURE_PREFIX = PurePosixPath("tests/fixtures/public")
PRIVATE_ROOTS = (
    PurePosixPath(".dvc/cache"),
    PurePosixPath(".dvc/tmp"),
    PurePosixPath("artifacts"),
    PurePosixPath("data/interim"),
    PurePosixPath("data/private"),
    PurePosixPath("data/processed"),
    PurePosixPath("data/raw"),
    PurePosixPath("mlruns"),
    PurePosixPath("models"),
    PurePosixPath("runs"),
)
PRIVATE_DVC_PATHS = {
    PurePosixPath(".dvc/config.local"),
}
PRIVATE_SUFFIXES = {
    ".a",
    ".avi",
    ".db",
    ".dll",
    ".dylib",
    ".exe",
    ".h5",
    ".hdf5",
    ".joblib",
    ".keras",
    ".key",
    ".lib",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".o",
    ".onnx",
    ".parquet",
    ".pem",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".pyc",
    ".pyd",
    ".sqlite",
    ".sqlite3",
    ".safetensors",
    ".so",
    ".tflite",
    ".webm",
    ".whl",
}
PUBLIC_FIXTURE_SUFFIXES = {
    ".avi",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".webm",
}
BINARY_SUFFIXES = PRIVATE_SUFFIXES | {
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".tar",
    ".webp",
    ".zip",
}
MACHINE_PATH_PATTERNS = (
    re.compile(rb"(?i)(?<![a-z0-9])[a-z]:[\\/]"),
    re.compile(rb"(?i)(?:^|[\s\"'=(])/(?:users|home)/[^/\s]+(?:/|\\)"),
    re.compile(rb"(?i)file:///(?:[a-z]:|users/|home/)"),
)
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{36,255}"),
)
DVC_METADATA_NAMES = {"dvc.lock", "dvc.yaml"}
DVC_NATIVE_NEWLINE_PATHS = {
    PurePosixPath(".dvc/.gitignore"),
    PurePosixPath(".dvc/config"),
    PurePosixPath("dvc.lock"),
}
DVC_PRIVATE_KEY_PATTERN = re.compile(
    rb"(?i)(?:access[_-]?key(?:[_-]?id)?|secret[_-]?key|session[_-]?token|credential|password)\s*[:=]"
)
DVC_REMOTE_PATTERN = re.compile(rb"(?i)(?:s3|gs|azure|ssh|hdfs|webhdfs|webdav|webdavs|https?)://")


@dataclass(frozen=True, order=True)
class Violation:
    """A path-scoped, non-sensitive repository-policy failure."""

    path: str
    rule: str
    message: str


def _is_within(path: PurePosixPath, parent: PurePosixPath) -> bool:
    return path == parent or parent in path.parents


def inspect_tracked_file(relative_path: str, content: bytes) -> tuple[Violation, ...]:
    """Inspect one tracked blob without returning any matching secret or path value."""
    path = PurePosixPath(relative_path)
    policy_path = PurePosixPath(relative_path.lower())
    violations: list[Violation] = []
    is_public_fixture = _is_within(policy_path, PUBLIC_FIXTURE_PREFIX)

    if len(content) > MAX_TRACKED_BYTES:
        violations.append(
            Violation(relative_path, "size", "tracked file exceeds the 1 MiB Git limit")
        )
    if any(_is_within(policy_path, private_root) for private_root in PRIVATE_ROOTS):
        violations.append(
            Violation(relative_path, "private-root", "private/generated directory is tracked")
        )
    if policy_path in PRIVATE_DVC_PATHS:
        violations.append(
            Violation(relative_path, "dvc-local-config", "local DVC configuration is tracked")
        )
    if policy_path.suffix == ".dvc":
        violations.append(
            Violation(relative_path, "dvc-pointer", "DVC data pointer is tracked publicly")
        )
    if policy_path.name.startswith(".env") and path.name != ".env.example":
        violations.append(Violation(relative_path, "secret-file", "environment file is tracked"))
    suffix = policy_path.suffix
    allowed_public_fixture = is_public_fixture and suffix in PUBLIC_FIXTURE_SUFFIXES
    if suffix in PRIVATE_SUFFIXES and not allowed_public_fixture:
        violations.append(
            Violation(relative_path, "artifact-type", "private/generated artifact type is tracked")
        )
    if suffix not in BINARY_SUFFIXES and b"\0" not in content:
        # DVC's Windows writers use the platform text newline. Git still stores
        # these exact generated files as LF through the repository attributes.
        if b"\r\n" in content and policy_path not in DVC_NATIVE_NEWLINE_PATHS:
            violations.append(Violation(relative_path, "line-ending", "text file contains CRLF"))
        if any(pattern.search(content) for pattern in MACHINE_PATH_PATTERNS):
            violations.append(
                Violation(relative_path, "machine-path", "text contains an absolute machine path")
            )
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            violations.append(
                Violation(relative_path, "secret", "text contains a high-confidence secret pattern")
            )
        if policy_path.name in DVC_METADATA_NAMES or policy_path == PurePosixPath(".dvc/config"):
            if DVC_PRIVATE_KEY_PATTERN.search(content):
                violations.append(
                    Violation(
                        relative_path,
                        "dvc-credential",
                        "DVC metadata contains a credential-like setting",
                    )
                )
            if DVC_REMOTE_PATTERN.search(content):
                violations.append(
                    Violation(
                        relative_path,
                        "dvc-remote",
                        "DVC metadata contains a physical remote location",
                    )
                )
    return tuple(violations)


def _candidate_paths(repository_root: Path) -> tuple[str, ...]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        check=True,
        capture_output=True,
    )
    return tuple(entry.decode("utf-8") for entry in result.stdout.split(b"\0") if entry)


def check_repository(repository_root: Path) -> tuple[Violation, ...]:
    """Inspect every tracked working-tree file and return stable relative failures."""
    violations: list[Violation] = []
    for relative_path in _candidate_paths(repository_root):
        path = repository_root / Path(relative_path)
        if not path.is_file():
            violations.append(
                Violation(relative_path, "missing", "tracked working-tree file is missing")
            )
            continue
        violations.extend(inspect_tracked_file(relative_path, path.read_bytes()))
    return tuple(sorted(violations))


def is_checkout_clean(repository_root: Path) -> bool:
    """Return whether tracked and non-ignored files match the current commit."""
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
    )
    return not result.stdout


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that the checkout is safe to publish.")
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="also fail when generated commands changed the checkout",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the repository guard without printing matching sensitive content."""
    args = _parse_args(argv)
    try:
        repository_root = Path.cwd()
        violations = check_repository(repository_root)
        checkout_clean = not args.require_clean or is_checkout_clean(repository_root)
    except (OSError, subprocess.SubprocessError):
        print("Repository hygiene check could not inspect the checkout.", file=sys.stderr)
        return 2
    if violations:
        print("Repository hygiene check failed:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation.path}: {violation.message} [{violation.rule}]", file=sys.stderr)
        return 1
    if not checkout_clean:
        print(
            "Repository hygiene check failed: the checkout changed during validation.",
            file=sys.stderr,
        )
        return 1
    print("Repository hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
