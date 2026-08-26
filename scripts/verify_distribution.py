"""Inspect built archives and smoke-test the wheel in an isolated environment."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath

MAX_MEMBER_BYTES = 1_048_576
EXPECTED_TAXONOMY_SCHEMAS = {
    "annotation-taxonomy-binding-1.schema.json",
    "bundle-taxonomy-binding-1.schema.json",
    "collection-taxonomy-binding-1.schema.json",
    "evaluation-taxonomy-binding-1.schema.json",
    "gesture-taxonomy-1.schema.json",
    "public-copy-taxonomy-binding-1.schema.json",
    "taxonomy-reference-1.schema.json",
    "training-taxonomy-binding-1.schema.json",
}
FORBIDDEN_REPOSITORY_ROOTS = {
    "artifacts",
    "data",
    "mlruns",
    "models",
    "runs",
}
FORBIDDEN_ANY_PARTS = {".dvc", ".env"}
FORBIDDEN_SUFFIXES = {
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
    ".safetensors",
    ".so",
    ".sqlite",
    ".sqlite3",
    ".tflite",
    ".webm",
    ".whl",
}


def validate_member_names(member_names: Iterable[str]) -> tuple[str, ...]:
    """Return sanitized policy failures for archive member names."""
    errors: list[str] = []
    for member_name in member_names:
        path = PurePosixPath(member_name)
        lowered_parts = tuple(part.lower() for part in path.parts)
        if path.is_absolute() or ".." in path.parts:
            errors.append("archive contains a non-portable member path")
        repository_parts = lowered_parts
        if repository_parts and repository_parts[0].startswith("signlab-"):
            repository_parts = repository_parts[1:]
        has_forbidden_root = bool(
            repository_parts and repository_parts[0] in FORBIDDEN_REPOSITORY_ROOTS
        )
        if (
            set(lowered_parts) & FORBIDDEN_ANY_PARTS
            or has_forbidden_root
            or path.suffix.lower() in FORBIDDEN_SUFFIXES
        ):
            errors.append("archive contains a private or generated artifact")
    return tuple(sorted(set(errors)))


def _inspect_wheel(wheel: Path) -> tuple[str, ...]:
    errors: list[str] = []
    with zipfile.ZipFile(wheel) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        errors.extend(validate_member_names(names))
        if not any(name.endswith("signlab/py.typed") for name in names):
            errors.append("wheel is missing the typed-package marker")
        if not any(name.startswith("signlab/commands/") for name in names):
            errors.append("wheel is missing CLI command modules")
        taxonomy_members = {
            PurePosixPath(name).name
            for name in names
            if name.startswith("signlab/resources/taxonomies/") and name.endswith(".json")
        }
        if taxonomy_members != {"signlab-five-1.0.0.json"}:
            errors.append("wheel is missing the built-in gesture taxonomy")
        schema_members = {
            PurePosixPath(name).name
            for name in names
            if name.startswith("signlab/resources/schemas/") and name.endswith(".json")
        }
        if schema_members != EXPECTED_TAXONOMY_SCHEMAS:
            errors.append("wheel does not contain the exact generated taxonomy schema set")
        if not any(name.endswith(".dist-info/METADATA") for name in names):
            errors.append("wheel is missing distribution metadata")
        entry_point_members = [
            member for member in members if member.filename.endswith(".dist-info/entry_points.txt")
        ]
        if len(entry_point_members) != 1:
            errors.append("wheel is missing unambiguous console-script metadata")
        else:
            entry_points = archive.read(entry_point_members[0]).decode("utf-8")
            if (
                re.search(
                    r"(?m)^signlab\s*=\s*signlab\.cli:main\s*$",
                    entry_points,
                )
                is None
            ):
                errors.append("wheel has an invalid SignLab console-script entry point")
        if any(member.file_size > MAX_MEMBER_BYTES for member in members):
            errors.append("wheel contains a member larger than 1 MiB")
    return tuple(sorted(set(errors)))


def _inspect_sdist(sdist: Path) -> tuple[str, ...]:
    errors: list[str] = []
    with tarfile.open(sdist, mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        errors.extend(validate_member_names(member.name for member in members))
        if not any(member.name.endswith("/src/signlab/py.typed") for member in members):
            errors.append("source distribution is missing the typed-package marker")
        if any(member.size > MAX_MEMBER_BYTES for member in members):
            errors.append("source distribution contains a member larger than 1 MiB")
    return tuple(sorted(set(errors)))


def _venv_executable(environment: Path, name: str) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / f"{name}.exe"
    return environment / "bin" / name


def _run(
    command: Sequence[str],
    *,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        encoding="utf-8",
        env=environment,
        text=True,
    )


def _install_and_smoke_test(wheel: Path) -> None:
    environment = os.environ.copy()
    environment["NO_COLOR"] = "1"
    environment["PYTHONUTF8"] = "1"
    with tempfile.TemporaryDirectory(prefix="signlab-wheel-") as temporary_directory:
        virtual_environment = Path(temporary_directory) / "venv"
        _run(
            ["uv", "venv", "--python", sys.executable, str(virtual_environment)],
            environment=environment,
        )
        python = _venv_executable(virtual_environment, "python")
        _run(
            ["uv", "pip", "install", "--python", str(python), str(wheel)],
            environment=environment,
        )
        version = _run(
            [str(python), "-m", "signlab.cli", "--version"],
            environment=environment,
        )
        if not version.stdout.strip():
            raise RuntimeError("isolated wheel did not expose a version")
        console_script = _venv_executable(virtual_environment, "signlab")
        _run([str(console_script), "--version"], environment=environment)
        for command in ("data", "train", "evaluate", "export", "doctor", "taxonomy"):
            _run(
                [str(console_script), command, "--help"],
                environment=environment,
            )
        _run(
            [str(console_script), "doctor", "check"],
            environment=environment,
        )
        _run(
            [str(console_script), "taxonomy", "validate"],
            environment=environment,
        )
        _run(
            [str(console_script), "taxonomy", "validate-resources"],
            environment=environment,
        )


def verify_distribution(directory: Path) -> tuple[str, ...]:
    """Inspect exactly one wheel and source archive, then install the wheel cleanly."""
    wheels = sorted(directory.glob("signlab-*.whl"))
    source_archives = sorted(directory.glob("signlab-*.tar.gz"))
    if len(wheels) != 1 or len(source_archives) != 1:
        return ("expected exactly one SignLab wheel and one source distribution",)
    errors = (*_inspect_wheel(wheels[0]), *_inspect_sdist(source_archives[0]))
    if errors:
        return tuple(sorted(set(errors)))
    _install_and_smoke_test(wheels[0])
    return ()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default=Path("dist"), type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        errors = verify_distribution(arguments.directory)
    except (
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        tarfile.TarError,
        zipfile.BadZipFile,
    ):
        print("Distribution verification could not complete.", file=sys.stderr)
        return 2
    if errors:
        print("Distribution verification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Distribution archives and isolated wheel install verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
