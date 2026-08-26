"""Generate the root DVC pipeline from SignLab's typed stage registry."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

from signlab.reproducibility.stages import render_dvc_pipeline


def write_pipeline(repository_root: Path, *, check: bool) -> bool:
    """Check or atomically write the generated root ``dvc.yaml``."""

    target = repository_root / "dvc.yaml"
    expected = render_dvc_pipeline()
    if check:
        try:
            return target.read_text(encoding="utf-8") == expected
        except OSError:
            return False
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            dir=repository_root,
            prefix=".dvc.yaml.",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
            newline="\n",
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(expected)
        temporary_path.replace(target)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if not write_pipeline(Path.cwd(), check=arguments.check):
        print("Generated dvc.yaml is missing or stale.")
        return 1
    print("Generated dvc.yaml is current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
