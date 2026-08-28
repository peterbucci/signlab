"""Regenerate or check the project-authored portable-feature golden corpus."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEST_DIRECTORY = REPOSITORY_ROOT / "tests"
GOLDEN_PATH = (
    TEST_DIRECTORY / "fixtures" / "public" / "features" / "portable-landmark-goldens-v1.json"
)


def _render() -> str:
    sys.path.insert(0, str(TEST_DIRECTORY))
    try:
        from test_feature_goldens import golden_document

        return json.dumps(golden_document(), indent=2, sort_keys=True) + "\n"
    finally:
        sys.path.pop(0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the golden corpus drifts")
    parser.add_argument("--output", type=Path, default=GOLDEN_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write the exact corpus or return nonzero when check mode finds drift."""

    arguments = _build_parser().parse_args(argv)
    expected = _render()
    if arguments.check:
        try:
            return int(arguments.output.read_text(encoding="utf-8") != expected)
        except OSError:
            return 1
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(expected, encoding="utf-8", newline="\n")
    print(arguments.output.relative_to(REPOSITORY_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
