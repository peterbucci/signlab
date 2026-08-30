"""Regenerate or check the shared candidate-runtime JSON and tiny ONNX probe."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
JSON_PATH = TESTS / "fixtures/public/parity/candidate-runtime-goldens-v1.json"
MODEL_PATH = TESTS / "fixtures/public/parity/candidate-runtime-v1.onnx"


def _artifacts() -> tuple[bytes, bytes]:
    sys.path.insert(0, str(TESTS))
    try:
        from test_candidate_runtime_goldens import golden_document, onnx_model_bytes

        rendered = json.dumps(
            golden_document(), allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True
        )
        return (rendered + "\n").encode(), onnx_model_bytes()
    finally:
        sys.path.pop(0)


def main(argv: Sequence[str] | None = None) -> int:
    """Write both artifacts, or return nonzero if either checked artifact drifts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when either artifact drifts")
    arguments = parser.parse_args(argv)
    expected_json, expected_model = _artifacts()
    if arguments.check:
        try:
            return int(
                JSON_PATH.read_bytes() != expected_json or MODEL_PATH.read_bytes() != expected_model
            )
        except OSError:
            return 1
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_bytes(expected_json)
    MODEL_PATH.write_bytes(expected_model)
    print(JSON_PATH.relative_to(ROOT).as_posix())
    print(MODEL_PATH.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
