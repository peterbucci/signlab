import hashlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from typer.testing import CliRunner

from signlab import cli
from signlab import model_parity as parity
from signlab.contracts.canonical import canonical_json_bytes, parse_json_object
from signlab.experiments.calibration import load_calibration_config
from signlab.model_bundle import BrowserModelBundleManifestV1, DecisionPolicyV1

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/popsign-constructed-calibration-v1.json"
POLICY = ROOT / "docs/reports/popsign-constructed-calibration-policy-v1.json"
MANIFEST = (
    ROOT / "src/signlab/resources/model_bundles/examples/browser-model-bundle-manifest.example.json"
)
CHECKPOINT = b"exact-checkpoint"


def _error(code: str, action: Any) -> None:
    with pytest.raises(parity.ModelParityError) as caught:
        action()
    assert (caught.value.code, str(caught.value)) == (code, code)


def _manifest() -> BrowserModelBundleManifestV1:
    payload = cast(dict[str, Any], parse_json_object(MANIFEST.read_bytes()))
    candidate = cast(dict[str, Any], payload["candidate"])
    candidate["research_checkpoint_sha256"] = "sha256:" + hashlib.sha256(CHECKPOINT).hexdigest()
    return BrowserModelBundleManifestV1.model_validate_json(
        canonical_json_bytes(payload), strict=True
    )


def _probabilities() -> np.ndarray[Any, np.dtype[np.float32]]:
    values = np.full((18, 6), 0.02, dtype=np.float32)
    values[np.arange(18), np.arange(18) % 6] = 0.9
    return values


def _install(
    monkeypatch: pytest.MonkeyPatch,
    portable: np.ndarray[Any, np.dtype[np.float32]] | None = None,
) -> tuple[list[int], list[int]]:
    manifest = _manifest()
    policy = DecisionPolicyV1.model_validate_json(POLICY.read_bytes(), strict=True)
    matrix = np.zeros((18, 64, 126), dtype=np.float32)
    matrix[:, 0, 0] = np.arange(18)
    labels = np.concatenate((np.repeat(np.arange(5), 3), np.full(3, 5))).astype(np.int64)
    native = _probabilities()
    portable_values = native.copy() if portable is None else portable
    portable_values[:, 0] += 5e-7
    portable_values[:, 1] -= 5e-7
    native_calls: list[int] = []
    onnx_calls: list[int] = []

    class Model:
        def predict(self, row: np.ndarray[Any, Any], *, verbose: int) -> np.ndarray[Any, Any]:
            assert (row.shape, verbose) == ((1, 64, 126), 0)
            index = int(row[0, 0, 0])
            native_calls.append(index)
            return native[index : index + 1]

    def load_model(path: Path) -> Model:
        assert (path.name, path.read_bytes()) == ("candidate.keras", CHECKPOINT)
        return Model()

    class Session:
        def __init__(self, raw: bytes, *, providers: list[str]) -> None:
            assert (raw, providers) == (b"onnx", ["CPUExecutionProvider"])

        def run(self, outputs: list[str], inputs: dict[str, np.ndarray[Any, Any]]) -> list[Any]:
            row = inputs["input"]
            index = int(row[0, 0, 0])
            onnx_calls.append(index)
            return [portable_values[index : index + 1]]

    def device(name: str) -> Any:
        assert name == "/CPU:0"
        return nullcontext()

    runtime = parity._Runtime(
        SimpleNamespace(models=SimpleNamespace(load_model=load_model)),
        SimpleNamespace(InferenceSession=Session),
        SimpleNamespace(device=device),
    )
    monkeypatch.setattr(
        parity, "validate_browser_bundle", lambda _root: (manifest, "sha256:" + "b" * 64)
    )
    monkeypatch.setattr(parity, "_bundle_assets", lambda *_args: (b"onnx", policy))
    monkeypatch.setattr(
        parity,
        "_development_matrix",
        lambda *_args: (matrix, labels, ("target",) * 15 + ("constructed_other",) * 3),
    )
    monkeypatch.setattr(parity, "_runtime", lambda: runtime)
    return native_calls, onnx_calls


def _run(tmp_path: Path) -> parity.ModelParityResult:
    checkpoint = tmp_path / "model.keras"
    checkpoint.write_bytes(CHECKPOINT)
    return parity.run_native_onnx_parity(
        tmp_path, checkpoint, CONFIG, "corpus", "external", tmp_path / "report.json"
    )


def test_parity_runs_exact_rows_on_cpu_and_writes_sanitized_canonical_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native_calls, onnx_calls = _install(monkeypatch)
    result = _run(tmp_path)
    raw = result.report_path.read_bytes()
    report = cast(dict[str, Any], parse_json_object(raw))
    comparison = cast(dict[str, Any], report["comparison"])
    rows = cast(list[dict[str, Any]], report["rows"])
    assert native_calls == onnx_calls == list(range(18))
    assert (result.row_count, comparison["probability_element_count"]) == (18, 108)
    assert comparison["native_abstentions"] == comparison["onnx_abstentions"] == 0
    assert [row["alias"] for row in rows] == list(parity._ALIASES)
    assert raw == canonical_json_bytes(report) + b"\n"
    assert str(tmp_path).encode() not in raw and b"sample_" not in raw  # noqa: PT018


def test_matrix_loader_requests_only_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_calibration_config(CONFIG)[0]
    seen: list[tuple[str, ...]] = []

    def stop_after_partition_check(*args: Any) -> tuple[Any, ...]:
        seen.append(cast(tuple[str, ...], args[-1]))
        raise OSError

    monkeypatch.setattr(parity, "_load_inputs", lambda *_args: (object(), object()))
    monkeypatch.setattr(parity, "_load_samples", stop_after_partition_check)
    with pytest.raises(parity.ModelParityError, match="parity_development_inputs_invalid"):
        parity._development_matrix(config, Path("corpus"), Path("manifest"))
    assert seen == [("validation",)]


@pytest.mark.parametrize(
    ("mode", "code"),
    [
        ("probability", "validation_target_001_probability_mismatch"),
        ("invalid", "validation_target_001_onnx_probabilities_invalid"),
    ],
)
def test_row_failures_are_sanitized_and_leave_no_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, code: str
) -> None:
    portable = _probabilities()
    if mode == "probability":
        portable[0] = np.roll(portable[0], 1)
    else:
        portable[0, 0] = np.nan
    _install(monkeypatch, portable)
    _error(code, lambda: _run(tmp_path))
    assert not (tmp_path / "report.json").exists()


def test_identity_conflict_and_cli_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    config, raw = load_calibration_config(CONFIG)
    _error(
        "parity_input_identity_mismatch",
        lambda: parity._require_identities(_manifest(), config, raw, b"wrong-checkpoint"),
    )
    monkeypatch.setattr(
        parity,
        "run_native_onnx_parity",
        lambda *_args: (_ for _ in ()).throw(parity.ModelParityError("blocked")),
    )
    result = CliRunner(env={"NO_COLOR": "1"}).invoke(
        cli.app,
        "evaluate native-onnx-parity bundle model config --corpus-root corpus --external-manifest external --report report".split(),  # noqa: E501, SIM905
    )
    assert (result.exit_code, result.output.strip()) == (1, "Native/ONNX parity failed: blocked.")


def test_critical_runtime_boundaries_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "model.onnx").write_bytes(b"x")
    (tmp_path / "decision-policy.json").write_bytes(b"x")
    _error("parity_bundle_identity_mismatch", lambda: parity._bundle_assets(tmp_path, _manifest()))
    invalid = np.zeros((1, 6), np.float32)
    _error("native_probabilities_invalid", lambda: parity._checked_probabilities(invalid, "native"))

    def unavailable(_name: str) -> Any:
        raise ImportError

    monkeypatch.setattr("signlab.model_parity.importlib.import_module", unavailable)
    _error("parity_runtime_unavailable", parity._runtime)
