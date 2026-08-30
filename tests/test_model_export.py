from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import onnx
import onnxruntime  # type: ignore[import-untyped]
import pytest
from typer.testing import CliRunner

from signlab import cli
from signlab import model_export as exporter
from signlab.contracts.canonical import parse_json_object
from signlab.model_bundle import ModelBundleError, validate_browser_bundle

ROOT = Path(__file__).resolve().parents[1]
DOSSIER = ROOT / "configs/evaluation/popsign-tcn-portable-export-candidate-v1.json"
REPORT = ROOT / "docs/reports/popsign-tcn-portable-export-nomination-v1.json"
EXPECTED_FILES = {path for _role, path, _media in exporter._ASSETS} | {"manifest.json"}


def _graph_bytes(suffix: int, mode: str) -> bytes:
    helper, tensor = onnx.helper, onnx.TensorProto
    names = {name: f"{name}_{suffix}" for name in ("reduce_axes", "axis", "weight", "bias")}
    initializers = [
        helper.make_tensor(names["reduce_axes"], tensor.INT64, [2], [1, 2]),
        helper.make_tensor(names["axis"], tensor.INT64, [1], [1]),
        helper.make_tensor(names["weight"], tensor.FLOAT, [1, 6], [0.0] * 6),
        helper.make_tensor(
            names["bias"], tensor.FLOAT, [6], [float("nan")] * 6 if mode == "nan" else [0.0] * 6
        ),
    ]
    values = {name: f"{name}_{suffix}" for name in ("mean", "row", "scores")}
    nodes = [
        helper.make_node(
            "ReduceMean", ["input", names["reduce_axes"]], [values["mean"]], keepdims=0
        ),
        helper.make_node("Unsqueeze", [values["mean"], names["axis"]], [values["row"]]),
        helper.make_node(
            "Gemm", [values["row"], names["weight"], names["bias"]], [values["scores"]]
        ),
        helper.make_node("Softmax", [values["scores"]], ["Identity:0"], axis=1),
    ]
    if mode == "operator":
        nodes[-1].domain = "custom"
    graph = helper.make_graph(
        nodes,
        f"volatile_graph_{suffix}",
        [helper.make_tensor_value_info("input", tensor.FLOAT, [1, 64, 126])],
        [helper.make_tensor_value_info("Identity:0", tensor.FLOAT, [1, 6])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("ai.onnx.ml", 2)],
    )
    return cast(bytes, model.SerializeToString())


def _setup(monkeypatch: pytest.MonkeyPatch, mode: str = "valid") -> None:
    calls = [0]

    class Model:
        input_shape = (None, 64, 126) if mode != "model" else (None, 64, 127)
        output_shape = (None, 6)
        output_names = ("probabilities",)

        def count_params(self) -> int:
            return 29_094

        def export(self, path: Path, **options: object) -> None:
            assert options["opset_version"] == 18
            calls[0] += 1
            path.write_bytes(_graph_bytes(calls[0], mode))

    def load_model(_path: Path) -> Model:
        assert _path.name == "candidate.keras"
        assert _path.read_bytes() == b"local-checkpoint"
        return Model()

    def input_spec(**options: object) -> dict[str, object]:
        assert options == {"shape": (1, 64, 126), "dtype": "float32", "name": "input"}
        return options

    keras = SimpleNamespace(InputSpec=input_spec, models=SimpleNamespace(load_model=load_model))
    monkeypatch.setattr(
        exporter,
        "_runtime",
        lambda: exporter._Runtime(keras, np, onnx, onnxruntime),
    )
    report = cast(dict[str, object], parse_json_object(REPORT.read_bytes()))
    if mode == "report":
        report = {**report, "candidate_status": "nomination_blocked"}
    monkeypatch.setattr(exporter, "build_candidate_nomination_report", lambda *_a, **_k: report)
    if mode != "report":
        dossier = cast(Any, exporter).load_candidate_dossier(DOSSIER)
        monkeypatch.setattr(
            exporter,
            "_verified_candidate",
            lambda *_a: (dossier, REPORT.read_bytes(), b"local-checkpoint"),
        )


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str = "bundle") -> Any:
    checkpoint = tmp_path / "model.keras"
    checkpoint.write_bytes(b"local-checkpoint")
    return exporter.export_browser_candidate_bundle(
        DOSSIER, REPORT, checkpoint, ROOT, tmp_path / name
    )


def test_export_is_complete_independently_valid_and_byte_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup(monkeypatch)
    first, second = _run(tmp_path, monkeypatch, "first"), _run(tmp_path, monkeypatch, "second")
    assert (first.bundle_sha256, first.model_sha256) == (
        second.bundle_sha256,
        second.model_sha256,
    )
    assert {
        path.relative_to(first.bundle_root).as_posix()
        for path in first.bundle_root.rglob("*")
        if path.is_file()
    } == EXPECTED_FILES
    for name in EXPECTED_FILES:
        assert (first.bundle_root / name).read_bytes() == (second.bundle_root / name).read_bytes()
    manifest, digest = validate_browser_bundle(first.bundle_root)
    assert digest == first.bundle_sha256
    assert manifest.licenses[-1].distribution == "local_evaluation_only"
    smoke = parse_json_object((first.bundle_root / "golden/smoke.json").read_bytes())
    assert smoke["evidence_scope"] == "onnx_self_smoke_only"
    assert cast(Any, smoke["output"])["probabilities_scope"] == "display_only_quantized_4dp"
    graph = onnx.load(first.bundle_root / "model.onnx")
    assert graph.graph.output[0].name == "probabilities"
    assert [(item.domain, item.version) for item in graph.opset_import] == [("", 18)]


@pytest.mark.parametrize(
    ("mode", "code"),
    [
        ("report", "candidate_nomination_invalid"),
        ("model", "candidate_model_invalid"),
        ("operator", "onnx_operator_unsupported"),
        ("nan", "onnx_smoke_invalid"),
        ("validation", "bundle_publication_failed"),
        ("existing", "destination_conflict"),
    ],
)
def test_every_prepublication_failure_is_atomic_and_path_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, code: str
) -> None:
    _setup(monkeypatch, mode)
    if mode == "validation":
        monkeypatch.setattr(
            exporter,
            "validate_browser_bundle",
            lambda _path: (_ for _ in ()).throw(ModelBundleError("invalid")),
        )
    output = tmp_path / "private-output"
    if mode == "existing":
        output.mkdir()
    with pytest.raises(exporter.ModelExportError) as caught:
        _run(tmp_path, monkeypatch, output.name)
    assert (caught.value.code, str(caught.value)) == (code, code)
    assert str(tmp_path) not in str(caught.value)
    assert output.is_dir() if mode == "existing" else not output.exists()
    assert not tuple(tmp_path.glob(".private-output.staging-*"))


def test_cli_reports_only_portable_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    result = exporter.ModelExportResult(Path("ignored"), "sha256:" + "a" * 64, "unused")
    monkeypatch.setattr(exporter, "export_browser_candidate_bundle", lambda *_a: result)
    runner = CliRunner(env={"NO_COLOR": "1"})
    args = (
        "export browser-bundle dossier report model --repository-root . --output-root out".split()  # noqa: SIM905
    )
    invoked = runner.invoke(cli.app, args)
    assert (invoked.exit_code, invoked.output.strip()) == (
        0,
        f"Candidate bundle exported and verified: {result.bundle_sha256}.",
    )
    monkeypatch.setattr(
        exporter,
        "export_browser_candidate_bundle",
        lambda *_a: (_ for _ in ()).throw(exporter.ModelExportError("blocked")),
    )
    failed = runner.invoke(cli.app, args)
    assert (failed.exit_code, failed.output.strip()) == (
        1,
        "Candidate bundle export failed: blocked.",
    )
