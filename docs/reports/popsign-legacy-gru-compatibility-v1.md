# PopSign legacy GRU compatibility v1

> Compatibility/export smoke only. This 50-train, 15-validation run is not model-quality, product-performance, or sign-language evidence.

## What was reproduced

- Recovered legacy run: `20251222_154233_gru_phase_3_run_001`
- Recovered model: `sha256:f69c07838a477df0853a6cdb71b1acb9a933e0de1491359da0cae5462584e46c`
- Same functional architecture: two forward GRU layers with 128 units, attention pooling, and a five-class softmax.
- Required input adaptation: legacy `30 x 126` effective tensors became current-contract `64 x 134` tensors.
- The legacy config's `input_dim=63` did not match its saved model's width of 126.
- Attention reduction uses an exporter-friendly dot product equivalent to the legacy weighted sum.
- The legacy Adam run did not actually apply its declared weight decay; this run does not add it.

## Reproducibility identity

- Source commit: `c7d455b903071c49ce654234a9a6f13f9b681693` (dirty: `false`)
- Configuration: `sha256:2358c943f7ec74d0f214864c934c62e1e83ecd736b63f793abad04e86b40928b`
- Frozen split: `sha256:673c777c67e47715127b75f2bf18aaa794a15e1604458d30da164ec7f90ead20`
- Feature plan: `sha256:ba8bedde078d73e9b5946d9aa115a463cf05eea50a39d5fb6ae01f950bcd01e6`
- Seed: `42`
- Opened data: 50 training clips and 15 validation clips; final test stayed sealed.

## Result

- Epochs completed: `16`
- Best epoch: `8`
- Validation accuracy: `1.000`
- Validation loss at the best epoch: `0.417`
- Validation macro-F1: `1.000`
- Keras artifact: `sha256:5452a3d7f4de29f3364bd49d8c96d73b4e313fa4c352705186bdb0f648a0d85c`
- ONNX artifact: `sha256:eb53b637c35116f096694ab3bb5b2a1a0ae0c529ab53695716812c1816fbb4e7`
- ONNX parity: all 15 validation outputs passed at `1e-5`; maximum absolute difference `3.28e-07`; predicted labels were identical.

## Decision

The outcome and exporter risk are recorded in [ADR 0001](../decisions/0001-training-framework.md).

## Reproduce

```shell
uv run --locked --extra legacy-compatibility signlab train legacy-gru-compatibility configs/experiments/popsign-legacy-gru-compatibility-v1.json --corpus-root <frozen-split-root> --external-manifest <external-dataset-manifest.json> --output-root runs/popsign-legacy-gru-compatibility-v1
```
