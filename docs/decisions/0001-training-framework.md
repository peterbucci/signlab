# ADR 0001: Use Keras 3 for the next temporal-model experiments

- Status: Accepted
- Date: 2026-08-29
- Story: [#26](https://github.com/peterbucci/signlab/issues/26)
- Evidence: [PopSign legacy GRU compatibility v1](../reports/popsign-legacy-gru-compatibility-v1.md)

## Context

SignLab needed one training framework for the next bounded temporal-model work. The
legacy project used Keras, while PyTorch would require a rewrite. Browser inference
will consume ONNX, so the deciding question was whether a current Keras model could
train reproducibly and cross that boundary without changing its predictions.

The available 50-training/15-validation corpus is too small to compare framework or
model quality. This decision therefore uses compatibility, implementation cost, and
export risk—not validation score—as its evidence.

## Decision

Use **Keras 3 with the TensorFlow backend** for the next temporal-model experiments,
with fixed-shape ONNX as the deployment boundary. Keep the exact versions locked and
run a real Keras-to-ONNX-to-ONNX-Runtime smoke test on Linux and Windows CI.

Do not add PyTorch, a framework-neutral trainer, or dual implementations now.

## Evidence

The reviewed compatibility command performed one seeded fit of the recovered
two-layer GRU architecture on 50 training clips and evaluated 15 validation clips.
It stopped after 16 epochs, restored epoch 8, and recorded validation loss 0.417,
accuracy 1.000, and macro-F1 1.000. Those scores are smoke evidence only.

The exported fixed `1 x 64 x 134` ONNX model passed the full ONNX checker. ONNX
Runtime's CPU provider matched all 15 Keras validation outputs with a maximum
absolute difference of `3.28e-7`, below the declared `1e-5` tolerance, and produced
identical labels. The final-test partition was never requested.

Keras officially supports ONNX through `Model.export()`, but its TensorFlow route
uses tf2onnx. The [tf2onnx project](https://github.com/onnx/tensorflow-onnx) currently
documents an older tested TensorFlow range and is seeking maintainers. That is a
real maintenance risk even though the pinned stack works today. See the official
[Keras export API](https://keras.io/api/models/model_saving_apis/export/),
[ONNX checker API](https://onnx.ai/onnx/api/checker.html), and
[ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html).

## Consequences

- The next model story can build on a working, measured path instead of first
  translating the legacy architecture.
- ONNX—not Keras serialization—is the boundary used by later inference work.
- The heavy compatibility dependencies remain optional and pinned.
- The validation score cannot support a model-quality or production claim.
- Reconsider PyTorch only if a required architecture cannot be expressed cleanly in
  Keras, the pinned exporter fails, or the tf2onnx maintenance risk becomes an
  observed blocker. Such a change must be a concrete migration, not a second generic
  training framework.

## Alternatives not chosen

- **PyTorch now:** plausible for future research, but it would add an untested rewrite
  when the measured Keras path already satisfies the immediate export requirement.
- **Support both frameworks:** rejected because no current consumer needs it and it
  would double training, export, and test work.
