# Project charter

## Problem

The legacy application successfully demonstrated five hand gestures, model
experimentation, live segmentation, and user feedback. Its results are difficult
to reproduce or generalize because experiment artifacts outgrew the UI, clips were
not grouped by signer/session during evaluation, inference preprocessing could
drift from training, and idle/unknown behavior was underspecified.

## Product claim

Until validated by fluent signers and a documented language source, SignLab is:

> A five-class isolated hand-gesture recognition system with continuous webcam
> segmentation and calibrated rejection.

It is not described as sign-language translation, a complete accessibility
solution, or a system that understands a sign language.

## Primary outcome

A recruiter or reviewer can open a URL, understand the research question and
limitations, run the model locally in their browser, and trace every published
metric to a reproducible dataset, split, configuration, commit, and model bundle.

## Success measures

- Signer-held-out macro-F1 and per-class recall.
- Event-level precision, recall, and F1 on continuous sessions.
- False activations per hour on negative-only sessions.
- Coverage and accepted error rate at the deployed rejection policy.
- p50/p95 landmark, model, and post-gesture decision latency.
- Python/ONNX/TypeScript prediction parity within declared tolerances.
- A clean clone can run a synthetic end-to-end fixture in CI.

## Non-goals for the core release

- Continuous sign-language translation.
- Public cloud training or server-side video inference.
- A large model zoo or unbounded hyperparameter sweeps.
- Kubernetes, microservice sprawl, or a desktop installer.
- Uploading webcam frames by default.

## Operating principles

1. The CLI and versioned contracts are the source of truth; UIs only invoke or present them.
2. Preserve raw signals and derive representations in versioned feature stages.
3. Never select a model or threshold using the locked final test set.
4. Prefer one justified experiment over a broad architecture sweep.
5. Make privacy, consent, failure modes, and limitations visible in the product.
