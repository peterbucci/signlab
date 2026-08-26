# Project charter

## Problem

The legacy application successfully demonstrated five hand gestures, model
experimentation, live segmentation, and user feedback. Its results are difficult
to reproduce or generalize because experiment artifacts outgrew the UI, clips were
not grouped by signer/session during evaluation, inference preprocessing could
drift from training, and idle/unknown behavior was underspecified.

## Product claim

Until the gestures are reviewed by a qualified signer against documented lexical
sources, SignLab's approved claim is:

> SignLab is a research prototype for recognizing isolated performances of five
> predefined hand gestures within continuous webcam video. It separates non-target
> events (`other`) from no active event (`inactive`) and uncertain decisions
> (`abstain`); no sign-language or translation capability is claimed.

The five targets plus learned `other` produce six classifier outputs. `inactive`
belongs to event detection and `abstain` to the decision policy; neither is a class.
SignLab is not described as a complete accessibility solution or as understanding
a named sign language.

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
