# Configurations

This directory owns reviewed, portable input configuration. Committed examples must
contain no resolved machine paths, credentials, participant identifiers, or private
artifact locations. Versioned runtime schemas and concrete configuration files arrive
with the contract stories that consume them.

`pipeline/synthetic-dvc.json` is the reviewed, fixture-only input to the public DVC
clean-room proof. It is not a production extraction, quality, feature, or split
configuration; its values only influence deterministic smoke-test receipts. It must
never contain environment-specific values. Private remote locations belong only in
ignored `.dvc/config.local`; credentials remain in the provider credential chain.

`experiments/popsign-reference-baselines-v1.json` is the single reviewed input for
the frozen five-target reference benchmark. It binds the corpus, external manifest,
split, feature plan, taxonomy, seed, two logistic-regression `C` candidates, and
latency procedure. Dataset and output paths remain command-line inputs because they
are local state, not part of the experiment identity.

`experiments/popsign-sequence-baselines-v1.json` freezes the one-seed engineering
feasibility comparison between a small GRU and a similarly sized causal residual
TCN. It fixes the same train/validation inputs, model shapes, training budget,
checkpoint rule, metrics, and CPU-latency procedure; it does not authorize test
access, tuning, or model selection.

`experiments/popsign-representation-ablations-v1.json` freezes three signer-grouped
development folds and six declared model/representation conditions: 18 total fits,
including nine neural fits. It compares only hand-local, body-relative, and combined
views; test access, winner selection, checkpoints, and export are excluded.

`evaluation/candidate-event-detector-v1.json` freezes one engineering configuration for
the frame-by-frame candidate-event state machine. Its thresholds are exercised only by
constructed conformance evidence; they are not tuned natural-use parameters.
