# Dataset manifests and Parquet tables

SignLab datasets are portable, versioned contracts rather than folders inferred
from filenames. The current `dataset-manifest/2` commits to six normalized tables:
participants, sessions, recordings, clips, annotations, and derived artifacts. The
published `dataset-manifest/1` reader and fixture remain unchanged so historical
pipeline records continue to validate.

## Identity boundary

Each table has two deliberately separate identities:

- `content_sha256` hashes its validated, primary-key-sorted logical rows using
  RFC 8785 and a table-schema domain separator.
- The table artifact reference hashes the exact Parquet bytes and records their
  byte count and portable locator.

The dataset `data_sha256` includes the six semantic table identities and the exact
sample projection, but excludes storage locators and Parquet encoding details.
Moving a dataset or rewriting equivalent rows with a different valid Parquet
encoding therefore preserves its semantic identity. The enclosing contract digest
still changes when an exact table artifact or locator changes.

## Time, labels, and grouping

Absolute times use canonical UTC. Media bounds use integer microseconds in the
root recording's coordinate system and half-open intervals: `[start_us, end_us)`.
Intervals must be non-empty, remain inside their recording, and, when an annotation
targets a clip, remain inside that clip.

Classifier labels are exactly `hello`, `no`, `please`, `thank_you`, `yes`, and
`other`. `inactive` is detector state and `abstain` is decision-policy output;
neither is a dataset class label. Ambiguous and ignored annotations carry coded
reasons and cannot enter the trainable sample projection.

Participants are pseudonymous. Sessions and recordings carry only coarse,
non-identifying device and capture facts. Every recording embeds its exact
recording-level consent grant. Names, contact details, signatures, device serials,
hostnames, and identity-vault mappings never belong in these tables.
Every v2 row artifact uses an enforceable content-addressed location with no
user-selected path segments:

```text
objects/sha256/p-<first-two-hex>/sha256-<64-hex>/<opaque-artifact-id>
signlab://objects/sha256/p-<first-two-hex>/sha256-<64-hex>/<opaque-artifact-id>
```

The hash directory must match the artifact reference and the final segment must
match its opaque artifact ID. Media type is explicit metadata, so the path needs
no extension. Dataset-table locations are likewise fixed to
`tables/<registered-table-name>.parquet` or
`signlab://tables/<registered-table-name>`. Backend, tenant, hostname, participant,
and free-text labels are resolver configuration, never locator content. The
retained `dataset-manifest/1` sample contract is not retroactively rewritten by
this v2 policy.

Participant, session, and source-recording membership is reconciled through every
foreign key. A crop, augmentation, or window also records its immutable root
recording and its split/partition. Validation walks the lineage graph and rejects
orphans, cycles, alternate roots, or derived samples that cross a frozen split.

## Parquet profile and validation

SignLab writes explicit Arrow schemas; it never relies on dataframe or Parquet
type inference. The writer uses Parquet 2.6, Zstandard level 3, microsecond UTC
timestamps, fixed row-group sizing, stored Arrow schemas, and page checksums.
Parquet bytes are an interchange representation, not the semantic dataset hash.

Full validation proceeds fail-closed:

1. Resolve workspace-relative paths beneath an explicit workspace root.
2. Hash the captured bytes and verify both their size and SHA-256.
3. Verify every present Parquet page checksum and the exact Arrow schema, field
   order, nullability, field IDs, and approved metadata. Exact outer SHA-256 is
   mandatory even for equivalent third-party encodings that omit page checksums.
4. Revalidate every row through its strict contract model.
5. Reconcile table counts and semantic hashes, foreign keys, intervals, taxonomy,
   consent snapshots, lineage, sample projection, and optional frozen split.

The current reference validator materializes all six logical tables in memory. It
is appropriate for contract fixtures and early experiments; production-scale
collections will require batched Parquet scanning, streaming semantic hashes, and
out-of-core relationship joins. This limitation is explicit rather than hidden
behind a successful validation state.

A logical `signlab://` locator requires an explicitly configured artifact resolver;
the URI is a portable logical name and never selects a physical host.
Structural hashes never establish current consent authorization. A positive current
authorization result additionally requires an authenticated consent receipt,
complete event log, and the governance verifier supplied by the identity-vault
boundary.

Validate the packaged review resources and create a temporary synthetic bundle:

```shell
uv run signlab data validate-resources
uv run signlab data write-example-dataset ../signlab-example-dataset
uv run signlab data validate-dataset ../signlab-example-dataset/dataset-manifest.json --workspace-root ../signlab-example-dataset
```

The validator reports Parquet table-byte integrity, semantic integrity, referenced
row-artifact integrity, split compatibility, and current consent authorization as
separate states. It never upgrades an omitted external check to “verified.”
The writer builds and validates a sibling staging directory, writes its manifest
as the final completion marker, then atomically publishes the directory.
The default byte-mismatch gate covers the six manifest-bound Parquet tables.
After an authorized DVC pull, add `--verify-row-artifacts` to stream every recording,
materialized clip, and derived/sample file from its content-addressed workspace
locator. The verifier rejects links, reparse points, hardlink aliases, special files,
path escapes, size drift, SHA-256 drift, and files that change while open. Logical
`signlab://` locators still require an explicit storage adapter. This byte check
remains separate from authenticated current consent. See
[data versioning](data-versioning.md) for the protected pull workflow.

## Public fixtures and private data

The repository packages generated JSON Schemas, Arrow-schema snapshots, and small
human-readable synthetic examples. Tests build canonical Parquet bundles in
temporary directories and seed independent checksum, label, interval, consent,
and lineage defects. Generated Parquet files and participant media are not shipped
in the Python package or committed as project data.

Regenerate the committed JSON review artifacts after an authoritative table or
Arrow schema change. A manifest-envelope change requires both generators because
its JSON Schema belongs to the shared pipeline-contract resource set:

```shell
uv run python scripts/generate_contract_resources.py
uv run python scripts/generate_dataset_resources.py
```

Real participant collection remains blocked by the governance readiness gate. The
synthetic fixtures prove format and validator behavior only; they do not represent
institutional approval, authenticated consent operations, durable private storage,
or a production dataset.
