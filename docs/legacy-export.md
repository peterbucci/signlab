# Legacy evidence export

Story #8 preserves the useful evidence from the school project without importing its
Streamlit/PyQt architecture or publishing participant-derived artifacts. The export
has two deliberately separate layers.

## Public evidence layer

`docs/legacy/export/v1/` is safe to review and version in Git. It contains:

- a sharded index of every legacy run, its sanitized configuration and metrics,
  portable `legacy://` artifact locators, availability, and explicit validity notes;
- the historical preprocessing plans and label maps used by the four promoted runs;
- content hashes and sizes for the promoted models, without model bytes;
- a receipt for the private quarantine, with aggregate counts and component digests;
- closed JSON Schemas for every public and private record format.

The run index is historical development evidence. Its sample-level split lacked
signer/session grouping, the saved test split was repeatedly inspected, the trained
models had no learned negative class, and the legacy `mamba` name did not describe a
Mamba state-space model. No row is eligible for a locked final-test claim.
Each row has an ordinal `record_id` and a unique `source_run_id`; the latter rebuilds
the legacy sweep/run identity used by promoted artifacts and live records, so those
references can be checked across stores even when the raw `run_name` was reused.

## Private quarantine layer

`data/private/legacy/v1/` is ignored by Git. The exporter creates a deterministic,
content-addressed quarantine containing:

- the four promoted Keras files and their small metadata inputs;
- every captured live-evaluation segment as an opaque byte object;
- all live attempts, annotations, earlier detections, and sessions as sanitized
  JSON Lines records; and
- a manifest that binds each `quarantine://sha256/...` URI to its byte count and
  SHA-256 digest.

The exporter never deserializes a model or landmark array. It opens SQLite sources
with immutable, query-only connections and copies approved files as bytes. Raw UUIDs,
database row IDs, participant/live-record wall-clock timestamps, absolute paths,
free-form notes, and legacy errors are excluded from the sanitized record streams.
Stable ordinal aliases preserve joins, while relative offsets, duration, confidence,
prediction, correction, segmentation settings, and source role retain the development
value of each attempt. Historical training-run and preprocessing-plan timestamps remain
as non-participant chronology and are never presented as capture times.

Historical label text is preserved exactly in this evidence layer, including the
spaced `thank you` label and the legacy idle token `nothing`. The taxonomy contract
defines the explicit import alias and keeps `nothing` quarantined; the exporter does
not silently relabel either value.

Quarantined feedback and segments are `development-only`; they cannot be used as a
locked final test. This local copy is not an off-machine backup. A separate authorized
archival-migration decision, not Story #14 or the Story #19 production collection,
must place any retained quarantine in approved private storage before the legacy
source can be retired.

## Export and verification

Supply every location explicitly so no command depends on the caller's current
directory or stores a machine path:

```shell
uv run signlab data export-legacy --legacy-root <legacy-root> --audit-snapshot docs/legacy/legacy-state.json --public-output docs/legacy/export/v1 --quarantine-output data/private/legacy/v1
```

The command first matches the source Git/database/artifact fingerprints recorded by
the immutable audit. Output targets must not already contain files. Generation is
atomic; a failed export leaves neither a partial public index nor a partial private
quarantine.

The committed public evidence can be checked on any clone without the legacy project:

```shell
uv run signlab data validate-legacy --public-root docs/legacy/export/v1
```

An operator who has the private quarantine can additionally verify its component
digests and referential integrity:

```shell
uv run signlab data validate-legacy --public-root docs/legacy/export/v1 --quarantine-root data/private/legacy/v1
```

Validation rejects unsupported versions, unrecognized fields, broken component
hashes, nonportable locators, raw identifiers, absolute machine paths, private values
in the public layer, and count or join mismatches.
