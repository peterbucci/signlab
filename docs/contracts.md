# Versioned pipeline contracts

SignLab treats research configuration and artifacts as portable data, not as
Python objects, local paths, or UI state. Pydantic models under
`src/signlab/contracts/` are the authoritative readers and writers. The committed
JSON Schemas and synthetic examples are review surfaces for other runtimes; they
do not replace application validation.

## Contract chain

The six v1 contracts form one digest-bound provenance chain:

```text
dataset manifest
  -> grouped split manifest
  -> preprocessing plan
  -> resolved configuration
  -> terminal run record
  -> research model manifest
```

| Contract | Identity and responsibility |
| --- | --- |
| `dataset-manifest/1` | A portable dataset envelope, exact sample membership, leakage groups, taxonomy/governance bindings, and a storage-independent `data_sha256` |
| `split-manifest/1` | Exact train/validation/test membership, participant/session/source-recording isolation, seed, and the exact dataset identities |
| `preprocessing-plan/1` | An ordered chain of versioned operations with explicit adjacent input/output schema compatibility |
| `resolved-configuration/1` | Fully expanded model, optimizer, trainer, and evaluator settings plus exact upstream references and deterministic seed policy |
| `run-record/1` | An immutable terminal result with clean code/lock identities, sanitized runtime facts, finite aggregate metrics, and content-addressed outputs |
| `model-manifest/1` | A research-model artifact bound to its successful run, configuration, dataset, split, preprocessing plan, taxonomy, label order, and input/output schemas |

`assert_model_compatible(...)` validates the complete chain. The narrower
`assert_split_compatible(...)`, `assert_resolved_configuration_compatible(...)`,
and `assert_run_compatible(...)` checks are available at earlier pipeline stages.
Valid hashes prove identity and integrity, not consent, authenticity, approval, or
promotion. Those decisions remain separate authenticated policy checks.

This foundation intentionally does not pre-empt later domain stories. The Phase 1
dataset story adds recording, clip, annotation, consent, quality, and Parquet table
contracts to the dataset envelope. Representation and extraction stories register
concrete preprocessing operations. The portable-inference epic defines the exact
ONNX browser-bundle component set and parity policy. Those contracts must reference
these identities rather than create a second provenance system.

## Canonical JSON and hashes

New pipeline contracts use the
[RFC 8785 JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785.html)
over validated I-JSON values. Each contract hash is:

```text
SHA-256(
  UTF8("signlab.contract/" + schema_version) + NUL + RFC8785(document)
)
```

The schema version is a domain separator, so identical JSON values belonging to
different contract families cannot share an identity. Digests use the full
`sha256:<64 lowercase hex>` form. Duplicate object keys, invalid UTF-8, lone
surrogates, non-finite numbers, non-string object keys, and integers outside the
exact IEEE-754 interoperable range are rejected before hashing.

Arrays whose order has meaning—especially preprocessing steps—retain that order.
Set-like tuples must already be unique and sorted; validators reject alternate
spellings instead of silently changing a submitted document. This makes the same
logical record serialize and hash identically on Windows, Linux, and a future
TypeScript consumer.

The published taxonomy and participant-governance v1 resources retain their
existing canonicalization and golden digests. Changing a published digest algorithm
in place would rewrite identity, so RFC 8785 begins at this new pipeline-contract
boundary.

### Dataset identities

The dataset envelope has two related identities:

- `data_sha256` hashes taxonomy/governance bindings, sample and grouping identities,
  labels, and artifact byte identities. It excludes storage locators and dataset
  display/version metadata, so moving the same bytes does not change the data.
- The dataset contract digest hashes the complete manifest, including its portable
  locators and `data_sha256`. A `ContractRefV1` binds this exact document identity.

Splits, configurations, runs, and models carry both where needed. Substituting a
new manifest under an old dataset ID or pointing a split at different data fails
the compatibility checks.

## Portable locations

Physical machine paths never cross a contract boundary. Every artifact uses one
of two discriminated locator objects:

```json
{"kind": "workspace_relative", "path": "fixtures/features/sample_01.json"}
```

```json
{"kind": "artifact_uri", "uri": "signlab://fixture-store/features/sample_01"}
```

Workspace paths use forward slashes and are resolved only against an explicit
workspace root by an adapter. Drive paths, drive-relative paths, UNC paths, leading
slashes, backslashes, traversal, empty/dot segments, Windows reserved device names,
and trailing dots/spaces are invalid.

Artifact URIs are logical `signlab://` identifiers governed by a deliberately
narrow [RFC 3986](https://www.rfc-editor.org/rfc/rfc3986.html) profile. They are
lowercase, credential-free, query-free, fragment-free, and percent-encoding-free.
A URI is a locator—not an integrity claim—and is always paired with a full content
digest.

## Compatibility and migration

The JSON Schema dialect marker (`$schema`), schema resource identity (`$id`), and
instance contract version (`schema_version`) have different jobs. Every v1 document
contains an exact discriminator such as `split-manifest/1`. Readers dispatch on it
before Pydantic validation.

| Contract family | Current writer | Retained readers | Automatic migration |
| --- | --- | --- | --- |
| Dataset manifest | `dataset-manifest/1` | `/1` | Never |
| Split manifest | `split-manifest/1` | `/1` | Never |
| Preprocessing plan | `preprocessing-plan/1` | `/1` | Never |
| Resolved configuration | `resolved-configuration/1` | `/1` | Never |
| Run record | `run-record/1` | `/1` | Never |
| Model manifest | `model-manifest/1` | `/1` | Never |

Unknown, missing, legacy, and future versions fail closed with the supported reader
and this migration section. Because v1 is the first published pipeline-contract
set, no fictional v0 migration is supplied. The packaged v1 examples, schemas, and
golden hashes are the backward-compatibility corpus.

The current-writer registry is deliberately separate from the retained-reference
registry. When a future writer advances to `/2`, existing `/1` references remain
valid for as long as the `/1` reader is retained; changing the current writer must
never silently retire historical provenance.

A future migration must:

1. retain and validate the original reader and golden fixture;
2. expose an explicit directional migration outside normal validation;
3. record the source schema version and source digest as provenance;
4. validate the migrated output against the new reader; and
5. write a new document without mutating the retained input.

An incompatible semantic change requires a new instance discriminator and a new,
immutable absolute `$id`. Publishing different schema bytes under an existing `$id`
is forbidden. JSON Schema migration annotations are documentation only; application
code owns migration behavior.

## Validation surfaces

Generated schemas use JSON Schema Draft 2020-12 and reject closed-object, type,
constant, length, pattern, and locally expressible portability defects. Pydantic and
application validation additionally enforce canonical ordering, hashes, exact split
coverage, leakage-group closure, adjacent preprocessing schemas, terminal run state,
and cross-document provenance.

Run both layers through the CLI:

```shell
uv run signlab contracts versions
uv run signlab contracts validate-resources
uv run signlab contracts validate path/to/contract.json
```

Regenerate review artifacts after an authoritative model or synthetic example
change:

```shell
uv run python scripts/generate_contract_resources.py
```

The test suite compares generated resources to the committed package data,
validates the coherent six-contract example chain, freezes every example digest,
and exercises the installed wheel and source distribution. Generation must leave
the checkout unchanged.

## Privacy and reproducibility boundary

Only synthetic examples belong in the package. Contract identifiers are opaque and
must not contain names, contact information, hostnames, usernames, working
directories, secrets, environment substitutions, exception messages, or stack
traces. Run failures use registered code/stage identifiers. Runtime records name an
OS family and accelerator class only; they do not record a device or machine name.

`frozen=True` prevents normal model attribute assignment but does not make arbitrary
nested Python objects trustworthy. Public validators reconstruct a JSON object and
revalidate it. `model_copy(update=...)` is never an accepted validation or digest
boundary.

Canonical timestamps are semantic run facts and therefore affect run identity.
Temporary paths, locale, timezone, host state, and wall-clock generation time are
absent from deterministic dataset, split, preprocessing, and configuration
identities.
