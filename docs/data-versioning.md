# Data versioning and reproducible stages

SignLab uses DVC 3.67.1 for content-addressed data transport and stage caching.
Story #14 establishes the DVC wiring with a small public synthetic fixture. It does
not implement the production data transforms, provision private infrastructure,
authorize participant-data use, or log a real experiment. Story #19 owns the
production dataset version and the authorized private-data gate described below;
that gate is intentionally not an acceptance check for Story #14.

## Scope and ownership

| System | Owns | Does not prove |
| --- | --- | --- |
| Public Git | Code, reviewed configuration, the stage registry, generated `dvc.yaml`, the synthetic `dvc.lock`, schemas, and public fixtures | That participant data is approved, available, or reproducible |
| Protected metadata Git | Reviewed private `.dvc` pointers and production `dvc.lock` history | Consent authorization or storage-policy compliance |
| Private DVC remote | Approved raw and derived bytes in content-addressed storage | Dataset validity, experiment validity, or permission to use the bytes |
| Experiment tracker | Later records run parameters, metrics, artifacts, and tracker-neutral Git/DVC metadata | Private media, credentials, or consent decisions |
| Identity vault | Names, contacts, signatures, and identity-to-pseudonym mappings | Research artifacts or model outputs |

The protected metadata Git and private DVC remote rows describe future Story #19
production systems, not infrastructure deployed or verified by Story #14.

The protected metadata repository must be a separate access-controlled repository,
not a branch of this public repository. Remote locations, private pointers, private
lock entries, and provider credentials must not appear in public commits, issues,
pull requests, logs, or reports.

DVC's native hashes identify cache objects and transport state. They do not replace
the dataset manifest's portable paths and SHA-256 identities. A successful pull also
does not establish current consent authorization; those checks remain separate.

## Fixture-only stage scaffold

`STAGE_REGISTRY` in `src/signlab/reproducibility/stages.py` is the source for the
generated root `dvc.yaml`:

```text
ingest -> validate -> extract -> quality -> split -> feature
```

Each current stage reads a synthetic input and writes a deterministic receipt marked
`fixture_only: true` and `implementation: fixture-smoke/1`. The receipts test graph
wiring, dependency invalidation, DVC caching, and transport. They do not ingest real
media, run MediaPipe, assess landmark quality, compute research features, or produce
a leakage-resistant production split.

Those behaviors belong to later stories:

| Boundary | Production owner |
| --- | --- |
| Ingest and validate | Capture, annotation, and dataset-import tooling (#17) |
| Extract | Version-pinned MediaPipe Tasks extraction (#23) |
| Quality | Missing-frame, timing, and landmark-quality policy (#20) |
| Split | Signer-grouped, leakage-resistant split manifests (#25) |
| Feature | Portable landmark representations and golden fixtures (#22) |

Splitting remains before any dataset-fitted feature transform so learned statistics
can be fitted on the training partition only. Later implementations should keep the
registered boundaries and delegate work to importable services instead of creating a
second pipeline graph.

The #17 importable service implements the ingest/validate boundary independently of
the DVC fixture adapter. Its current public entry point accepts only an explicitly
synthetic `collection-sidecar/1`, resolves an ephemeral opaque source map beneath an
explicit source root, and atomically publishes `raw-dataset-manifest/1`. That raw
manifest is the handoff to extraction; it is not a sample-bearing
`dataset-manifest/2`, and it does not add another DVC graph. The registered `ingest`
and `validate` stages remain fixture receipts until their graph adapters are replaced
deliberately alongside the downstream production stages. An authorized real-data
adapter remains blocked on the governance and storage gates described below.

The #23 [landmark extraction service](landmark-extraction.md) likewise implements
the canonical `extract` service boundary independently of the root receipt. It consumes an
exact raw-manifest identity, verifies pinned local MediaPipe task bytes, preserves
source PTS/time-base and orientation facts, and emits semantic plus exact-byte
Parquet evidence. It does not modify `STAGE_REGISTRY`, `dvc.yaml`, or the public
fixture lock, and `dvc repro` still runs `fixture-smoke/1` for `extract`. Wiring a
private stage adapter is a later, reviewed change after the Story #19 authorization
and storage gate; it must call the same importable service rather than create a
parallel graph.

The #20 [landmark quality service](landmark-quality.md) similarly implements the
canonical `quality` boundary without replacing the fixture receipt. It validates the
exact raw and extraction bundles, recomputes every per-sequence and dataset finding
from immutable landmark rows, and atomically publishes a report-only manifest. It
does not rewrite landmark Parquet or modify `STAGE_REGISTRY`, `dvc.yaml`, or
`dvc.lock`; a future protected adapter must delegate to this same service.

Install the reproducibility dependencies and check the public scaffold with:

```shell
uv sync --locked --all-groups
uv run --locked python scripts/generate_dvc_pipeline.py --check
uv run --locked dvc repro --force --no-run-cache
uv run --locked dvc status --json
```

The final status must be `{}`. Public fixture outputs under `data/` and the local DVC
cache remain ignored; only the reviewed synthetic graph and lock belong in public Git.

## Local clean-room proof

`scripts/verify_dvc_clean_room.py` proves only the public fixture workflow. It:

1. requires a clean committed source checkout;
2. clones an isolated producer and forcibly reproduces the fixture graph;
3. pushes the fixture outputs to a temporary local DVC remote;
4. clones an empty consumer and pulls from that temporary remote;
5. compares the producer and consumer output SHA-256 values and clean Git/DVC state;
6. writes a path-free report containing the Git commit, DVC/control identities,
   stage identities, completed phases, and `consent: not_checked`.

CI runs this proof on Linux and Windows and retains a report on both success and
failure. A failure identifies an allowlisted phase without printing subprocess output
or private values. This proof uses a temporary local remote; it does not contact,
configure, or validate the private production remote.

## Private remote configuration

Private storage is an operator-managed prerequisite. The current adapter supports S3
and S3-compatible storage and writes remote metadata only to ignored
`.dvc/config.local`.

Story #14 unit-tests this adapter without contacting a live remote. Story #19 owns
storage provisioning, policy review, and the first live private configuration.

| Variable | Purpose |
| --- | --- |
| `SIGNLAB_DVC_REMOTE_URL` | Required credential-free `s3://` bucket/prefix |
| `SIGNLAB_DVC_ENDPOINT_URL` | Optional HTTPS S3-compatible endpoint; loopback HTTP is accepted only for local tests |
| `SIGNLAB_DVC_REGION` | Optional canonical region |
| Provider credential chain | Short-lived role, workload identity, or standard provider environment credentials used by DVC |

Configure the local checkout with:

```shell
uv run --locked signlab data configure-private-remote
```

The command rejects credentials embedded in the URL, verifies that
`.dvc/config.local` is ignored, and restores the previous local configuration if DVC
configuration fails. It intentionally does not provision the bucket, grant roles, or
verify encryption, audit logging, versioning, retention, deletion, or backup policy.
Those controls must be established and reviewed in the storage environment before
participant data is uploaded.

## Future authorized private-data gate (Story #19)

The public clean-room proof completes the transport requirement for Story #14. The
authorized private-data gate is a separate Story #19 acceptance check that can run
only after the approved production dataset, protected metadata repository, private
remote, storage controls, and authenticated consent integration exist. Its downstream
contract is:

1. Check out the reviewed protected-metadata revision containing the approved DVC
   pointers and production lock.
2. Confirm that storage controls and a least-privilege reader role have been approved.
3. Configure the ignored local remote and authenticate through the provider credential
   chain.
4. Pull the exact reviewed targets and require `dvc status --json` to return `{}`.
5. Validate the dataset manifest and its referenced bytes:

   ```shell
   uv run --locked signlab data validate-dataset DATASET_MANIFEST \
     --workspace-root DATASET_ROOT --verify-row-artifacts
   ```

6. Separately verify the authenticated consent receipt, recording grant, and consent
   event log for the intended purpose and time.
7. Capture a protected metadata snapshot only after the reviewed state is final:

   ```shell
   uv run --locked signlab data capture-reproduction-snapshot \
     --repository-role protected-metadata
   ```

If the protected repository, approved version, remote, role, storage controls, or
consent evidence is unavailable, Story #19's private-data criterion remains
unverified. Do not provision infrastructure or upload participant data merely to
satisfy Story #14, and do not substitute the public fixture or a successful DVC pull
for the later production evidence.

## Tracker-neutral reproduction metadata

The `dvc-snapshot/1` document records a clean metadata Git commit, DVC version,
SHA-256 identities for `uv.lock`, `dvc.yaml`, and `dvc.lock`, and one canonical hash
for each registered lock entry. Public-fixture snapshots identify this repository;
protected snapshots omit their repository locator.

Capture an ignored snapshot with:

```shell
uv run --locked signlab data capture-reproduction-snapshot \
  --repository-role public-fixture
```

`dvc_experiment_metadata()` exposes the commit, DVC lock/snapshot identities, and
per-stage hashes as tracker-neutral strings for a run actually governed by that DVC
snapshot. The minimal Story #27 ledger records the explicit corpus, split, feature,
configuration, code, environment, seed, hardware, metric, and artifact identities
used by the licensed public baseline. It deliberately does not claim that the
fixture-only DVC snapshot governs that separate corpus. A future DVC-backed run must
carry the tracker-neutral DVC mapping in its portable configuration/report before
logging it. Story #14 does not select or integrate the tracker.
