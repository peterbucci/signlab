# Data versioning and reproducible stages

SignLab uses DVC 3.67.1 for content-addressed data transport and deterministic
stage caching. The public repository proves that mechanism with a tiny, explicitly
synthetic fixture. It does **not** publish participant-data pointers, private hashes,
filenames, sizes, or stage history.

## Ownership and trust boundaries

| System | Owns | Must not own |
| --- | --- | --- |
| Public Git | Code, reviewed configuration, the typed stage registry, generated `dvc.yaml`, the synthetic `dvc.lock`, schemas, and public fixtures | Participant bytes, participant DVC pointers, private lock entries, remote locations, or credentials |
| Protected metadata Git | Approved private `.dvc` pointers, production `dvc.lock`, and opaque dataset-version records | Identity-vault data or cloud credentials |
| DVC remote | Encrypted approved raw/derived bytes and the content-addressed cache | Consent decisions, identity mappings, or experiment metrics |
| MLflow | Immutable parameters, aggregate metrics, and sanitized Git/DVC/dataset/split identities | Media, row-level predictions, private paths, bucket URLs, or credentials |
| Future Prefect adapter | Scheduling, retries, and calls to the registered stage services or DVC targets | A second pipeline DAG or duplicated stage logic |
| Identity vault | Names, contacts, signatures, and identity-to-pseudonym mapping | Research artifacts or model outputs |

DVC native MD5, ETag, and provider checksums are cache and transport identities.
They are not SignLab's security boundary. SignLab streams every materialized row
artifact after a pull and verifies its exact size and SHA-256. A successful DVC pull
does not establish current consent authorization.

The protected metadata repository must be a separate access-controlled repository;
a branch in this public repository is not a private boundary. Its directory and
object names must be opaque. Even pseudonyms and content hashes are linkable
research data under the governance policy.

These adapters reject observed symlinks, Windows reparse points, hardlink aliases,
path escapes, and file-identity changes. Cross-platform path APIs cannot make an
untrusted process running as the same operating-system account harmless between
every namespace check and file operation. Run private-data commands only in a
dedicated access-controlled checkout, with no concurrent writer and a trusted local
account; treat unexpected filesystem activity as a failed verification.

## One stage graph

The authoritative registry is `STAGE_REGISTRY` in
`src/signlab/reproducibility/stages.py`. The generated root `dvc.yaml` must match it
byte-for-byte:

```text
ingest -> validate -> extract -> quality -> split -> feature
```

Splitting precedes any future dataset-fitted feature normalization so statistics
can be learned from the training partition only. Each DVC command calls a thin CLI
adapter backed by an importable Python service. A later orchestrator imports the
same registry or invokes DVC targets; it does not redefine these edges.

The current transforms are named `fixture-only/1` and every receipt carries
`fixture_only: true`. They prove command execution, lineage, atomic publication,
cache behavior, and cross-platform determinism. They do not claim that production
ingestion, MediaPipe extraction, quality scoring, feature engineering, or grouped
splitting is implemented; those remain separate reviewed stories.

Regenerate and reproduce the public graph with:

```shell
uv run --locked python scripts/generate_dvc_pipeline.py --check
uv run --locked dvc repro --force --no-run-cache
uv run --locked dvc status --json
```

`--force --no-run-cache` is intentional: an ordinary reproduction may skip a stage
or restore a run-cache result and therefore cannot prove that every command ran.
The public fixture outputs remain ignored under `data/`; DVC's cache remains ignored
under `.dvc/cache/`.

## Clean-room proof

CI runs `scripts/verify_dvc_clean_room.py` on both Linux and Windows. The verifier:

1. clones the exact Git commit without local Git hardlinks;
2. recomputes the producer graph with no DVC run-cache reuse;
3. pushes only synthetic outputs to an isolated temporary local remote;
4. creates a second clone with an empty DVC cache and no outputs;
5. pulls and verifies every synthetic output from that remote;
6. makes the remote unavailable, clears only the temporary consumer cache/outputs,
   and recomputes all six stages offline;
7. requires identical SHA-256 results, unchanged `dvc.yaml` and `dvc.lock`, clean
   DVC status, and clean Git status; and
8. emits a sanitized ephemeral report with no paths, URLs, native DVC hashes,
   participant identifiers, or exception text.

The report says `consent: not_checked`. It is workflow evidence, not
permission to collect, train, publish, or reuse participant data.

## Private remote setup

Use S3 or an S3-compatible private object store with encryption, audit logging,
versioning/retention, and prefix-scoped roles. Reader and publisher roles should be
separate; deletion permission is reserved for authorized maintenance and withdrawal
workflows. DVC remote garbage collection must never be run casually because it can
break reproducibility or conflict with withdrawal retention evidence.

Configure the protected metadata checkout through environment-provided values:

| Variable | Purpose | Persistence |
| --- | --- | --- |
| `SIGNLAB_DVC_REMOTE_URL` | Required credential-free `s3://` bucket/prefix | Written only to ignored `.dvc/config.local` |
| `SIGNLAB_DVC_ENDPOINT_URL` | Optional HTTPS S3-compatible origin | Written only to ignored `.dvc/config.local` |
| `SIGNLAB_DVC_REGION` | Optional canonical region | Written only to ignored `.dvc/config.local` |
| Provider credential chain | Short-lived role, workload identity, or standard AWS environment credentials | Used by later `dvc pull`/`dvc push` commands; never passed as command arguments or written by SignLab |

Then run:

```shell
uv run --locked signlab data configure-private-remote
```

The adapter accepts only a credential-free S3 URL, HTTPS endpoints (or loopback HTTP
for local tests), and a canonical region. It writes only the `private` remote to
`.dvc/config.local`, rejects credential-like or unexpected settings, captures DVC
output, and restores the prior local file if any step fails. Its local-only DVC
configuration subprocess receives a minimal environment with neither provider
credentials nor repository/Python override variables. It never prints the remote
location. The tracked `.dvc/config` disables analytics and update checks, keeps
Studio offline, disables experiment auto-push/autostage, and permits only
`reflink,copy` cache materialization.

## Authorized pull and verification

The production acceptance check occurs only on an authorized machine in the
protected metadata repository:

1. Check out the reviewed Git revision containing the exact private DVC metadata.
2. Assume a least-privilege reader role through the provider credential chain.
3. Configure the ignored local remote and pull the exact approved DVC targets.
4. Run `dvc status --json` and require no missing or changed stage state.
5. Validate the table-backed dataset and all row bytes:

   ```shell
   uv run --locked signlab data validate-dataset DATASET_MANIFEST \
     --workspace-root DATASET_ROOT --verify-row-artifacts
   ```

6. Separately run authenticated consent-receipt, recording-grant, and event-log
   verification for the intended purpose and time. `Referenced row artifacts:
   verified` does not imply `Current consent authorization: verified`.
7. Capture the committed, opaque metadata Git/DVC identities only after the
   lockfile is final:

   ```shell
   uv run --locked signlab data capture-reproduction-snapshot \
     --repository-role protected-metadata
   ```

The public repository cannot automate this gate because it receives neither private
metadata nor credentials. A sanitized attestation may record only an opaque private
version reference, the public Git revision, boolean verification results, reviewer,
and review time. It must omit private hashes, paths, bucket names, participant IDs,
row counts, and remote listings.

## Experiment provenance

`dvc-snapshot/1` is a standalone frozen contract; published `run-record/1` and
`SourceIdentityV1` remain unchanged. A snapshot records:

- an explicit metadata-repository role, the public SignLab locator only for the
  public fixture, an opaque 40-character metadata Git commit, and clean Git/DVC
  assertions;
- exact DVC 3.67.1;
- portable SHA-256 identities for `uv.lock`, `dvc.yaml`, and `dvc.lock` (DVC's
  native Windows newlines normalize to the Git LF representation);
- the exact six-stage inventory; and
- a domain-separated RFC 8785/SHA-256 identity plus native dependency/output cache
  identities for every lock entry.

Capture an ignored JSON snapshot from a clean workspace:

```shell
uv run --locked signlab data capture-reproduction-snapshot \
  --repository-role public-fixture
```

The service observes Git and DVC state before and after reading the control files,
requires the declared role to match the Git origin, requires the effective cache type
to remain `reflink,copy`, and fails if anything changes. A protected snapshot never
stores its repository locator; its Git commit is meaningful only inside the approved
environment. The existing run-record source identity remains responsible for the
public code revision, so a private metadata commit is never mislabeled as a public
SignLab commit. The MLflow projection prepares immutable parameters for the complete
DVC identities and low-cardinality tags only for search; the later MLflow integration
story is responsible for logging that projection on a real run. DVC Experiments and
DVCLive are deliberately not introduced alongside MLflow.

## Failure rules

Stop rather than weaken a check when the protected metadata repository, private
remote, approved data version, retention/deletion policy, least-privilege role,
authenticated current consent, or real production stage is absent. Never use the
synthetic graph, a cache hit, a DVC pull, an unauthenticated hash, or an ignored local
file as substitute evidence.
