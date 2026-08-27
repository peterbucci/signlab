# Licensed external datasets

SignLab has a separate boundary for public datasets whose use is authorized by a
dataset license. It does not reuse the participant `raw-dataset-manifest/1`: a
public license is not SignLab participant consent, and provider consent records are
not SignLab consent records.

This boundary is deliberately offline. SignLab creates a reviewed download plan,
but it never fetches human video or website previews. An operator downloads the
listed official archives, reviews the governing terms, and explicitly acknowledges
the license before the importer will read local files.

## Registered release

The first source is
[PopSign ASL v1.0](https://signdata.cc.gatech.edu/view/datasets/popsign_v1_0/index.html),
using its official download identity `popsign_v1_0`. The publisher data card reports
200,686 isolated videos, 250 signs, and 47 signers. It also identifies the upper-body
videos as identifiable human data and says the release is suitable for isolated
sign recognition, not continuous recognition or sign-to-English translation.

The dataset is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). SignLab's registered
attribution is:

> PopSign ASL v1.0, Georgia Institute of Technology and Deaf Professional Arts
> Network, licensed under CC BY 4.0.

Reuse must preserve attribution, link the license, and indicate changes. A future
license or publisher change requires a new reviewed source-resource version; the
published v1 resource cannot be edited in place.

The
[official download guide](https://signdata.cc.gatech.edu/view/guides/downloading_popsign/index.html)
defines per-sign tar archives under `game` or `non-game` and `train`, `val`, or
`test`. It warns that the website's transformed preview videos are not appropriate
dataset inputs. The registered plan therefore points only to official per-sign tar
archives and forbids preview media.

## Reviewed five-target selection

The first technical corpus uses only the publisher's `game` category. Source labels
remain intact and are mapped explicitly into the narrower SignLab taxonomy:

| PopSign source label | SignLab target |
| --- | --- |
| `hello` | `hello` |
| `no` | `no` |
| `please` | `please` |
| `thankyou` | `thank_you` |
| `yes` | `yes` |

These are reviewed gloss alignments for five predefined gesture targets. They do
not assert linguistic equivalence, dialect coverage, translation ability, or a
general named-language capability. No public source label is silently converted to
SignLab's learned `other` class.

## Offline workflow

Create the deterministic 15-archive plan:

```shell
uv run signlab data plan-external-dataset popsign-asl-v1 --output popsign-plan.json
```

Read the plan, the source data card, and the CC BY 4.0 terms. Download each listed
archive with a tool of your choice and save it at its plan-relative path beneath a
local archive root. SignLab itself performs no network request.

Import already-downloaded archives only after explicit license acknowledgement:

```shell
uv run signlab data import-popsign popsign-plan.json \
  --archive-root LOCAL_ARCHIVES \
  --output data/raw/external/popsign-v1 \
  --accept-license CC-BY-4.0
```

Validate copied media offline. Supplying the archive root also re-hashes every
original archive; omitting it validates the frozen archive records and every copied
media byte without overclaiming that the external archive files were re-read.

```shell
uv run signlab data validate-external-dataset \
  data/raw/external/popsign-v1/external-dataset-manifest.json \
  --workspace-root data/raw/external/popsign-v1 \
  --archive-root LOCAL_ARCHIVES
```

CLI output contains only aggregate counts, status, and content identities. It does
not print local paths, upstream filenames, timestamp tokens, or signer identifiers.

## Import and security guarantees

The importer treats every tar as hostile input. It streams regular MP4 members and
rejects traversal, absolute or backslash paths, links, devices, FIFOs, unexpected
files, duplicate or case-colliding names, malformed publisher filenames, and
configured count or size limits. It never calls a bulk extraction API.

Archive and member bytes receive exact SHA-256 identities. The publisher does not
provide archive checksums for this endpoint, so the first local hash is explicitly
recorded as trust-on-first-use and then frozen. Upstream signer and recording tokens
are used only in memory to derive source-namespaced opaque IDs. The public manifest
does not retain publisher filenames or recording timestamps. A signer appearing in
more than one official split fails closed.

Copied media is stored at content-addressed, workspace-relative paths. Publication
uses a sibling staging directory, writes the manifest last, and atomically renames a
complete bundle. Repeating the same import is unchanged; an occupied destination
with different content is a conflict.

## What this story does not do

`external-dataset-manifest/1` is an authorization and raw-media boundary. It does
not create landmarks, quality decisions, portable features, trainable samples,
split overrides, DVC production pointers, or model results. Story #74 bridges a
reviewed external manifest into the extraction-ready public-data corpus after Story
#22 defines portable representations.

Results from this path support only a licensed-public-data technical claim. They do
not prove that SignLab ran its own participant study, validated natural continuous
sessions, or collected an original production dataset.
