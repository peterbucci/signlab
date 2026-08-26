# Version-pinned landmark extraction

Story #23 defines the current public reproducible boundary between a validated
synthetic `raw-dataset-manifest/1` and raw hand/body landmark observations. It is an
importable Python service with strict contracts and deterministic Parquet storage;
it is not a quality policy, feature representation, browser application, or second
DVC pipeline.

## Reproducibility lock

The default `mediapipe-extraction-config/1` fixes every SignLab-controlled runtime
choice recorded by the v1 contract:

| Component | Exact identity |
| --- | --- |
| Python task runtime | `mediapipe==1.0.1` |
| Browser task runtime | `@mediapipe/tasks-vision@1.0.1` |
| Video decoder | `av==18.1.0` |
| Delegate and running mode | `CPU`, `VIDEO` |
| Task cardinality | two hands, one pose |
| All hand and pose detection/presence/tracking thresholds | `0.5` |
| Stable-hand tracker | `deterministic_wrist_mcp_centroid_minimum_cost@1.0.0` |
| Tracker numeric pins | spatial cost `0.25`, handedness penalty `0.05`, ambiguity margin `1e-9` |

The six `0.5` values configure vendor detection and tracking. They do not decide
whether an extracted sequence is good enough for research; that decision belongs to
Story #20.

The Python and future browser paths use the same exact MediaPipe task bundles:

| Task | Revision and exact bytes |
| --- | --- |
| Hand Landmarker Full | `hand_landmarker.task`, revision `1.0.0`, 7,819,105 bytes, `sha256:fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |
| Pose Landmarker Lite | `pose_landmarker_lite.task`, revision `1.0.0`, 5,777,746 bytes, `sha256:59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a` |

The packaged model lock records the immutable Google-hosted
[Hand Landmarker bundle](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task?generation=1682480004222387)
and
[Pose Landmarker Lite bundle](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task?generation=1682624736756847),
their model cards, byte sizes, hashes, compatible runtimes, and Apache-2.0
provenance. The task files themselves are external assets: they are not committed,
packaged in the wheel, or downloaded by extraction code. An operator acquires them
before a run, and the runtime reads them into local buffers only after both size and
SHA-256 checks pass.

The upstream [Python Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)
and [Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)
guides document VIDEO mode and timestamped frame calls. The corresponding
[Web Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/web_js)
and [Web Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/web_js)
guides establish that the same `.task` model-asset form is accepted in the browser.
SignLab additionally pins package versions and exact task bytes instead of using a
moving `latest` dependency or model URL.

The exact detector fields follow the official
[HandLandmarkerOptions](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarkerOptions)
and
[PoseLandmarkerOptions](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/PoseLandmarkerOptions)
APIs. MediaPipe
[BaseOptions](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/BaseOptions)
provides the local model-buffer and CPU-delegate boundary used here. Upstream source
and its license remain available in the
[MediaPipe repository](https://github.com/google-ai-edge/mediapipe) and
[Apache-2.0 license](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE).

## Canonical video timeline

The research batch uses PyAV to decode exactly one video stream and preserves each
frame's source presentation timestamp (`PTS`) and rational time base. Frames must
have strictly increasing PTS values and one constant time base. Relative time is
computed from the first decoded PTS with integer arithmetic:

```text
relative_us = floor((pts - first_pts) * time_base_numerator * 1_000_000
                    / time_base_denominator)
```

MediaPipe VIDEO calls require increasing millisecond timestamps. Multiple source
frames can floor to the same millisecond, so SignLab applies the versioned
collision-free recurrence:

```text
task_ms[0] = 0
task_ms[i] = max(floor(relative_us[i] / 1_000), task_ms[i - 1] + 1)
```

The output retains source PTS, time-base numerator and denominator, relative
microseconds, and the task milliseconds for every frame. Missing PTS, changing time
bases, non-increasing PTS, empty video, multiple video streams, malformed task
results, and incompatible package or model versions fail closed. `LIVE_STREAM` is
not used: extraction is a synchronous, frame-complete batch boundary.

## Raw observation contract

Each `landmark-frame/1` row contains exactly two ordered stable identities,
`hand_0` and `hand_1`. A present hand retains its detector index, tracked slot,
reported handedness and score, 21 image-space points, and 21 world-space points.
The deterministic tracker uses wrist and palm geometry plus the pinned handedness
penalty to keep identities stable between adjacent frames; it never chooses only the
first vendor detection.

Each frame also contains these six pose anchors in fixed order:

1. left shoulder;
2. right shoulder;
3. left elbow;
4. right elbow;
5. left wrist;
6. right wrist.

Points retain finite `x`, `y`, and `z` values. Task-provided `visibility` and
`presence` remain nullable because they are not available for every landmark type.
Coordinates are not clipped or reinterpreted.

Absence is data, not an implicit fill operation. An absent hand has `present: false`
and all detection fields set to null. An absent body anchor has null image/world
points. An invalid source or inference frame has a controlled reason, zero hand and
anchor observations, and explicit masks. No previous observation is repeated, no
gap is interpolated, and no quality threshold is applied here. The manifest records
zero-, one-, and two-hand frame counts, invalid-frame counts, total observations,
and ordered presence counts for all six anchors as diagnostics only.

Source `mirror_state` and `rotation_degrees` are copied into each sequence reference.
Extraction does not silently flip, rotate, normalize, center, scale, smooth, or
window coordinates. Story #20 owns missing-frame, timing, and landmark-quality
policy; Story #22 owns portable feature representations and cross-runtime feature
goldens.

## Semantic and physical evidence

`landmark-frames-table/1` is the semantic authority for one source recording. Its
RFC 8785, domain-separated SHA-256 binds ordered validated rows independently of a
specific Parquet encoding. The Parquet writer separately fixes the Arrow schema and
field IDs, metadata allowlist, compression and writer options, row count, exact byte
size, and file SHA-256. A reader verifies all of those facts before reconstructing
and revalidating semantic rows.

For portable Parquet encoding, an absent hand's fixed-size image/world arrays contain
exactly 21 null point elements behind `present: false`; the verified reader restores
the contract-level null arrays and rejects any non-null data hidden behind that mask.
This avoids platform-dependent nullable fixed-size-list behavior without weakening
the explicit absence semantics.

`landmark-extraction-manifest/1` then binds:

- the exact raw dataset ID, version, semantic digest, and raw-manifest digest;
- the complete extraction configuration and its digest;
- both MediaPipe task identities and exact hashes;
- each source recording's input SHA-256, size, mirroring, and rotation;
- each derived Parquet artifact's lineage, semantic row digest, byte digest, size,
  schema version, and observation counts; and
- its own canonical self-digest.

This distinguishes three questions that must not be conflated: whether source data
is authorized, whether landmark rows have the same meaning, and whether a particular
Parquet file contains the same bytes.

## Batch commands

The public command accepts only a fully validated synthetic raw bundle whose every
recording grant includes `derived_features: true`. The model directory must already
contain the two exact `.task` files listed above; the command never downloads them.

```shell
uv run signlab data extract-landmarks path/to/raw-dataset-manifest.json --raw-bundle-root path/to/raw-bundle --model-root path/to/verified-model-assets --output path/to/landmark-bundle
```

Running the same command again validates the existing bundle and reports
`unchanged`. It never patches a partial or conflicting destination. Independently
revalidate the raw source, manifest, exact file inventory, Parquet bytes, semantic
rows, counts, and lineage with:

```shell
uv run signlab data validate-extraction path/to/landmark-bundle/landmark-extraction-manifest.json --workspace-root path/to/landmark-bundle --raw-manifest path/to/raw-dataset-manifest.json --raw-bundle-root path/to/raw-bundle
```

Command output is deliberately aggregate and path-free: status, manifest/config/raw
digests, sequence/frame/invalid-frame counts, and integrity status. Participant,
session, recording, source-path, and vendor error details are never printed.

## Privacy and consent boundary

Google's [MediaPipe API privacy terms](https://developers.google.com/edge/mediapipe/legal/tos)
state that input processing happens on-device and input media is not sent to Google.
They separately describe possible service contacts and performance/utilization
metrics, including an operator responsibility to obtain any legally required user
consent. SignLab therefore does not treat the upstream package alone as a zero-
network guarantee.

For private research extraction, acquire and verify the two model assets first,
then run the batch in a network-isolated environment with approved local source and
output storage. The extraction code has no downloader and initializes MediaPipe from
verified local model buffers. This network isolation is a SignLab operating control,
not a claim about every upstream MediaPipe distribution or platform.

The public repository exercises this boundary only with synthetic fixtures. The
current service rejects every `fixture_only: false` bundle; it has no flag or caller
hook that can bypass that rule. A future protected adapter may enable participant
media only after the approved private importer and an authenticated consent decision
authorize the intended purpose and time. Landmark output is a consent-bound
`derived_features` asset, so the recording grant must set `derived_features: true`;
permission to capture or retain raw video alone is insufficient.
`derived_features_redistribution` is a separate permission and is not implied by
extraction.

Derived Parquet and manifests remain restricted research data, use pseudonymous
lineage, and stay discoverable by the withdrawal graph. None of these contracts
authorize publication, demonstration, redistribution, training, or evaluation.

## DVC boundary

Story #23 does not add or rewrite a DVC graph. The root
`ingest -> validate -> extract -> quality -> split -> feature` registry created in
Story #14 remains a fixture-only receipt scaffold. Its `extract` receipt does not run
MediaPipe. A later deliberate adapter may call this importable extraction service
while preserving the existing stage boundary; the production protected-metadata
and private-remote gate remains owned by Story #19.
