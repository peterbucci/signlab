# Browser release security boundary

Status: release contract for Story #51; provider application and live verification belong to
Story #57. The machine-readable header values are
[`contracts/web-release-headers.v1.json`](contracts/web-release-headers.v1.json).

## Claim and scope

The production demo is a static browser application. Camera frames, derived landmarks,
decisions, and optional feedback stay in the browser unless the user explicitly downloads a
feedback package. The application has no account, application server, analytics, telemetry,
upload endpoint, advertising, remote font, beacon, or WebSocket.

This claim covers SignLab's application code and configured release assets. It does not claim
that an operating system, browser, extension, network, hosting provider, or compromised release
origin is trustworthy. It is a privacy boundary, not a claim that gesture predictions are safe
for consequential use.

## Data and trust boundaries

| Item | Path and lifetime | Release boundary |
| --- | --- | --- |
| Camera frames | Camera to video element to landmark worker | User starts the camera; raw frames are not recorded, persisted, or uploaded. |
| Frame bitmaps | Main thread to landmark worker | At most one in flight and the newest waiting frame are retained; replaced and processed images are closed. |
| Landmarks | Landmark worker to event detector | Kept only for the bounded event unless the user separately opts to save that event locally. |
| Decisions | ONNX worker to the page | The latest event result is displayed locally; it is not transmitted. |
| Feedback | Explicit per-event save to IndexedDB | Raw video and free text are excluded. Landmarks require a separate opt-in. Records remain local until deleted by the user or site storage is cleared. |
| Feedback package | Explicit second consent to a Blob download | Maximum serialized size is 16 MiB. Download does not authorize upload, research use, or training. |
| Model bundle | Same-origin manifest and assets to Cache Storage and workers | Every declared asset has an exact size and SHA-256 check before activation; cached and rollback bytes are reverified. |
| MediaPipe tasks | Same-origin release files to verified buffers | Exact expected size and SHA-256 are checked before worker initialization. |
| Runtime code | Release origin to the browser | React, MediaPipe JavaScript/WASM, and ONNX Runtime JavaScript/WASM are bundled release assets. |

The release trust anchors are TLS and control of the release origin, the reviewed source and
locked dependencies used to build it, the approved model-manifest digests supplied to the
release build, and the committed task-asset hashes. Hash checks detect bytes that differ from
those anchors; they do not make a maliciously approved artifact safe.

## Production network allowlist

The production browser may make only `GET` requests to its own origin for:

- the HTML document and generated JavaScript, CSS, and WASM assets;
- the configured model-bundle manifest and its declared assets; and
- the two pinned MediaPipe task files staged by Story #57.

The release build accepts `VITE_SIGNLAB_MODEL_BUNDLE_URL` only on the page origin and requires
its full manifest digest in the comma-separated
`VITE_SIGNLAB_TRUSTED_MODEL_MANIFEST_SHA256` allowlist. The task files resolve relative to the
deployed document at `models/mediapipe/hand_landmarker.task` and
`models/mediapipe/pose_landmarker_lite.task`, including subpath deployments.

Camera capture, inference, IndexedDB feedback, Cache Storage, and Blob downloads do not require
network requests. A release must not make `POST`, `PUT`, beacon, WebSocket, analytics, telemetry,
font, or tracker requests. The production CSP therefore uses `connect-src 'self'`.

Development may fetch the two exact digest-pinned task files from
`https://storage.googleapis.com/mediapipe-models/`. That fallback exposes ordinary request
metadata such as an IP address to Google and is not part of the production privacy claim. A
build that still needs that fallback is not release-ready. Story #57 owns copying the reviewed
bytes to the release origin, configuring their paths, applying the header contract, and checking
the deployed site.

## Required response headers

The provider-neutral JSON contract is authoritative. Its CSP denies everything by default,
allows only same-origin connections, scripts, styles, and workers, and permits the narrow
`'wasm-unsafe-eval'` source needed for WebAssembly. It does not allow wildcards, inline scripts,
inline styles, or the general `'unsafe-eval'` source. `frame-ancestors 'none'` and
`X-Frame-Options: DENY` prevent framing. `Permissions-Policy` limits camera access to this
origin and denies microphone and geolocation. `Referrer-Policy: no-referrer`,
`X-Content-Type-Options: nosniff`, and `Cross-Origin-Resource-Policy: same-origin` reduce
metadata leakage, MIME confusion, and cross-origin reuse of release responses.

The ONNX runtime deliberately imports the WASM build, selects only the WASM execution provider,
and sets `numThreads=1`. Cross-origin isolation is therefore not required. The contract
intentionally omits COOP and COEP; enabling WASM threads later requires a separate measured
decision and an asset-compatibility review.

## Existing limits and failure behavior

- Only one landmark frame is processed while at most one newer frame waits.
- Candidate events stop at the configured four-second maximum and become a fixed `64 x 126`
  tensor before inference.
- Model manifests are capped at 64 KiB and declared bundle assets at 8 MiB total. Network reads
  stop at those limits before parsing, hashing, or persistence.
- Bundle and task bytes fail closed on type, structure, size, digest, or compatibility mismatch.
- Partial cache writes never become the active bundle; activation and rollback reverify bytes.
- Feedback export fails above 16 MiB and never deletes or changes the saved local records.
- Camera tracks, workers, frame handles, and model resources are released when the session stops.

## Threats and controls

| Threat | Control | Remaining exposure |
| --- | --- | --- |
| Accidental camera use | Permission is requested only after the user starts the demo; tracks stop on stop, navigation, hiding, or interruption. | Browser permission indicators and controls remain browser-owned. |
| Frame or landmark exfiltration | Same-origin CSP, explicit request allowlist, no upload/telemetry code, and browser tests for forbidden request methods/channels. | A compromised origin, dependency, browser, or extension could bypass application intent. |
| Tampered model/task bytes | Exact byte length and SHA-256 verification before use; strict manifest and runtime contracts. | A compromised reviewed hash or release build remains trusted. |
| Unbounded frame or asset pressure | One in-flight plus one newest-waiting bitmap; bounded stream readers stop manifests and bundle assets at their reviewed limits. | The browser and network stack may transiently buffer transport data outside application control. |
| Feedback used without consent | Per-event local consent, separate landmark opt-in, separate export consent, and non-trainable quarantine on Python import. | IndexedDB is not encrypted and has no application retention timer; users or same-origin code can access local records. |
| UI embedded for deceptive permission prompts | CSP `frame-ancestors 'none'` plus `X-Frame-Options: DENY`. | Top-level phishing on another origin is outside this application. |
| Unnecessary browser capabilities | Camera is self-only; microphone and geolocation are denied; no service worker or backend exists. | Hosting-provider logs still contain normal HTTP request metadata. |

## Release check

Story #57 must translate the JSON values without weakening them, serve all runtime and model
assets from the release origin with correct MIME types, and inspect the deployed page for
unexpected requests. Any required third-party origin, upload path, analytics, WebGPU backend,
WASM threading, service worker, user account, or server API is a new design decision—not an
implicit exception to this contract.
