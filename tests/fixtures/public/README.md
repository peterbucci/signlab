# Public test fixtures

Only tiny synthetic, explicitly consented, or separately licensed fixtures may be
committed here. Every non-text fixture must include provenance and license notes in
the story that adds it. The repository guard limits each tracked file to 1 MiB; real
participant media, extracted features, and trained weights remain outside Git.

`collection/` contains an identity-free, no-camera rehearsal for the draft
[collection and annotation protocol](../../../docs/collection-protocol.md). Its
session plan is evidence for the protocol, not a production sidecar contract, and
its annotation table uses only invented observations and identifiers. The fixture
does not represent consent, collection approval, or participant data.

`ingest/` contains one project-authored 73-byte payload whose contents explicitly
state that it is not video and was not produced by a person or camera, plus an
opaque relative source map. The `.webm` suffix exercises media-path handling only;
the importer does not claim to probe its codec. These MIT-licensed synthetic bytes
drive the cross-platform raw-import golden test and do not represent consent,
collection approval, or usable media.

`extraction/` contains invented detector results, timestamps, and coordinate seeds.
It is the authoritative input to the scripted batch decoder and inference fake,
including explicit source-frame and task-inference failures. It exercises
deterministic hand association, absence and invalid masks, replay, and output
serialization without shipping a model or representing a person. The scripted
backend exists only in tests and is not selectable from the SignLab command line.

`features/` contains MIT-licensed expected hashes and aggregate counts for nine
project-authored synthetic landmark scenarios. The inputs are invented numeric hand
and pose shapes defined by the test builders; they were not captured from a person,
camera, public corpus, or participant. Alongside exact canonical hashes, the corpus
stores complete expected quantized value arrays so another runtime can apply the
declared one-quantum tolerance. The golden covers all three representations, one and
two hands, mirrored equivalence, an approved gap, a suspected-swap barrier, missing
pose, irregular timing, optional geometry and derivatives, and padding.

`events/` contains an MIT-licensed stream of invented timestamps, hand masks, quality
flags, and quantized motion. It exercises candidate-event state transitions without
including camera, landmark, public-corpus, or participant-derived values. Its metrics
are conformance evidence only, not natural-use performance.

`feedback/` contains one MIT-licensed package of invented records emitted by the real
browser serializer. Browser and Python tests share its exact bytes to prove local
download compatibility without including a person, camera, or participant data.

`replay/` contains MIT-licensed invented truth intervals and final decisions for one
mixed and one negative-only session. It proves scoring counts and grouping mechanics;
its rates and timing values are not measurements of people, models, or runtimes. It
also contains one compact post-landmark motion plan that reuses the public synthetic
parity template to exercise the live browser pipeline without a person or camera.

`parity/` contains one MIT-licensed, no-person candidate-runtime golden and a tiny
deterministic ONNX probe. The JSON shares raw synthetic frames, compact expected
features, masks, decisions, and exact resource identities with Python and TypeScript.
The ONNX bytes test runtime mechanics only; they are not a trained candidate model.
