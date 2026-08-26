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
