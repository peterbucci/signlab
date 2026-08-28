# Landmark quality policy

Story #20 assesses the immutable raw landmark evidence produced by the
[MediaPipe extraction boundary](landmark-extraction.md). It does not rewrite that
evidence, fill its nulls in place, create a feature tensor, choose a model input
representation, or change the public DVC fixture graph. Story #22 owns feature
materialization and may consume the plan defined here.

## Evidence and policy choices

MediaPipe's documented detection, presence, and tracking thresholds configure
whether its tasks consider their internal operations successful. They are not
per-frame quality measurements exposed by the task result. The
[Hand Landmarker result](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HandLandmarkerResult)
contains handedness classifications plus image and world landmarks; it does not
return the internal detection, presence, or tracking scores. Accordingly, SignLab
names the available hand value `handedness_confidence` and never relabels it as
detection confidence. The
[MediaPipe landmark contract](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/components/containers/Landmark)
defines nullable visibility and presence scores, so reports retain separate score
denominators instead of treating an unsupported null as zero.

Research on optical motion capture supports the limited conclusion that simple
interpolation is more defensible for short gaps than long gaps; it does not provide
a transferable SignLab threshold. For example, a
[published gap-recovery comparison](https://pmc.ncbi.nlm.nih.gov/articles/PMC3813748/)
found interpolation accuracy depended strongly on gap length and recording
conditions. Every numeric limit below is therefore a versioned pilot-screening
choice, not a scientific truth. Story #21 must review the observed pilot
distribution before a production policy is frozen.

## Immutable input and honest denominators

Assessment accepts a validated extraction sequence, its exact sequence reference,
and the corresponding raw recording row. It returns new strict report models and
does not mutate any input model or source Parquet bytes.

At this boundary, one extraction sequence is one recording and is the phase's
per-sample quality-report unit. Sample, clip, split, and feature identities do not
exist yet; later stages must preserve the recording lineage when they create them.

Reports distinguish:

- valid frames, source-invalid frames, and task-inference failures;
- expected-hand coverage over every source frame, including invalid frames;
- raw presence counts and all-frame coverage for each of the six pose anchors;
- evaluated and low-score counts for nullable confidence channels;
- gap actions and the longest preserved internal hand gap; and
- elapsed-time timestamp, wrist/palm continuity, and suspected-swap diagnostics.

The handedness-confidence fraction is nullable when no handedness scores were
available. Pose confidence uses explicit evaluated and low-score counts because
visibility and presence may independently be null; an unsupported score is not
silently changed to zero. The version 1 report does not claim valid-frame,
elapsed-time-weighted, any-hand, or two-hand coverage metrics.

Expected hand cardinality comes from the recording's declared performed
handedness: `both` expects two hands; `left`, `right`, and `unknown` expect at
least one. Expected-hand observations are capped at that cardinality for each
frame, so a second incidental detection cannot inflate a one-hand recording's
coverage.

## Gaps and interpolation eligibility

A gap is evaluated independently from the raw `present` mask for `hand_0`,
`hand_1`, and each of the six pose anchors. A present low-confidence observation
is not relabeled as missing. Every report records the boundary kind, source frame
and timestamp bounds, missing count, elapsed duration, decision, barrier flags,
and sorted reasons.

Only an internal gap with observed endpoints can be eligible for coordinate
interpolation. Both the missing-frame count and elapsed bridge duration use
inclusive configured limits. The policy preserves a gap as missing when it is at
the beginning or end, exceeds either limit, contains an invalid frame, crosses a
timestamp discontinuity, crosses a suspected hand swap, or includes an identity or
confidence barrier. Confidence is checked at the observed interpolation bounds;
high-confidence same-slot label conflict is recorded as identity ambiguity.
Leading and trailing values are never extrapolated or
forward-filled.

Coordinates are the only values for which the policy exposes a linear interpolation
helper. Slot identity, detector index, handedness, handedness confidence, visibility,
and presence are never interpolated. The raw masks and uninterpolated values remain
authoritative.

This guard matters with common numerical APIs: the documented default of
[`numpy.interp`](https://numpy.org/doc/stable/reference/generated/numpy.interp.html)
returns endpoint values outside the input range. SignLab instead rejects an
unbracketed target so an edge gap cannot silently become stationary motion.

## Source time and resampling plan

Quality calculations use `relative_timestamp_us`, never frame index or MediaPipe's
collision-avoidance task milliseconds. This follows the source timeline established
by #23: [PyAV documents](https://pyav.org/docs/stable/api/time.html) presentation
time as presentation timestamp multiplied by its rational time base.

The policy builds a compact rational target grid at the configured rate, currently
30 Hz. It begins at zero, advances using integer half-up rounding of each nominal
target, and includes the exact final observed timestamp. When the final timestamp
is off the nominal grid, it is appended instead of shortening or stretching the
sequence. The summary records the target rate, target count, observed span,
declared media duration, trailing unobserved duration, endpoints, and a constant-time
commitment to the uniquely determined complete target grid.

The declared media duration can extend beyond the last decoded presentation
timestamp. That interval is recorded as a `preserve_missing` tail decision, not as
fabricated source frames; the policy does not repeat the last hand position to
reach the container duration. This makes duration accounting visible without
manufacturing motion.

General Fourier resampling is not used. The
[SciPy resampling documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html)
requires equidistant input samples and directs non-constant intervals to
interpolation methods. The
[portable feature boundary](landmark-representations.md) consumes this exact grid
and the approved coordinate gaps; any later filtering or anti-aliasing remains an
explicit preprocessing choice.

## Continuity and suspected swaps

For consecutive available two-hand observations, the diagnostic compares the
existing slot assignment with the crossed alternative using #23's wrist-and-palm
continuity geometry. A crossed assignment that wins by the configured margin,
supported by high-confidence crossed handedness labels, is recorded as a
suspected-swap event. Detector-index reversal by itself is not a swap.

These events are deliberately not called proven identity switches. Standard
tracking evaluation defines an identity switch relative to ground-truth identity,
as explained in the primary
[HOTA metrics paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7881978/). SignLab has
no per-frame ground-truth hand identity at this boundary. Suspected swaps therefore
must retain that qualifier. In the packaged pilot policy they warn or quarantine;
that profile has no suspected-swap reject threshold.

The report contains aggregate suspected-swap and wrist/palm temporal-discontinuity
counts. Gap records retain identity-ambiguity and suspected-swap barriers where
applicable; version 1 does not expose a separate label-flip series or speed series.
MediaPipe's
[handedness documentation](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md#multi_handedness)
also explains why mirror state must remain visible when interpreting labels.

## Run and validate

Use the command help as the authoritative interface reference:

```console
uv run signlab data assess-landmark-quality --help
uv run signlab data validate-landmark-quality --help
```

CLI summaries are aggregate and path-free. Recording-level evidence remains in the
strict quality report artifacts rather than being printed into logs.

## Pilot-screening defaults

The packaged policy is intentionally conservative and fully versioned. Its initial
values are operating hypotheses for the synthetic fixtures and pilot:

| Setting | Pilot value |
| --- | ---: |
| Target cadence | 30 Hz |
| Maximum interpolated internal missing frames | 2 |
| Maximum interpolated bridge | 100 ms |
| Pose visibility and presence qualification | 0.5 each |
| Handedness evidence threshold | 0.8 |
| Timestamp gap | non-increasing, or greater than both 100 ms and 3 times median cadence |
| Crossed-assignment margin | 0.01 |
| Wrist/palm discontinuity speed | 12 normalized-image units per second |

The initial triage bands are likewise configuration values:

| Metric | Warning | Quarantine | Reject |
| --- | ---: | ---: | ---: |
| Invalid-frame fraction | `> 0.01` | `> 0.05` | `> 0.20` |
| Expected-hand all-frame coverage | `< 0.95` | `< 0.80` | `< 0.50` |
| Minimum raw pose-anchor all-frame coverage | `< 0.90` | `< 0.70` | none |
| Low-handedness-confidence fraction | `> 0.10` | `> 0.30` | none |
| Longest preserved internal hand gap | `> 100 ms` | `> 500 ms` | none |
| Timestamp gaps | at least 1 | at least 3 | none |
| Wrist/palm temporal discontinuities | at least 1 | at least 3 | none |
| Suspected hand swaps | at least 1 | at least 3 | none |

The contract requires every valid policy to reject zero-valid-frame sequences and
sequences with no expected hand observations. Under the packaged pilot defaults,
completely absent pose causes quarantine but not rejection solely because pose is
absent; #22 may make an explicit reviewed hand-only representation choice.

Each finding contains the rule and metric identifiers, metric direction, observed
integer value, violated threshold, and severity. Equality with a threshold is not
a violation. A sequence receives the most severe applicable disposition: `pass`,
`warning`, `quarantine`, or `reject`.

Dataset reports use weighted invalid-frame and expected-hand denominators, the
minimum sequence pose coverage, the maximum preserved internal hand gap, summed
transition diagnostics, and complete disposition counts. A quarantine or reject
blocks dataset readiness; reports never delete or hide rejected lineage.
