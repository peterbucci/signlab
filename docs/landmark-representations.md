# Portable landmark representations

Story #22 turns immutable landmark observations into fixed-shape model inputs. It
does not change extraction, decide which recordings belong in a dataset, or run a
public corpus. Those later responsibilities remain with the dataset bridge and
experiment stories.

## Boundary

The feature stage consumes five exact inputs:

1. one `landmark-frames-table/1`;
2. its `landmark-sequence-reference/1` lineage;
3. the matching `sequence-quality-report/1`;
4. one `landmark-feature-plan/1`; and
5. the extraction configuration SHA-256 from the validated
   `landmark-extraction-manifest/1` that contains that sequence reference.

It emits `portable-feature-sequence/1`. The output binds the source media,
semantic landmark rows, extraction configuration, quality policy and report,
feature plan, and optional fitted statistics by SHA-256. A different decision at
any of those boundaries therefore produces a different cache identity.

The published `preprocessing-plan/1` contract is unchanged. A later experiment
story will register the feature operation and bind the resolved landmark-feature-
plan digest into experiment plans. Story #22 does not reinterpret, migrate, or wire
that existing contract.

## Three selectable representations

### Hand-local shape

For each stable `hand_0` and `hand_1` slot, the transform:

- reads the 21 MediaPipe **hand-world** points in their registered order;
- undoes a source mirror on the world x-axis exactly once;
- subtracts wrist landmark 0;
- divides by the 3-D wrist-to-middle-MCP (landmark 9) distance; and
- corrects MediaPipe's vendor-reported handedness using the recorded source mirror
  state, then reflects the x-axis when that corrected label is `left`.

This removes camera translation, uniform hand scale, and the orientation encoded by
the corrected handedness label while retaining the full 21-by-3 local shape.
[MediaPipe defines handedness under a mirrored/selfie-input assumption](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md#multi_handedness),
so a `not_mirrored` source swaps `left` and `right`, while a `mirrored` source keeps
the reported label. This correction still does not make the vendor classification
independently verified anatomical truth. A missing hand or a zero normalization
distance yields zero storage with false masks; it never yields NaN.

### Body-relative wrist and palm trajectory

This representation deliberately uses **image x/y** for both the hand and pose
points. MediaPipe hand-world and pose-world coordinates have different origins and
must not be subtracted from each other.

The shoulder midpoint is the origin and the 2-D shoulder width is the scale. Each
hand contributes a wrist point and a palm centroid over landmarks 0, 5, 9, and 17.
A mirrored source is restored with `x = 1 - x` exactly once.

If either shoulder is unavailable or shoulder width is zero, body-relative
channels keep their fixed positions but are masked. Hand-local channels in a
combined representation remain valid. No prior pose is copied forward.

### Combined

Combined features are exactly the registered hand-local channels followed by the
registered body-relative channels. Stable extraction slots remain stable. Detector
order and suspected-swap diagnostics never cause the feature stage to reorder or
"repair" hands.

## Optional channels

A plan may append a small registered set of features without changing raw
extraction:

- five finger flexion angles per hand;
- five wrist-to-fingertip distances per hand;
- backward elapsed-time velocity for position channels; and
- backward elapsed-time acceleration for position channels.

Acceleration requires velocity. Derivatives use the actual microsecond duration
between target timestamps and are masked whenever a required adjacent value is
missing. They never cross a gap or padding boundary. The registered order is
resample, derive velocities and accelerations on the full grid, select long-sequence
frames, then pad; selection never changes the derivative interval.

## Timing, gaps, and fixed shape

The target grid is the exact 30 Hz elapsed-time grid committed by the quality
report, including an off-grid final observed endpoint. Coordinates between ordinary
observed frames may be linearly resampled. A missing interval is filled only when
the quality report explicitly marks that exact signal gap
`interpolate_linear`. Leading, trailing, long, invalid, discontinuous, low-
confidence, or identity-ambiguous gaps stay missing.

Long grids use deterministic uniform index selection that preserves both
endpoints. Short grids receive neutral right padding. Padding continues nominal
timestamps so time stays strictly increasing, but every feature and source-
availability mask is false.

Each output separately records:

- `valid_mask`: a numeric value is usable;
- `observed_mask`: the value came from an exact source observation;
- `interpolated_mask`: the coordinate was linearly resampled;
- `hand_present_mask`: a sampled hand value is available for each stable slot after
  approved resampling, independent of whether every derived value was usable;
- `body_available_mask`: both normalization shoulders were usable; and
- `padding_mask`: the row is synthetic shape padding.

Masked values are always integer zero. Observed and interpolated masks cannot both
be true.

## Portable numbers and hashes

Feature values are quantized at a scale of 1,000,000 with round-half-away-from-zero
and serialized as safe JSON integers. Consumers recover the represented number by
dividing by that scale. This removes differences in float-to-text formatting from
canonical identities while keeping tolerances much tighter than landmark noise.

Every self-digested feature document uses domain-separated RFC 8785 canonical JSON
and SHA-256. Paths, drive letters, mtimes, and cache location are not part of the
identity.

## Training-only statistics

`feature-statistics/1` fits masked z-score statistics only from inputs carrying
explicit `train` membership. The fitter requires a caller-supplied, validated
`SplitManifestV1` and the exact validated `LandmarkFeaturePlanV1`, whose learned-
statistics mode must explicitly enable train-only masked z-score fitting. For every
training wrapper, it verifies that the recorded split-manifest digest identifies
that exact manifest and that the source recording belongs to its train partition.
Validation or test membership is rejected rather than silently included. Inputs
must bind that plan's digest, channel order, and quantization and must not already be
standardized. Applying statistics repeats the same plan checks.

Invalid and padded cells do not contribute to counts. A channel with no training
observations uses mean zero and scale one; a constant channel uses scale one. The
statistics artifact records the verified split-manifest digest and sorted
training-sequence identities, and standardized outputs bind its digest.

Story #25 now freezes the authoritative PopSign smoke train/validation/test membership.
The participant-backed `SplitManifestV1` remains a separate governed contract. Story
#22 verifies caller-supplied split evidence; it does not invent or revise membership.

## Cache behavior

`feature-cache-key/1` includes:

- source recording and media digest;
- semantic landmark digest;
- extraction configuration digest;
- quality policy and report digests;
- landmark feature plan digest; and
- fitted-statistics digest, when used.

Cache files live beneath a directory derived only from that digest. Publication is
atomic and idempotent. Loading validates both the feature contract and every key
binding; missing, corrupt, symlinked, or mismatched content fails closed.

## Story #74 handoff

This story supplies a pure, importable feature service and synthetic portable
goldens. Story #74 will run that service over licensed PopSign inputs, publish the
resulting corpus and reports, and add the production DVC boundary. No public media,
participant media, DVC lock, split assignment, training run, ONNX artifact, or UI is
created here.
