import {
  BODY_ANCHOR_NAMES,
  HAND_SLOT_IDS,
  type LandmarkFrameMessage,
  type LandmarkPoint,
  type ReportedHandedness,
} from "../landmarks/protocol";
import candidateFeaturePlan from "../../../../src/signlab/resources/features/config/hand-local-64-1.default.json";

const FRAME_COUNT = 64;
const HAND_WIDTH = 63;
const FEATURE_WIDTH = 126;
const SCALE = 1_000_000;
const SHAPE = [1, FRAME_COUNT, FEATURE_WIDTH] as const;
const GAP_SIGNALS = [...HAND_SLOT_IDS, "left_shoulder", "right_shoulder"] as const;

export type SourceMirrorState = "mirrored" | "not_mirrored";
export type CandidateGapSignal = (typeof GAP_SIGNALS)[number];
export type CandidateFrame = Pick<
  LandmarkFrameMessage,
  "relativeTimestampUs" | "valid" | "hands" | "bodyAnchors"
>;

export interface CandidateGapEvidence {
  readonly signal: CandidateGapSignal;
  readonly decision: "interpolate_linear";
  readonly leftObservedFrameIndex: number;
  readonly leftObservedTimestampUs: number;
  readonly rightObservedFrameIndex: number;
  readonly rightObservedTimestampUs: number;
}

export interface CandidateQualityEvidence {
  readonly timestampDiscontinuityCount: number;
  readonly gaps: readonly CandidateGapEvidence[];
}

type Point = readonly [number, number, number];
type HandValue = { readonly world: readonly Point[]; readonly handedness: ReportedHandedness };
type Sample<T> = { readonly value: T; readonly evidence: "observed" | "interpolated" };

function deepEqual(value: unknown, expected: unknown): boolean {
  if (Array.isArray(expected)) {
    return (
      Array.isArray(value) &&
      value.length === expected.length &&
      expected.every((item, index) => deepEqual(value[index], item))
    );
  }
  if (typeof expected === "object" && expected !== null) {
    if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
    const actual = value as Record<string, unknown>;
    const entries = Object.entries(expected);
    return (
      Object.keys(actual).length === entries.length &&
      entries.every(([key, item]) => deepEqual(actual[key], item))
    );
  }
  return value === expected;
}

function requireCandidatePlan(plan: unknown): void {
  if (!deepEqual(plan, candidateFeaturePlan)) {
    throw new Error("candidate.preprocessing.unsupported_feature_plan");
  }
}

const safeTimestamp = (value: number) => Number.isSafeInteger(value) && value >= 0;

function xyz(point: LandmarkPoint | null): Point {
  if (point === null || ![point.x, point.y, point.z].every(Number.isFinite)) {
    throw new Error("candidate.preprocessing.invalid_landmarks");
  }
  return [point.x, point.y, point.z];
}

function handAt(frame: CandidateFrame, slotIndex: number): HandValue | null {
  const slot = HAND_SLOT_IDS[slotIndex];
  const hand = frame.hands[slotIndex];
  if (slot === undefined || hand?.slotId !== slot) {
    throw new Error("candidate.preprocessing.invalid_hand_slots");
  }
  if (!frame.valid || !hand.present) return null;
  if (
    (hand.handedness !== "left" && hand.handedness !== "right") ||
    hand.handednessConfidence === null ||
    !Number.isFinite(hand.handednessConfidence) ||
    hand.handednessConfidence < 0 ||
    hand.handednessConfidence > 1 ||
    hand.worldLandmarks?.length !== 21
  ) {
    throw new Error("candidate.preprocessing.invalid_hand");
  }
  return { world: hand.worldLandmarks.map(xyz), handedness: hand.handedness };
}

function anchorAt(frame: CandidateFrame, name: "left_shoulder" | "right_shoulder"): Point | null {
  const index = BODY_ANCHOR_NAMES.indexOf(name);
  const anchor = frame.bodyAnchors[index];
  if (anchor?.name !== name) throw new Error("candidate.preprocessing.invalid_body_anchors");
  return !frame.valid || !anchor.present ? null : xyz(anchor.imagePoint);
}

function signalAt(frame: CandidateFrame, signal: CandidateGapSignal): HandValue | Point | null {
  if (signal === "hand_0") return handAt(frame, 0);
  if (signal === "hand_1") return handAt(frame, 1);
  return anchorAt(frame, signal);
}

function validateInputs(
  frames: readonly CandidateFrame[],
  quality: CandidateQualityEvidence,
): void {
  if (
    frames.length === 0 ||
    frames[0]?.relativeTimestampUs !== 0 ||
    quality.timestampDiscontinuityCount !== 0
  ) {
    throw new Error("candidate.preprocessing.invalid_timeline");
  }
  frames.forEach((frame, index) => {
    if (
      !safeTimestamp(frame.relativeTimestampUs) ||
      (index > 0 &&
        (frames[index - 1]?.relativeTimestampUs ?? Infinity) >= frame.relativeTimestampUs) ||
      !HAND_SLOT_IDS.every((slot, slotIndex) => frame.hands[slotIndex]?.slotId === slot) ||
      !BODY_ANCHOR_NAMES.every((name, anchorIndex) => frame.bodyAnchors[anchorIndex]?.name === name)
    ) {
      throw new Error("candidate.preprocessing.invalid_timeline");
    }
    HAND_SLOT_IDS.forEach((_, slotIndex) => void handAt(frame, slotIndex));
  });
  quality.gaps.forEach((gap) => {
    const interior = frames.slice(gap.leftObservedFrameIndex + 1, gap.rightObservedFrameIndex);
    if (
      !GAP_SIGNALS.includes(gap.signal) ||
      gap.decision !== "interpolate_linear" ||
      ![
        gap.leftObservedFrameIndex,
        gap.rightObservedFrameIndex,
        gap.leftObservedTimestampUs,
        gap.rightObservedTimestampUs,
      ].every(safeTimestamp) ||
      gap.rightObservedFrameIndex <= gap.leftObservedFrameIndex + 1 ||
      frames[gap.leftObservedFrameIndex]?.relativeTimestampUs !== gap.leftObservedTimestampUs ||
      frames[gap.rightObservedFrameIndex]?.relativeTimestampUs !== gap.rightObservedTimestampUs ||
      signalAt(frames[gap.leftObservedFrameIndex]!, gap.signal) === null ||
      signalAt(frames[gap.rightObservedFrameIndex]!, gap.signal) === null ||
      interior.some((frame) => signalAt(frame, gap.signal) !== null)
    ) {
      throw new Error("candidate.preprocessing.invalid_gap_evidence");
    }
  });
}

function roundHalfUp(numerator: number, denominator: number): number {
  return Math.floor((2 * numerator + denominator) / (2 * denominator));
}

function elapsedGrid(finalUs: number): number[] {
  const maximumSpan = roundHalfUp((1_000_000 - 1) * SCALE, 30);
  if (finalUs > maximumSpan) throw new Error("candidate.preprocessing.timeline_too_long");
  const output: number[] = [];
  for (let index = 0; ; index += 1) {
    const timestamp = roundHalfUp(index * SCALE, 30);
    if (timestamp > finalUs) break;
    output.push(timestamp);
  }
  if (output.at(-1) !== finalUs) output.push(finalUs);
  return output;
}

function lowerBound(frames: readonly CandidateFrame[], targetUs: number): number {
  let low = 0;
  let high = frames.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if ((frames[middle]?.relativeTimestampUs ?? Infinity) < targetUs) low = middle + 1;
    else high = middle;
  }
  return low;
}

function sample<T>(
  frames: readonly CandidateFrame[],
  quality: CandidateQualityEvidence,
  signal: CandidateGapSignal,
  targetUs: number,
  read: (frame: CandidateFrame) => T | null,
  blend: (left: T, right: T, ratio: number) => T | null,
): Sample<T> | null {
  const position = lowerBound(frames, targetUs);
  const exact = frames[position];
  if (exact?.relativeTimestampUs === targetUs) {
    const value = read(exact);
    if (value !== null) return { value, evidence: "observed" };
  }
  const gap = quality.gaps.find((item) => {
    return (
      item.signal === signal &&
      item.leftObservedTimestampUs <= targetUs &&
      targetUs <= item.rightObservedTimestampUs
    );
  });
  const leftIndex = gap?.leftObservedFrameIndex ?? position - 1;
  const rightIndex = gap?.rightObservedFrameIndex ?? position;
  const leftFrame = frames[leftIndex];
  const rightFrame = frames[rightIndex];
  if (leftFrame === undefined || rightFrame === undefined) return null;
  const left = read(leftFrame);
  const right = read(rightFrame);
  const elapsed = rightFrame.relativeTimestampUs - leftFrame.relativeTimestampUs;
  if (
    left === null ||
    right === null ||
    elapsed <= 0 ||
    targetUs < leftFrame.relativeTimestampUs ||
    targetUs > rightFrame.relativeTimestampUs
  )
    return null;
  const value = blend(left, right, (targetUs - leftFrame.relativeTimestampUs) / elapsed);
  return value === null ? null : { value, evidence: "interpolated" };
}

function interpolatePoint(left: Point, right: Point, ratio: number): Point {
  return [0, 1, 2].map(
    (axis) => left[axis]! + (right[axis]! - left[axis]!) * ratio,
  ) as unknown as Point;
}

function sampleHand(
  frames: readonly CandidateFrame[],
  quality: CandidateQualityEvidence,
  slotIndex: number,
  targetUs: number,
): Sample<HandValue> | null {
  return sample(
    frames,
    quality,
    HAND_SLOT_IDS[slotIndex] ?? "hand_0",
    targetUs,
    (frame) => handAt(frame, slotIndex),
    (left, right, ratio) =>
      left.handedness !== right.handedness
        ? null
        : {
            handedness: left.handedness,
            world: left.world.map((point, index) =>
              interpolatePoint(point, right.world[index]!, ratio),
            ),
          },
  );
}

function localPoints(hand: HandValue, mirrored: boolean): readonly Point[] | null {
  const points = hand.world.map(([x, y, z]) => [mirrored ? -x : x, y, z] as const);
  const wrist = points[0];
  const middleMcp = points[9];
  if (wrist === undefined || middleMcp === undefined) return null;
  const scale = Math.hypot(
    middleMcp[0] - wrist[0],
    middleMcp[1] - wrist[1],
    middleMcp[2] - wrist[2],
  );
  if (!Number.isFinite(scale) || scale <= 0) return null;
  const corrected = mirrored ? hand.handedness : hand.handedness === "left" ? "right" : "left";
  const sign = corrected === "left" ? -1 : 1;
  return points.map(([x, y, z]) => [
    (sign * (x - wrist[0])) / scale,
    (y - wrist[1]) / scale,
    (z - wrist[2]) / scale,
  ]);
}

function modelValue(value: number): number {
  const scaled = Math.abs(value) * SCALE;
  if (!Number.isFinite(scaled) || scaled > Number.MAX_SAFE_INTEGER) {
    throw new Error("candidate.preprocessing.invalid_feature_value");
  }
  const integer = Math.floor(scaled + 0.5) * (value < 0 ? -1 : 1);
  return Math.fround(integer === 0 ? 0 : integer / SCALE);
}

function selectionIndices(count: number): number[] {
  if (count <= FRAME_COUNT) return Array.from({ length: count }, (_, index) => index);
  return Array.from({ length: FRAME_COUNT }, (_, index) =>
    Math.floor((2 * index * (count - 1) + FRAME_COUNT - 1) / (2 * (FRAME_COUNT - 1))),
  );
}

export function preprocessCandidate(
  frames: readonly CandidateFrame[],
  sourceMirrorState: SourceMirrorState,
  quality: CandidateQualityEvidence,
  featurePlan: unknown,
) {
  requireCandidatePlan(featurePlan);
  if (sourceMirrorState !== "mirrored" && sourceMirrorState !== "not_mirrored") {
    throw new Error("candidate.preprocessing.invalid_mirror_state");
  }
  validateInputs(frames, quality);
  const sourceGrid = elapsedGrid(frames.at(-1)!.relativeTimestampUs);
  const selected = selectionIndices(sourceGrid.length);
  const values = new Float32Array(FRAME_COUNT * FEATURE_WIDTH);
  const validMask = new Uint8Array(values.length);
  const observedMask = new Uint8Array(values.length);
  const interpolatedMask = new Uint8Array(values.length);
  const handPresentMask = new Uint8Array(FRAME_COUNT * 2);
  const bodyAvailableMask = new Uint8Array(FRAME_COUNT);
  const paddingMask = new Uint8Array(FRAME_COUNT);
  const timestampsUs = selected.map((index) => sourceGrid[index]!);
  const mirrored = sourceMirrorState === "mirrored";

  selected.forEach((sourceIndex, outputIndex) => {
    const targetUs = sourceGrid[sourceIndex]!;
    HAND_SLOT_IDS.forEach((_, slotIndex) => {
      const sampled = sampleHand(frames, quality, slotIndex, targetUs);
      if (sampled === null) return;
      handPresentMask[outputIndex * 2 + slotIndex] = 1;
      localPoints(sampled.value, mirrored)?.forEach((coordinates, landmarkIndex) =>
        coordinates.forEach((value, axisIndex) => {
          const index =
            outputIndex * FEATURE_WIDTH + slotIndex * HAND_WIDTH + landmarkIndex * 3 + axisIndex;
          values[index] = modelValue(value);
          validMask[index] = 1;
          if (sampled.evidence === "observed") observedMask[index] = 1;
          else interpolatedMask[index] = 1;
        }),
      );
    });
    const shoulder = (name: "left_shoulder" | "right_shoulder") =>
      sample(frames, quality, name, targetUs, (frame) => anchorAt(frame, name), interpolatePoint)
        ?.value;
    const left = shoulder("left_shoulder");
    const right = shoulder("right_shoulder");
    if (left !== undefined && right !== undefined) {
      const leftX = mirrored ? 1 - left[0] : left[0];
      const rightX = mirrored ? 1 - right[0] : right[0];
      const bodyScale = Math.hypot(rightX - leftX, right[1] - left[1]);
      if (Number.isFinite(bodyScale) && bodyScale > 0) {
        bodyAvailableMask[outputIndex] = 1;
      }
    }
  });

  const lastSelectedTimestamp = timestampsUs.at(-1)!;
  for (let index = selected.length; index < FRAME_COUNT; index += 1) {
    paddingMask[index] = 1;
    const offset = index - selected.length + 1;
    timestampsUs.push(lastSelectedTimestamp + roundHalfUp(offset * SCALE, 30));
  }
  if (!values.every(Number.isFinite)) throw new Error("candidate.preprocessing.invalid_tensor");
  return {
    values,
    shape: SHAPE,
    validMask,
    observedMask,
    interpolatedMask,
    handPresentMask,
    bodyAvailableMask,
    paddingMask,
    timestampsUs,
  };
}
