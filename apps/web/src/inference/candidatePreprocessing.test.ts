import { describe, expect, it } from "vitest";

import plan from "../../../../src/signlab/resources/features/config/hand-local-64-1.default.json";
import {
  absentBodyAnchors,
  absentHandSlots,
  type HandSlot,
  type HandSlotId,
  type LandmarkPoint,
} from "../landmarks/protocol";
import {
  preprocessCandidate,
  type CandidateFrame,
  type CandidateGapEvidence,
} from "./candidatePreprocessing";

const point = (x: number, y: number, z: number): LandmarkPoint => ({
  x,
  y,
  z,
  visibility: null,
  presence: null,
});

function hand(
  slotId: HandSlotId,
  handedness: "left" | "right",
  mirrored = false,
  zeroScale = false,
  invalid = false,
  landmarkOneY = 0.05,
): HandSlot {
  const landmarks = Array.from({ length: 21 }, (_, index) => {
    const [x, y, z] = zeroScale
      ? [0, 0, 0]
      : index === 9
        ? [1, 0, 0]
        : [index / 10, index === 1 ? landmarkOneY : index / 20, -index / 40];
    return point(
      invalid && index === 1 ? Number.NaN : mirrored ? -(3 + 2 * x) : 3 + 2 * x,
      4 + 2 * y,
      2 * z,
    );
  });
  return {
    slotId,
    present: true,
    detectorIndex: slotId === "hand_0" ? 0 : 1,
    trackingId: slotId,
    handedness,
    handednessConfidence: 0.9,
    imageLandmarks: landmarks,
    worldLandmarks: landmarks,
  };
}

function frame(
  timestamp: number,
  first: HandSlot | null,
  second: HandSlot | null = null,
): CandidateFrame {
  const absent = absentHandSlots();
  return {
    relativeTimestampUs: timestamp,
    valid: true,
    hands: [first ?? absent[0], second ?? absent[1]],
    bodyAnchors: absentBodyAnchors(),
  };
}

const quality = (gaps: readonly CandidateGapEvidence[] = [], discontinuities = 0) => ({
  timestampDiscontinuityCount: discontinuities,
  gaps,
});
const count = (values: Uint8Array, start: number, length: number) =>
  values.slice(start, start + length).reduce((sum, value) => sum + value, 0);

describe("candidate preprocessing contract", () => {
  it("normalizes fixed hand slots, mirror state, elapsed time, quantization, and padding", () => {
    const unmirroredHands = [hand("hand_0", "left"), hand("hand_1", "right")] as const;
    const mirroredHands = [hand("hand_0", "right", true), hand("hand_1", "left", true)] as const;
    const run = (hands: readonly [HandSlot, HandSlot], mirror: "mirrored" | "not_mirrored") =>
      preprocessCandidate([frame(0, ...hands), frame(50_000, ...hands)], mirror, quality(), plan);
    const result = run(unmirroredHands, "not_mirrored");
    const mirrored = run(mirroredHands, "mirrored");
    const halfStep = preprocessCandidate(
      [
        frame(
          0,
          hand("hand_0", "left", false, false, false, 0.0000005),
          hand("hand_1", "right", false, false, false, -0.0000005),
        ),
      ],
      "not_mirrored",
      quality(),
      plan,
    );

    expect(result.shape).toEqual([1, 64, 126]);
    expect(result.values).toHaveLength(8_064);
    expect(Array.from(result.values.slice(0, 126))).toEqual(
      Array.from(mirrored.values.slice(0, 126)),
    );
    expect(result.values[3]).toBe(Math.fround(0.1));
    expect(result.values[66]).toBe(Math.fround(-0.1));
    expect(halfStep.values[4]).toBe(Math.fround(0.000001));
    expect(halfStep.values[67]).toBe(Math.fround(-0.000001));
    expect(result.timestampsUs.slice(0, 4)).toEqual([0, 33_333, 50_000, 83_333]);
    expect(count(result.interpolatedMask, 126, 126)).toBe(126);
    expect(count(result.validMask, 0, 126)).toBe(126);
    expect(result.bodyAvailableMask[0]).toBe(0);
    expect(Array.from(result.paddingMask.slice(0, 4))).toEqual([0, 0, 0, 1]);
  });

  it("bridges only an approved gap and masks missing or zero-scale hands", () => {
    const rows = [
      frame(0, hand("hand_0", "left", false, false, false, 0)),
      frame(33_333, null),
      frame(66_666, hand("hand_0", "left", false, false, false, 0.2)),
    ];
    const gap: CandidateGapEvidence = {
      signal: "hand_0",
      decision: "interpolate_linear",
      leftObservedFrameIndex: 0,
      leftObservedTimestampUs: 0,
      rightObservedFrameIndex: 2,
      rightObservedTimestampUs: 66_666,
    };
    const blocked = preprocessCandidate(rows, "not_mirrored", quality(), plan);
    const approved = preprocessCandidate(rows, "not_mirrored", quality([gap]), plan);
    const invalidScale = preprocessCandidate(
      [frame(0, hand("hand_0", "left"), hand("hand_1", "right", false, true))],
      "not_mirrored",
      quality(),
      plan,
    );

    expect(count(blocked.validMask, 126, 63)).toBe(0);
    expect(count(approved.interpolatedMask, 126, 63)).toBe(63);
    expect(approved.values[130]).toBe(Math.fround(0.1));
    expect(approved.handPresentMask[2]).toBe(1);
    expect(invalidScale.handPresentMask[1]).toBe(1);
    expect(count(invalidScale.validMask, 63, 63)).toBe(0);
  });

  it("preserves long-grid endpoints and rejects unsafe inputs or contract variants", () => {
    const timestamp = (index: number) => Math.floor((2 * index * 1_000_000 + 30) / 60);
    const rows = Array.from({ length: 66 }, (_, index) =>
      frame(timestamp(index), hand("hand_0", "left")),
    );
    const result = preprocessCandidate(rows, "not_mirrored", quality(), plan);
    const changedPlan = structuredClone(plan);
    changedPlan.feature_order.reverse();
    const sparsePlan = { ...plan, feature_order: new Array(126) };
    const rejects =
      (
        candidateFrames: readonly CandidateFrame[],
        candidateQuality = quality(),
        candidatePlan: unknown = plan,
      ) =>
      () =>
        preprocessCandidate(candidateFrames, "not_mirrored", candidateQuality, candidatePlan);

    expect([result.timestampsUs[0], result.timestampsUs[63]]).toEqual([0, timestamp(65)]);
    expect(count(result.paddingMask, 0, 64)).toBe(0);
    [
      rejects([]),
      rejects([frame(0, null), frame(0, null)]),
      rejects([frame(0, hand("hand_0", "left", false, false, true))]),
      rejects([frame(0, null)], quality([], 1)),
      rejects([frame(0, null)], quality(), changedPlan),
      rejects([frame(0, null)], quality(), sparsePlan),
    ].forEach((invoke) => expect(invoke).toThrow());
  });
});
