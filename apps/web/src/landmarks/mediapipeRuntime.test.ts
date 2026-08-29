import { HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";
import { describe, expect, it, vi } from "vitest";

import {
  PINNED_MEDIAPIPE_CONFIG,
  createMediaPipeDetector,
  normalizeHandResult,
  normalizePoseResult,
} from "./mediapipeRuntime";

function points(count: number, x = 0.25) {
  return Array.from({ length: count }, () => ({ x, y: 0.5, z: -0.1, visibility: 0.8 }));
}

function oneHandResult() {
  return {
    landmarks: [points(21)],
    worldLandmarks: [points(21, 0.75)],
    handedness: [[{ score: 0.91, index: 0, categoryName: "Left", displayName: "" }]],
  };
}

function onePoseResult() {
  return { landmarks: [points(33)], worldLandmarks: [points(33, 0.75)] };
}

describe("MediaPipe landmark normalization", () => {
  it("returns two-hand-compatible observations and the six configured pose anchors", () => {
    const hands = normalizeHandResult(oneHandResult());
    const anchors = normalizePoseResult(onePoseResult());

    expect(hands).toHaveLength(1);
    expect(hands[0]).toMatchObject({
      detectorIndex: 0,
      reportedHandedness: "left",
      handednessConfidence: 0.91,
    });
    expect(hands[0]?.imageLandmarks).toHaveLength(21);
    expect(hands[0]?.imageLandmarks[0]).toEqual({
      x: 0.25,
      y: 0.5,
      z: -0.1,
      visibility: 0.8,
      presence: null,
    });
    expect(anchors.map(({ name }) => name)).toEqual([
      "left_shoulder",
      "right_shoulder",
      "left_elbow",
      "right_elbow",
      "left_wrist",
      "right_wrist",
    ]);
    expect(anchors.every(({ present }) => present)).toBe(true);
  });

  it("represents an absent pose and fails closed on malformed observations", () => {
    expect(normalizePoseResult({ landmarks: [], worldLandmarks: [] })).toMatchObject(
      Array.from({ length: 6 }, () => ({ present: false })),
    );
    expect(() =>
      normalizeHandResult({
        ...oneHandResult(),
        landmarks: [points(20)],
      }),
    ).toThrow("extraction.result.invalid");
    expect(() =>
      normalizeHandResult({
        ...oneHandResult(),
        handedness: [[{ score: Number.NaN, index: 0, categoryName: "Left", displayName: "" }]],
      }),
    ).toThrow("extraction.result.invalid");
  });
});

describe("createMediaPipeDetector", () => {
  it("initializes both pinned VIDEO tasks once and closes their resources once", async () => {
    const handResult = oneHandResult();
    const poseResult = { ...onePoseResult(), close: vi.fn() };
    const handDetect = vi.fn(() => handResult);
    const poseDetect = vi.fn(() => poseResult);
    const handClose = vi.fn();
    const poseClose = vi.fn();
    const handTask = {
      detectForVideo: handDetect,
      close: handClose,
    } as unknown as HandLandmarker;
    const poseTask = {
      detectForVideo: poseDetect,
      close: poseClose,
    } as unknown as PoseLandmarker;
    const createHand = vi.spyOn(HandLandmarker, "createFromOptions").mockResolvedValue(handTask);
    const createPose = vi.spyOn(PoseLandmarker, "createFromOptions").mockResolvedValue(poseTask);

    const detector = await createMediaPipeDetector({
      handModelBuffer: Uint8Array.from([1, 2]).buffer,
      poseModelBuffer: Uint8Array.from([3, 4]).buffer,
    });
    const frame = {} as ImageBitmap;
    const result = detector.infer(frame, 12);
    handClose.mockImplementationOnce(() => {
      throw new Error("close failed");
    });
    expect(() => detector.close()).toThrow("close failed");
    detector.close();

    expect(createHand).toHaveBeenCalledTimes(1);
    expect(createPose).toHaveBeenCalledTimes(1);
    expect(createHand.mock.calls[0]?.[0].wasmLoaderPath).not.toBe(
      createPose.mock.calls[0]?.[0].wasmLoaderPath,
    );
    expect(createHand.mock.calls[0]?.[0].wasmBinaryPath).toBe(
      createPose.mock.calls[0]?.[0].wasmBinaryPath,
    );
    expect(createHand.mock.calls[0]?.[1]).toMatchObject({
      baseOptions: { delegate: "CPU" },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    expect(createPose.mock.calls[0]?.[1]).toMatchObject({
      baseOptions: { delegate: "CPU" },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      outputSegmentationMasks: false,
    });
    expect(handDetect).toHaveBeenCalledWith(frame, 12);
    expect(poseDetect).toHaveBeenCalledWith(frame, 12);
    expect(result.hands).toHaveLength(1);
    expect(result.bodyAnchors).toHaveLength(PINNED_MEDIAPIPE_CONFIG.bodyAnchorIndices.length);
    expect(poseResult.close).toHaveBeenCalledTimes(1);
    expect(handClose).toHaveBeenCalledTimes(1);
    expect(poseClose).toHaveBeenCalledTimes(1);
  });
});
