import { describe, expect, it, vi } from "vitest";

import type { LandmarkDetector } from "./mediapipeRuntime";
import {
  LANDMARK_WORKER_PROTOCOL_VERSION,
  absentBodyAnchors,
  type LandmarkWorkerOutputMessage,
  type ProcessLandmarkFrame,
} from "./protocol";
import type { HandDetection } from "./tracking";
import { LandmarkWorkerSession } from "./workerSession";

function frame() {
  const close = vi.fn();
  return { image: { close } as unknown as ImageBitmap, close };
}

function hand(detectorIndex: number, x: number, side: "left" | "right"): HandDetection {
  const points = Array.from({ length: 21 }, () => ({
    x,
    y: 0.5,
    z: 0,
    visibility: null,
    presence: null,
  }));
  return {
    detectorIndex,
    imageLandmarks: points,
    worldLandmarks: points,
    reportedHandedness: side,
    handednessConfidence: 0.9,
  };
}

function processMessage(
  image: ImageBitmap,
  frameId: number,
  taskTimestampMs: number,
): ProcessLandmarkFrame {
  return {
    type: "process-frame",
    protocolVersion: LANDMARK_WORKER_PROTOCOL_VERSION,
    frameId,
    relativeTimestampUs: 0,
    taskTimestampMs,
    frame: image,
  };
}

describe("LandmarkWorkerSession", () => {
  it("initializes once, preserves identity through a failed frame, and releases resources", async () => {
    const outputs: LandmarkWorkerOutputMessage[] = [];
    const first = frame();
    const failed = frame();
    const third = frame();
    const infer = vi
      .fn<LandmarkDetector["infer"]>()
      .mockReturnValueOnce({
        hands: [hand(0, 0.2, "left"), hand(1, 0.8, "right")],
        bodyAnchors: absentBodyAnchors(),
      })
      .mockImplementationOnce(() => {
        throw new Error("private runtime detail");
      })
      .mockReturnValueOnce({
        hands: [hand(5, 0.78, "right")],
        bodyAnchors: absentBodyAnchors(),
      });
    const detectorClose = vi.fn();
    const detector: LandmarkDetector = { infer, close: detectorClose };
    const factory = vi.fn(() => Promise.resolve(detector));
    const session = new LandmarkWorkerSession(factory, (message) => outputs.push(message));

    await session.handle({
      type: "initialize",
      protocolVersion: LANDMARK_WORKER_PROTOCOL_VERSION,
      handModelBuffer: new ArrayBuffer(1),
      poseModelBuffer: new ArrayBuffer(1),
    });
    await session.handle(processMessage(first.image, 0, 0));
    await session.handle(processMessage(failed.image, 1, 1));
    await session.handle(processMessage(third.image, 2, 2));
    await session.handle({
      type: "stop",
      protocolVersion: LANDMARK_WORKER_PROTOCOL_VERSION,
    });

    expect(factory).toHaveBeenCalledTimes(1);
    expect(outputs.map(({ type }) => type)).toEqual([
      "ready",
      "frame",
      "frame",
      "frame",
      "stopped",
    ]);
    expect(outputs[2]).toMatchObject({
      valid: false,
      invalidReason: "task_inference_failed",
      failureCode: "extraction.runtime.inference.failed",
      failureCount: 1,
    });
    expect(outputs[3]).toMatchObject({
      valid: true,
      hands: [
        { slotId: "hand_0", present: false },
        { slotId: "hand_1", present: true, detectorIndex: 5 },
      ],
    });
    expect(first.close).toHaveBeenCalledTimes(1);
    expect(failed.close).toHaveBeenCalledTimes(1);
    expect(third.close).toHaveBeenCalledTimes(1);
    expect(detectorClose).toHaveBeenCalledTimes(1);
    expect(JSON.stringify(outputs)).not.toContain("private runtime detail");
  });

  it("rejects a timestamp outside the exact recurrence and still closes the frame", async () => {
    const outputs: LandmarkWorkerOutputMessage[] = [];
    const image = frame();
    const infer = vi.fn(() => ({ hands: [], bodyAnchors: absentBodyAnchors() }));
    const detector: LandmarkDetector = {
      infer,
      close: vi.fn(),
    };
    const session = new LandmarkWorkerSession(
      () => Promise.resolve(detector),
      (message) => outputs.push(message),
    );
    await session.handle({
      type: "initialize",
      protocolVersion: LANDMARK_WORKER_PROTOCOL_VERSION,
      handModelBuffer: new ArrayBuffer(1),
      poseModelBuffer: new ArrayBuffer(1),
    });

    await session.handle(processMessage(image.image, 0, 7));

    expect(outputs.at(-1)).toMatchObject({
      type: "failure",
      stage: "protocol",
      code: "extraction.protocol.invalid",
    });
    expect(infer).not.toHaveBeenCalled();
    expect(image.close).toHaveBeenCalledTimes(1);
  });
});
