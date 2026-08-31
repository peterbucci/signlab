import { describe, expect, it, vi } from "vitest";

import config from "../../../../configs/evaluation/candidate-event-detector-v1.json";
import type { CandidateInferenceClientEvent } from "../inference/CandidateInferenceWorkerClient";
import {
  CANDIDATE_INFERENCE_PROTOCOL_VERSION,
  type CandidateInferenceInput,
  type CandidateInferenceResult,
} from "../inference/candidateInferenceProtocol";
import type { LandmarkClientEvent } from "../landmarks/LandmarkWorkerClient";
import {
  LANDMARK_WORKER_PROTOCOL_VERSION,
  absentBodyAnchors,
  absentHandSlots,
  type HandSlot,
  type LandmarkFrameMessage,
  type LandmarkPoint,
} from "../landmarks/protocol";
import type { VerifiedModelBundle } from "../modelBundle/modelBundleSession";
import {
  LiveRecognitionSession,
  type LiveInferenceClient,
  type LiveLandmarkClient,
  type LiveRecognitionSnapshot,
} from "./liveRecognitionSession";

const SEGMENTER_SHA256 = "sha256:0443badf68d34347a00096682cf049b6f49b5253c12e47bf61b068a597aa162d";
const QUALITY_SHA256 = "sha256:680b0904e1cc5d8e03119032e92920a3a0185917a600c4293323b7925da9a545";

const bundle = {
  id: "test-bundle",
  version: "1",
  manifest: {
    components: {
      segmenter_sha256: SEGMENTER_SHA256,
      quality_policy_sha256: QUALITY_SHA256,
    },
  },
  bytesByRole: { segmenter: new Blob([JSON.stringify(config)]) },
} as unknown as VerifiedModelBundle;

const point = (x: number): LandmarkPoint => ({
  x,
  y: 0,
  z: 0,
  visibility: null,
  presence: null,
});

function hand(x: number): HandSlot {
  const landmarks = Array.from({ length: 21 }, () => point(x));
  return {
    slotId: "hand_0",
    present: true,
    detectorIndex: 0,
    trackingId: "hand_0",
    handedness: "left",
    handednessConfidence: 0.9,
    imageLandmarks: landmarks,
    worldLandmarks: landmarks,
  };
}

function frame(
  frameId: number,
  timestampUs: number,
  x: number | null,
  valid = true,
): LandmarkFrameMessage {
  const hands = absentHandSlots();
  const base = {
    type: "frame" as const,
    protocolVersion: LANDMARK_WORKER_PROTOCOL_VERSION,
    frameId,
    relativeTimestampUs: timestampUs,
    taskTimestampMs: Math.floor(timestampUs / 1_000),
    hands: x === null ? hands : ([hand(x), hands[1]] as const),
    bodyAnchors: absentBodyAnchors(),
    processingMs: 1,
    failureCount: 0,
  };
  return valid
    ? { ...base, valid: true, invalidReason: null, failureCode: null }
    : {
        ...base,
        valid: false,
        invalidReason: "task_inference_failed",
        failureCode: "extraction.runtime.inference.failed",
      };
}

const result = (requestId: number): CandidateInferenceResult => ({
  type: "result",
  protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
  requestId,
  bundle: { id: bundle.id, version: bundle.version },
  backend: "wasm",
  decision: { kind: "target", label: "hello", confidence: 0.9 },
  reason: "accepted_target",
  rankedScores: [{ label: "hello", confidence: 0.9 }],
  timings: { preprocessingMs: 1, inferenceMs: 2, decisionMs: 1, totalMs: 4 },
});

function harness() {
  let emitLandmark: (event: LandmarkClientEvent) => void = () => undefined;
  let emitInference: (event: CandidateInferenceClientEvent) => void = () => undefined;
  let frameId = 0;
  const snapshots: LiveRecognitionSnapshot[] = [];
  const classifyCalls: Array<{ requestId: number; input: CandidateInferenceInput }> = [];
  const landmark: LiveLandmarkClient = {
    initialize: vi.fn(() =>
      emitLandmark({
        type: "ready",
        protocolVersion: LANDMARK_WORKER_PROTOCOL_VERSION,
        startupMs: 1,
        failureCount: 0,
      }),
    ),
    submitFrame: vi.fn(() => frameId++),
    close: vi.fn(() => Promise.resolve()),
  };
  const inference: LiveInferenceClient = {
    initialize: vi.fn(() => {
      emitInference({
        type: "ready",
        protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
        bundle: { id: bundle.id, version: bundle.version },
        backend: "wasm",
        startupMs: 1,
      });
      return Promise.resolve();
    }),
    classify: vi.fn((requestId: number, input: CandidateInferenceInput) =>
      classifyCalls.push({ requestId, input }),
    ),
    close: vi.fn(),
  };
  const session = new LiveRecognitionSession({
    bundle,
    taskBuffers: { handModelBuffer: new ArrayBuffer(1), poseModelBuffer: new ArrayBuffer(1) },
    onState: (snapshot) => snapshots.push(snapshot),
    createLandmarkClient: (onEvent) => {
      emitLandmark = onEvent;
      return landmark;
    },
    createInferenceClient: (onEvent) => {
      emitInference = onEvent;
      return inference;
    },
  });
  return {
    session,
    snapshots,
    classifyCalls,
    landmark,
    inference,
    emitLandmark: (event: LandmarkClientEvent) => emitLandmark(event),
    emitInference: (event: CandidateInferenceClientEvent) => emitInference(event),
  };
}

function emitActiveEvent(emit: (event: LandmarkClientEvent) => void, invalidInside = false): void {
  [0, 50_000, 100_000, 150_000, 200_000].forEach((timestampUs, index) =>
    emit(frame(index, timestampUs, index * 0.05)),
  );
  if (invalidInside) {
    emit(frame(5, 225_000, null, false));
    emit(frame(6, 250_000, 0.25));
    [275_000, 300_000, 325_000].forEach((timestampUs, offset) =>
      emit(frame(offset + 7, timestampUs, null, false)),
    );
  } else {
    [225_000, 250_000, 275_000].forEach((timestampUs, offset) =>
      emit(frame(offset + 5, timestampUs, null, false)),
    );
  }
}

describe("LiveRecognitionSession", () => {
  it("publishes current-session diagnostics from existing worker events", async () => {
    const runtime = harness();
    await runtime.session.initialize();
    expect(runtime.snapshots.at(-1)?.diagnostics).toEqual({
      detectorState: "inactive",
      landmarkState: "waiting",
      detectedHands: 0,
      droppedFrames: 0,
      backend: "wasm",
      bundle: { id: bundle.id, version: bundle.version },
    });

    const snapshotCount = runtime.snapshots.length;
    runtime.emitLandmark({ type: "frame-dropped", frameId: 0, droppedFrames: 2 });
    expect(runtime.snapshots).toHaveLength(snapshotCount);
    runtime.emitLandmark(frame(0, 0, 0.4));
    expect(runtime.snapshots.at(-1)?.diagnostics).toMatchObject({
      detectorState: "inactive",
      landmarkState: "usable",
      detectedHands: 1,
      droppedFrames: 2,
    });

    runtime.emitLandmark(frame(1, 50_000, null));
    expect(runtime.snapshots.at(-1)?.diagnostics).toMatchObject({
      landmarkState: "no_hands",
      detectedHands: 0,
      droppedFrames: 2,
    });
  });

  it("classifies one exact event with rebased frames and publishes its result", async () => {
    const runtime = harness();
    await runtime.session.initialize();
    expect(runtime.snapshots.at(-1)?.phase).toBe("ready");

    emitActiveEvent(runtime.emitLandmark);

    expect(runtime.classifyCalls).toHaveLength(1);
    const request = runtime.classifyCalls[0]!;
    expect(request.input.frames.map((candidate) => candidate.relativeTimestampUs)).toEqual([
      0, 50_000, 100_000, 150_000, 200_000,
    ]);
    expect(request.input).toMatchObject({
      sourceMirrorState: "not_mirrored",
      quality: { timestampDiscontinuityCount: 0, gaps: [] },
    });
    expect(runtime.snapshots.at(-1)?.phase).toBe("classifying");

    const stable = result(request.requestId);
    runtime.emitInference(stable);
    expect(runtime.snapshots.at(-1)).toMatchObject({ phase: "result", stableResult: stable });
  });

  it("discards an event containing an invalid frame without classifying", async () => {
    const runtime = harness();
    await runtime.session.initialize();
    emitActiveEvent(runtime.emitLandmark, true);
    expect(runtime.classifyCalls).toHaveLength(0);
    expect(runtime.snapshots.at(-1)).toMatchObject({
      phase: "result",
      stableResult: null,
      failureCode: "live.recognition.candidate.invalid",
    });
  });

  it("keeps only the configured timestamp window", async () => {
    const runtime = harness();
    await runtime.session.initialize();
    for (let index = 0; index <= 10; index += 1) {
      runtime.emitLandmark(frame(index, index * 1_000_000, null));
    }
    expect(Reflect.get(runtime.session, "bufferedFrames")).toHaveLength(5);
  });

  it("keeps the last result stable while watching the next frames", async () => {
    const runtime = harness();
    await runtime.session.initialize();
    emitActiveEvent(runtime.emitLandmark);
    const stable = result(runtime.classifyCalls[0]!.requestId);
    runtime.emitInference(stable);
    runtime.emitLandmark(frame(20, 500_000, null));
    expect(runtime.snapshots.at(-1)).toMatchObject({ phase: "watching", stableResult: stable });
  });

  it("fails closed, clears buffered data, and closes both workers", async () => {
    const runtime = harness();
    await runtime.session.initialize();
    runtime.emitLandmark(frame(0, 0, null));
    runtime.emitLandmark({
      type: "failure",
      protocolVersion: LANDMARK_WORKER_PROTOCOL_VERSION,
      stage: "protocol",
      code: "extraction.protocol.invalid",
      failureCount: 1,
    });

    expect(runtime.snapshots.at(-1)).toMatchObject({
      phase: "failed",
      failureCode: "extraction.protocol.invalid",
    });
    expect(Reflect.get(runtime.session, "bufferedFrames")).toHaveLength(0);
    expect(runtime.landmark.close).toHaveBeenCalledOnce();
    expect(runtime.inference.close).toHaveBeenCalledOnce();
    await runtime.session.close();
    expect(runtime.landmark.close).toHaveBeenCalledOnce();
    expect(runtime.inference.close).toHaveBeenCalledOnce();
  });
});
