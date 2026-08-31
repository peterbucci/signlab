/// <reference types="node" />
// @vitest-environment node

import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import eventConfig from "../../../../configs/evaluation/candidate-event-detector-v1.json";
import decisionPolicy from "../../../../docs/reports/popsign-constructed-calibration-policy-v1.json";
import featurePlan from "../../../../src/signlab/resources/features/config/hand-local-64-1.default.json";
import parityFixture from "../../../../tests/fixtures/public/parity/candidate-runtime-goldens-v1.json";
import replayFixture from "../../../../tests/fixtures/public/replay/live-post-landmark-v1.json";
import {
  CandidateInferenceWorkerClient,
  type CandidateInferenceWorkerPort,
} from "../inference/CandidateInferenceWorkerClient";
import {
  type CandidateInferenceInput,
  type CandidateInferenceResult,
  type CandidateInferenceWorkerInput,
  type CandidateInferenceWorkerOutput,
} from "../inference/candidateInferenceProtocol";
import {
  CandidateInferenceSession,
  createCandidateInferenceEngine,
} from "../inference/candidateInferenceSession";
import type { LandmarkClientEvent } from "../landmarks/LandmarkWorkerClient";
import {
  LANDMARK_WORKER_PROTOCOL_VERSION,
  absentHandSlots,
  type BodyAnchors,
  type HandSlots,
  type LandmarkFrameMessage,
} from "../landmarks/protocol";
import type { VerifiedModelBundle } from "../modelBundle/modelBundleSession";
import { LiveRecognitionSession, type LiveRecognitionSnapshot } from "./liveRecognitionSession";

type ReplayStep = {
  readonly relativeTimestampUs: number;
  readonly handsPresent: boolean;
  readonly translateImageX?: number;
};

const repositoryRoot = resolve(process.cwd(), "../..");
const replay = replayFixture as unknown as typeof replayFixture & {
  readonly steps: readonly ReplayStep[];
};
const templateCase = parityFixture.preprocessingCases.find(
  ({ id }) => id === replay.template.preprocessingCaseId,
);
const template =
  templateCase?.frames[replay.template.frameIndex] ??
  (() => {
    throw new Error("missing replay template");
  })();

const encode = (value: unknown) => new TextEncoder().encode(JSON.stringify(value));

class InProcessInferencePort implements CandidateInferenceWorkerPort {
  readonly classifications: CandidateInferenceInput[] = [];
  readonly stopped: Promise<void>;
  private messageListener: (message: CandidateInferenceWorkerOutput) => void = () => undefined;
  private errorListener: () => void = () => undefined;
  private readonly worker: CandidateInferenceSession;
  private resolveStopped: () => void = () => undefined;

  constructor() {
    this.stopped = new Promise((resolveStopped) => (this.resolveStopped = resolveStopped));
    let clock = 0;
    this.worker = new CandidateInferenceSession(
      createCandidateInferenceEngine,
      (message) => {
        this.messageListener(message);
        if (message.type === "stopped") this.resolveStopped();
      },
      () => ++clock,
    );
  }

  post(message: CandidateInferenceWorkerInput): void {
    if (message.type === "classify") this.classifications.push(message.input);
    void this.worker.handle(message).catch(() => this.errorListener());
  }

  onMessage(listener: (message: CandidateInferenceWorkerOutput) => void): () => void {
    this.messageListener = listener;
    return () => (this.messageListener = () => undefined);
  }

  onError(listener: () => void): () => void {
    this.errorListener = listener;
    return () => (this.errorListener = () => undefined);
  }

  terminate(): void {}
}

function frame(step: ReplayStep, frameId: number): LandmarkFrameMessage {
  const hands = step.handsPresent
    ? (template.hands.map((hand) => ({
        ...hand,
        imageLandmarks: hand.imageLandmarks?.map((point) => ({
          ...point,
          x: point.x + (step.translateImageX ?? 0),
        })),
      })) as unknown as HandSlots)
    : absentHandSlots();
  return {
    type: "frame",
    protocolVersion: LANDMARK_WORKER_PROTOCOL_VERSION,
    frameId,
    relativeTimestampUs: step.relativeTimestampUs,
    taskTimestampMs: Math.floor(step.relativeTimestampUs / 1_000),
    hands,
    bodyAnchors: template.bodyAnchors as unknown as BodyAnchors,
    processingMs: 0,
    failureCount: 0,
    valid: true,
    invalidReason: null,
    failureCode: null,
  };
}

async function modelBundle(): Promise<VerifiedModelBundle> {
  const model = await readFile(resolve(repositoryRoot, parityFixture.resources.testModel.path));
  return {
    id: replay.bundle.id,
    version: replay.bundle.version,
    manifest: {
      components: {
        segmenter_sha256: parityFixture.resources.segmenter.semanticSha256,
        quality_policy_sha256: parityFixture.resources.qualityPolicy.semanticSha256,
      },
    },
    bytesByRole: {
      model: new Blob([Uint8Array.from(model)]),
      feature_plan: new Blob([encode(featurePlan)]),
      decision_policy: new Blob([encode(decisionPolicy)]),
      segmenter: new Blob([encode(eventConfig)]),
    },
  } as unknown as VerifiedModelBundle;
}

async function runReplay() {
  let emitLandmark: (event: LandmarkClientEvent) => void = () => undefined;
  let port: InProcessInferencePort | undefined;
  let resolveReady: () => void = () => undefined;
  let resolveResult: (result: CandidateInferenceResult) => void = () => undefined;
  const ready = new Promise<void>((resolve) => (resolveReady = resolve));
  const result = new Promise<CandidateInferenceResult>((resolve) => (resolveResult = resolve));
  const emittedFrames = replay.steps.map(frame);
  const bundle = await modelBundle();
  const session = new LiveRecognitionSession({
    bundle,
    taskBuffers: { handModelBuffer: new ArrayBuffer(0), poseModelBuffer: new ArrayBuffer(0) },
    onState: (snapshot: LiveRecognitionSnapshot) => {
      if (snapshot.phase === "ready") resolveReady();
      if (snapshot.phase === "result" && snapshot.stableResult !== null)
        resolveResult(snapshot.stableResult);
    },
    createLandmarkClient: (onEvent) => {
      emitLandmark = onEvent;
      return {
        initialize: () =>
          onEvent({
            type: "ready",
            protocolVersion: LANDMARK_WORKER_PROTOCOL_VERSION,
            startupMs: 0,
            failureCount: 0,
          }),
        submitFrame: () => 0,
        close: () => Promise.resolve(),
      };
    },
    createInferenceClient: (onEvent) => {
      port = new InProcessInferencePort();
      return new CandidateInferenceWorkerClient(port, onEvent);
    },
  });

  await session.initialize();
  await ready;
  emittedFrames.forEach(emitLandmark);
  const stableResult = await result;
  await session.close();
  if (port === undefined) throw new Error("missing inference port");
  await port.stopped;
  return { emittedFrames, inputs: port.classifications, result: stableResult };
}

describe("live post-landmark replay", () => {
  it(`${replay.fixtureId} deterministically replays ${replay.bundle.id}@${replay.bundle.version}`, async () => {
    const first = await runReplay();
    const second = await runReplay();
    const expected = replay.expected;
    expect(replay.template.path).toBe(
      "tests/fixtures/public/parity/candidate-runtime-goldens-v1.json",
    );
    expect(first.inputs).toHaveLength(1);
    const input = first.inputs[0]!;

    expect(input.frames.map(({ relativeTimestampUs }) => relativeTimestampUs)).toEqual(
      expected.event.inferenceTimestampsUs,
    );
    expect(input.frames).toHaveLength(
      expected.event.lastFrameIndex - expected.event.firstFrameIndex + 1,
    );
    input.frames.forEach((candidateFrame, index) => {
      const source = first.emittedFrames[expected.event.firstFrameIndex + index]!;
      expect(candidateFrame.hands).toBe(source.hands);
      expect(candidateFrame.bodyAnchors).toBe(source.bodyAnchors);
    });
    expect(first.emittedFrames[expected.event.firstFrameIndex]?.relativeTimestampUs).toBe(
      expected.event.firstTimestampUs,
    );
    expect(first.emittedFrames[expected.event.lastFrameIndex]?.relativeTimestampUs).toBe(
      expected.event.lastTimestampUs,
    );
    expect(first.result).toMatchObject({
      bundle: replay.bundle,
      backend: expected.backend,
      decision: { kind: expected.decision.kind, label: expected.decision.label },
      reason: expected.decision.reason,
    });
    expect(first.result.rankedScores).toHaveLength(expected.rankedScores.length);
    first.result.rankedScores.forEach((score, index) => {
      expect(score.label).toBe(expected.rankedScores[index]?.label);
      expect(
        Math.abs(score.confidence - expected.rankedScores[index]!.confidence),
      ).toBeLessThanOrEqual(expected.scoreAbsoluteTolerance);
    });
    expect(second.inputs).toEqual(first.inputs);
    expect(second.result).toEqual(first.result);
  });
});
