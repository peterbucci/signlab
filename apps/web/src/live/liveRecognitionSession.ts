import {
  createCandidateInferenceWorkerClient,
  type CandidateInferenceClientEvent,
  type CandidateInferenceWorkerClient,
} from "../inference/CandidateInferenceWorkerClient";
import {
  CandidateObservationProjector,
  createCandidateEventDetector,
  type CandidateEvent,
  type CandidateEventDetector,
} from "../inference/candidateEvents";
import type {
  CandidateInferenceInput,
  CandidateInferenceResult,
} from "../inference/candidateInferenceProtocol";
import {
  createLandmarkWorkerClient,
  type LandmarkClientEvent,
  type LandmarkWorkerClient,
} from "../landmarks/LandmarkWorkerClient";
import type { LandmarkModelAssetBuffers } from "../landmarks/landmarkModelAssets";
import type { LandmarkFrameMessage } from "../landmarks/protocol";
import type { VerifiedModelBundle } from "../modelBundle/modelBundleSession";

export type LiveRecognitionPhase =
  "loading" | "ready" | "watching" | "recording" | "classifying" | "result" | "failed";

export interface LiveRecognitionSnapshot {
  readonly phase: LiveRecognitionPhase;
  readonly stableResult: CandidateInferenceResult | null;
  readonly failureCode: string | null;
}

export type LiveLandmarkClient = Pick<LandmarkWorkerClient, "initialize" | "submitFrame" | "close">;
export type LiveInferenceClient = Pick<
  CandidateInferenceWorkerClient,
  "initialize" | "classify" | "close"
>;
type Factory<Event, Client> = (onEvent: (event: Event) => void) => Client;

export interface LiveRecognitionSessionOptions {
  readonly bundle: VerifiedModelBundle;
  readonly taskBuffers: LandmarkModelAssetBuffers;
  readonly onState: (snapshot: LiveRecognitionSnapshot) => void;
  readonly createLandmarkClient?: Factory<LandmarkClientEvent, LiveLandmarkClient>;
  readonly createInferenceClient?: Factory<CandidateInferenceClientEvent, LiveInferenceClient>;
}

type BufferedFrame = readonly [detectorIndex: number, frame: LandmarkFrameMessage];

const QUALITY_EVIDENCE = { timestampDiscontinuityCount: 0, gaps: [] } as const;
const INITIAL_SNAPSHOT = { phase: "loading", stableResult: null, failureCode: null } as const;

export class LiveRecognitionSession {
  private snapshotValue: LiveRecognitionSnapshot = INITIAL_SNAPSHOT;
  private readonly projector = new CandidateObservationProjector();
  private detector: CandidateEventDetector | null = null;
  private landmarkClient: LiveLandmarkClient | null = null;
  private inferenceClient: LiveInferenceClient | null = null;
  private readyWorkers = 0;
  private closed = false;
  private landmarkClosePromise: Promise<void> = Promise.resolve();
  private nextDetectorIndex = 0;
  private nextRequestId = 0;
  private activeRequestId: number | null = null;
  private bufferedFrames: BufferedFrame[] = [];
  private retentionUs = 0;

  constructor(private readonly options: LiveRecognitionSessionOptions) {
    options.onState(this.snapshotValue);
  }

  async initialize(): Promise<void> {
    try {
      this.detector = await createCandidateEventDetector(this.options.bundle);
      if (this.closed) return;
      const config = this.detector.config;
      this.retentionUs =
        config.maximum_event_duration_us +
        config.pre_roll_us +
        config.finalization_duration_us +
        config.maximum_gap_us;
      const landmarkFactory = this.options.createLandmarkClient ?? createLandmarkWorkerClient;
      const inferenceFactory =
        this.options.createInferenceClient ?? createCandidateInferenceWorkerClient;
      this.landmarkClient = landmarkFactory((event) => this.handleLandmarkEvent(event));
      this.inferenceClient = inferenceFactory((event) => this.handleInferenceEvent(event));
      this.landmarkClient.initialize(
        this.options.taskBuffers.handModelBuffer,
        this.options.taskBuffers.poseModelBuffer,
      );
      await this.inferenceClient.initialize(this.options.bundle);
    } catch {
      this.fail("live.recognition.initialization.failed");
    }
  }

  submitFrame(frame: ImageBitmap, captureTimestampMs: number): number | null {
    if (
      this.closed ||
      this.landmarkClient === null ||
      ["failed", "loading", "classifying"].includes(this.snapshotValue.phase)
    ) {
      frame.close();
      return null;
    }
    if (["ready", "result"].includes(this.snapshotValue.phase)) this.publish("watching");
    try {
      return this.landmarkClient.submitFrame(frame, captureTimestampMs);
    } catch {
      this.fail("live.recognition.frame_submission.failed");
      return null;
    }
  }

  close(): Promise<void> {
    if (!this.closed) {
      this.closed = true;
      this.bufferedFrames = [];
      this.projector.reset();
      if (this.snapshotValue.phase !== "failed") this.closeClients();
    }
    return this.landmarkClosePromise;
  }

  private handleLandmarkEvent(event: LandmarkClientEvent): void {
    if (this.closed || this.snapshotValue.phase === "failed") return;
    switch (event.type) {
      case "ready":
        if (++this.readyWorkers === 2) this.publish("ready");
        return;
      case "frame-dropped":
        return;
      case "frame":
        if (this.snapshotValue.phase !== "classifying") this.handleLandmarkFrame(event);
        return;
      case "failure":
      case "worker-transport-failure":
        return this.fail(event.code);
      default:
        return this.fail("live.recognition.landmark_worker.stopped");
    }
  }

  private handleLandmarkFrame(frame: LandmarkFrameMessage): void {
    const detector = this.detector!;
    try {
      const detectorIndex = this.nextDetectorIndex++;
      this.bufferedFrames.push([detectorIndex, frame]);
      const cutoff = frame.relativeTimestampUs - this.retentionUs;
      while ((this.bufferedFrames[0]?.[1].relativeTimestampUs ?? cutoff) < cutoff)
        this.bufferedFrames.shift();
      const event = detector.push(this.projector.project(frame));
      if (event !== null) this.classify(event);
      else
        this.publish(
          ["arming", "recording", "finalizing"].includes(detector.state) ? "recording" : "watching",
        );
    } catch {
      this.fail("live.recognition.landmark_frame.invalid");
    }
  }

  private classify(event: CandidateEvent): void {
    const input = this.buildInput(event);
    while ((this.bufferedFrames[0]?.[0] ?? event.lastFrameIndex + 1) <= event.lastFrameIndex)
      this.bufferedFrames.shift();
    if (input === null) return this.publish("result", null, "live.recognition.candidate.invalid");
    const client = this.inferenceClient;
    if (client === null || this.activeRequestId !== null)
      return this.fail("live.recognition.inference.unavailable");
    const requestId = this.nextRequestId++;
    this.activeRequestId = requestId;
    this.publish("classifying");
    try {
      client.classify(requestId, input);
    } catch {
      this.fail("live.recognition.classification.failed");
    }
  }

  private buildInput(event: CandidateEvent): CandidateInferenceInput | null {
    const expectedCount = event.lastFrameIndex - event.firstFrameIndex + 1;
    const frames = this.bufferedFrames.filter(
      ([index]) => index >= event.firstFrameIndex && index <= event.lastFrameIndex,
    );
    if (
      expectedCount <= 0 ||
      frames.length !== expectedCount ||
      frames[0]?.[1].relativeTimestampUs !== event.firstTimestampUs ||
      frames.at(-1)?.[1].relativeTimestampUs !== event.lastTimestampUs ||
      frames.some(
        ([detectorIndex, frame], index) =>
          detectorIndex !== event.firstFrameIndex + index ||
          !frame.valid ||
          (index > 0 && frame.relativeTimestampUs <= frames[index - 1]![1].relativeTimestampUs),
      )
    )
      return null;
    const firstTimestampUs = frames[0][1].relativeTimestampUs;
    return {
      frames: frames.map(([, frame]) => ({
        relativeTimestampUs: frame.relativeTimestampUs - firstTimestampUs,
        valid: frame.valid,
        hands: frame.hands,
        bodyAnchors: frame.bodyAnchors,
      })),
      sourceMirrorState: "not_mirrored",
      quality: QUALITY_EVIDENCE,
    };
  }

  private handleInferenceEvent(event: CandidateInferenceClientEvent): void {
    if (this.closed || this.snapshotValue.phase === "failed") return;
    switch (event.type) {
      case "ready":
        if (++this.readyWorkers === 2) this.publish("ready");
        return;
      case "result":
        if (event.requestId !== this.activeRequestId)
          return this.fail("live.recognition.inference.result_mismatch");
        this.activeRequestId = null;
        return this.publish("result", event);
      case "failure":
      case "worker-transport-failure":
        return this.fail(event.code);
      default:
        return this.fail("live.recognition.inference_worker.stopped");
    }
  }

  private publish(
    phase: LiveRecognitionPhase,
    stableResult = this.snapshotValue.stableResult,
    failureCode: string | null = null,
  ): void {
    this.snapshotValue = Object.freeze({ phase, stableResult, failureCode });
    this.options.onState(this.snapshotValue);
  }

  private fail(code: string): void {
    if (this.closed || this.snapshotValue.phase === "failed") return;
    this.bufferedFrames = [];
    this.projector.reset();
    this.activeRequestId = null;
    this.publish("failed", this.snapshotValue.stableResult, code);
    this.closeClients();
  }

  private closeClients(): void {
    this.inferenceClient?.close();
    this.landmarkClosePromise = this.landmarkClient?.close() ?? Promise.resolve();
  }
}
