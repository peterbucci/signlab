import {
  HAND_SLOT_IDS,
  LANDMARK_WORKER_PROTOCOL_VERSION,
  absentBodyAnchors,
  absentHandSlots,
  type HandSlot,
  type HandSlots,
  type LandmarkWorkerInputMessage,
  type LandmarkWorkerOutputMessage,
  type ProcessLandmarkFrame,
} from "./protocol";
import type { LandmarkDetector, LandmarkModelBuffers } from "./mediapipeRuntime";
import { HandIdentityTracker, type HandDetection, type TrackedDetections } from "./tracking";

export type LandmarkDetectorFactory = (models: LandmarkModelBuffers) => Promise<LandmarkDetector>;

export type LandmarkWorkerPost = (message: LandmarkWorkerOutputMessage) => void;

function now(): number {
  return performance.now();
}

function presentHandSlot(
  slotId: (typeof HAND_SLOT_IDS)[number],
  detection: HandDetection,
): HandSlot {
  return {
    slotId,
    present: true,
    detectorIndex: detection.detectorIndex,
    trackingId: slotId,
    handedness: detection.reportedHandedness,
    handednessConfidence: detection.handednessConfidence,
    imageLandmarks: detection.imageLandmarks,
    worldLandmarks: detection.worldLandmarks,
  };
}

function handSlots(detections: TrackedDetections): HandSlots {
  return HAND_SLOT_IDS.map((slotId, index) => {
    const detection = detections[index];
    return detection === null || detection === undefined
      ? absentHandSlots()[index]
      : presentHandSlot(slotId, detection);
  }) as unknown as HandSlots;
}

function isNonNegativeSafeInteger(value: number): boolean {
  return Number.isSafeInteger(value) && value >= 0;
}

export class LandmarkWorkerSession {
  private readonly tracker = new HandIdentityTracker();
  private detector: LandmarkDetector | undefined;
  private initializing = false;
  private stopped = false;
  private failureCount = 0;
  private previousTaskTimestampMs = -1;

  constructor(
    private readonly createDetector: LandmarkDetectorFactory,
    private readonly post: LandmarkWorkerPost,
  ) {}

  async handle(message: LandmarkWorkerInputMessage): Promise<void> {
    if (message.protocolVersion !== LANDMARK_WORKER_PROTOCOL_VERSION) {
      this.failProtocol();
      if (message.type === "process-frame") message.frame.close();
      return;
    }
    if (message.type === "initialize") {
      await this.initialize(message.handModelBuffer, message.poseModelBuffer);
      return;
    }
    if (message.type === "process-frame") {
      this.processFrame(message);
      return;
    }
    this.stop();
  }

  private async initialize(
    handModelBuffer: ArrayBuffer,
    poseModelBuffer: ArrayBuffer,
  ): Promise<void> {
    if (this.stopped || this.initializing || this.detector !== undefined) {
      this.failProtocol();
      return;
    }
    this.initializing = true;
    const startedAt = now();
    try {
      const detector = await this.createDetector({ handModelBuffer, poseModelBuffer });
      if (this.stopped) {
        detector.close();
        return;
      }
      this.detector = detector;
      this.post({
        type: "ready",
        protocolVersion: LANDMARK_WORKER_PROTOCOL_VERSION,
        startupMs: now() - startedAt,
        failureCount: this.failureCount,
      });
    } catch {
      this.failureCount += 1;
      this.post({
        type: "failure",
        protocolVersion: LANDMARK_WORKER_PROTOCOL_VERSION,
        stage: "initialization",
        code: "extraction.runtime.initialization.failed",
        failureCount: this.failureCount,
      });
    } finally {
      this.initializing = false;
    }
  }

  private processFrame(message: ProcessLandmarkFrame): void {
    const startedAt = now();
    try {
      if (
        this.stopped ||
        this.detector === undefined ||
        !isNonNegativeSafeInteger(message.frameId) ||
        !isNonNegativeSafeInteger(message.relativeTimestampUs) ||
        !isNonNegativeSafeInteger(message.taskTimestampMs) ||
        message.taskTimestampMs !==
          Math.max(
            Math.floor(message.relativeTimestampUs / 1_000),
            this.previousTaskTimestampMs + 1,
          )
      ) {
        this.failProtocol();
        return;
      }
      this.previousTaskTimestampMs = message.taskTimestampMs;
      const detected = this.detector.infer(message.frame, message.taskTimestampMs);
      const tracked = this.tracker.track(detected.hands);
      this.post({
        type: "frame",
        protocolVersion: LANDMARK_WORKER_PROTOCOL_VERSION,
        frameId: message.frameId,
        relativeTimestampUs: message.relativeTimestampUs,
        taskTimestampMs: message.taskTimestampMs,
        valid: true,
        invalidReason: null,
        failureCode: null,
        hands: handSlots(tracked),
        bodyAnchors: detected.bodyAnchors,
        processingMs: now() - startedAt,
        failureCount: this.failureCount,
      });
    } catch {
      this.failureCount += 1;
      this.post({
        type: "frame",
        protocolVersion: LANDMARK_WORKER_PROTOCOL_VERSION,
        frameId: message.frameId,
        relativeTimestampUs: message.relativeTimestampUs,
        taskTimestampMs: message.taskTimestampMs,
        valid: false,
        invalidReason: "task_inference_failed",
        failureCode: "extraction.runtime.inference.failed",
        hands: absentHandSlots(),
        bodyAnchors: absentBodyAnchors(),
        processingMs: now() - startedAt,
        failureCount: this.failureCount,
      });
    } finally {
      message.frame.close();
    }
  }

  private failProtocol(): void {
    this.failureCount += 1;
    this.post({
      type: "failure",
      protocolVersion: LANDMARK_WORKER_PROTOCOL_VERSION,
      stage: "protocol",
      code: "extraction.protocol.invalid",
      failureCount: this.failureCount,
    });
  }

  private stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    try {
      this.detector?.close();
    } catch {
      this.failureCount += 1;
    } finally {
      this.detector = undefined;
      this.tracker.reset();
      this.post({
        type: "stopped",
        protocolVersion: LANDMARK_WORKER_PROTOCOL_VERSION,
        failureCount: this.failureCount,
      });
    }
  }
}
