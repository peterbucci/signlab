export const LANDMARK_WORKER_PROTOCOL_VERSION = "signlab-landmark-worker/1" as const;

export const HAND_SLOT_IDS = ["hand_0", "hand_1"] as const;
export const BODY_ANCHOR_NAMES = [
  "left_shoulder",
  "right_shoulder",
  "left_elbow",
  "right_elbow",
  "left_wrist",
  "right_wrist",
] as const;

export type HandSlotId = (typeof HAND_SLOT_IDS)[number];
export type BodyAnchorName = (typeof BODY_ANCHOR_NAMES)[number];
export type ReportedHandedness = "left" | "right";

export interface LandmarkPoint {
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly visibility: number | null;
  readonly presence: number | null;
}

export interface HandSlot {
  readonly slotId: HandSlotId;
  readonly present: boolean;
  readonly detectorIndex: number | null;
  readonly trackingId: HandSlotId | null;
  readonly handedness: ReportedHandedness | null;
  readonly handednessConfidence: number | null;
  readonly imageLandmarks: readonly LandmarkPoint[] | null;
  readonly worldLandmarks: readonly LandmarkPoint[] | null;
}

export interface BodyAnchor {
  readonly name: BodyAnchorName;
  readonly present: boolean;
  readonly imagePoint: LandmarkPoint | null;
  readonly worldPoint: LandmarkPoint | null;
}

export type HandSlots = readonly [HandSlot, HandSlot];
export type BodyAnchors = readonly [
  BodyAnchor,
  BodyAnchor,
  BodyAnchor,
  BodyAnchor,
  BodyAnchor,
  BodyAnchor,
];

export interface InitializeLandmarkWorker {
  readonly type: "initialize";
  readonly protocolVersion: typeof LANDMARK_WORKER_PROTOCOL_VERSION;
  readonly handModelBuffer: ArrayBuffer;
  readonly poseModelBuffer: ArrayBuffer;
}

export interface ProcessLandmarkFrame {
  readonly type: "process-frame";
  readonly protocolVersion: typeof LANDMARK_WORKER_PROTOCOL_VERSION;
  readonly frameId: number;
  readonly relativeTimestampUs: number;
  readonly taskTimestampMs: number;
  readonly frame: ImageBitmap;
}

export interface StopLandmarkWorker {
  readonly type: "stop";
  readonly protocolVersion: typeof LANDMARK_WORKER_PROTOCOL_VERSION;
}

export type LandmarkWorkerInputMessage =
  InitializeLandmarkWorker | ProcessLandmarkFrame | StopLandmarkWorker;

export interface LandmarkWorkerReady {
  readonly type: "ready";
  readonly protocolVersion: typeof LANDMARK_WORKER_PROTOCOL_VERSION;
  readonly startupMs: number;
  readonly failureCount: number;
}

interface LandmarkFrameBase {
  readonly type: "frame";
  readonly protocolVersion: typeof LANDMARK_WORKER_PROTOCOL_VERSION;
  readonly frameId: number;
  readonly relativeTimestampUs: number;
  readonly taskTimestampMs: number;
  readonly hands: HandSlots;
  readonly bodyAnchors: BodyAnchors;
  readonly processingMs: number;
  readonly failureCount: number;
}

export interface ValidLandmarkFrame extends LandmarkFrameBase {
  readonly valid: true;
  readonly invalidReason: null;
  readonly failureCode: null;
}

export interface InvalidLandmarkFrame extends LandmarkFrameBase {
  readonly valid: false;
  readonly invalidReason: "task_inference_failed";
  readonly failureCode: "extraction.runtime.inference.failed";
}

export type LandmarkFrameMessage = ValidLandmarkFrame | InvalidLandmarkFrame;

export interface LandmarkWorkerFailure {
  readonly type: "failure";
  readonly protocolVersion: typeof LANDMARK_WORKER_PROTOCOL_VERSION;
  readonly stage: "initialization" | "protocol";
  readonly code: "extraction.runtime.initialization.failed" | "extraction.protocol.invalid";
  readonly failureCount: number;
}

export interface LandmarkWorkerStopped {
  readonly type: "stopped";
  readonly protocolVersion: typeof LANDMARK_WORKER_PROTOCOL_VERSION;
  readonly failureCount: number;
}

export type LandmarkWorkerOutputMessage =
  LandmarkWorkerReady | LandmarkFrameMessage | LandmarkWorkerFailure | LandmarkWorkerStopped;

export function absentHandSlots(): HandSlots {
  return HAND_SLOT_IDS.map((slotId) => ({
    slotId,
    present: false,
    detectorIndex: null,
    trackingId: null,
    handedness: null,
    handednessConfidence: null,
    imageLandmarks: null,
    worldLandmarks: null,
  })) as unknown as HandSlots;
}

export function absentBodyAnchors(): BodyAnchors {
  return BODY_ANCHOR_NAMES.map((name) => ({
    name,
    present: false,
    imagePoint: null,
    worldPoint: null,
  })) as unknown as BodyAnchors;
}
