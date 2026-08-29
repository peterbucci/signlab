import {
  HandLandmarker,
  PoseLandmarker,
  type HandLandmarkerResult,
  type Landmark,
  type NormalizedLandmark,
  type PoseLandmarkerResult,
} from "@mediapipe/tasks-vision";
import wasmBinaryUrl from "@mediapipe/tasks-vision/vision_wasm_module_internal.wasm?url";
import wasmLoaderUrl from "@mediapipe/tasks-vision/vision_wasm_module_internal.js?url";

import {
  BODY_ANCHOR_NAMES,
  type BodyAnchor,
  type BodyAnchors,
  type LandmarkPoint,
  type ReportedHandedness,
} from "./protocol";
import type { HandDetection } from "./tracking";

const BODY_ANCHOR_INDICES = [11, 12, 13, 14, 15, 16] as const;
const HAND_LANDMARK_COUNT = 21;
const POSE_LANDMARK_COUNT = 33;

export const PINNED_MEDIAPIPE_CONFIG = {
  packageVersion: "1.0.1",
  delegate: "CPU",
  runningMode: "VIDEO",
  numHands: 2,
  numPoses: 1,
  minHandDetectionConfidence: 0.5,
  minHandPresenceConfidence: 0.5,
  minHandTrackingConfidence: 0.5,
  minPoseDetectionConfidence: 0.5,
  minPosePresenceConfidence: 0.5,
  minPoseTrackingConfidence: 0.5,
  bodyAnchorIndices: BODY_ANCHOR_INDICES,
} as const;

export interface LandmarkDetectorResult {
  readonly hands: readonly HandDetection[];
  readonly bodyAnchors: BodyAnchors;
}

export interface LandmarkDetector {
  infer(frame: ImageBitmap, taskTimestampMs: number): LandmarkDetectorResult;
  close(): void;
}

export interface LandmarkModelBuffers {
  readonly handModelBuffer: ArrayBuffer;
  readonly poseModelBuffer: ArrayBuffer;
}

type HandResultLike = Pick<HandLandmarkerResult, "landmarks" | "worldLandmarks" | "handedness">;
type PoseResultLike = Pick<PoseLandmarkerResult, "landmarks" | "worldLandmarks">;
type PointLike = Landmark | NormalizedLandmark;

function requireFinite(value: number): number {
  if (!Number.isFinite(value)) throw new Error("extraction.result.invalid");
  return value;
}

function optionalUnitInterval(value: number | undefined): number | null {
  if (value === undefined) return null;
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error("extraction.result.invalid");
  }
  return value;
}

function requiredUnitInterval(value: number | undefined): number {
  const normalized = optionalUnitInterval(value);
  if (normalized === null) throw new Error("extraction.result.invalid");
  return normalized;
}

function normalizePoint(point: PointLike): LandmarkPoint {
  return {
    x: requireFinite(point.x),
    y: requireFinite(point.y),
    z: requireFinite(point.z),
    visibility: optionalUnitInterval(point.visibility),
    // The JavaScript 1.0.1 landmark API does not expose a presence channel.
    presence: null,
  };
}

function normalizePointList(
  points: readonly PointLike[],
  expectedCount: number,
): readonly LandmarkPoint[] {
  if (points.length !== expectedCount) throw new Error("extraction.result.invalid");
  return points.map(normalizePoint);
}

function normalizeHandedness(label: string): ReportedHandedness {
  const normalized = label.toLowerCase();
  if (normalized !== "left" && normalized !== "right") {
    throw new Error("extraction.result.invalid");
  }
  return normalized;
}

export function normalizeHandResult(result: HandResultLike): readonly HandDetection[] {
  const { landmarks, worldLandmarks, handedness } = result;
  if (
    landmarks.length > PINNED_MEDIAPIPE_CONFIG.numHands ||
    landmarks.length !== worldLandmarks.length ||
    landmarks.length !== handedness.length
  ) {
    throw new Error("extraction.result.invalid");
  }

  return landmarks.map((imagePoints, detectorIndex) => {
    const worldPoints = worldLandmarks[detectorIndex];
    const categories = handedness[detectorIndex];
    const category = categories?.[0];
    if (worldPoints === undefined || category === undefined) {
      throw new Error("extraction.result.invalid");
    }
    return {
      detectorIndex,
      imageLandmarks: normalizePointList(imagePoints, HAND_LANDMARK_COUNT),
      worldLandmarks: normalizePointList(worldPoints, HAND_LANDMARK_COUNT),
      reportedHandedness: normalizeHandedness(category.categoryName),
      handednessConfidence: requiredUnitInterval(category.score),
    };
  });
}

function absentAnchor(name: (typeof BODY_ANCHOR_NAMES)[number]): BodyAnchor {
  return { name, present: false, imagePoint: null, worldPoint: null };
}

export function normalizePoseResult(result: PoseResultLike): BodyAnchors {
  const { landmarks, worldLandmarks } = result;
  if (
    landmarks.length > PINNED_MEDIAPIPE_CONFIG.numPoses ||
    landmarks.length !== worldLandmarks.length
  ) {
    throw new Error("extraction.result.invalid");
  }
  if (landmarks.length === 0) {
    return BODY_ANCHOR_NAMES.map(absentAnchor) as unknown as BodyAnchors;
  }

  const imagePose = landmarks[0];
  const worldPose = worldLandmarks[0];
  if (
    imagePose === undefined ||
    worldPose === undefined ||
    imagePose.length !== POSE_LANDMARK_COUNT ||
    worldPose.length !== POSE_LANDMARK_COUNT
  ) {
    throw new Error("extraction.result.invalid");
  }

  return BODY_ANCHOR_NAMES.map((name, anchorIndex) => {
    const pointIndex = BODY_ANCHOR_INDICES[anchorIndex];
    const imagePoint = pointIndex === undefined ? undefined : imagePose[pointIndex];
    const worldPoint = pointIndex === undefined ? undefined : worldPose[pointIndex];
    if (imagePoint === undefined || worldPoint === undefined) {
      throw new Error("extraction.result.invalid");
    }
    return {
      name,
      present: true,
      imagePoint: normalizePoint(imagePoint),
      worldPoint: normalizePoint(worldPoint),
    };
  }) as unknown as BodyAnchors;
}

class BrowserMediaPipeDetector implements LandmarkDetector {
  private closed = false;

  constructor(
    private readonly handLandmarker: HandLandmarker,
    private readonly poseLandmarker: PoseLandmarker,
  ) {}

  infer(frame: ImageBitmap, taskTimestampMs: number): LandmarkDetectorResult {
    if (this.closed) throw new Error("extraction.runtime.closed");
    const handResult = this.handLandmarker.detectForVideo(frame, taskTimestampMs);
    const poseResult = this.poseLandmarker.detectForVideo(frame, taskTimestampMs);
    try {
      return {
        hands: normalizeHandResult(handResult),
        bodyAnchors: normalizePoseResult(poseResult),
      };
    } finally {
      poseResult.close();
    }
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    closeTaskPair(this.handLandmarker, this.poseLandmarker);
  }
}

function closeTaskPair(
  handLandmarker: HandLandmarker | undefined,
  poseLandmarker: PoseLandmarker | undefined,
): void {
  try {
    handLandmarker?.close();
  } finally {
    poseLandmarker?.close();
  }
}

export async function createMediaPipeDetector(
  models: LandmarkModelBuffers,
): Promise<LandmarkDetector> {
  // MediaPipe clears its loader factory after each task is created. Distinct
  // import URLs make the module loader execute once for each task.
  const loaderSeparator = wasmLoaderUrl.includes("?") ? "&" : "?";
  const wasmFileset = (task: "hand" | "pose") => ({
    wasmLoaderPath: `${wasmLoaderUrl}${loaderSeparator}task=${task}`,
    wasmBinaryPath: wasmBinaryUrl,
  });
  let handLandmarker: HandLandmarker | undefined;
  let poseLandmarker: PoseLandmarker | undefined;
  try {
    handLandmarker = await HandLandmarker.createFromOptions(wasmFileset("hand"), {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(models.handModelBuffer),
        delegate: PINNED_MEDIAPIPE_CONFIG.delegate,
      },
      runningMode: PINNED_MEDIAPIPE_CONFIG.runningMode,
      numHands: PINNED_MEDIAPIPE_CONFIG.numHands,
      minHandDetectionConfidence: PINNED_MEDIAPIPE_CONFIG.minHandDetectionConfidence,
      minHandPresenceConfidence: PINNED_MEDIAPIPE_CONFIG.minHandPresenceConfidence,
      minTrackingConfidence: PINNED_MEDIAPIPE_CONFIG.minHandTrackingConfidence,
    });
    poseLandmarker = await PoseLandmarker.createFromOptions(wasmFileset("pose"), {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(models.poseModelBuffer),
        delegate: PINNED_MEDIAPIPE_CONFIG.delegate,
      },
      runningMode: PINNED_MEDIAPIPE_CONFIG.runningMode,
      numPoses: PINNED_MEDIAPIPE_CONFIG.numPoses,
      minPoseDetectionConfidence: PINNED_MEDIAPIPE_CONFIG.minPoseDetectionConfidence,
      minPosePresenceConfidence: PINNED_MEDIAPIPE_CONFIG.minPosePresenceConfidence,
      minTrackingConfidence: PINNED_MEDIAPIPE_CONFIG.minPoseTrackingConfidence,
      outputSegmentationMasks: false,
    });
    return new BrowserMediaPipeDetector(handLandmarker, poseLandmarker);
  } catch {
    try {
      closeTaskPair(handLandmarker, poseLandmarker);
    } catch {
      // Initialization still reports one stable public failure code.
    }
    throw new Error("extraction.runtime.initialization.failed");
  }
}
