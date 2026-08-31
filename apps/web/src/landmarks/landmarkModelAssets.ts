import landmarkerConfig from "../../../../src/signlab/resources/extraction/config/mediapipe-extraction-config-1.default.json";

export interface LandmarkModelAssetEnvironment {
  readonly fetch: (input: URL) => Promise<Response>;
  readonly subtle?: SubtleCrypto;
}

export interface LandmarkModelAssetBuffers {
  readonly handModelBuffer: ArrayBuffer;
  readonly poseModelBuffer: ArrayBuffer;
}

const TASKS = [
  {
    url: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task?generation=1682480004222387",
    spec: landmarkerConfig.hand_task_asset,
  },
  {
    url: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task?generation=1682624736756847",
    spec: landmarkerConfig.pose_task_asset,
  },
] as const;

function fail(code: string): never {
  throw new Error(code);
}

function digestString(buffer: ArrayBuffer): string {
  return `sha256:${Array.from(new Uint8Array(buffer), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("")}`;
}

async function loadOne(
  task: (typeof TASKS)[number],
  environment: LandmarkModelAssetEnvironment,
): Promise<ArrayBuffer> {
  let buffer: ArrayBuffer;
  try {
    const response = await environment.fetch(new URL(task.url));
    if (!response.ok) fail("landmark.models.unavailable");
    buffer = await response.arrayBuffer();
  } catch {
    fail("landmark.models.unavailable");
  }
  if (buffer.byteLength !== task.spec.size_bytes) fail("landmark.models.size_mismatch");

  let digest: ArrayBuffer;
  try {
    if (environment.subtle === undefined) fail("landmark.models.environment.unsupported");
    digest = await environment.subtle.digest("SHA-256", new Uint8Array(buffer));
  } catch {
    fail("landmark.models.environment.unsupported");
  }
  if (digestString(digest) !== task.spec.sha256) fail("landmark.models.digest_mismatch");
  return buffer;
}

export async function loadLandmarkModelAssets(
  environment: LandmarkModelAssetEnvironment = {
    fetch: (input) => globalThis.fetch(input),
    subtle: globalThis.crypto?.subtle,
  },
): Promise<LandmarkModelAssetBuffers> {
  if (environment.subtle === undefined) fail("landmark.models.environment.unsupported");
  const [handModelBuffer, poseModelBuffer] = await Promise.all([
    loadOne(TASKS[0], environment),
    loadOne(TASKS[1], environment),
  ]);
  return { handModelBuffer, poseModelBuffer };
}
