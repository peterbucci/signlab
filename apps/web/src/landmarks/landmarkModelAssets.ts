import landmarkerConfig from "../../../../src/signlab/resources/extraction/config/mediapipe-extraction-config-1.default.json";
import { readBoundedResponse } from "../network/readBoundedResponse";

export interface LandmarkModelAssetEnvironment {
  readonly fetch: (input: URL, init?: RequestInit) => Promise<Response>;
  readonly subtle?: SubtleCrypto;
  readonly production?: boolean;
  readonly documentBaseUrl?: string;
}

export interface LandmarkModelAssetBuffers {
  readonly handModelBuffer: ArrayBuffer;
  readonly poseModelBuffer: ArrayBuffer;
}

const TASKS = [
  {
    developmentUrl:
      "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task?generation=1682480004222387",
    productionPath: "models/mediapipe/hand_landmarker.task",
    spec: landmarkerConfig.hand_task_asset,
  },
  {
    developmentUrl:
      "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task?generation=1682624736756847",
    productionPath: "models/mediapipe/pose_landmarker_lite.task",
    spec: landmarkerConfig.pose_task_asset,
  },
] as const;

class LandmarkModelAssetError extends Error {}

function fail(code: string): never {
  throw new LandmarkModelAssetError(code);
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
    const url =
      environment.production === true
        ? new URL(task.productionPath, environment.documentBaseUrl)
        : new URL(task.developmentUrl);
    const response =
      environment.production === true
        ? await environment.fetch(url, { redirect: "error" })
        : await environment.fetch(url);
    if (!response.ok) fail("landmark.models.unavailable");
    buffer = await readBoundedResponse(response, task.spec.size_bytes, () =>
      fail("landmark.models.size_mismatch"),
    );
  } catch (error) {
    if (error instanceof LandmarkModelAssetError) throw error;
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
    fetch: (input, init) => globalThis.fetch(input, init),
    subtle: globalThis.crypto?.subtle,
    production: import.meta.env.PROD,
    documentBaseUrl: globalThis.document?.baseURI,
  },
): Promise<LandmarkModelAssetBuffers> {
  if (environment.subtle === undefined) fail("landmark.models.environment.unsupported");
  const [handModelBuffer, poseModelBuffer] = await Promise.all([
    loadOne(TASKS[0], environment),
    loadOne(TASKS[1], environment),
  ]);
  return { handModelBuffer, poseModelBuffer };
}
