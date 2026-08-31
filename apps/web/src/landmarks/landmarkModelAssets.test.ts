import landmarkerConfig from "../../../../src/signlab/resources/extraction/config/mediapipe-extraction-config-1.default.json";
import { describe, expect, it, vi } from "vitest";

import { loadLandmarkModelAssets, type LandmarkModelAssetEnvironment } from "./landmarkModelAssets";

const HAND_SIZE = landmarkerConfig.hand_task_asset.size_bytes;
const POSE_SIZE = landmarkerConfig.pose_task_asset.size_bytes;
const HAND_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task?generation=1682480004222387";
const POSE_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task?generation=1682624736756847";
const RELEASE_HAND_URL = "https://example.test/signlab/models/mediapipe/hand_landmarker.task";
const RELEASE_POSE_URL = "https://example.test/signlab/models/mediapipe/pose_landmarker_lite.task";

function bytesForSha256(value: string): ArrayBuffer {
  const hex = value.replace("sha256:", "");
  return Uint8Array.from(hex.match(/.{2}/g) ?? [], (byte) => Number.parseInt(byte, 16)).buffer;
}

function digestForSize(size: number): ArrayBuffer {
  return bytesForSha256(
    size === HAND_SIZE
      ? landmarkerConfig.hand_task_asset.sha256
      : landmarkerConfig.pose_task_asset.sha256,
  );
}

function mockSubtle(digest: SubtleCrypto["digest"]): SubtleCrypto {
  return { digest } as unknown as SubtleCrypto;
}

function environment(
  overrides: Partial<LandmarkModelAssetEnvironment> = {},
): LandmarkModelAssetEnvironment {
  return {
    fetch: vi.fn((url: URL) => {
      const size = [HAND_URL, RELEASE_HAND_URL].includes(url.href) ? HAND_SIZE : POSE_SIZE;
      return Promise.resolve(new Response(new Uint8Array(size)));
    }),
    subtle: mockSubtle(
      vi.fn((_algorithm: AlgorithmIdentifier, data: BufferSource) =>
        Promise.resolve(digestForSize(data.byteLength)),
      ),
    ),
    ...overrides,
  };
}

async function expectedCode(promise: Promise<unknown>, code: string): Promise<void> {
  await expect(promise).rejects.toThrow(code);
}

describe("loadLandmarkModelAssets", () => {
  it("loads the exact supplied pair after checking canonical sizes and digests", async () => {
    const injected = environment();

    const result = await loadLandmarkModelAssets(injected);

    expect(result.handModelBuffer.byteLength).toBe(HAND_SIZE);
    expect(result.poseModelBuffer.byteLength).toBe(POSE_SIZE);
    expect(injected.fetch).toHaveBeenCalledTimes(2);
    expect(injected.fetch).toHaveBeenNthCalledWith(1, new URL(HAND_URL));
    expect(injected.fetch).toHaveBeenNthCalledWith(2, new URL(POSE_URL));
  });

  it("uses fixed same-origin, subpath-safe model URLs in production", async () => {
    const injected = environment({
      production: true,
      documentBaseUrl: "https://example.test/signlab/",
    });

    await loadLandmarkModelAssets(injected);

    expect(injected.fetch).toHaveBeenNthCalledWith(1, new URL(RELEASE_HAND_URL), {
      redirect: "error",
    });
    expect(injected.fetch).toHaveBeenNthCalledWith(2, new URL(RELEASE_POSE_URL), {
      redirect: "error",
    });
  });

  it.each([
    ["short", HAND_SIZE - 1],
    ["oversized streamed", HAND_SIZE + 1],
  ])("rejects a %s task without revealing its URL", async (_case, size) => {
    const injected = environment({
      fetch: vi.fn(() => Promise.resolve(new Response(new Uint8Array(size)))),
    });
    const promise = loadLandmarkModelAssets(injected);

    await expectedCode(promise, "landmark.models.size_mismatch");
    await expect(promise).rejects.not.toThrow("models.example");
  });

  it("rejects a digest mismatch", async () => {
    const injected = environment({
      subtle: mockSubtle(vi.fn(() => Promise.resolve(new Uint8Array(32).buffer))),
    });

    await expectedCode(loadLandmarkModelAssets(injected), "landmark.models.digest_mismatch");
  });

  it("rejects unavailable or failing cryptography with one stable code", async () => {
    const withoutCrypto = environment({ subtle: undefined });
    await expectedCode(
      loadLandmarkModelAssets(withoutCrypto),
      "landmark.models.environment.unsupported",
    );
    expect(withoutCrypto.fetch).not.toHaveBeenCalled();
    const subtle = mockSubtle(vi.fn(() => Promise.reject(new Error("implementation detail"))));
    await expectedCode(
      loadLandmarkModelAssets(environment({ subtle })),
      "landmark.models.environment.unsupported",
    );
  });

  it("sanitizes network and response failures", async () => {
    for (const fetch of [
      vi.fn(() => Promise.reject(new Error(`failed at ${HAND_URL}`))),
      vi.fn(() => Promise.resolve(new Response(null, { status: 503 }))),
    ]) {
      const promise = loadLandmarkModelAssets(environment({ fetch }));
      await expectedCode(promise, "landmark.models.unavailable");
      await expect(promise).rejects.not.toThrow("models.example");
    }
  });
});
