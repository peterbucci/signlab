import { webcrypto } from "node:crypto";
import { describe, expect, it, vi } from "vitest";

import candidateDecisionPolicy from "../../../../docs/reports/popsign-constructed-calibration-policy-v1.json";
import candidateEventConfig from "../../../../configs/evaluation/candidate-event-detector-v1.json";
import candidateFeaturePlan from "../../../../src/signlab/resources/features/config/hand-local-64-1.default.json";
import mediaPipeConfig from "../../../../src/signlab/resources/extraction/config/mediapipe-extraction-config-1.default.json";
import manifestExample from "../../../../src/signlab/resources/model_bundles/examples/browser-model-bundle-manifest.example.json";
import qualityPolicy from "../../../../src/signlab/resources/quality/config/landmark-quality-policy-1.default.json";
import {
  ModelBundleSession,
  type ModelBundleAssetRole,
  type ModelBundleEnvironment,
  type ModelBundleStatus,
} from "./modelBundleSession";

const BASE_URL = "https://example.test/models/candidate/";
const MANIFEST_URL = `${BASE_URL}manifest.json`;
const encoder = new TextEncoder();

function jsonBytes(value: unknown): Uint8Array {
  return encoder.encode(JSON.stringify(value));
}

function arrayBuffer(bytes: Uint8Array): ArrayBuffer {
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
}

async function sha256(bytes: Uint8Array): Promise<string> {
  const digest = await webcrypto.subtle.digest("SHA-256", arrayBuffer(bytes));
  return `sha256:${Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("")}`;
}

async function fixture() {
  const bytes: Record<ModelBundleAssetRole, Uint8Array> = {
    decision_policy: jsonBytes(candidateDecisionPolicy),
    feature_plan: jsonBytes(candidateFeaturePlan),
    golden_smoke: jsonBytes({ format: "synthetic-golden-smoke/1" }),
    landmarker: jsonBytes(mediaPipeConfig),
    model: encoder.encode("tiny structural ONNX fixture"),
    model_card: encoder.encode("# Synthetic model card\n"),
    quality_policy: jsonBytes(qualityPolicy),
    segmenter: jsonBytes(candidateEventConfig),
  };
  const manifest = structuredClone(manifestExample);
  for (const asset of manifest.assets) {
    const raw = bytes[asset.role as ModelBundleAssetRole];
    asset.size_bytes = raw.byteLength;
    asset.sha256 = await sha256(raw);
  }
  return { manifest, bytes };
}

function harness(
  manifest: typeof manifestExample,
  bytes: Record<ModelBundleAssetRole, Uint8Array>,
  manifestBody: unknown = manifest,
) {
  const requests: string[] = [];
  const responses = { ...bytes };
  const failures = new Set<ModelBundleAssetRole>();
  const fetch = vi.fn((input: URL): Promise<Response> => {
    requests.push(input.href);
    if (input.href === MANIFEST_URL) {
      const body = typeof manifestBody === "string" ? manifestBody : JSON.stringify(manifestBody);
      return Promise.resolve(new Response(body, { status: 200 }));
    }
    const asset = manifest.assets.find(
      (candidate) => new URL(candidate.locator.path, MANIFEST_URL).href === input.href,
    );
    if (asset === undefined) return Promise.resolve(new Response(null, { status: 404 }));
    const role = asset.role as ModelBundleAssetRole;
    if (failures.has(role)) return Promise.reject(new TypeError("simulated network interruption"));
    return Promise.resolve(new Response(arrayBuffer(responses[role]), { status: 200 }));
  });
  const environment: ModelBundleEnvironment = {
    fetch,
    subtle: webcrypto.subtle as SubtleCrypto,
  };
  return {
    session: new ModelBundleSession(environment),
    fetch,
    requests,
    responses,
    failures,
  };
}

describe("ModelBundleSession", () => {
  it("loads the manifest first and atomically exposes immutable verified bytes", async () => {
    const data = await fixture();
    const test = harness(data.manifest, data.bytes);
    const statuses: ModelBundleStatus[] = [];

    const loaded = await test.session.load(BASE_URL, (status) => statuses.push(status));

    expect(test.requests[0]).toBe(MANIFEST_URL);
    expect(test.requests).toHaveLength(9);
    expect(statuses.map((status) => status.phase)).toEqual(["loading", "verifying", "ready"]);
    expect(loaded).toBe(test.session.active);
    expect(loaded.id).toBe(data.manifest.bundle_id);
    expect(loaded.version).toBe(data.manifest.version);
    expect(Object.isFrozen(loaded)).toBe(true);
    expect(Object.isFrozen(loaded.manifest)).toBe(true);
    expect(Object.isFrozen(loaded.bytesByRole)).toBe(true);
    expect(await loaded.bytesByRole.model.text()).toBe("tiny structural ONNX fixture");
  });

  it.each([
    [
      "unsupported version",
      (value: typeof manifestExample) => (value.format = "browser-model-bundle/2"),
      "bundle.manifest.unsupported",
    ],
    [
      "missing role",
      (value: typeof manifestExample) => value.assets.pop(),
      "bundle.manifest.invalid",
    ],
    [
      "duplicate role",
      (value: typeof manifestExample) => (value.assets[1] = structuredClone(value.assets[0]!)),
      "bundle.manifest.invalid",
    ],
    [
      "wrong asset identity",
      (value: typeof manifestExample) => (value.assets[0]!.artifact_id = "different_asset"),
      "bundle.manifest.invalid",
    ],
    [
      "unsafe path",
      (value: typeof manifestExample) => (value.assets[4]!.locator.path = "../model.onnx"),
      "bundle.manifest.invalid",
    ],
    [
      "wrong label order",
      (value: typeof manifestExample) => value.labels.reverse(),
      "bundle.component.incompatible",
    ],
    [
      "wrong ONNX shape",
      (value: typeof manifestExample) => (value.onnx.input_shape[2] = 127),
      "bundle.manifest.invalid",
    ],
    [
      "wrong component identity",
      (value: typeof manifestExample) =>
        (value.components.landmarker_sha256 = `sha256:${"0".repeat(64)}`),
      "bundle.component.incompatible",
    ],
  ])("rejects a %s before requesting assets", async (_name, mutate, code) => {
    const data = await fixture();
    mutate(data.manifest);
    const test = harness(data.manifest, data.bytes);

    await expect(test.session.load(BASE_URL)).rejects.toMatchObject({ code });
    expect(test.requests).toEqual([MANIFEST_URL]);
    expect(test.session.active).toBeNull();
  });

  it("rejects malformed manifest JSON before requesting assets", async () => {
    const data = await fixture();
    const test = harness(data.manifest, data.bytes, "{broken");

    await expect(test.session.load(BASE_URL)).rejects.toMatchObject({
      code: "bundle.manifest.invalid",
    });
    expect(test.requests).toEqual([MANIFEST_URL]);
  });

  it("fails before asset downloads when Web Crypto is unavailable", async () => {
    const data = await fixture();
    const test = harness(data.manifest, data.bytes);
    const session = new ModelBundleSession({ fetch: test.fetch });

    await expect(session.load(BASE_URL)).rejects.toMatchObject({
      code: "bundle.environment.unsupported",
    });
    expect(test.requests).toEqual([MANIFEST_URL]);
  });

  it.each([
    ["size", "bundle.asset.size_mismatch"],
    ["digest", "bundle.asset.digest_mismatch"],
    ["network", "bundle.asset.unavailable"],
  ])("rejects an asset %s failure", async (failure, code) => {
    const data = await fixture();
    const test = harness(data.manifest, data.bytes);
    if (failure === "size") test.responses.model = encoder.encode("short");
    if (failure === "digest") {
      const altered = data.bytes.model.slice();
      altered[0] = altered[0]! ^ 1;
      test.responses.model = altered;
    }
    if (failure === "network") test.failures.add("model");

    await expect(test.session.load(BASE_URL)).rejects.toMatchObject({ code });
    expect(test.session.active).toBeNull();
    expect(test.session.status).toMatchObject({ phase: "error", active: null });
  });

  it("rejects an altered pinned component after its raw-byte checks pass", async () => {
    const data = await fixture();
    const altered = { ...candidateEventConfig, start_motion_q: 250001 };
    data.bytes.segmenter = jsonBytes(altered);
    const segmenter = data.manifest.assets.find((asset) => asset.role === "segmenter")!;
    segmenter.size_bytes = data.bytes.segmenter.byteLength;
    segmenter.sha256 = await sha256(data.bytes.segmenter);
    const test = harness(data.manifest, data.bytes);

    await expect(test.session.load(BASE_URL)).rejects.toMatchObject({
      code: "bundle.component.incompatible",
    });
    expect(test.session.active).toBeNull();
  });

  it("preserves the previous active bundle when a replacement fails", async () => {
    const data = await fixture();
    const test = harness(data.manifest, data.bytes);
    const first = await test.session.load(BASE_URL);
    const altered = test.responses.model.slice();
    altered[0] = altered[0]! ^ 1;
    test.responses.model = altered;

    await expect(test.session.load(BASE_URL)).rejects.toMatchObject({
      code: "bundle.asset.digest_mismatch",
    });

    expect(test.session.active).toBe(first);
    expect(test.session.status).toEqual({
      phase: "error",
      active: { id: first.id, version: first.version },
      failureReason: "A model bundle file failed its integrity check.",
    });
  });
});
