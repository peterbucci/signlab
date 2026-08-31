import { createHash, webcrypto } from "node:crypto";
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
  type VerifiedModelBundle,
} from "./modelBundleSession";

const BASE_URL = "https://example.test/models/candidate/";
const MANIFEST_URL = `${BASE_URL}manifest.json`;
const CACHE_POINTER = "https://model-bundle-cache.signlab.invalid/v1/pointer";
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

class MemoryCacheStorage {
  readonly entries = new Map<string, Response>();
  failPointerWrites = false;
  failDeletes = false;
  pointerWrites = 0;
  deleteAttempts = 0;
  pointerWriteGate?: Promise<void>;
  readonly cache = {
    match: vi.fn((request: RequestInfo | URL) =>
      Promise.resolve(this.entries.get(this.key(request))?.clone()),
    ),
    put: vi.fn(async (request: RequestInfo | URL, response: Response) => {
      const key = this.key(request);
      if (key === CACHE_POINTER) {
        this.pointerWrites += 1;
        await this.pointerWriteGate;
      }
      if (this.failPointerWrites && key === CACHE_POINTER) {
        throw new DOMException("quota reached", "QuotaExceededError");
      }
      this.entries.set(key, response.clone());
    }),
    keys: vi.fn(() => Promise.resolve([...this.entries.keys()].map((key) => new Request(key)))),
    delete: vi.fn((request: RequestInfo | URL) => {
      this.deleteAttempts += 1;
      if (this.failDeletes) return Promise.reject(new Error("simulated cleanup failure"));
      return Promise.resolve(this.entries.delete(this.key(request)));
    }),
  } as unknown as Cache;
  readonly storage = {
    open: vi.fn(() => Promise.resolve(this.cache)),
  } as unknown as CacheStorage;

  seedPointer(value: unknown) {
    this.entries.set(CACHE_POINTER, new Response(JSON.stringify(value)));
  }

  async pointer(): Promise<Record<string, unknown> | null> {
    const response = this.entries.get(CACHE_POINTER);
    return response === undefined
      ? null
      : ((await response.clone().json()) as Record<string, unknown>);
  }

  replace(identity: string, role: ModelBundleAssetRole, body: BodyInit) {
    const encoded = encodeURIComponent(identity);
    const key = [...this.entries.keys()].find(
      (candidate) => candidate.includes(`/bundles/${encoded}/`) && candidate.endsWith(`/${role}`),
    );
    if (key === undefined) throw new Error(`Missing cached ${role}`);
    this.entries.set(key, new Response(body));
  }

  private key(request: RequestInfo | URL): string {
    if (typeof request === "string") return request;
    return request instanceof URL ? request.href : request.url;
  }
}

async function fixture(version = manifestExample.version) {
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
  manifest.version = version;
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
  cache?: MemoryCacheStorage,
) {
  const requests: string[] = [];
  const responses = { ...bytes };
  const failures = new Set<ModelBundleAssetRole>();
  const network = { manifestUnavailable: false };
  const fetch = vi.fn((input: URL): Promise<Response> => {
    requests.push(input.href);
    if (input.href === MANIFEST_URL) {
      if (network.manifestUnavailable) {
        return Promise.reject(new TypeError("simulated offline endpoint"));
      }
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
    cacheStorage: cache?.storage,
  };
  return {
    session: new ModelBundleSession(environment),
    fetch,
    requests,
    responses,
    failures,
    network,
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
      source: "network",
      rollbackAvailable: false,
      cacheWarning:
        "Offline model storage is unavailable, so this model may need to be downloaded again.",
    });
  });

  it("stores exact verified manifest bytes and reuses matching cached assets", async () => {
    const data = await fixture();
    const cache = new MemoryCacheStorage();
    const first = harness(data.manifest, data.bytes, data.manifest, cache);
    const manifestText = JSON.stringify(data.manifest);

    const saved = await first.session.load(BASE_URL);

    expect(await saved.manifestBytes.text()).toBe(manifestText);
    expect(saved.manifestSha256).toBe(
      `sha256:${createHash("sha256")
        .update(await saved.manifestBytes.text())
        .digest("hex")}`,
    );
    expect(await cache.pointer()).toMatchObject({
      format: "signlab-model-bundle-pointer/1",
      active: saved.manifestSha256,
      previous: null,
    });
    expect(first.session.status).toMatchObject({
      phase: "ready",
      source: "network",
      rollbackAvailable: false,
    });
    expect(first.session.status.cacheWarning).toBeUndefined();

    const warm = harness(data.manifest, data.bytes, data.manifest, cache);
    const reused = await warm.session.load(BASE_URL);

    expect(reused.manifestSha256).toBe(saved.manifestSha256);
    expect(warm.requests).toEqual([MANIFEST_URL]);
    expect(warm.session.status).toMatchObject({ phase: "ready", source: "cache" });
  });

  it("restores and re-verifies the active cache when the model endpoint is unavailable", async () => {
    const data = await fixture();
    const cache = new MemoryCacheStorage();
    const online = harness(data.manifest, data.bytes, data.manifest, cache);
    const saved = await online.session.load(BASE_URL);
    const offline = harness(data.manifest, data.bytes, data.manifest, cache);
    offline.network.manifestUnavailable = true;

    const restored = await offline.session.load(BASE_URL);

    expect(restored.manifestSha256).toBe(saved.manifestSha256);
    expect(offline.requests).toEqual([MANIFEST_URL]);
    expect(offline.session.status).toMatchObject({
      phase: "ready",
      source: "fallback",
      rollbackAvailable: false,
    });
  });

  it("repairs a corrupt warm entry from verified network bytes", async () => {
    const data = await fixture();
    const cache = new MemoryCacheStorage();
    const saved = await harness(data.manifest, data.bytes, data.manifest, cache).session.load(
      BASE_URL,
    );
    cache.replace(saved.manifestSha256, "model", "damaged cache bytes");
    const warm = harness(data.manifest, data.bytes, data.manifest, cache);

    const repaired = await warm.session.load(BASE_URL);

    expect(repaired.manifestSha256).toBe(saved.manifestSha256);
    expect(warm.requests).toHaveLength(9);
    expect(warm.session.status).toMatchObject({ phase: "ready", source: "network" });
    const offline = harness(data.manifest, data.bytes, data.manifest, cache);
    offline.network.manifestUnavailable = true;
    await expect(offline.session.load(BASE_URL)).resolves.toMatchObject({
      manifestSha256: saved.manifestSha256,
    });
  });

  it("retains one previous bundle and verifies it before an explicit rollback", async () => {
    const cache = new MemoryCacheStorage();
    const version1 = await fixture("1.0.0");
    const first = harness(version1.manifest, version1.bytes, version1.manifest, cache);
    const saved1 = await first.session.load(BASE_URL);
    const version2 = await fixture("1.1.0");
    const second = harness(version2.manifest, version2.bytes, version2.manifest, cache);
    const saved2 = await second.session.load(BASE_URL);

    expect(second.session.status.rollbackAvailable).toBe(true);
    cache.failPointerWrites = true;
    const warm = harness(version2.manifest, version2.bytes, version2.manifest, cache);
    await warm.session.load(BASE_URL);
    expect(warm.session.status).toMatchObject({ rollbackAvailable: true });
    expect(warm.session.status.cacheWarning).toBeUndefined();
    cache.failPointerWrites = false;
    const restored = await second.session.rollback();

    expect(restored.version).toBe("1.0.0");
    expect(second.session.status).toMatchObject({
      phase: "ready",
      source: "rollback",
      rollbackAvailable: true,
    });
    expect(await cache.pointer()).toMatchObject({
      active: saved1.manifestSha256,
      previous: saved2.manifestSha256,
    });
  });

  it("keeps rollback available when an offline active cache entry is corrupt", async () => {
    const cache = new MemoryCacheStorage();
    const version1 = await fixture("1.0.0");
    await harness(version1.manifest, version1.bytes, version1.manifest, cache).session.load(
      BASE_URL,
    );
    const version2 = await fixture("1.1.0");
    const online = harness(version2.manifest, version2.bytes, version2.manifest, cache);
    const active = await online.session.load(BASE_URL);
    cache.replace(active.manifestSha256, "model", "damaged cache bytes");
    const offline = harness(version2.manifest, version2.bytes, version2.manifest, cache);
    offline.network.manifestUnavailable = true;

    await expect(offline.session.load(BASE_URL)).rejects.toMatchObject({
      code: "bundle.cache.corrupt",
    });
    expect(offline.session.status).toMatchObject({
      phase: "error",
      source: "fallback",
      rollbackAvailable: true,
    });

    await expect(offline.session.rollback()).resolves.toMatchObject({ version: "1.0.0" });
    expect(offline.session.status.source).toBe("rollback");
  });

  it("keeps an unpersisted network replacement ready without offering the wrong rollback", async () => {
    const cache = new MemoryCacheStorage();
    const version1 = await fixture("1.0.0");
    const persisted = await harness(
      version1.manifest,
      version1.bytes,
      version1.manifest,
      cache,
    ).session.load(BASE_URL);
    cache.failPointerWrites = true;
    const version2 = await fixture("1.1.0");
    const test = harness(version2.manifest, version2.bytes, version2.manifest, cache);

    await expect(test.session.load(BASE_URL)).resolves.toMatchObject({ version: "1.1.0" });
    expect(test.session.status).toMatchObject({
      phase: "ready",
      source: "network",
      rollbackAvailable: false,
    });
    expect(test.session.status.cacheWarning).toMatch(/Offline model storage is unavailable/);
    expect(await cache.pointer()).toMatchObject({
      active: persisted.manifestSha256,
      previous: null,
    });
    await expect(test.session.rollback()).rejects.toMatchObject({
      code: "bundle.cache.rollback_unavailable",
    });
  });

  it("leaves unsupported pointer metadata untouched and reports a recoverable warning", async () => {
    const data = await fixture();
    const cache = new MemoryCacheStorage();
    const futurePointer = {
      format: "signlab-model-bundle-pointer/2",
      active: `sha256:${"a".repeat(64)}`,
      previous: null,
    };
    cache.seedPointer(futurePointer);
    const test = harness(data.manifest, data.bytes, data.manifest, cache);

    await expect(test.session.load(BASE_URL)).resolves.toMatchObject({
      version: data.manifest.version,
    });
    expect(await cache.pointer()).toEqual(futurePointer);
    expect(cache.entries.size).toBe(1);
    expect(test.session.status.cacheWarning).toMatch(/Offline model storage is unavailable/);
  });

  it("keeps only active and previous entries while cleanup failure stays recoverable", async () => {
    const cache = new MemoryCacheStorage();
    const saved: VerifiedModelBundle[] = [];
    for (const version of ["1.0.0", "1.1.0", "1.2.0"]) {
      const data = await fixture(version);
      saved.push(
        await harness(data.manifest, data.bytes, data.manifest, cache).session.load(BASE_URL),
      );
    }
    expect(
      [...cache.entries.keys()].some((key) =>
        key.includes(encodeURIComponent(saved[0]!.manifestSha256)),
      ),
    ).toBe(false);

    cache.failDeletes = true;
    const version4 = await fixture("1.3.0");
    const latest = harness(version4.manifest, version4.bytes, version4.manifest, cache);
    const active = await latest.session.load(BASE_URL);

    expect(await cache.pointer()).toMatchObject({
      active: active.manifestSha256,
      previous: saved[2]!.manifestSha256,
    });
    expect(cache.deleteAttempts).toBeGreaterThan(0);
    expect(latest.session.status).toMatchObject({ phase: "ready", source: "network" });
    expect(latest.session.status.cacheWarning).toBeUndefined();
  });

  it("does not activate a load superseded during the persistent pointer commit", async () => {
    const data = await fixture();
    const cache = new MemoryCacheStorage();
    let releasePointer!: () => void;
    cache.pointerWriteGate = new Promise<void>((resolve) => {
      releasePointer = resolve;
    });
    const test = harness(data.manifest, data.bytes, data.manifest, cache);

    const superseded = test.session.load(BASE_URL);
    await vi.waitFor(() => expect(cache.pointerWrites).toBe(1));
    data.manifest.version = "1.1.0";
    const current = test.session.load(BASE_URL);
    releasePointer();

    await expect(superseded).rejects.toMatchObject({ code: "bundle.load.superseded" });
    const active = await current;
    expect(test.session.active).toBe(active);
    expect(active.version).toBe("1.1.0");
    expect(await cache.pointer()).toMatchObject({ active: active.manifestSha256 });
    expect(test.session.status).toMatchObject({ phase: "ready", active: { id: active.id } });
  });
});
