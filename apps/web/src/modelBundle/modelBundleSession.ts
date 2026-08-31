import Ajv2020 from "ajv/dist/2020.js";

import candidateDecisionPolicy from "../../../../docs/reports/popsign-constructed-calibration-policy-v1.json";
import candidateEventConfig from "../../../../configs/evaluation/candidate-event-detector-v1.json";
import candidateFeaturePlan from "../../../../src/signlab/resources/features/config/hand-local-64-1.default.json";
import mediaPipeConfig from "../../../../src/signlab/resources/extraction/config/mediapipe-extraction-config-1.default.json";
import manifestSchema from "../../../../src/signlab/resources/model_bundles/schemas/browser-model-bundle-manifest-1.schema.json";
import qualityPolicy from "../../../../src/signlab/resources/quality/config/landmark-quality-policy-1.default.json";
import {
  ModelBundleCache,
  type CachedModelBundle,
  type ModelBundleCacheState,
} from "./modelBundleCache";

const LABELS = ["hello", "no", "please", "thank_you", "yes", "other"] as const;
const ASSETS = [
  ["decision_policy", "decision-policy.json", "application/json"],
  ["feature_plan", "feature-plan.json", "application/json"],
  ["golden_smoke", "golden/smoke.json", "application/json"],
  ["landmarker", "landmarker.json", "application/json"],
  ["model", "model.onnx", "application/onnx"],
  ["model_card", "model-card.md", "text/markdown"],
  ["quality_policy", "quality-policy.json", "application/json"],
  ["segmenter", "segmenter.json", "application/json"],
] as const;
const COMPONENTS = {
  landmarker_sha256: "sha256:7343cd8bb724313b4063a3ebd5d7f7470a78b00f2eeda275a15e5f9b2e66e94c",
  quality_policy_sha256: "sha256:680b0904e1cc5d8e03119032e92920a3a0185917a600c4293323b7925da9a545",
  feature_plan_sha256: "sha256:1c62d2738ce0609168967b675fa0dcd1797f8fbe881cd9b5c775d4e2a83e4a3e",
  segmenter_sha256: "sha256:0443badf68d34347a00096682cf049b6f49b5253c12e47bf61b068a597aa162d",
  decision_policy_sha256: "sha256:6eb700443dbb50de5094868d564e8a72aff6af0182c8394ab8c6a28844572e41",
} as const;
const TAXONOMY_SHA256 = "sha256:c0f6cbddfe43e3a6eb3de01dbbbbc1ceebcb83d50cc197999776f58e3d9ce20d";
const ONNX = {
  format: "onnx",
  opset: 18,
  input_name: "input",
  input_shape: [1, 64, 126],
  input_dtype: "float32",
  input_semantics: "hand_local_feature_sequence",
  output_name: "probabilities",
  output_shape: [1, 6],
  output_dtype: "float32",
  output_semantics: "uncalibrated_class_probabilities",
} as const;
const LICENSES = [
  { scope: "mediapipe", spdx: "Apache-2.0", distribution: "redistributable" },
  {
    scope: "popsign_source_data",
    spdx: "CC-BY-4.0",
    distribution: "redistributable_with_attribution",
  },
  { scope: "signlab_code", spdx: "MIT", distribution: "redistributable" },
  { scope: "trained_model", spdx: "NOASSERTION", distribution: "local_evaluation_only" },
] as const;
const PINNED_COMPONENTS = {
  decision_policy: candidateDecisionPolicy,
  feature_plan: candidateFeaturePlan,
  landmarker: mediaPipeConfig,
  quality_policy: qualityPolicy,
  segmenter: candidateEventConfig,
} as const;

export type ModelBundleAssetRole = (typeof ASSETS)[number][0];
export type ModelBundlePhase = "idle" | "loading" | "verifying" | "ready" | "error";
export type ModelBundleSource = "network" | "cache" | "fallback" | "rollback";
export type ModelBundleFailureCode =
  | "bundle.environment.unsupported"
  | "bundle.load.superseded"
  | "bundle.manifest.unavailable"
  | "bundle.manifest.invalid"
  | "bundle.manifest.unsupported"
  | "bundle.asset.unavailable"
  | "bundle.asset.size_mismatch"
  | "bundle.asset.digest_mismatch"
  | "bundle.component.incompatible"
  | "bundle.cache.corrupt"
  | "bundle.cache.rollback_unavailable";

interface ManifestAsset {
  readonly artifact_id: string;
  readonly role: ModelBundleAssetRole;
  readonly media_type: string;
  readonly sha256: string;
  readonly size_bytes: number;
  readonly locator: { readonly kind: "workspace_relative"; readonly path: string };
}

export interface ModelBundleManifest {
  readonly format: "browser-model-bundle/1";
  readonly bundle_id: string;
  readonly version: string;
  readonly candidate: Readonly<Record<string, string>>;
  readonly components: typeof COMPONENTS;
  readonly onnx: typeof ONNX;
  readonly labels: readonly string[];
  readonly licenses: readonly unknown[];
  readonly assets: readonly ManifestAsset[];
}

export interface VerifiedModelBundle {
  readonly id: string;
  readonly version: string;
  readonly manifest: ModelBundleManifest;
  readonly manifestBytes: Blob;
  readonly manifestSha256: string;
  readonly bytesByRole: Readonly<Record<ModelBundleAssetRole, Blob>>;
}

export interface ModelBundleStatus {
  readonly phase: ModelBundlePhase;
  readonly active: { readonly id: string; readonly version: string } | null;
  readonly failureReason?: string;
  readonly source?: ModelBundleSource;
  readonly rollbackAvailable?: boolean;
  readonly cacheWarning?: string;
}

export interface ModelBundleEnvironment {
  readonly fetch: (input: URL) => Promise<Response>;
  readonly subtle?: SubtleCrypto;
  readonly documentBaseUrl?: string;
  readonly cacheStorage?: Pick<CacheStorage, "open">;
}

const FAILURE_MESSAGES: Record<ModelBundleFailureCode, string> = {
  "bundle.environment.unsupported": "This browser cannot verify model files.",
  "bundle.load.superseded": "A newer model bundle load replaced this request.",
  "bundle.manifest.unavailable": "The model bundle manifest could not be loaded.",
  "bundle.manifest.invalid": "The model bundle manifest is incomplete or invalid.",
  "bundle.manifest.unsupported": "This model bundle version is not supported.",
  "bundle.asset.unavailable": "A required model bundle file could not be loaded.",
  "bundle.asset.size_mismatch": "A model bundle file has the wrong size.",
  "bundle.asset.digest_mismatch": "A model bundle file failed its integrity check.",
  "bundle.component.incompatible": "The model bundle is incompatible with this demo.",
  "bundle.cache.corrupt": "The cached model bundle is incomplete or damaged.",
  "bundle.cache.rollback_unavailable": "No verified previous model bundle is available.",
};
const CACHE_WARNING =
  "Offline model storage is unavailable, so this model may need to be downloaded again.";

export class ModelBundleLoadError extends Error {
  constructor(readonly code: ModelBundleFailureCode) {
    super(FAILURE_MESSAGES[code]);
    this.name = "ModelBundleLoadError";
  }
}

const schemaValidator = new Ajv2020({ allErrors: false });
// Pydantic emits this annotation; the surrounding oneOf still performs the validation.
schemaValidator.addKeyword({ keyword: "discriminator", schemaType: "object", valid: true });
const validateSchema = schemaValidator.compile(manifestSchema);

function equalJson(value: unknown, expected: unknown): boolean {
  if (Array.isArray(expected)) {
    return (
      Array.isArray(value) &&
      value.length === expected.length &&
      expected.every((item, index) => equalJson(value[index], item))
    );
  }
  if (typeof expected === "object" && expected !== null) {
    if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
    const actual = value as Record<string, unknown>;
    const entries = Object.entries(expected);
    return (
      Object.keys(actual).length === entries.length &&
      entries.every(([key, item]) => equalJson(actual[key], item))
    );
  }
  return value === expected;
}

function deepFreeze<T>(value: T): Readonly<T> {
  if (typeof value === "object" && value !== null && !Object.isFrozen(value)) {
    Object.values(value).forEach((nested) => deepFreeze(nested));
    Object.freeze(value);
  }
  return value;
}

function fail(code: ModelBundleFailureCode): never {
  throw new ModelBundleLoadError(code);
}

function validateManifest(value: unknown): ModelBundleManifest {
  if (typeof value === "object" && value !== null && "format" in value) {
    if ((value as { format?: unknown }).format !== "browser-model-bundle/1") {
      fail("bundle.manifest.unsupported");
    }
  }
  if (!validateSchema(value)) fail("bundle.manifest.invalid");
  const manifest = value as unknown as ModelBundleManifest;
  if (
    !equalJson(manifest.labels, LABELS) ||
    !equalJson(manifest.components, COMPONENTS) ||
    !equalJson(manifest.onnx, ONNX) ||
    !equalJson(manifest.licenses, LICENSES) ||
    manifest.candidate.input_feature_plan_sha256 !== COMPONENTS.feature_plan_sha256 ||
    manifest.candidate.taxonomy_sha256 !== TAXONOMY_SHA256 ||
    manifest.assets.length !== ASSETS.length
  ) {
    fail("bundle.component.incompatible");
  }
  manifest.assets.forEach((asset, index) => {
    const expected = ASSETS[index];
    if (
      expected === undefined ||
      asset.role !== expected[0] ||
      asset.artifact_id !== expected[0] ||
      asset.locator.kind !== "workspace_relative" ||
      asset.locator.path !== expected[1] ||
      asset.media_type !== expected[2] ||
      asset.size_bytes <= 0
    ) {
      fail("bundle.manifest.invalid");
    }
  });
  return deepFreeze(manifest);
}

function bundleManifestUrl(base: string, fallbackBase?: string): URL {
  try {
    const url = new URL(base, fallbackBase);
    if (
      !["http:", "https:"].includes(url.protocol) ||
      url.username !== "" ||
      url.password !== "" ||
      url.search !== "" ||
      url.hash !== ""
    ) {
      fail("bundle.manifest.invalid");
    }
    if (!url.pathname.endsWith("/")) url.pathname += "/";
    return new URL("manifest.json", url);
  } catch (error) {
    if (error instanceof ModelBundleLoadError) throw error;
    return fail("bundle.manifest.invalid");
  }
}

function assetUrl(manifestUrl: URL, path: string): URL {
  const base = new URL(".", manifestUrl);
  const url = new URL(path, base);
  if (url.origin !== base.origin || !url.pathname.startsWith(base.pathname)) {
    fail("bundle.manifest.invalid");
  }
  return url;
}

function parseJson(buffer: ArrayBuffer, code: ModelBundleFailureCode): unknown {
  try {
    return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(buffer)) as unknown;
  } catch {
    return fail(code);
  }
}

function verifyComponents(
  manifest: ModelBundleManifest,
  buffers: Readonly<Record<ModelBundleAssetRole, ArrayBuffer>>,
): void {
  for (const [role, expected] of Object.entries(PINNED_COMPONENTS)) {
    const parsed = parseJson(
      buffers[role as ModelBundleAssetRole],
      "bundle.component.incompatible",
    );
    if (!equalJson(parsed, expected)) fail("bundle.component.incompatible");
  }
  const identities = candidateDecisionPolicy.identities;
  const candidate = manifest.candidate;
  if (
    candidate.research_checkpoint_sha256 !== identities.model_sha256 ||
    candidate.configuration_sha256 !== identities.configuration_sha256 ||
    candidate.corpus_sha256 !== identities.corpus_sha256 ||
    candidate.derivative_set_sha256 !== identities.derivative_set_sha256 ||
    candidate.split_sha256 !== identities.split_sha256 ||
    candidate.input_feature_plan_sha256 !== identities.input_feature_plan_sha256 ||
    candidate.source_feature_plan_sha256 !== identities.source_feature_plan_sha256 ||
    candidate.taxonomy_sha256 !== identities.taxonomy_sha256
  ) {
    fail("bundle.component.incompatible");
  }
}

function statusFor(
  phase: ModelBundlePhase,
  active: VerifiedModelBundle | null,
  failureReason?: string,
  source?: ModelBundleSource,
  rollbackAvailable?: boolean,
  cacheWarning?: string,
): ModelBundleStatus {
  return Object.freeze({
    phase,
    active: active === null ? null : Object.freeze({ id: active.id, version: active.version }),
    ...(failureReason === undefined ? {} : { failureReason }),
    ...(source === undefined ? {} : { source }),
    ...(rollbackAvailable === undefined ? {} : { rollbackAvailable }),
    ...(cacheWarning === undefined ? {} : { cacheWarning }),
  });
}

interface PreparedManifest {
  readonly manifest: ModelBundleManifest;
  readonly buffer: ArrayBuffer;
  readonly sha256: string;
}

type StatusUpdate = (
  phase: ModelBundlePhase,
  reason?: string,
  source?: ModelBundleSource,
  rollbackAvailable?: boolean,
  cacheWarning?: string | null,
) => void;

export class ModelBundleSession {
  private activeValue: VerifiedModelBundle | null = null;
  private statusValue = statusFor("idle", null);
  private attempt = 0;
  private readonly cache: ModelBundleCache;
  private cacheMutation = Promise.resolve();

  constructor(
    private readonly environment: ModelBundleEnvironment = {
      fetch: (input) => globalThis.fetch(input),
      subtle: globalThis.crypto?.subtle,
      documentBaseUrl: globalThis.document?.baseURI,
      cacheStorage: globalThis.caches,
    },
  ) {
    this.cache = new ModelBundleCache(environment.cacheStorage);
  }

  get active(): VerifiedModelBundle | null {
    return this.activeValue;
  }

  get status(): ModelBundleStatus {
    return this.statusValue;
  }

  async load(
    baseUrl: string,
    onStatus: (status: ModelBundleStatus) => void = () => undefined,
  ): Promise<VerifiedModelBundle> {
    const attempt = ++this.attempt;
    const update = this.updater(attempt, onStatus);

    try {
      update("loading");
      const manifestUrl = bundleManifestUrl(baseUrl, this.environment.documentBaseUrl);
      let manifestBuffer: ArrayBuffer;
      try {
        manifestBuffer = await this.fetchBytes(manifestUrl, "bundle.manifest.unavailable");
      } catch (error) {
        return await this.restoreActive(error, update, attempt);
      }
      const prepared = await this.prepareManifest(manifestBuffer);

      const warm = await this.cache.read(prepared.sha256);
      if (warm !== null) {
        try {
          const verified = await this.verifyCached(warm, prepared.sha256, update);
          const activation = await this.mutateCache(attempt, () =>
            this.cache.activate(verified.manifestSha256),
          );
          return this.activate(
            verified,
            "cache",
            activation.saved ? activation.state : null,
            update,
            attempt,
            activation.saved ? undefined : CACHE_WARNING,
          );
        } catch (error) {
          if (this.failureCode(error) !== "bundle.cache.corrupt") throw error;
        }
      }

      let verified: VerifiedModelBundle;
      try {
        verified = await this.verifyBundle(
          prepared,
          async (asset) =>
            await this.fetchBytes(
              assetUrl(manifestUrl, asset.locator.path),
              "bundle.asset.unavailable",
            ),
          update,
        );
      } catch (error) {
        if (this.failureCode(error) === "bundle.asset.unavailable") {
          return await this.restoreActive(error, update, attempt);
        }
        throw error;
      }
      const activation = await this.mutateCache(attempt, () =>
        this.cache.saveAndActivate({
          identity: verified.manifestSha256,
          manifest: verified.manifestBytes,
          bytesByRole: verified.bytesByRole,
        }),
      );
      return this.activate(
        verified,
        "network",
        activation.saved ? activation.state : null,
        update,
        attempt,
        activation.saved ? undefined : CACHE_WARNING,
      );
    } catch (error) {
      return this.finishFailure(error, "bundle.asset.unavailable", attempt, onStatus);
    }
  }

  async rollback(
    onStatus: (status: ModelBundleStatus) => void = () => undefined,
  ): Promise<VerifiedModelBundle> {
    const attempt = ++this.attempt;
    const update = this.updater(attempt, onStatus);
    try {
      update("loading", undefined, "rollback");
      const cached = await this.cache.readSelected("previous");
      if (cached === null) fail("bundle.cache.rollback_unavailable");
      const verified = await this.verifyCached(cached, cached.identity, update);
      const state = await this.mutateCache(attempt, () => this.cache.rollback(cached.identity));
      if (state === null) fail("bundle.cache.rollback_unavailable");
      return this.activate(verified, "rollback", state, update, attempt);
    } catch (error) {
      return this.finishFailure(error, "bundle.cache.rollback_unavailable", attempt, onStatus);
    }
  }

  private async prepareManifest(buffer: ArrayBuffer): Promise<PreparedManifest> {
    const manifest = validateManifest(parseJson(buffer, "bundle.manifest.invalid"));
    return { manifest, buffer, sha256: await this.digest(buffer) };
  }

  private async verifyBundle(
    prepared: PreparedManifest,
    readAsset: (asset: ManifestAsset) => Promise<ArrayBuffer>,
    update: StatusUpdate,
  ): Promise<VerifiedModelBundle> {
    const downloads = await Promise.all(
      prepared.manifest.assets.map(async (asset) => ({ asset, buffer: await readAsset(asset) })),
    );
    update("verifying");
    const buffers = {} as Record<ModelBundleAssetRole, ArrayBuffer>;
    for (const { asset, buffer } of downloads) {
      if (buffer.byteLength !== asset.size_bytes) fail("bundle.asset.size_mismatch");
      if ((await this.digest(buffer)) !== asset.sha256) fail("bundle.asset.digest_mismatch");
      buffers[asset.role] = buffer;
    }
    verifyComponents(prepared.manifest, buffers);
    const bytesByRole = Object.freeze(
      Object.fromEntries(
        prepared.manifest.assets.map((asset) => [
          asset.role,
          new Blob([buffers[asset.role]], { type: asset.media_type }),
        ]),
      ) as Record<ModelBundleAssetRole, Blob>,
    );
    return Object.freeze({
      id: prepared.manifest.bundle_id,
      version: prepared.manifest.version,
      manifest: prepared.manifest,
      manifestBytes: new Blob([prepared.buffer], { type: "application/json" }),
      manifestSha256: prepared.sha256,
      bytesByRole,
    });
  }

  private async verifyCached(
    cached: CachedModelBundle,
    expectedIdentity: string,
    update: StatusUpdate,
  ): Promise<VerifiedModelBundle> {
    try {
      const prepared = await this.prepareManifest(await cached.manifest.arrayBuffer());
      if (prepared.sha256 !== expectedIdentity) fail("bundle.cache.corrupt");
      return await this.verifyBundle(
        prepared,
        async (asset) => {
          const bytes = await cached.asset(asset.role);
          if (bytes === null) fail("bundle.cache.corrupt");
          return await bytes.arrayBuffer();
        },
        update,
      );
    } catch (error) {
      const code = this.failureCode(error);
      if (code === "bundle.environment.unsupported" || code === "bundle.load.superseded") {
        throw error;
      }
      return fail("bundle.cache.corrupt");
    }
  }

  private async restoreActive(
    error: unknown,
    update: StatusUpdate,
    attempt: number,
  ): Promise<VerifiedModelBundle> {
    const cached = await this.cache.readSelected("active");
    if (cached === null) throw error;
    update(
      "loading",
      undefined,
      "fallback",
      cached.state?.previous !== null && cached.state?.previous !== undefined,
    );
    return this.activate(
      await this.verifyCached(cached, cached.identity, update),
      "fallback",
      cached.state,
      update,
      attempt,
    );
  }

  private activate(
    verified: VerifiedModelBundle,
    source: ModelBundleSource,
    state: ModelBundleCacheState | null,
    update: StatusUpdate,
    attempt: number,
    cacheWarning?: string,
  ): VerifiedModelBundle {
    if (attempt !== this.attempt) fail("bundle.load.superseded");
    this.activeValue = verified;
    update(
      "ready",
      undefined,
      source,
      state?.previous !== null && state?.previous !== undefined,
      cacheWarning ?? null,
    );
    return verified;
  }

  private updater(attempt: number, onStatus: (status: ModelBundleStatus) => void): StatusUpdate {
    return (
      phase,
      reason,
      source = this.statusValue.source,
      rollbackAvailable = this.statusValue.rollbackAvailable,
      cacheWarning = this.statusValue.cacheWarning,
    ) => {
      if (attempt !== this.attempt) fail("bundle.load.superseded");
      this.statusValue = statusFor(
        phase,
        this.activeValue,
        reason,
        source,
        rollbackAvailable,
        cacheWarning ?? undefined,
      );
      onStatus(this.statusValue);
    };
  }

  private mutateCache<T>(attempt: number, mutation: () => Promise<T>): Promise<T> {
    const result = this.cacheMutation.then(() => {
      if (attempt !== this.attempt) fail("bundle.load.superseded");
      return mutation();
    });
    this.cacheMutation = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  private finishFailure(
    error: unknown,
    fallback: ModelBundleFailureCode,
    attempt: number,
    onStatus: (status: ModelBundleStatus) => void,
  ): never {
    const failure =
      error instanceof ModelBundleLoadError ? error : new ModelBundleLoadError(fallback);
    if (attempt === this.attempt) {
      this.statusValue = statusFor(
        "error",
        this.activeValue,
        failure.message,
        this.statusValue.source,
        this.statusValue.rollbackAvailable,
        this.statusValue.cacheWarning,
      );
      onStatus(this.statusValue);
    }
    throw failure;
  }

  private async digest(buffer: ArrayBuffer): Promise<string> {
    const subtle = this.environment.subtle;
    if (subtle === undefined) fail("bundle.environment.unsupported");
    try {
      const digest = await subtle.digest("SHA-256", new Uint8Array(buffer));
      return `sha256:${Array.from(new Uint8Array(digest), (byte) =>
        byte.toString(16).padStart(2, "0"),
      ).join("")}`;
    } catch {
      return fail("bundle.environment.unsupported");
    }
  }

  private failureCode(error: unknown): ModelBundleFailureCode | undefined {
    return error instanceof ModelBundleLoadError ? error.code : undefined;
  }

  private async fetchBytes(url: URL, code: ModelBundleFailureCode): Promise<ArrayBuffer> {
    try {
      const response = await this.environment.fetch(url);
      if (!response.ok) fail(code);
      return await response.arrayBuffer();
    } catch (error) {
      if (error instanceof ModelBundleLoadError) throw error;
      return fail(code);
    }
  }
}
