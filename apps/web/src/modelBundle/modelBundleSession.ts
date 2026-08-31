import Ajv2020 from "ajv/dist/2020.js";

import candidateDecisionPolicy from "../../../../docs/reports/popsign-constructed-calibration-policy-v1.json";
import candidateEventConfig from "../../../../configs/evaluation/candidate-event-detector-v1.json";
import candidateFeaturePlan from "../../../../src/signlab/resources/features/config/hand-local-64-1.default.json";
import mediaPipeConfig from "../../../../src/signlab/resources/extraction/config/mediapipe-extraction-config-1.default.json";
import manifestSchema from "../../../../src/signlab/resources/model_bundles/schemas/browser-model-bundle-manifest-1.schema.json";
import qualityPolicy from "../../../../src/signlab/resources/quality/config/landmark-quality-policy-1.default.json";

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
export type ModelBundleFailureCode =
  | "bundle.environment.unsupported"
  | "bundle.load.superseded"
  | "bundle.manifest.unavailable"
  | "bundle.manifest.invalid"
  | "bundle.manifest.unsupported"
  | "bundle.asset.unavailable"
  | "bundle.asset.size_mismatch"
  | "bundle.asset.digest_mismatch"
  | "bundle.component.incompatible";

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
  readonly bytesByRole: Readonly<Record<ModelBundleAssetRole, Blob>>;
}

export interface ModelBundleStatus {
  readonly phase: ModelBundlePhase;
  readonly active: { readonly id: string; readonly version: string } | null;
  readonly failureReason?: string;
}

export interface ModelBundleEnvironment {
  readonly fetch: (input: URL) => Promise<Response>;
  readonly subtle?: SubtleCrypto;
  readonly documentBaseUrl?: string;
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
};

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
): ModelBundleStatus {
  return Object.freeze({
    phase,
    active: active === null ? null : Object.freeze({ id: active.id, version: active.version }),
    ...(failureReason === undefined ? {} : { failureReason }),
  });
}

export class ModelBundleSession {
  private activeValue: VerifiedModelBundle | null = null;
  private statusValue = statusFor("idle", null);
  private attempt = 0;

  constructor(
    private readonly environment: ModelBundleEnvironment = {
      fetch: (input) => globalThis.fetch(input),
      subtle: globalThis.crypto?.subtle,
      documentBaseUrl: globalThis.document?.baseURI,
    },
  ) {}

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
    const update = (phase: ModelBundlePhase, reason?: string) => {
      if (attempt !== this.attempt) fail("bundle.load.superseded");
      this.statusValue = statusFor(phase, this.activeValue, reason);
      onStatus(this.statusValue);
    };

    try {
      update("loading");
      const manifestUrl = bundleManifestUrl(baseUrl, this.environment.documentBaseUrl);
      const manifestBuffer = await this.fetchBytes(manifestUrl, "bundle.manifest.unavailable");
      if (attempt !== this.attempt) fail("bundle.load.superseded");
      const manifest = validateManifest(parseJson(manifestBuffer, "bundle.manifest.invalid"));
      const subtle = this.environment.subtle;
      if (subtle === undefined) fail("bundle.environment.unsupported");
      const downloads = await Promise.all(
        manifest.assets.map(async (asset) => ({
          asset,
          buffer: await this.fetchBytes(
            assetUrl(manifestUrl, asset.locator.path),
            "bundle.asset.unavailable",
          ),
        })),
      );
      update("verifying");
      const buffers = {} as Record<ModelBundleAssetRole, ArrayBuffer>;
      for (const { asset, buffer } of downloads) {
        if (buffer.byteLength !== asset.size_bytes) fail("bundle.asset.size_mismatch");
        let digest: ArrayBuffer;
        try {
          digest = await subtle.digest("SHA-256", new Uint8Array(buffer));
        } catch {
          fail("bundle.environment.unsupported");
        }
        const actual = `sha256:${Array.from(new Uint8Array(digest), (byte) =>
          byte.toString(16).padStart(2, "0"),
        ).join("")}`;
        if (actual !== asset.sha256) fail("bundle.asset.digest_mismatch");
        buffers[asset.role] = buffer;
      }
      verifyComponents(manifest, buffers);
      if (attempt !== this.attempt) fail("bundle.load.superseded");
      const bytesByRole = Object.freeze(
        Object.fromEntries(
          manifest.assets.map((asset) => [
            asset.role,
            new Blob([buffers[asset.role]], { type: asset.media_type }),
          ]),
        ) as Record<ModelBundleAssetRole, Blob>,
      );
      const verified = Object.freeze({
        id: manifest.bundle_id,
        version: manifest.version,
        manifest,
        bytesByRole,
      });
      this.activeValue = verified;
      update("ready");
      return verified;
    } catch (error) {
      const failure =
        error instanceof ModelBundleLoadError
          ? error
          : new ModelBundleLoadError("bundle.asset.unavailable");
      if (attempt === this.attempt) {
        this.statusValue = statusFor("error", this.activeValue, failure.message);
        onStatus(this.statusValue);
      }
      throw failure;
    }
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
