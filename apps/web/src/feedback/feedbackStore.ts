import type { CandidateInferenceResult } from "../inference/candidateInferenceProtocol";
import { BODY_ANCHOR_NAMES, HAND_SLOT_IDS } from "../landmarks/protocol";
import type { LiveFeedbackContext } from "../live/liveRecognitionSession";

// prettier-ignore
export const FEEDBACK_LABELS = ["hello", "no", "please", "thank_you", "yes", "other", "inactive"] as const;
export type FeedbackLabel = (typeof FEEDBACK_LABELS)[number];

export interface FeedbackSubmission {
  readonly result: CandidateInferenceResult;
  readonly context: LiveFeedbackContext;
  readonly correction: FeedbackLabel;
  readonly includeLandmarks: boolean;
  readonly previewMirrored: boolean;
}

export interface FeedbackListResult {
  readonly records: readonly FeedbackRecord[];
  readonly damagedKeys: readonly IDBValidKey[];
}

export interface FeedbackStore {
  save(submission: FeedbackSubmission): Promise<FeedbackRecord>;
  list(): Promise<FeedbackListResult>;
  updateCorrection(id: string, correction: FeedbackLabel): Promise<void>;
  delete(key: IDBValidKey): Promise<void>;
  clear(): Promise<void>;
}

export class FeedbackStoreError extends Error {
  constructor(readonly code: "unavailable" | "blocked" | "quota" | "version") {
    super(`feedback.storage.${code}`);
  }
}

function buildRecord(submission: FeedbackSubmission, id: string, savedAt: string) {
  const { result, context } = submission;
  const record = {
    format: "signlab-feedback-record/1" as const,
    id,
    savedAt,
    bundle: { ...result.bundle },
    prediction: { decision: structuredClone(result.decision), reason: result.reason },
    scores: result.rankedScores.map(({ label, confidence }) => ({ label, confidence })),
    timings: { ...result.timings },
    event: {
      ...context.event,
      durationUs: context.event.lastTimestampUs - context.event.firstTimestampUs,
    },
    conditions: {
      sourceMirrorState: context.input.sourceMirrorState,
      previewMirrored: submission.previewMirrored,
    },
    correction: submission.correction,
    consent: {
      mode: "per_event" as const,
      scope: "local_feedback_only" as const,
      grantedAt: savedAt,
      landmarksIncluded: submission.includeLandmarks,
    },
  };
  return submission.includeLandmarks
    ? { ...record, landmarks: structuredClone(context.input.frames) }
    : record;
}

export type FeedbackRecord = Readonly<ReturnType<typeof buildRecord>>;

const object = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

type Rule = ((value: unknown) => boolean) | { readonly [field: string]: Rule };
function matches(value: unknown, rule: Rule): boolean {
  if (typeof rule === "function") return rule(value);
  if (!object(value)) return false;
  const fields = Object.entries(rule);
  return (
    Object.keys(value).length === fields.length &&
    fields.every(([field, child]) => Object.hasOwn(value, field) && matches(value[field], child))
  );
}

const oneOf =
  (...allowed: unknown[]): Rule =>
  (value) =>
    allowed.includes(value);
const nullable =
  (rule: Rule): Rule =>
  (value) =>
    value === null || matches(value, rule);
const list =
  (rule: Rule): Rule =>
  (value) =>
    Array.isArray(value) && value.every((item) => matches(item, rule));
const tuple =
  (...rules: Rule[]): Rule =>
  (value) =>
    Array.isArray(value) &&
    value.length === rules.length &&
    rules.every((rule, index) => matches(value[index], rule));
const text = (value: unknown) => typeof value === "string";
const bool = (value: unknown) => typeof value === "boolean";
const iso = (value: unknown) =>
  typeof value === "string" &&
  !Number.isNaN(Date.parse(value)) &&
  new Date(value).toISOString() === value;
const finite = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);
const nonnegative = (value: unknown): value is number => finite(value) && value >= 0;
const whole = (value: unknown): value is number =>
  nonnegative(value) && Number.isSafeInteger(value);
const probability = (value: unknown): value is number => nonnegative(value) && value <= 1;

// prettier-ignore
const pointRule = { x: finite, y: finite, z: finite, visibility: nullable(probability), presence: nullable(probability) } satisfies Rule;
// prettier-ignore
const handRule = (slotId: string): Rule => ({ slotId: oneOf(slotId), present: bool, detectorIndex: nullable(whole), trackingId: nullable(oneOf(...HAND_SLOT_IDS)), handedness: nullable(oneOf("left", "right")), handednessConfidence: nullable(probability), imageLandmarks: nullable(list(pointRule)), worldLandmarks: nullable(list(pointRule)) });
// prettier-ignore
const anchorRule = (name: string): Rule => ({ name: oneOf(name), present: bool, imagePoint: nullable(pointRule), worldPoint: nullable(pointRule) });
// prettier-ignore
const frameRule = { relativeTimestampUs: whole, valid: bool, hands: tuple(...HAND_SLOT_IDS.map(handRule)), bodyAnchors: tuple(...BODY_ANCHOR_NAMES.map(anchorRule)) } satisfies Rule;
// prettier-ignore
const decisionRule: Rule = (value) => matches(value, { kind: oneOf("abstain") }) || matches(value, { kind: oneOf("other"), label: oneOf("other"), confidence: probability }) || matches(value, { kind: oneOf("target"), label: oneOf(...FEEDBACK_LABELS.slice(0, 5)), confidence: probability });
// prettier-ignore
const RECORD_RULE = {
  format: oneOf("signlab-feedback-record/1"), id: text, savedAt: iso,
  bundle: { id: text, version: text },
  prediction: { decision: decisionRule, reason: oneOf("accepted_target", "accepted_other", "below_threshold") },
  scores: list({ label: oneOf(...FEEDBACK_LABELS.slice(0, 6)), confidence: probability }),
  timings: { preprocessingMs: nonnegative, inferenceMs: nonnegative, decisionMs: nonnegative, totalMs: nonnegative },
  event: { firstFrameIndex: whole, lastFrameIndex: whole, firstTimestampUs: whole, lastTimestampUs: whole, terminationReason: oneOf("settled", "signal_gap", "max_duration", "stream_end"), configSha256: text, durationUs: nonnegative },
  conditions: { sourceMirrorState: oneOf("mirrored", "not_mirrored"), previewMirrored: bool },
  correction: oneOf(...FEEDBACK_LABELS),
  consent: { mode: oneOf("per_event"), scope: oneOf("local_feedback_only"), grantedAt: iso, landmarksIncluded: bool },
} satisfies Record<string, Rule>;

function isRecord(value: unknown): value is FeedbackRecord {
  const hasLandmarks = object(value) && Object.hasOwn(value, "landmarks");
  if (!matches(value, hasLandmarks ? { ...RECORD_RULE, landmarks: list(frameRule) } : RECORD_RULE))
    return false;
  const record = value as FeedbackRecord;
  return (
    record.consent.grantedAt === record.savedAt &&
    record.consent.landmarksIncluded === hasLandmarks &&
    record.scores.length === 6 &&
    new Set(record.scores.map(({ label }) => label)).size === 6 &&
    record.event.lastFrameIndex >= record.event.firstFrameIndex &&
    record.event.durationUs === record.event.lastTimestampUs - record.event.firstTimestampUs
  );
}

export function mapFeedbackStoreError(error: unknown): FeedbackStoreError {
  const name = object(error) ? error.name : "";
  if (name === "QuotaExceededError") return new FeedbackStoreError("quota");
  if (name === "VersionError") return new FeedbackStoreError("version");
  return error instanceof FeedbackStoreError ? error : new FeedbackStoreError("unavailable");
}

const requested = <T>(request: IDBRequest<T>) =>
  new Promise<T>((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(mapFeedbackStoreError(request.error));
  });

export class NativeFeedbackStore implements FeedbackStore {
  constructor(
    private readonly factory: IDBFactory | null = globalThis.indexedDB ?? null,
    private readonly databaseName = "signlab-feedback",
    private readonly now = () => new Date().toISOString(),
    private readonly createId: () => string = () => globalThis.crypto.randomUUID(),
  ) {}

  private open(): Promise<IDBDatabase> {
    if (this.factory === null) return Promise.reject(new FeedbackStoreError("unavailable"));
    return new Promise((resolve, reject) => {
      let blocked = false;
      let request: IDBOpenDBRequest;
      try {
        request = this.factory!.open(this.databaseName, 1);
      } catch (error) {
        reject(mapFeedbackStoreError(error));
        return;
      }
      request.onupgradeneeded = () => {
        if (!request.result.objectStoreNames.contains("events"))
          request.result.createObjectStore("events", { keyPath: "id" });
      };
      request.onblocked = () => {
        blocked = true;
        reject(new FeedbackStoreError("blocked"));
      };
      request.onerror = () => reject(mapFeedbackStoreError(request.error));
      request.onsuccess = () => {
        if (blocked) return request.result.close();
        request.result.onversionchange = () => request.result.close();
        resolve(request.result);
      };
    });
  }

  private async transact<T>(
    mode: IDBTransactionMode,
    action: (store: IDBObjectStore) => Promise<T>,
  ): Promise<T> {
    const database = await this.open();
    try {
      const transaction = database.transaction("events", mode);
      const completed = new Promise<void>((resolve, reject) => {
        transaction.oncomplete = () => resolve();
        transaction.onabort = transaction.onerror = () =>
          reject(mapFeedbackStoreError(transaction.error));
      });
      const [result] = await Promise.all([action(transaction.objectStore("events")), completed]);
      return result;
    } catch (error) {
      throw mapFeedbackStoreError(error);
    } finally {
      database.close();
    }
  }

  async save(submission: FeedbackSubmission): Promise<FeedbackRecord> {
    const record = buildRecord(submission, this.createId(), this.now());
    if (!isRecord(record)) throw new FeedbackStoreError("unavailable");
    return this.transact("readwrite", (store) => requested(store.put(record)).then(() => record));
  }

  list(): Promise<FeedbackListResult> {
    return this.transact(
      "readonly",
      (store) =>
        new Promise((resolve, reject) => {
          const records: FeedbackRecord[] = [];
          const damagedKeys: IDBValidKey[] = [];
          const request = store.openCursor();
          request.onerror = () => reject(mapFeedbackStoreError(request.error));
          request.onsuccess = () => {
            const cursor = request.result;
            if (cursor === null) return resolve({ records, damagedKeys });
            if (isRecord(cursor.value)) records.push(cursor.value);
            else damagedKeys.push(cursor.primaryKey);
            cursor.continue();
          };
        }),
    );
  }

  updateCorrection(id: string, correction: FeedbackLabel): Promise<void> {
    return this.transact("readwrite", async (store) => {
      const record: unknown = await requested(store.get(id));
      if (!isRecord(record)) throw new FeedbackStoreError("unavailable");
      await requested(store.put({ ...record, correction }));
    });
  }

  delete(key: IDBValidKey): Promise<void> {
    return this.transact("readwrite", async (store) => void (await requested(store.delete(key))));
  }

  clear(): Promise<void> {
    return this.transact("readwrite", async (store) => void (await requested(store.clear())));
  }
}
