import { IDBFactory } from "fake-indexeddb";
import { afterEach, describe, expect, it, vi } from "vitest";

import { absentBodyAnchors, absentHandSlots } from "../landmarks/protocol";
import type { LiveFeedbackContext } from "../live/liveRecognitionSession";
import {
  CANDIDATE_INFERENCE_PROTOCOL_VERSION,
  type CandidateInferenceResult,
} from "../inference/candidateInferenceProtocol";
import {
  mapFeedbackStoreError,
  NativeFeedbackStore,
  type FeedbackSubmission,
} from "./feedbackStore";

const result: CandidateInferenceResult = {
  type: "result",
  protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
  requestId: 4,
  bundle: { id: "candidate", version: "1" },
  backend: "wasm",
  decision: { kind: "target", label: "hello", confidence: 0.7 },
  reason: "accepted_target",
  rankedScores: [
    { label: "hello", confidence: 0.7 },
    { label: "no", confidence: 0.1 },
    { label: "please", confidence: 0.08 },
    { label: "thank_you", confidence: 0.05 },
    { label: "yes", confidence: 0.04 },
    { label: "other", confidence: 0.03 },
  ],
  timings: { preprocessingMs: 1, inferenceMs: 2, decisionMs: 1, totalMs: 4 },
};

const context: LiveFeedbackContext = {
  event: {
    firstFrameIndex: 2,
    lastFrameIndex: 2,
    firstTimestampUs: 10,
    lastTimestampUs: 10,
    terminationReason: "settled",
    configSha256: "sha256:detector",
  },
  input: {
    frames: [
      {
        relativeTimestampUs: 0,
        valid: true,
        hands: absentHandSlots(),
        bodyAnchors: absentBodyAnchors(),
      },
    ],
    sourceMirrorState: "not_mirrored",
    quality: { timestampDiscontinuityCount: 0, gaps: [] },
  },
};

const submission = (includeLandmarks = false): FeedbackSubmission => ({
  result,
  context,
  correction: "yes",
  includeLandmarks,
  previewMirrored: true,
});

function store(factory = new IDBFactory(), name = "feedback-test") {
  let id = 0;
  return new NativeFeedbackStore(
    factory,
    name,
    () => "2026-08-31T12:00:00.000Z",
    () => `feedback-${++id}`,
  );
}

function open(factory: IDBFactory, name: string, version: number): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = factory.open(name, version);
    request.onerror = () => reject(request.error ?? new Error("database open failed"));
    request.onsuccess = () => resolve(request.result);
  });
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("NativeFeedbackStore", () => {
  it("writes only the disclosed v1 fields and causes no network request", async () => {
    const feedback = store();
    const fetchSpy = vi.fn();
    const socketSpy = vi.fn();
    const beaconSpy = vi.fn();
    const xhrSpy = vi.spyOn(XMLHttpRequest.prototype, "send");
    vi.stubGlobal("fetch", fetchSpy);
    vi.stubGlobal("WebSocket", socketSpy);
    vi.stubGlobal("navigator", { sendBeacon: beaconSpy });

    const saved = await feedback.save({
      ...submission(),
      rawVideo: new Blob(["private"]),
      deviceId: "camera-name",
    } as FeedbackSubmission);
    expect(Object.keys(saved).sort()).toEqual([
      "bundle",
      "conditions",
      "consent",
      "correction",
      "event",
      "format",
      "id",
      "prediction",
      "savedAt",
      "scores",
      "timings",
    ]);
    expect(saved).not.toHaveProperty("landmarks");
    expect(JSON.stringify(saved)).not.toContain("private");
    expect(JSON.stringify(saved)).not.toContain("camera-name");

    const withLandmarks = await feedback.save(submission(true));
    expect(withLandmarks).toHaveProperty("landmarks.0.hands");
    expect((await feedback.list()).records).toHaveLength(2);
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(xhrSpy).not.toHaveBeenCalled();
    expect(socketSpy).not.toHaveBeenCalled();
    expect(beaconSpy).not.toHaveBeenCalled();
  });

  it("updates, deletes, clears, and isolates damaged records", async () => {
    const factory = new IDBFactory();
    const feedback = store(factory);
    const first = await feedback.save(submission());
    const second = await feedback.save(submission());
    await feedback.updateCorrection(first.id, "inactive");
    await feedback.delete(second.id);
    expect((await feedback.list()).records[0]?.correction).toBe("inactive");

    const database = await open(factory, "feedback-test", 1);
    const transaction = database.transaction("events", "readwrite");
    // prettier-ignore
    const damaged: Record<string, unknown>[] = [
      { ...structuredClone(first), id: "bad-time", savedAt: "garbage", consent: { ...first.consent, grantedAt: "garbage" } },
      { ...structuredClone(first), id: "no-scores", scores: [] },
      { ...structuredClone(first), id: "cyclic" },
    ];
    damaged[2]!.prediction = { decision: damaged[2], reason: "accepted_target" };
    damaged.forEach((record) => transaction.objectStore("events").put(record));
    await new Promise<void>((resolve, reject) => {
      transaction.oncomplete = () => resolve();
      transaction.onabort = () => reject(transaction.error ?? new Error("transaction failed"));
    });
    database.close();
    expect((await feedback.list()).records).toHaveLength(1);
    expect((await feedback.list()).damagedKeys).toEqual(["bad-time", "cyclic", "no-scores"]);
    await feedback.clear();
    expect(await feedback.list()).toEqual({ records: [], damagedKeys: [] });
  });

  it("reports unavailable, quota, and newer-version storage safely", async () => {
    await expect(new NativeFeedbackStore(null).list()).rejects.toMatchObject({
      code: "unavailable",
    });
    expect(mapFeedbackStoreError(new DOMException("full", "QuotaExceededError")).code).toBe(
      "quota",
    );
    const factory = new IDBFactory();
    (await open(factory, "future-feedback", 2)).close();
    await expect(store(factory, "future-feedback").list()).rejects.toMatchObject({
      code: "version",
    });
  });
});
