import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { IDBFactory } from "fake-indexeddb";
import { createHash } from "node:crypto";
import { describe, expect, it, vi } from "vitest";

import browserFeedbackGolden from "../../../../tests/fixtures/public/feedback/browser-feedback-package-v1.json";

import {
  CANDIDATE_INFERENCE_PROTOCOL_VERSION,
  type CandidateInferenceResult,
} from "../inference/candidateInferenceProtocol";
import { CANDIDATE_LABELS } from "../inference/candidateDecision";
import type { LiveFeedbackContext } from "../live/liveRecognitionSession";
import { FeedbackPage, FeedbackSaveForm } from "./Feedback";
import {
  createFeedbackPackage,
  downloadFeedbackPackage,
  summarizeFeedbackPackage,
} from "./feedbackPackage";
import { NativeFeedbackStore } from "./feedbackStore";

const result: CandidateInferenceResult = {
  type: "result",
  protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
  requestId: 1,
  bundle: { id: "candidate", version: "1" },
  backend: "wasm",
  decision: { kind: "target", label: "hello", confidence: 0.8 },
  reason: "accepted_target",
  rankedScores: CANDIDATE_LABELS.map((label) => ({ label, confidence: 0.8 })),
  timings: { preprocessingMs: 1, inferenceMs: 2, decisionMs: 1, totalMs: 4 },
};
const context: LiveFeedbackContext = {
  event: {
    firstFrameIndex: 1,
    lastFrameIndex: 2,
    firstTimestampUs: 10,
    lastTimestampUs: 20,
    terminationReason: "settled",
    configSha256: "sha256:detector",
  },
  input: {
    frames: [],
    sourceMirrorState: "not_mirrored",
    quality: { timestampDiscontinuityCount: 0, gaps: [] },
  },
};

function store(name: string) {
  let id = 0;
  return new NativeFeedbackStore(
    new IDBFactory(),
    name,
    () => "2026-08-31T12:00:00.000Z",
    () => `${name}-${++id}`,
  );
}

function submitFeedback() {
  fireEvent.change(screen.getByLabelText("Correct outcome"), { target: { value: "yes" } });
  fireEvent.click(screen.getByLabelText(/Save these fields in this browser/));
  fireEvent.submit(screen.getByRole("button", { name: "Save in this browser" }).closest("form")!);
}

describe("local feedback UI", () => {
  it("requires event consent and keeps landmarks separately off by default", async () => {
    const feedback = store("save-form");
    const props = { result, context, store: feedback };
    const view = render(<FeedbackSaveForm {...props} previewMirrored />);
    fireEvent.change(screen.getByLabelText("Correct outcome"), { target: { value: "yes" } });
    fireEvent.submit(screen.getByRole("button", { name: "Save in this browser" }).closest("form")!);
    expect((await feedback.list()).records).toHaveLength(0);

    view.rerender(<FeedbackSaveForm {...props} previewMirrored={false} />);
    submitFeedback();
    await screen.findByText("Feedback saved in this browser.");
    expect((await feedback.list()).records[0]).not.toHaveProperty("landmarks");
    expect((await feedback.list()).records[0]?.conditions.previewMirrored).toBe(true);

    view.rerender(
      <FeedbackSaveForm
        {...props}
        key="next-event"
        result={{ ...result, requestId: 2 }}
        previewMirrored={false}
      />,
    );
    fireEvent.click(screen.getByLabelText("Include derived landmark coordinates"));
    submitFeedback();
    await waitFor(async () => expect((await feedback.list()).records).toHaveLength(2));
    expect((await feedback.list()).records[1]).toHaveProperty("landmarks");
  });

  it("lists, inspects, updates, deletes, and clears local records", async () => {
    const feedback = store("feedback-page");
    await feedback.save({
      result,
      context,
      correction: "hello",
      includeLandmarks: false,
      previewMirrored: true,
    });
    await feedback.save({
      result,
      context,
      correction: "yes",
      includeLandmarks: false,
      previewMirrored: true,
    });
    render(<FeedbackPage store={feedback} />);

    await screen.findByText("2 feedback record(s) saved locally.");
    fireEvent.change(screen.getAllByLabelText("Correct outcome")[0]!, {
      target: { value: "inactive" },
    });
    await waitFor(async () =>
      expect(
        (await feedback.list()).records.some(({ correction }) => correction === "inactive"),
      ).toBe(true),
    );
    fireEvent.click(screen.getAllByText("Inspect every saved field")[0]!);
    expect(await screen.findByText(/signlab-feedback-record\/1/)).toBeInTheDocument();

    fireEvent.click(screen.getAllByRole("button", { name: "Delete this record" })[0]!);
    await screen.findByText("1 feedback record(s) saved locally.");
    fireEvent.click(screen.getByRole("button", { name: "Clear all local feedback" }));
    await screen.findByText("No feedback is saved in this browser.");
    expect((await feedback.list()).records).toHaveLength(0);
  });

  it("requires export consent and exports the current list after deletion", async () => {
    const feedback = store("feedback-export");
    await feedback.save({
      result,
      context,
      correction: "hello",
      includeLandmarks: true,
      previewMirrored: false,
    });
    await feedback.save({
      result,
      context,
      correction: "yes",
      includeLandmarks: false,
      previewMirrored: false,
    });
    const exporter = vi.fn().mockResolvedValue(undefined);
    render(<FeedbackPage store={feedback} exporter={exporter} />);

    await screen.findByText("2 feedback record(s) saved locally.");
    const download = screen.getByRole("button", { name: "Download feedback package" });
    fireEvent.submit(download.closest("form")!);
    expect(exporter).not.toHaveBeenCalled();
    fireEvent.click(screen.getByLabelText(/Include stored landmark coordinates in this download/));
    expect(screen.getByText("Yes (1)")).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText(/I consent to this local download/));
    fireEvent.click(screen.getAllByRole("button", { name: "Delete this record" })[0]!);
    expect(download).toBeDisabled();
    await screen.findByText("1 feedback record(s) saved locally.");
    fireEvent.click(download);

    await screen.findByText(/Nothing was uploaded/);
    expect(exporter).toHaveBeenCalledOnce();
    expect(exporter.mock.calls[0]?.[0]).toHaveLength(1);
    expect(exporter.mock.calls[0]?.[1]).toBe(true);
  });

  it("sanitizes export failures and disables an empty export", async () => {
    const feedback = store("feedback-export-failure");
    await feedback.save({
      result,
      context,
      correction: "yes",
      includeLandmarks: false,
      previewMirrored: false,
    });
    const exporter = vi.fn().mockRejectedValue(new Error("private digest detail"));
    const view = render(<FeedbackPage store={feedback} exporter={exporter} />);
    await screen.findByText("1 feedback record(s) saved locally.");
    fireEvent.click(screen.getByLabelText(/I consent to this local download/));
    fireEvent.click(screen.getByRole("button", { name: "Download feedback package" }));
    expect(await screen.findByText(/Saved feedback was not changed/)).toBeInTheDocument();
    expect(screen.queryByText(/private digest detail/)).not.toBeInTheDocument();

    view.unmount();
    render(<FeedbackPage store={store("feedback-export-empty")} />);
    await screen.findByText("There are no valid records to export.");
    expect(screen.getByRole("button", { name: "Download feedback package" })).toBeDisabled();
  });

  it("explains unavailable storage without falling back", async () => {
    render(<FeedbackPage store={new NativeFeedbackStore(null)} />);
    expect(await screen.findByText(/Local feedback storage is unavailable/)).toBeInTheDocument();
  });
});

describe("feedback package", () => {
  it("has deterministic identity, summaries, and explicit landmark stripping", async () => {
    const feedback = store("feedback-package");
    const withLandmarks = await feedback.save({
      result: { ...result, bundle: { id: "candidate", version: "2" } },
      context,
      correction: "hello",
      includeLandmarks: true,
      previewMirrored: false,
    });
    const withoutLandmarks = await feedback.save({
      result,
      context,
      correction: "yes",
      includeLandmarks: false,
      previewMirrored: false,
    });
    const records = [withoutLandmarks, withLandmarks];
    const feedbackPackage = await createFeedbackPackage(records, false, "2026-08-31T13:00:00.000Z");
    const payload = JSON.parse(feedbackPackage.payloadJson) as Record<string, unknown>[];

    expect(Object.keys(feedbackPackage)).toEqual([
      "format",
      "manifest",
      "payloadJson",
      "payloadSha256",
    ]);
    expect(Object.keys(feedbackPackage.manifest)).toEqual([
      "recordFormat",
      "recordCount",
      "bundleVersions",
      "fields",
      "localConsentScope",
      "landmarkRecordCount",
      "landmarksIncluded",
      "exportConsent",
    ]);
    expect(payload.map(({ id }) => id)).toEqual([withLandmarks.id, withoutLandmarks.id]);
    expect(payload.every((record) => !Object.hasOwn(record, "landmarks"))).toBe(true);
    expect(
      payload.map((record) => (record.consent as { landmarksIncluded: boolean }).landmarksIncluded),
    ).toEqual([false, false]);
    expect(feedbackPackage.manifest).toMatchObject({
      recordFormat: "signlab-feedback-record/1",
      recordCount: 2,
      bundleVersions: [
        { id: "candidate", version: "1" },
        { id: "candidate", version: "2" },
      ],
      fields: Object.keys(payload[0]!).sort(),
      localConsentScope: "local_feedback_only",
      landmarkRecordCount: 0,
      landmarksIncluded: false,
      exportConsent: {
        statementVersion: "signlab-feedback-export-consent/1",
        granted: true,
        grantedAt: "2026-08-31T13:00:00.000Z",
        scope: "manual_research_review",
        trainingUse: "requires_review_and_approval",
      },
    });
    expect(feedbackPackage.payloadSha256).toBe(
      `sha256:${createHash("sha256").update(feedbackPackage.payloadJson).digest("hex")}`,
    );
    expect(feedbackPackage.payloadJson).not.toMatch(/rawVideo|audio|freeText|fingerprint/);
    expect(summarizeFeedbackPackage(records, true)).toMatchObject({
      landmarkRecordCount: 1,
      landmarksIncluded: true,
    });
    const included = await createFeedbackPackage(records, true, "2026-08-31T13:00:00.000Z");
    expect(included).toEqual(browserFeedbackGolden);
    expect(
      (JSON.parse(included.payloadJson) as Record<string, unknown>[]).some((record) =>
        Object.hasOwn(record, "landmarks"),
      ),
    ).toBe(true);
  });

  it("downloads readable JSON locally without a network request", async () => {
    const feedback = store("feedback-download");
    const record = await feedback.save({
      result,
      context,
      correction: "yes",
      includeLandmarks: false,
      previewMirrored: false,
    });
    let downloaded: Blob | undefined;
    const target = {
      createObjectURL: vi.fn((blob: Blob) => {
        downloaded = blob;
        return "blob:feedback";
      }),
      revokeObjectURL: vi.fn(),
      click: vi.fn(),
    };
    const network = vi.fn();
    vi.stubGlobal("fetch", network);
    await downloadFeedbackPackage([record], false, "2026-08-31T13:00:00.000Z", target);

    const contents = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(reader.error ?? new Error("FileReader failed"));
      reader.onload = () =>
        typeof reader.result === "string"
          ? resolve(reader.result)
          : reject(new Error("FileReader returned non-text"));
      reader.readAsText(downloaded!);
    });
    expect(contents.endsWith("\n")).toBe(true);
    expect(JSON.parse(contents)).toMatchObject({ format: "signlab-feedback-package/1" });
    expect(target.click).toHaveBeenCalledWith(
      "blob:feedback",
      "signlab-feedback-2026-08-31.signlab-feedback.json",
    );
    expect(target.revokeObjectURL).toHaveBeenCalledWith("blob:feedback");
    expect(network).not.toHaveBeenCalled();
  });

  it("rejects an oversized export before creating a download Blob", async () => {
    const feedback = store("feedback-oversized");
    const record = await feedback.save({
      result,
      context,
      correction: "yes",
      includeLandmarks: false,
      previewMirrored: false,
    });
    const target = {
      createObjectURL: vi.fn(() => "blob:feedback"),
      revokeObjectURL: vi.fn(),
      click: vi.fn(),
    };

    await expect(
      downloadFeedbackPackage(
        [{ ...record, id: "x".repeat(16 * 1024 * 1024 + 1) }],
        false,
        "2026-08-31T13:00:00.000Z",
        target,
      ),
    ).rejects.toThrow("Feedback package exceeds the 16 MiB export limit.");
    expect(target.createObjectURL).not.toHaveBeenCalled();
    expect(target.click).not.toHaveBeenCalled();
  });
});
