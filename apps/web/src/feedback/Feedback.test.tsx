import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { IDBFactory } from "fake-indexeddb";
import { describe, expect, it } from "vitest";

import {
  CANDIDATE_INFERENCE_PROTOCOL_VERSION,
  type CandidateInferenceResult,
} from "../inference/candidateInferenceProtocol";
import { CANDIDATE_LABELS } from "../inference/candidateDecision";
import type { LiveFeedbackContext } from "../live/liveRecognitionSession";
import { FeedbackPage, FeedbackSaveForm } from "./Feedback";
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

  it("explains unavailable storage without falling back", async () => {
    render(<FeedbackPage store={new NativeFeedbackStore(null)} />);
    expect(await screen.findByText(/Local feedback storage is unavailable/)).toBeInTheDocument();
  });
});
