import { useEffect, useRef, useState, type FormEvent } from "react";

import type { CandidateInferenceResult } from "../inference/candidateInferenceProtocol";
import type { LiveFeedbackContext } from "../live/liveRecognitionSession";
import {
  FEEDBACK_LABELS,
  NativeFeedbackStore,
  type FeedbackLabel,
  type FeedbackListResult,
  type FeedbackRecord,
  type FeedbackStore,
} from "./feedbackStore";

const browserStore = new NativeFeedbackStore();

function FeedbackOptions() {
  return FEEDBACK_LABELS.map((value) => (
    <option key={value} value={value}>
      {value.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase())}
    </option>
  ));
}

function feedbackStorageMessage(error: unknown): string {
  const code = String(Reflect.get(Object(error), "code"));
  if (code.includes("quota")) return "This browser has no space available for feedback.";
  if (code.includes("blocked")) return "Close other SignLab tabs, then try again.";
  if (code.includes("version")) return "This feedback was created by a newer SignLab version.";
  return "Local feedback storage is unavailable. Nothing was saved.";
}

export function FeedbackSaveForm({
  result,
  context,
  previewMirrored,
  store = browserStore,
}: {
  result: CandidateInferenceResult;
  context: LiveFeedbackContext;
  previewMirrored: boolean;
  store?: FeedbackStore;
}) {
  const previewMirroredAtResult = useRef(previewMirrored).current;
  const [status, setStatus] = useState<"idle" | "saving" | "saved">("idle");
  const [message, setMessage] = useState("");
  const save = async (event: FormEvent) => {
    event.preventDefault();
    const fields = new FormData(event.currentTarget as HTMLFormElement);
    const correction = fields.get("correction");
    if (
      typeof correction !== "string" ||
      !FEEDBACK_LABELS.includes(correction as FeedbackLabel) ||
      !fields.has("consent") ||
      status !== "idle"
    )
      return;
    setStatus("saving");
    try {
      await store.save({
        result,
        context,
        correction: correction as FeedbackLabel,
        includeLandmarks: fields.has("landmarks"),
        previewMirrored: previewMirroredAtResult,
      });
      setStatus("saved");
      setMessage("Feedback saved in this browser.");
    } catch (error) {
      setStatus("idle");
      setMessage(feedbackStorageMessage(error));
    }
  };

  return (
    <form className="feedback-save" onSubmit={(event) => void save(event)}>
      <h3>Save feedback locally</h3>
      <fieldset disabled={status !== "idle"}>
        <label>
          Correct outcome
          <select name="correction" defaultValue="" required>
            <option value="">Choose an outcome</option>
            <FeedbackOptions />
          </select>
        </label>
        <p>
          Saved: record ID/time, model bundle, prediction/reason, scores/timings, correction, event
          boundaries/duration/termination/detector version, mirror conditions, consent, and—if
          selected—landmarks. Never: video, audio, device details, or free text.
        </p>
        <label className="feedback-check">
          <input name="landmarks" type="checkbox" /> Include derived landmark coordinates
        </label>
        <label className="feedback-check">
          <input name="consent" type="checkbox" required /> Save these fields in this browser
          only—not for upload or training.
        </label>
        <button type="submit">
          {status === "saving" ? "Saving" : status === "saved" ? "Saved" : "Save in this browser"}
        </button>
      </fieldset>
      {message && <p role="status">{message}</p>}
    </form>
  );
}

function RecordDetails({ record }: { record: FeedbackRecord }) {
  const [open, setOpen] = useState(false);
  return (
    <details onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>Inspect every saved field</summary>
      {open && <pre>{JSON.stringify(record, null, 2)}</pre>}
    </details>
  );
}

export function FeedbackPage({ store = browserStore }: { store?: FeedbackStore }) {
  const headingRef = useRef<HTMLHeadingElement>(null);
  const [items, setItems] = useState<FeedbackListResult>({ records: [], damagedKeys: [] });
  const [message, setMessage] = useState("Loading local feedback.");
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    document.title = "Feedback | SignLab";
    headingRef.current?.focus();
    void store.list().then(
      (next) => {
        setItems(next);
        setMessage(
          next.records.length
            ? `${next.records.length} feedback record(s) saved locally.`
            : "No feedback is saved in this browser.",
        );
      },
      (error: unknown) => setMessage(feedbackStorageMessage(error)),
    );
  }, [revision, store]);
  const mutate = async (operation: () => Promise<void>) => {
    try {
      await operation();
      setRevision((value) => value + 1);
    } catch (error) {
      setMessage(feedbackStorageMessage(error));
    }
  };

  return (
    <section className="content-page feedback-page" aria-labelledby="feedback-heading">
      <div className="content-intro">
        <p className="eyebrow">Local by default</p>
        <h1 id="feedback-heading" ref={headingRef} tabIndex={-1}>
          Feedback stays local by default
        </h1>
        <p className="page-summary">
          Local corrections are not uploaded or training data. The browser may clear them.
        </p>
        <p className="status-banner" role="status">
          {message}
        </p>
        <button
          className="danger-action"
          type="button"
          disabled={items.records.length + items.damagedKeys.length === 0}
          onClick={() => void mutate(() => store.clear())}
        >
          Clear all local feedback
        </button>
      </div>
      <section className="feedback-list" aria-label="Locally saved feedback">
        {items.records.map((record) => (
          <article key={record.id}>
            <strong>{new Date(record.savedAt).toLocaleString()}</strong>
            <label>
              Correct outcome
              <select
                value={record.correction}
                onChange={(event) =>
                  void mutate(() =>
                    store.updateCorrection(record.id, event.target.value as FeedbackLabel),
                  )
                }
              >
                <FeedbackOptions />
              </select>
            </label>
            <RecordDetails record={record} />
            <button type="button" onClick={() => void mutate(() => store.delete(record.id))}>
              Delete this record
            </button>
          </article>
        ))}
        {items.damagedKeys.map((key, index) => (
          <article key={index}>
            <strong>Unreadable local record</strong>
            <p>This entry is damaged or from an unsupported format.</p>
            <button type="button" onClick={() => void mutate(() => store.delete(key))}>
              Delete damaged record
            </button>
          </article>
        ))}
      </section>
    </section>
  );
}
