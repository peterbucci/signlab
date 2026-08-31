import { useEffect, useRef, useState, type FormEvent } from "react";

import type { CandidateInferenceResult } from "../inference/candidateInferenceProtocol";
import type { LiveFeedbackContext } from "../live/liveRecognitionSession";
import { downloadFeedbackPackage, summarizeFeedbackPackage } from "./feedbackPackage";
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

function feedbackCountMessage(count: number): string {
  return count
    ? `${count} feedback record(s) saved locally.`
    : "No feedback is saved in this browser.";
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

type FeedbackExporter = typeof downloadFeedbackPackage;

export function FeedbackPage({
  store = browserStore,
  exporter = downloadFeedbackPackage,
}: {
  store?: FeedbackStore;
  exporter?: FeedbackExporter;
}) {
  const headingRef = useRef<HTMLHeadingElement>(null);
  const [items, setItems] = useState<FeedbackListResult>({ records: [], damagedKeys: [] });
  const [message, setMessage] = useState("Loading local feedback.");
  const [includeLandmarks, setIncludeLandmarks] = useState(false);
  const [exportMessage, setExportMessage] = useState("");
  const [mutating, setMutating] = useState(false);
  useEffect(() => {
    document.title = "Feedback | SignLab";
    headingRef.current?.focus();
    void store.list().then(
      (next) => {
        setItems(next);
        setMessage(feedbackCountMessage(next.records.length));
      },
      (error: unknown) => setMessage(feedbackStorageMessage(error)),
    );
  }, [store]);
  const mutate = async (operation: () => Promise<void>) => {
    if (mutating) return;
    setMutating(true);
    try {
      await operation();
      const next = await store.list();
      setItems(next);
      setMessage(feedbackCountMessage(next.records.length));
    } catch (error) {
      setItems({ records: [], damagedKeys: [] });
      setMessage(feedbackStorageMessage(error));
    } finally {
      setMutating(false);
    }
  };
  const summary = summarizeFeedbackPackage(items.records, includeLandmarks);
  const storedLandmarkCount = items.records.filter((record) => "landmarks" in record).length;
  const bundleVersions =
    summary.bundleVersions.map(({ id, version }) => `${id} ${version}`).join(", ") || "None";
  const exportedFields = summary.fields.join(", ") || "None";
  const exporting = exportMessage === "Creating local download.";
  const exportFeedback = async (event: FormEvent) => {
    event.preventDefault();
    if (
      !new FormData(event.currentTarget as HTMLFormElement).has("exportConsent") ||
      items.records.length === 0 ||
      exporting ||
      mutating
    )
      return;
    setExportMessage("Creating local download.");
    try {
      await exporter(items.records, includeLandmarks);
      setExportMessage("Feedback package downloaded locally. Nothing was uploaded.");
    } catch {
      setExportMessage(
        "The feedback package could not be created. Saved feedback was not changed.",
      );
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
      <section aria-labelledby="feedback-export-heading">
        <h2 id="feedback-export-heading">Download for manual research review</h2>
        <p>
          This creates a file on this device only; SignLab does not upload it. Sharing it would be a
          manual contribution for review, and any possible training use requires later review and
          approval.
        </p>
        <dl>
          <dt>Valid records</dt>
          <dd>{summary.recordCount}</dd>
          <dt>Bundle versions</dt>
          <dd>{bundleVersions}</dd>
          <dt>Exported fields</dt>
          <dd>{exportedFields}</dd>
          <dt>Local consent scope</dt>
          <dd>{summary.localConsentScope}</dd>
          <dt>Stored records with landmarks</dt>
          <dd>{storedLandmarkCount}</dd>
          <dt>Landmarks included in this download</dt>
          <dd>{summary.landmarksIncluded ? `Yes (${summary.landmarkRecordCount})` : "No"}</dd>
        </dl>
        <form onSubmit={(event) => void exportFeedback(event)}>
          <label className="feedback-check">
            <input
              type="checkbox"
              checked={includeLandmarks}
              onChange={(event) => setIncludeLandmarks(event.target.checked)}
            />{" "}
            Include stored landmark coordinates in this download (off by default)
          </label>
          <label className="feedback-check">
            <input name="exportConsent" type="checkbox" required /> I consent to this local download
            for manual research review only. Possible training use requires later review and
            approval.
          </label>
          <button type="submit" disabled={items.records.length === 0 || exporting || mutating}>
            {exporting ? "Creating download" : "Download feedback package"}
          </button>
        </form>
        {items.records.length === 0 && <p>There are no valid records to export.</p>}
        {exportMessage && <p role="status">{exportMessage}</p>}
      </section>
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
