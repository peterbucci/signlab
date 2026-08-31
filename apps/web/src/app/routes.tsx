import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";

import { livePageDefinition, type StaticPageDefinition } from "./routeDefinitions";
import { FeedbackSaveForm } from "../feedback/Feedback";
import type { FeedbackStore } from "../feedback/feedbackStore";
import {
  useCameraSession,
  type CameraEnvironment,
  type CameraStatus,
} from "../camera/useCameraSession";
import {
  loadLandmarkModelAssets,
  type LandmarkModelAssetBuffers,
} from "../landmarks/landmarkModelAssets";
import {
  LiveRecognitionSession,
  type LiveRecognitionDiagnostics,
  type LiveRecognitionSnapshot,
} from "../live/liveRecognitionSession";
import {
  ModelBundleSession,
  type ModelBundleStatus,
  type VerifiedModelBundle,
} from "../modelBundle/modelBundleSession";

function usePageHeading(title: string) {
  const headingRef = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    document.title = `${title} | SignLab`;
    headingRef.current?.focus();
  }, [title]);

  return headingRef;
}

function StatusBanner({ children, label }: { children: ReactNode; label?: string }) {
  return (
    <div className="status-banner" role="status" aria-label={label}>
      <span aria-hidden="true" />
      {children}
    </div>
  );
}

const startableStatuses = new Set<CameraStatus>([
  "idle",
  "denied",
  "unavailable",
  "interrupted",
  "error",
]);

type BundleLoader = Pick<ModelBundleSession, "load" | "status"> &
  Partial<Pick<ModelBundleSession, "rollback">>;
type LiveSession = Pick<LiveRecognitionSession, "initialize" | "submitFrame" | "close">;

interface LiveRuntime {
  loadAssets(): Promise<LandmarkModelAssetBuffers>;
  createSession(
    bundle: VerifiedModelBundle,
    buffers: LandmarkModelAssetBuffers,
    onState: (snapshot: LiveRecognitionSnapshot) => void,
  ): LiveSession;
  request(video: HTMLVideoElement, callback: (captureTimestampMs: number) => void): number;
  cancel(video: HTMLVideoElement, requestId: number): void;
  capture(video: HTMLVideoElement): Promise<ImageBitmap>;
}

const browserLiveRuntime: LiveRuntime = {
  loadAssets: loadLandmarkModelAssets,
  createSession: (bundle, taskBuffers, onState) =>
    new LiveRecognitionSession({ bundle, taskBuffers, onState }),
  request(video, callback) {
    if (typeof video.requestVideoFrameCallback !== "function") {
      throw new Error("live.recognition.video_frames.unsupported");
    }
    return video.requestVideoFrameCallback((_now, metadata) =>
      callback(metadata.mediaTime * 1_000),
    );
  },
  cancel(video, requestId) {
    video.cancelVideoFrameCallback(requestId);
  },
  capture: (video) => globalThis.createImageBitmap(video),
};

function bundleStatusMessage(status: ModelBundleStatus): string {
  const active =
    status.active === null ? null : `${status.active.id} version ${status.active.version}`;
  if (status.phase === "idle") return "No model bundle is configured.";
  if (status.phase === "loading") return "Loading the model bundle manifest and files.";
  if (status.phase === "verifying") return "Verifying every model bundle file.";
  if (status.phase === "ready") {
    if (status.source === "fallback")
      return `Using verified cached model bundle ${active ?? ""} because the bundle endpoint is unavailable.`;
    if (status.source === "cache") return `Verified cached model bundle ${active ?? ""} is ready.`;
    if (status.source === "rollback")
      return `Restored previous verified model bundle ${active ?? ""}.`;
    if (status.source === "network")
      return `Verified model bundle ${active ?? ""} is ready${status.cacheWarning === undefined ? " and saved in this browser." : ` for this session. ${status.cacheWarning}`}`;
    return `Verified model bundle ${active ?? ""} is ready.`;
  }
  const fallback = "The model bundle could not be activated.";
  return `${status.failureReason ?? fallback}${active === null ? "" : ` ${active} remains active.`}`;
}

const livePhaseMessages = {
  loading: "Preparing the on-device models.",
  ready: "Models are ready. Hold still, then perform one gesture.",
  watching: "Watching for the start of a gesture.",
  recording: "Movement detected. Finish the gesture naturally.",
  classifying: "Classifying the completed gesture.",
  result: "Result ready. Hold still before the next gesture.",
  failed: "Recognition stopped safely. Retry or restart the camera.",
} as const;

const readableLabel = (label: string) =>
  label.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());

const diagnosticStateLabel = (state: string) =>
  state === "recording" ? "Gesture in progress" : readableLabel(state);

type StableResult = NonNullable<LiveRecognitionSnapshot["stableResult"]>;

function decisionTitle(result: StableResult): string {
  if (!("label" in result.decision)) return "No confident match";
  if (result.decision.kind === "other") return "Other movement";
  return readableLabel(result.decision.label);
}

function decisionReason(result: StableResult): string {
  if (result.reason === "below_threshold") return "The model abstained because confidence was low.";
  if (result.reason === "accepted_other") return "The event looked unlike the five target prompts.";
  return "The top calibrated score passed the decision threshold.";
}

function resultStatusMessage(result: StableResult): string {
  return `Result: ${decisionTitle(result)}. ${decisionReason(result)}`;
}

function landmarkSummary(diagnostics: LiveRecognitionDiagnostics | undefined): string {
  if (diagnostics === undefined || diagnostics.landmarkState === "waiting")
    return "Waiting for frames";
  if (diagnostics.landmarkState === "invalid") return "Latest frame unavailable";
  if (diagnostics.landmarkState === "no_hands") return "No hands detected";
  return `${diagnostics.detectedHands} ${diagnostics.detectedHands === 1 ? "hand" : "hands"} detected`;
}

export function LivePage({
  cameraEnvironment,
  modelBundleUrl,
  modelBundleSession,
  liveRuntime = browserLiveRuntime,
  feedbackStore,
}: {
  cameraEnvironment?: CameraEnvironment;
  modelBundleUrl?: string;
  modelBundleSession?: BundleLoader;
  liveRuntime?: LiveRuntime;
  feedbackStore?: FeedbackStore;
}) {
  const headingRef = usePageHeading(livePageDefinition.label);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [previewMirrored, setPreviewMirrored] = useState(true);
  const bundleLoader = useMemo(
    () => modelBundleSession ?? new ModelBundleSession(),
    [modelBundleSession],
  );
  const [bundleStatus, setBundleStatus] = useState(bundleLoader.status);
  const [verifiedBundle, setVerifiedBundle] = useState<VerifiedModelBundle | null>(null);
  const [liveSnapshot, setLiveSnapshot] = useState<LiveRecognitionSnapshot | null>(null);
  const [bundleAttempt, setBundleAttempt] = useState(0);
  const [runtimeAttempt, setRuntimeAttempt] = useState(0);
  const { state, canRequest, start, pause, resume, stop, switchCamera } =
    useCameraSession(cameraEnvironment);

  useEffect(() => {
    const configuredUrl = modelBundleUrl ?? import.meta.env.VITE_SIGNLAB_MODEL_BUNDLE_URL;
    if (configuredUrl === undefined || configuredUrl.trim() === "") return;
    let mounted = true;
    void bundleLoader
      .load(configuredUrl, (status) => {
        if (mounted) setBundleStatus(status);
      })
      .then((bundle) => {
        if (mounted) setVerifiedBundle(bundle);
      })
      .catch(() => undefined);
    return () => {
      mounted = false;
    };
  }, [bundleAttempt, bundleLoader, modelBundleUrl]);

  useEffect(() => {
    const video = videoRef.current;
    if (video === null) return;
    video.srcObject = state.stream;
    return () => {
      video.srcObject = null;
    };
  }, [state.stream]);

  useEffect(() => {
    const video = videoRef.current;
    if (
      state.status !== "active" ||
      state.stream === null ||
      verifiedBundle === null ||
      video === null
    ) {
      return;
    }

    let disposed = false;
    let halted = false;
    let frameRequestId: number | null = null;
    let liveSession: LiveSession | null = null;
    setLiveSnapshot({ phase: "loading", stableResult: null, failureCode: null });

    const stopPump = () => {
      halted = true;
      if (frameRequestId !== null) liveRuntime.cancel(video, frameRequestId);
      frameRequestId = null;
    };
    const failStartup = () => {
      if (disposed) return;
      stopPump();
      setLiveSnapshot((previous) => ({
        phase: "failed",
        stableResult: previous?.stableResult ?? null,
        failureCode: "live.recognition.startup.failed",
        diagnostics: previous?.diagnostics,
      }));
      void liveSession?.close();
    };
    const scheduleFrame = () => {
      if (disposed || halted || liveSession === null) return;
      try {
        frameRequestId = liveRuntime.request(video, (captureTimestampMs) => {
          frameRequestId = null;
          void (async () => {
            try {
              const bitmap = await liveRuntime.capture(video);
              if (disposed || halted) bitmap.close();
              else liveSession?.submitFrame(bitmap, captureTimestampMs);
            } catch {
              failStartup();
            }
            if (!disposed && !halted) scheduleFrame();
          })();
        });
      } catch {
        failStartup();
      }
    };

    void (async () => {
      try {
        const taskBuffers = await liveRuntime.loadAssets();
        if (disposed) return;
        liveSession = liveRuntime.createSession(verifiedBundle, taskBuffers, (snapshot) => {
          if (disposed) return;
          setLiveSnapshot(snapshot);
          if (snapshot.phase === "failed") stopPump();
        });
        await liveSession.initialize();
        if (disposed) await liveSession.close();
        else scheduleFrame();
      } catch {
        failStartup();
      }
    })();

    return () => {
      disposed = true;
      stopPump();
      void liveSession?.close();
    };
  }, [liveRuntime, runtimeAttempt, state.status, state.stream, verifiedBundle]);

  const hasStream = state.stream !== null;
  const canStart = canRequest && startableStatuses.has(state.status);
  const requestingCamera = state.status === "requesting";
  const stableResult = liveSnapshot?.stableResult ?? null;
  const feedbackContext = liveSnapshot?.feedbackContext ?? null;
  const recognitionMessage =
    liveSnapshot?.failureCode === "live.recognition.candidate.invalid"
      ? "That event could not be classified. Hold still, then try again."
      : state.status !== "active" || liveSnapshot === null
        ? "Recognition starts when the camera and model are ready."
        : liveSnapshot.phase === "result" && stableResult !== null
          ? resultStatusMessage(stableResult)
          : livePhaseMessages[liveSnapshot.phase];
  const diagnostics = liveSnapshot?.diagnostics;
  const bundleBlocked = bundleStatus.phase === "error" && bundleStatus.active === null;
  const runtimeFailed = state.status === "active" && liveSnapshot?.phase === "failed";
  const retryAvailable = bundleBlocked || runtimeFailed;
  const rollbackAvailable =
    bundleStatus.rollbackAvailable === true && bundleLoader.rollback !== undefined;
  const diagnosticBundle = diagnostics?.bundle ?? stableResult?.bundle ?? verifiedBundle;
  const retrySetup = () => {
    if (bundleBlocked) {
      setVerifiedBundle(null);
      setBundleAttempt((attempt) => attempt + 1);
    }
    if (runtimeFailed) setRuntimeAttempt((attempt) => attempt + 1);
  };
  const restorePrevious = () => {
    if (bundleLoader.rollback === undefined) return;
    void bundleLoader
      .rollback(setBundleStatus)
      .then((bundle) => {
        setLiveSnapshot(null);
        setVerifiedBundle(bundle);
      })
      .catch(() => undefined);
  };

  return (
    <section className="live-page" aria-labelledby="live-heading">
      <div className="live-intro">
        <p className="eyebrow">{livePageDefinition.eyebrow}</p>
        <h1 id="live-heading" ref={headingRef} tabIndex={-1}>
          {livePageDefinition.heading}
        </h1>
        <p className="page-summary">{livePageDefinition.summary}</p>
        <div className="privacy-notice">
          <strong>Your camera stays under your control.</strong>
          <p>
            SignLab asks for permission only after you select Start camera. This preview stays on
            this device; the page does not upload, save, or record raw video.
          </p>
        </div>
        <section className="live-guide" aria-labelledby="live-guide-heading">
          <h2 id="live-guide-heading">How to try it</h2>
          <ol>
            <li>Choose one prompt: Hello, No, Please, Thank you, or Yes.</li>
            <li>
              Keep your hands and upper body in frame with even light facing you, not behind you.
            </li>
            <li>Hold still, perform one prompt naturally, then hold still for the result.</li>
          </ol>
          <p>
            Mirroring changes only the preview. This five-prompt research prototype is not
            sign-language translation. <a href="#/limitations">Read its limitations.</a>
          </p>
        </section>
        <StatusBanner>{state.message}</StatusBanner>
        <div className="model-bundle-status">
          <p className="card-label">Model bundle</p>
          <StatusBanner label="Model bundle status">
            {bundleStatusMessage(bundleStatus)}
          </StatusBanner>
        </div>
        <div className="model-bundle-status">
          <p className="card-label">Recognition</p>
          <StatusBanner label="Recognition status">{recognitionMessage}</StatusBanner>
        </div>
        {retryAvailable ? (
          <div className="recovery-action">
            <button type="button" onClick={retrySetup}>
              Retry setup
            </button>
            <span>Retries the failed setup without reloading this page.</span>
          </div>
        ) : null}
        {rollbackAvailable ? (
          <div className="recovery-action">
            <button
              type="button"
              disabled={bundleStatus.phase === "loading" || bundleStatus.phase === "verifying"}
              onClick={restorePrevious}
            >
              Restore previous model
            </button>
            <span>Rechecks and restores the one previously verified model.</span>
          </div>
        ) : null}
      </div>

      <div className="camera-card">
        <div className="camera-preview">
          {hasStream ? (
            <video
              ref={videoRef}
              className={previewMirrored ? "is-mirrored" : undefined}
              aria-label="Local camera preview"
              autoPlay
              muted
              playsInline
            />
          ) : (
            <div className="camera-placeholder" aria-hidden="true">
              <span>Camera off</span>
            </div>
          )}
        </div>

        <div className="camera-controls" role="group" aria-label="Camera controls">
          {canStart || requestingCamera || hasStream ? (
            <button
              className={hasStream ? undefined : "primary-action"}
              type="button"
              aria-disabled={requestingCamera || undefined}
              onClick={() => {
                if (requestingCamera) return;
                if (hasStream) stop();
                else void start();
              }}
            >
              {hasStream ? "Stop camera" : requestingCamera ? "Starting camera" : "Start camera"}
            </button>
          ) : null}
          {state.status === "active" || state.status === "paused" ? (
            <button type="button" onClick={state.status === "active" ? pause : resume}>
              {state.status === "active" ? "Pause preview" : "Resume preview"}
            </button>
          ) : null}
        </div>

        {hasStream ? (
          <div className="camera-options">
            {state.devices.length > 1 ? (
              <label>
                Camera
                <select
                  value={state.selectedDeviceId ?? ""}
                  disabled={state.status !== "active"}
                  onChange={(event) => void switchCamera(event.target.value)}
                >
                  {state.devices.map((device) => (
                    <option key={device.deviceId} value={device.deviceId}>
                      {device.label}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
            <label className="mirror-option">
              <input
                type="checkbox"
                checked={previewMirrored}
                onChange={(event) => setPreviewMirrored(event.target.checked)}
              />
              Mirror preview
            </label>
          </div>
        ) : null}

        {stableResult === null ? null : (
          <section className="recognition-result" aria-label="Latest recognition result">
            <p className="card-label">Latest event</p>
            <strong>{decisionTitle(stableResult)}</strong>
            <p>{decisionReason(stableResult)}</p>
            <p className="score-note">
              Higher calibrated scores are stronger model matches, not guarantees. Alternatives show
              what the model considered; decision time covers on-device preprocessing, inference,
              and scoring for this completed event.
            </p>
            <ol aria-label="Top calibrated scores">
              {stableResult.rankedScores.slice(0, 3).map(({ label, confidence }) => (
                <li key={label}>
                  <span>{readableLabel(label)}</span>
                  <span>{Math.round(confidence * 100)}%</span>
                </li>
              ))}
            </ol>
            <dl>
              <div>
                <dt>Decision time</dt>
                <dd>{Math.round(stableResult.timings.totalMs)} ms</dd>
              </div>
              <div>
                <dt>Runtime</dt>
                <dd>{stableResult.backend.toUpperCase()}</dd>
              </div>
              <div>
                <dt>Bundle</dt>
                <dd>{`${stableResult.bundle.id} ${stableResult.bundle.version}`}</dd>
              </div>
            </dl>
            {feedbackContext === null ? null : (
              <FeedbackSaveForm
                key={`${stableResult.bundle.id}:${stableResult.requestId}:${feedbackContext.event.firstTimestampUs}`}
                result={stableResult}
                context={feedbackContext}
                previewMirrored={previewMirrored}
                store={feedbackStore}
              />
            )}
          </section>
        )}

        <details className="live-diagnostics">
          <summary>Session diagnostics</summary>
          <p>Most recent session facts only. They are not saved or uploaded.</p>
          <dl aria-label="Session diagnostics">
            <div>
              <dt>Recognition</dt>
              <dd>
                {liveSnapshot === null ? "Not started" : diagnosticStateLabel(liveSnapshot.phase)}
              </dd>
            </div>
            <div>
              <dt>Detector</dt>
              <dd>
                {diagnostics === undefined
                  ? "Not ready"
                  : diagnosticStateLabel(diagnostics.detectorState)}
              </dd>
            </div>
            <div>
              <dt>Landmarks</dt>
              <dd>{landmarkSummary(diagnostics)}</dd>
            </div>
            <div>
              <dt>Dropped frames</dt>
              <dd>{diagnostics?.droppedFrames ?? 0}</dd>
            </div>
            <div>
              <dt>Processed frames</dt>
              <dd>{diagnostics?.processedFrames ?? 0}</dd>
            </div>
            <div>
              <dt>Runtime</dt>
              <dd>{diagnostics?.backend?.toUpperCase() ?? "Waiting"}</dd>
            </div>
            <div>
              <dt>Bundle</dt>
              <dd>
                {diagnosticBundle === null
                  ? "Not ready"
                  : `${diagnosticBundle.id} ${diagnosticBundle.version}`}
              </dd>
            </div>
          </dl>
        </details>
      </div>
    </section>
  );
}

export function OverviewPage() {
  const headingRef = usePageHeading("Overview");

  return (
    <>
      <section className="hero" aria-labelledby="overview-heading">
        <div className="hero-copy">
          <p className="eyebrow">Reproducible research · Private by design</p>
          <h1 id="overview-heading" ref={headingRef} tabIndex={-1}>
            Five gestures, tested honestly.
          </h1>
          <p className="hero-summary">
            SignLab studies whether a small model can recognize isolated performances of five
            predefined gesture prompts in continuous webcam video—without confusing background
            movement or uncertainty for a result.
          </p>
          <StatusBanner>
            Browser interface in progress: the private camera-to-result path now runs on-device;
            replay and release checks remain.
          </StatusBanner>
        </div>

        <aside className="research-card" aria-label="Current research boundary">
          <p className="card-label">Current boundary</p>
          <strong>Live path connected. Evaluation next.</strong>
          <dl>
            <div>
              <dt>Vocabulary</dt>
              <dd>5 prompts</dd>
            </div>
            <div>
              <dt>Frozen split</dt>
              <dd>80 samples</dd>
            </div>
            <div>
              <dt>Browser model</dt>
              <dd>On-device</dd>
            </div>
          </dl>
        </aside>
      </section>

      <section className="principles" aria-labelledby="principles-heading">
        <div>
          <p className="section-index">01 / Approach</p>
          <h2 id="principles-heading">A demo with receipts.</h2>
        </div>
        <div className="principle-grid">
          <article>
            <span>01</span>
            <h3>Evaluate unseen people</h3>
            <p>
              Training and evaluation identities stay separated to prevent an easy-looking result.
            </p>
          </article>
          <article>
            <span>02</span>
            <h3>Allow uncertainty</h3>
            <p>The finished system may abstain instead of forcing every movement into a gesture.</p>
          </article>
          <article>
            <span>03</span>
            <h3>Run on device</h3>
            <p>
              The planned public demo processes camera input locally in the user&apos;s browser.
            </p>
          </article>
        </div>
      </section>
    </>
  );
}

export function StaticPage({ page }: { page: StaticPageDefinition }) {
  const headingId = `${page.path.slice(1)}-heading`;
  const headingRef = usePageHeading(page.label);

  return (
    <section className="content-page" aria-labelledby={headingId}>
      <div className="content-intro">
        <p className="eyebrow">{page.eyebrow}</p>
        <h1 id={headingId} ref={headingRef} tabIndex={-1}>
          {page.heading}
        </h1>
        <p className="page-summary">{page.summary}</p>
        <StatusBanner>{page.status}</StatusBanner>
        {page.note === undefined ? null : <p className="page-note">{page.note}</p>}
      </div>

      <aside className="detail-card">
        <p className="card-label">{page.detailsTitle}</p>
        <ul>
          {page.details.map((detail) => (
            <li key={detail}>{detail}</li>
          ))}
        </ul>
      </aside>
    </section>
  );
}

export function NotFoundPage() {
  const headingRef = usePageHeading("Page not found");

  return (
    <section className="content-page not-found" aria-labelledby="not-found-heading">
      <div className="content-intro">
        <p className="eyebrow">404</p>
        <h1 id="not-found-heading" ref={headingRef} tabIndex={-1}>
          Page not found
        </h1>
        <p className="page-summary">That SignLab page does not exist.</p>
        <Link className="text-link" to="/">
          Return to the overview
        </Link>
      </div>
    </section>
  );
}
