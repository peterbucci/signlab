import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";

import { livePageDefinition, type StaticPageDefinition } from "./routeDefinitions";
import {
  useCameraSession,
  type CameraEnvironment,
  type CameraStatus,
} from "../camera/useCameraSession";

function usePageHeading(title: string) {
  const headingRef = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    document.title = `${title} | SignLab`;
    headingRef.current?.focus({ preventScroll: true });
  }, [title]);

  return headingRef;
}

function StatusBanner({ children }: { children: ReactNode }) {
  return (
    <div className="status-banner" role="status">
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

export function LivePage({ cameraEnvironment }: { cameraEnvironment?: CameraEnvironment }) {
  const headingRef = usePageHeading(livePageDefinition.label);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [previewMirrored, setPreviewMirrored] = useState(true);
  const { state, canRequest, start, pause, resume, stop, switchCamera } =
    useCameraSession(cameraEnvironment);

  useEffect(() => {
    const video = videoRef.current;
    if (video === null) return;
    video.srcObject = state.stream;
    return () => {
      video.srcObject = null;
    };
  }, [state.stream]);

  const hasStream = state.stream !== null;
  const canStart = canRequest && startableStatuses.has(state.status);

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
        <StatusBanner>{state.message}</StatusBanner>
        <p className="page-note">
          The recognition model is not connected yet. Preview mirroring changes only what you see,
          never the camera stream or future model coordinates.
        </p>
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

        <div className="camera-controls" aria-label="Camera controls">
          {canStart ? (
            <button className="primary-action" type="button" onClick={() => void start()}>
              Start camera
            </button>
          ) : null}
          {state.status === "active" ? (
            <button type="button" onClick={pause}>
              Pause preview
            </button>
          ) : null}
          {state.status === "paused" ? (
            <button type="button" onClick={resume}>
              Resume preview
            </button>
          ) : null}
          {hasStream ? (
            <button type="button" onClick={() => stop()}>
              Stop camera
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
            Browser interface scaffold: camera and model inference are not connected yet.
          </StatusBanner>
        </div>

        <aside className="research-card" aria-label="Current research boundary">
          <p className="card-label">Current boundary</p>
          <strong>Data ready. Evaluation next.</strong>
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
              <dd>Pending</dd>
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
