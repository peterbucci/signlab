import { useEffect, useRef, type ReactNode } from "react";
import { Link } from "react-router-dom";

import type { StaticPageDefinition } from "./routeDefinitions";

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
