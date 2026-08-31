import evidence from "../../../../docs/reports/signlab-public-results-v1.json";

type Source = { path: string; sha256: string };
const sources: Record<string, Source> = evidence.sources;

function focusResultsHeading(heading: HTMLHeadingElement | null) {
  if (heading === null) return;
  document.title = "Results | SignLab";
  heading.focus();
}

function SourceLinks({ ids }: { ids: readonly string[] }) {
  return (
    <div className="evidence-sources">
      <strong>Exact sources</strong>
      <ul>
        {ids.map((id) => {
          const source = sources[id];
          if (source === undefined) return null;
          const href = `https://github.com/peterbucci/signlab/blob/${evidence.evidenceCommit}/${source.path}`;
          return (
            <li key={id}>
              <a href={href} title={source.sha256}>
                {source.path.split("/").at(-1)}
              </a>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function Metrics({ values }: { values: readonly (readonly string[])[] }) {
  return (
    <dl className="evidence-metrics">
      {values.map(([label, value]) => (
        <div key={`${label}:${value}`}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function ComparisonTable({
  caption,
  columns,
  rows,
}: {
  caption: string;
  columns: readonly string[];
  rows: readonly (readonly (number | string)[])[];
}) {
  return (
    <div className="evidence-table-wrap">
      <table aria-label={caption}>
        <thead>
          <tr>
            <th scope="col">Measure</th>
            {columns.map((column) => (
              <th scope="col" key={column}>
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, ...values]) => (
            <tr key={label}>
              <th scope="row">{label}</th>
              {values.map((value, index) => (
                <td key={`${label}:${columns[index]}`}>{value}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ResultsPage() {
  return (
    <article className="results-page" aria-labelledby="results-heading">
      <header className="results-intro">
        <p className="eyebrow">Reviewed evidence only</p>
        <h1 id="results-heading" ref={focusResultsHeading} tabIndex={-1}>
          Research results
        </h1>
        <p className="page-summary">
          A compact record of what SignLab has measured, which model each number belongs to, and
          what the evidence cannot support yet.
        </p>
        <aside className="results-boundary" aria-label="Current evidence boundary">
          <strong>No headline candidate accuracy is published.</strong>
          <p>
            The real held-out score below belongs to a simple reference baseline. Candidate evidence
            currently covers architecture development, constructed calibration, runtime equivalence,
            and one-machine browser measurements.
          </p>
          <SourceLinks ids={["datasetCard", "modelCard"]} />
        </aside>
      </header>

      <div className="evidence-grid">
        <section className="evidence-card" aria-labelledby="dataset-evidence-heading">
          <h2 id="dataset-evidence-heading">{evidence.dataset.label}</h2>
          <p>
            <strong>{`${evidence.dataset.clips} clips · ${evidence.dataset.signers} signers · ${evidence.dataset.split.train} / ${evidence.dataset.split.validation} / ${evidence.dataset.split.test} train/validation/test`}</strong>
          </p>
          <p>{evidence.dataset.statement}</p>
          <SourceLinks ids={evidence.dataset.sources} />
        </section>

        <section className="evidence-card" aria-labelledby="baseline-evidence-heading">
          <h2 id="baseline-evidence-heading">{evidence.baseline.label}</h2>
          <p>{evidence.baseline.statement}</p>
          <Metrics values={evidence.baseline.metrics} />
          <ComparisonTable
            caption="Aggregate test errors by prompt"
            columns={["Errors", "Support"]}
            rows={evidence.baseline.errors}
          />
          <SourceLinks ids={evidence.baseline.sources} />
        </section>

        {[evidence.architecture, evidence.calibration, evidence.runtime].map((claim) => (
          <section
            className="evidence-card"
            aria-labelledby={`${claim.scope}-heading`}
            key={claim.scope}
          >
            <h2 id={`${claim.scope}-heading`}>{claim.label}</h2>
            <p>{claim.statement}</p>
            <Metrics values={claim.metrics} />
            <SourceLinks ids={claim.sources} />
          </section>
        ))}

        <section
          className="evidence-card evidence-card-wide"
          aria-labelledby="browser-evidence-heading"
        >
          <h2 id="browser-evidence-heading">{evidence.browser.label}</h2>
          <p>{evidence.browser.statement}</p>
          <p>{evidence.browser.environment}</p>
          <p>{evidence.browser.sampleNote}</p>
          <ComparisonTable
            caption="Cold and warm browser reference runs"
            columns={evidence.browser.runLabels}
            rows={evidence.browser.metrics}
          />
          <ul className="evidence-limitations">
            {evidence.browser.limitations.map((limitation) => (
              <li key={limitation}>{limitation}</li>
            ))}
          </ul>
          <SourceLinks ids={evidence.browser.sources} />
        </section>

        <section
          className="evidence-card evidence-card-wide unavailable-evidence"
          aria-labelledby="unavailable-evidence-heading"
        >
          <h2 id="unavailable-evidence-heading">{evidence.unavailable.label}</h2>
          <p>These are unknowns, not zeroes and not implied successes.</p>
          <ul>
            {evidence.unavailable.items.map((item) => (
              <li key={item.id}>
                <h3>{item.label}</h3>
                <p>{item.statement}</p>
                <SourceLinks ids={item.sources} />
              </li>
            ))}
          </ul>
        </section>
      </div>
    </article>
  );
}
