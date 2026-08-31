import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptPath = fileURLToPath(import.meta.url);
const repositoryRoot = fileURLToPath(new URL("../../../", import.meta.url));
const artifactPath = resolve(repositoryRoot, "docs/reports/signlab-public-results-v1.json");
const reviewedArtifactDigest = "6544e447267e04cd1dc225c9d54cc3ce515368fa99c8b27048da40ff3ef6c179";

const requiredClaims = {
  dataset: ["Public dataset facts", "dataset_fact"],
  baseline: ["Signer-held-out logistic baseline", "signer_held_out_baseline"],
  architecture: ["Development architecture evidence", "development_architecture_support"],
  calibration: ["Constructed calibration mechanics", "constructed_mechanics"],
  runtime: ["Runtime equivalence evidence", "runtime_equivalence"],
  browser: ["Reference-machine browser measurement", "reference_machine_only"],
};
const requiredUnavailable =
  "exact_candidate_locked_test fairness natural_continuous_false_activations natural_other population_claims robustness".split(
    " ",
  );
const unsafeText =
  /(?:[a-z]:[\\/]|\\\\|\/(?:users|home|tmp)\/|gh[pousr]_[a-z0-9]{20,}|akia[a-z0-9]{16}|-----begin .*private key-----|\.(?:avi|bmp|gif|heic|jpe?g|mkv|mov|mp4|png|webm)\b|participant[_-]?media|raw[_-]?videos?)/i;
const identityKey = /"(?:email|participant_?id|signer_?id|user_?name)"\s*:/i;

export async function validateResearchEvidence(evidence, root = repositoryRoot) {
  const errors = [];
  const check = (condition, message) => {
    if (!condition) errors.push(message);
  };
  check(evidence.format === "signlab-public-results/1", "format must be signlab-public-results/1");
  check(
    evidence.evidenceCommit === "5fb187c5a36678f868b14200452fcdc7c8650f94",
    "evidence commit changed",
  );

  for (const [id, [label, scope]] of Object.entries(requiredClaims)) {
    const claim = evidence[id];
    check(claim?.label === label && claim?.scope === scope, `${id} claim label or scope changed`);
  }
  check(evidence.unavailable?.scope === "unavailable", "unavailable scope changed");
  const unavailableIds = (evidence.unavailable?.items ?? []).map(({ id }) => id).sort();
  check(unavailableIds.join() === requiredUnavailable.join(), "unavailable evidence list changed");

  const sourceIds = new Set(Object.keys(evidence.sources ?? {}));
  const owners = [
    ...Object.keys(requiredClaims).map((id) => evidence[id]),
    ...(evidence.unavailable?.items ?? []),
  ];
  owners.forEach((owner) =>
    (owner?.sources ?? []).forEach((id) => check(sourceIds.has(id), `unknown source: ${id}`)),
  );
  for (const [id, source] of Object.entries(evidence.sources ?? {})) {
    const pathAllowed = /^docs\/(?:cards|reports)\/[a-z0-9.-]+\.(?:json|md)$/.test(
      source.path ?? "",
    );
    check(pathAllowed, `${id} source path is not public evidence`);
    if (!pathAllowed) continue;
    const bytes = await readFile(resolve(root, source.path));
    const actual = `sha256:${createHash("sha256").update(bytes).digest("hex")}`;
    check(actual === source.sha256, `${id} source hash changed`);
  }

  const candidateText = JSON.stringify([
    evidence.dataset,
    evidence.architecture,
    evidence.calibration,
    evidence.runtime,
    evidence.browser,
  ]);
  check(
    !/(?:(?:test|locked-test).{0,40}(?:accuracy|f1|macro|\b0\.\d+)|(?:accuracy|f1|macro|\b0\.\d+).{0,40}(?:test|locked-test))/i.test(
      candidateText,
    ),
    "candidate test claim is not allowed",
  );
  const serialized = JSON.stringify(evidence);
  check(
    createHash("sha256").update(serialized).digest("hex") === reviewedArtifactDigest,
    "reviewed artifact content changed",
  );
  check(!unsafeText.test(serialized.replaceAll("\\\\", "\\")), "private or unsafe text found");
  check(!identityKey.test(serialized), "private identity field found");
  if (errors.length > 0) throw new Error(`research evidence invalid:\n- ${errors.join("\n- ")}`);
  return { claims: Object.keys(requiredClaims).length, sources: sourceIds.size };
}

if (process.argv[1] !== undefined && resolve(process.argv[1]) === scriptPath) {
  const evidence = JSON.parse(await readFile(artifactPath, "utf8"));
  const result = await validateResearchEvidence(evidence);
  console.log(`Research evidence verified: ${result.claims} claims, ${result.sources} sources.`);
}
