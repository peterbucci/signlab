// @vitest-environment node

import { readFile } from "node:fs/promises";

import { describe, expect, it } from "vitest";

import { validateResearchEvidence } from "./validateResearchEvidence.mjs";

const artifactUrl = new URL(
  "../../../docs/reports/signlab-public-results-v1.json",
  import.meta.url,
);
const evidence = JSON.parse(await readFile(artifactUrl, "utf8"));
const copy = () => structuredClone(evidence);
async function rejects(change, message) {
  const changed = copy();
  change(changed);
  await expect(validateResearchEvidence(changed)).rejects.toThrow(message);
}

describe("public research evidence", () => {
  it("accepts the reviewed artifact and its exact source bytes", async () => {
    await expect(validateResearchEvidence(copy())).resolves.toEqual({ claims: 6, sources: 9 });
  });

  it("rejects content drift and changed claim boundaries", async () => {
    await rejects((item) => {
      item.baseline.metrics[0][1] = "0.999";
    }, "reviewed artifact content changed");
    await rejects((item) => {
      item.architecture.scope = "candidate_performance";
    }, "claim label or scope changed");
    await rejects((item) => {
      item.browser.statement = "Candidate accuracy 0.900 on the locked test.";
    }, "candidate test claim is not allowed");
  });

  it("rejects private, secret, identity, and raw-media text", async () => {
    for (const unsafe of [
      ["C:", "Users", "person", "result.json"].join("\\"),
      ["", "home", "person", "result.json"].join("/"),
      "ghp_12345678901234567890",
      "raw_videos/signer.mov",
      "portrait.gif",
    ]) {
      await rejects((item) => {
        item.dataset.statement = unsafe;
      }, "private or unsafe text found");
    }
    await rejects((item) => {
      item.baseline.participantId = "hidden";
    }, "private identity field");
  });

  it("requires every named evidence gap", async () => {
    await rejects((item) => item.unavailable.items.pop(), "unavailable evidence list changed");
  });
});
