/// <reference types="node" />
// @vitest-environment node

import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import policy from "../../../../docs/reports/popsign-constructed-calibration-policy-v1.json";
import plan from "../../../../src/signlab/resources/features/config/hand-local-64-1.default.json";
import fixture from "../../../../tests/fixtures/public/parity/candidate-runtime-goldens-v1.json";
import { CANDIDATE_LABELS, decideCandidate } from "./candidateDecision";
import { createCandidateInferenceEngine } from "./candidateInferenceSession";
import { preprocessCandidate } from "./candidatePreprocessing";

type ExpectedRows = {
  readonly nonPaddingFrameCount: number;
  readonly [key: string]: number | readonly (readonly (number | boolean)[])[] | undefined;
};
type PreprocessArguments = Parameters<typeof preprocessCandidate>;

const repositoryRoot = resolve(process.cwd(), "../..");
const hash = (bytes: Uint8Array) => `sha256:${createHash("sha256").update(bytes).digest("hex")}`;
const rows = (expected: unknown, key: string, width: number) => {
  const typed = expected as ExpectedRows;
  const stored = typed[key];
  return Array.isArray(stored)
    ? stored.flat().map(Number)
    : Array<number>(typed.nonPaddingFrameCount * width).fill(0);
};
const paddedRows = (expected: unknown, key: string, width: number, padding: number) =>
  rows(expected, key, width).concat(Array<number>(padding * width).fill(0));
const expectedTensor = (expected: unknown) => {
  const output = new Float32Array(64 * 126);
  output.set(rows(expected, "valuesQ", 126).map((value) => Math.fround(value / 1e6)));
  return output;
};
const tensorHash = (values: Float32Array) => {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  values.forEach((value, index) => view.setFloat32(index * 4, value, true));
  return hash(bytes);
};
const selectedIndices = (count: number) =>
  count <= 64
    ? Array.from({ length: count }, (_, index) => index)
    : Array.from({ length: 64 }, (_, index) => Math.floor((2 * index * (count - 1) + 63) / 126));

describe("candidate runtime golden parity", () => {
  it("binds the fixture to the exact plans, policy, labels, and runtime assets", async () => {
    expect(fixture.format).toBe("signlab-candidate-runtime-goldens/1");
    expect(fixture.labels).toEqual(CANDIDATE_LABELS);
    expect(fixture.metadata.digestEncoding).toEqual({
      resources: "sha256 of exact file bytes, lowercase hex with sha256: prefix",
      tensor: "little-endian row-major float32 after valuesQ / 1000000",
    });
    const resources = [
      fixture.resources.featurePlan,
      fixture.resources.qualityPolicy,
      fixture.resources.decisionPolicy,
      fixture.resources.nativeOnnxEvidence,
      fixture.resources.segmenter,
      fixture.resources.testModel,
    ];
    for (const resource of resources) {
      const actualHash = hash(await readFile(resolve(repositoryRoot, resource.path)));
      expect(actualHash).toBe(resource.fileSha256);
    }
    expect(fixture.resources.decisionPolicy).toMatchObject({
      temperatureMilli: policy.temperature.temperature_milli,
      thresholdPercent: policy.abstention.threshold_percent,
      inclusive: policy.abstention.inclusive,
    });
  });

  it.each(fixture.preprocessingCases)("matches Python preprocessing for $id", (candidateCase) => {
    const expected = candidateCase.expected;
    const padding = expected.padding.frameCount;
    const actual = preprocessCandidate(
      candidateCase.frames as unknown as PreprocessArguments[0],
      candidateCase.sourceMirrorState as PreprocessArguments[1],
      candidateCase.quality as PreprocessArguments[2],
      plan,
    );
    const expectedQ = paddedRows(expected, "valuesQ", 126, padding);
    const actualQ = Array.from(actual.values, (value) => Math.round(value * 1e6));
    const maximumQuantumDifference = Math.max(
      ...actualQ.map((value, index) => Math.abs(value - expectedQ[index]!)),
    );

    expect(actual.shape).toEqual(expected.shape);
    expect(maximumQuantumDifference).toBeLessThanOrEqual(1);
    expect(actual.timestampsUs).toEqual(expected.timestampsUs);
    expect(Array.from(actual.validMask)).toEqual(paddedRows(expected, "validMask", 126, padding));
    expect(Array.from(actual.observedMask)).toEqual(
      paddedRows(expected, "observedMask", 126, padding),
    );
    expect(Array.from(actual.interpolatedMask)).toEqual(
      paddedRows(expected, "interpolatedMask", 126, padding),
    );
    expect(Array.from(actual.handPresentMask)).toEqual(
      paddedRows(expected, "handPresentMask", 2, padding),
    );
    expect(Array.from(actual.bodyAvailableMask)).toEqual(
      paddedRows(expected, "bodyAvailableMask", 1, padding),
    );
    expect(Array.from(actual.paddingMask)).toEqual(
      Array(expected.nonPaddingFrameCount).fill(0).concat(Array(padding).fill(1)),
    );
    expect(expected.selectedSourceIndices).toEqual(selectedIndices(expected.sourceGridFrameCount));
    expect(expected.rowEncoding).toBe(
      "valuesQ" in expected ? "explicit_nonpadding_rows" : "all_zero_nonpadding_rows",
    );
    expect(expected.padding).toEqual({
      side: "right",
      frameCount: 64 - expected.nonPaddingFrameCount,
      valueQ: 0,
      allMasksFalse: true,
      timestampRule: "continue_nominal_grid",
    });
    expect(expected.portableSequenceSha256).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(tensorHash(actual.values)).toBe(expected.tensorSha256);
  });

  it.each(fixture.decisionCases)("matches the checked decision for $id", (decisionCase) => {
    const actual = decideCandidate(
      decisionCase.candidateActive,
      decisionCase.probabilities,
      policy,
    );
    expect(actual.kind).toBe(decisionCase.expected.kind);
    if (
      "label" in decisionCase.expected &&
      "confidence" in decisionCase.expected &&
      decisionCase.expected.confidence !== undefined &&
      "label" in actual
    ) {
      expect(actual.label).toBe(decisionCase.expected.label);
      expect(actual.confidence).toBeCloseTo(decisionCase.expected.confidence, 12);
    }
  });

  it("proves the single-threaded WASM engine contract, not worker/browser integration", async () => {
    const model = await readFile(resolve(repositoryRoot, fixture.resources.testModel.path));
    const engine = await createCandidateInferenceEngine(Uint8Array.from(model).buffer);
    try {
      for (const runtimeCase of fixture.onnx.cases) {
        const source = fixture.preprocessingCases.find(
          (candidateCase) => candidateCase.id === runtimeCase.preprocessingCaseId,
        );
        if (source === undefined) throw new Error("missing preprocessing golden");
        const values = expectedTensor(source.expected);
        const probabilities = await engine.run(values);
        if (!(probabilities instanceof Float32Array)) throw new Error("invalid ONNX output");
        probabilities.forEach((value, index) => {
          const expected = runtimeCase.probabilities[index]!;
          const tolerance =
            fixture.onnx.tolerances.absolute +
            fixture.onnx.tolerances.relative * Math.abs(expected);
          expect(Math.abs(value - expected)).toBeLessThanOrEqual(tolerance);
        });
      }
    } finally {
      await engine.close();
    }
  });
});
