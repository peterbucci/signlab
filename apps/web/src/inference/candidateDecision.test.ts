import { describe, expect, it } from "vitest";

import { decideCandidate } from "./candidateDecision";
import policy from "../../../../docs/reports/popsign-constructed-calibration-policy-v1.json";

describe("candidate decision contract", () => {
  it("gives inactivity precedence over malformed policy and probability input", () => {
    expect(decideCandidate(false, "not probabilities", null)).toEqual({ kind: "inactive" });
  });

  it("temperature-scales valid probabilities and chooses the first equal maximum", () => {
    const probabilities = [0.4, 0.4, 0.05, 0.05, 0.05, 0.05];
    const decision = decideCandidate(true, probabilities, policy);
    const expected = 0.4 ** 20 / (2 * 0.4 ** 20 + 4 * 0.05 ** 20);

    expect(decision).toMatchObject({ kind: "target", label: "hello" });
    expect(decideCandidate(true, [1.000019, 0, 0, 0, 0, 0], policy)).toMatchObject({
      kind: "target",
    });
    if (decision.kind !== "target") throw new Error("expected target decision");
    expect(decision.confidence).toBeCloseTo(expected, 14);
  });

  it("keeps the selected other class distinct from target and abstain", () => {
    expect(
      decideCandidate(true, new Float32Array([0.01, 0.01, 0.01, 0.01, 0.01, 0.95]), policy),
    ).toMatchObject({ kind: "other", label: "other" });
  });

  it("abstains on malformed active probability vectors", () => {
    const malformed: unknown[] = [
      "not probabilities",
      [0.2, 0.2],
      [0.2, 0.2, 0.2, 0.2, 0.2, Number.NaN],
      [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    ];

    expect(malformed.map((value) => decideCandidate(true, value, policy))).toEqual(
      malformed.map(() => ({ kind: "abstain" })),
    );
  });

  it("rejects policy variants instead of silently changing runtime behavior", () => {
    const variants = [
      { format: "signlab-decision-policy/2" },
      { class_map: { ...policy.class_map, "0": "yes" } },
      { identities: { ...policy.identities, model_sha256: "sha256:changed" } },
      { unexpected: true },
      { decision_precedence: new Array(4) },
      {
        temperature: {
          method: "softmax_log_probability_scalar_temperature/1",
          temperature_milli: 51,
        },
      },
      {
        abstention: {
          inclusive: true,
          objective: "maximize_target_coverage_zero_observed_accepted_errors/1",
          threshold_percent: 1,
        },
      },
    ].map((change) => ({ ...policy, ...change }));

    expect(variants.map((variant) => decideCandidate(true, [1, 0, 0, 0, 0, 0], variant))).toEqual(
      variants.map(() => ({ kind: "abstain" })),
    );
  });
});
