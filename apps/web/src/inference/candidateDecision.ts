import candidateDecisionPolicy from "../../../../docs/reports/popsign-constructed-calibration-policy-v1.json";

export const CANDIDATE_LABELS = ["hello", "no", "please", "thank_you", "yes", "other"] as const;

export type CandidateLabel = (typeof CANDIDATE_LABELS)[number];
export type CandidateDecision =
  | { readonly kind: "inactive" | "abstain" }
  | {
      readonly kind: "target";
      readonly label: Exclude<CandidateLabel, "other">;
      readonly confidence: number;
    }
  | { readonly kind: "other"; readonly label: "other"; readonly confidence: number };
export interface ScoredCandidateDecision {
  readonly decision: Exclude<CandidateDecision, { readonly kind: "inactive" }>;
  readonly reason: "accepted_target" | "accepted_other" | "below_threshold";
  readonly rankedScores: readonly {
    readonly label: CandidateLabel;
    readonly confidence: number;
  }[];
}

const TEMPERATURE = candidateDecisionPolicy.temperature.temperature_milli / 1_000;
const THRESHOLD = candidateDecisionPolicy.abstention.threshold_percent / 100;
const PROBABILITY_SUM_TOLERANCE = 1e-5 + 1e-5;
const LOG_FLOOR = 1e-7;

export function exactJson(value: unknown, expected: unknown = candidateDecisionPolicy): boolean {
  if (Array.isArray(expected)) {
    return (
      Array.isArray(value) &&
      value.length === expected.length &&
      expected.every((item, index) => exactJson(value[index], item))
    );
  }
  if (typeof expected === "object" && expected !== null) {
    if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
    const actual = value as Record<string, unknown>;
    const entries = Object.entries(expected);
    return (
      Object.keys(actual).length === entries.length &&
      entries.every(([key, item]) => exactJson(actual[key], item))
    );
  }
  return value === expected;
}

function validProbabilities(value: unknown): number[] | null {
  if (!Array.isArray(value) && !(value instanceof Float32Array)) return null;
  const values: readonly unknown[] = value instanceof Float32Array ? Array.from(value) : value;
  if (values.length !== CANDIDATE_LABELS.length) return null;

  const probabilities: number[] = [];
  for (const entry of values) {
    if (typeof entry !== "number" || !Number.isFinite(entry) || entry < 0) return null;
    probabilities.push(entry);
  }
  const sum = probabilities.reduce((total, entry) => total + entry, 0);
  return Math.abs(sum - 1) <= PROBABILITY_SUM_TOLERANCE ? probabilities : null;
}

function calibratedProbabilities(probabilities: readonly number[]): number[] {
  const logits = probabilities.map(
    (probability) => Math.log(Math.min(1, Math.max(LOG_FLOOR, probability))) / TEMPERATURE,
  );
  const maximum = Math.max(...logits);
  const exponentials = logits.map((logit) => Math.exp(logit - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

export function scoreCandidate(
  probabilities: unknown,
  policy: unknown,
): ScoredCandidateDecision | null {
  const checked = validProbabilities(probabilities);
  if (!exactJson(policy) || checked === null) return null;
  const rankedScores = calibratedProbabilities(checked)
    .map((confidence, index) => ({ label: CANDIDATE_LABELS[index]!, confidence }))
    .sort((left, right) => right.confidence - left.confidence);
  const selected = rankedScores[0]!;
  if (selected.confidence < THRESHOLD) {
    return { decision: { kind: "abstain" }, reason: "below_threshold", rankedScores };
  }
  const decision =
    selected.label === "other"
      ? ({ kind: "other", label: "other", confidence: selected.confidence } as const)
      : ({ kind: "target", label: selected.label, confidence: selected.confidence } as const);
  return {
    decision,
    reason: decision.kind === "other" ? "accepted_other" : "accepted_target",
    rankedScores,
  };
}

/** Apply the one supported candidate policy without guessing on invalid runtime input. */
export function decideCandidate(
  candidateActive: boolean,
  probabilities: unknown,
  policy: unknown,
): CandidateDecision {
  if (!candidateActive) return { kind: "inactive" };
  return scoreCandidate(probabilities, policy)?.decision ?? { kind: "abstain" };
}
