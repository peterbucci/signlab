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

const TEMPERATURE = candidateDecisionPolicy.temperature.temperature_milli / 1_000;
const THRESHOLD = candidateDecisionPolicy.abstention.threshold_percent / 100;
const PROBABILITY_SUM_TOLERANCE = 1e-5 + 1e-5;
const LOG_FLOOR = 1e-7;

function exactPolicy(value: unknown, expected: unknown = candidateDecisionPolicy): boolean {
  if (Array.isArray(expected)) {
    return (
      Array.isArray(value) &&
      value.length === expected.length &&
      expected.every((item, index) => exactPolicy(value[index], item))
    );
  }
  if (typeof expected === "object" && expected !== null) {
    if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
    const actual = value as Record<string, unknown>;
    const entries = Object.entries(expected);
    return (
      Object.keys(actual).length === entries.length &&
      entries.every(([key, item]) => exactPolicy(actual[key], item))
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

/** Apply the one supported candidate policy without guessing on invalid runtime input. */
export function decideCandidate(
  candidateActive: boolean,
  probabilities: unknown,
  policy: unknown,
): CandidateDecision {
  if (!candidateActive) return { kind: "inactive" };

  const checkedProbabilities = validProbabilities(probabilities);
  if (!exactPolicy(policy) || checkedProbabilities === null) {
    return { kind: "abstain" };
  }

  const calibrated = calibratedProbabilities(checkedProbabilities);
  let selectedIndex = 0;
  for (let index = 1; index < calibrated.length; index += 1) {
    if (calibrated[index]! > calibrated[selectedIndex]!) selectedIndex = index;
  }
  const confidence = calibrated[selectedIndex]!;
  const label = CANDIDATE_LABELS[selectedIndex]!;

  if (confidence < THRESHOLD) return { kind: "abstain" };
  return label === "other"
    ? { kind: "other", label, confidence }
    : { kind: "target", label, confidence };
}
