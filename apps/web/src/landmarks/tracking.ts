import type { LandmarkPoint, ReportedHandedness } from "./protocol";

const PALM_LANDMARK_INDICES = [0, 5, 9, 13, 17] as const;

export const HAND_TRACKING_CONFIG = {
  maxSpatialCost: 0.25,
  handednessDisagreementPenalty: 0.05,
  ambiguityMargin: 1e-9,
} as const;

export interface HandDetection {
  readonly detectorIndex: number;
  readonly imageLandmarks: readonly LandmarkPoint[];
  readonly worldLandmarks: readonly LandmarkPoint[];
  readonly reportedHandedness: ReportedHandedness;
  readonly handednessConfidence: number;
}

export type TrackedDetections = readonly [HandDetection | null, HandDetection | null];

interface Assignment {
  readonly pairs: readonly (readonly [number, number])[];
  readonly cost: number;
}

function squaredDistance(
  first: readonly [number, number, number],
  second: readonly [number, number, number],
): number {
  const x = first[0] - second[0];
  const y = first[1] - second[1];
  const z = first[2] - second[2];
  return x * x + y * y + z * z;
}

function pointCoordinates(point: LandmarkPoint): readonly [number, number, number] {
  return [point.x, point.y, point.z];
}

function palmCentroid(detection: HandDetection): readonly [number, number, number] {
  let x = 0;
  let y = 0;
  let z = 0;
  for (const index of PALM_LANDMARK_INDICES) {
    const point = detection.imageLandmarks[index];
    if (point === undefined) {
      throw new Error("A hand detection must contain exactly 21 image landmarks");
    }
    x += point.x;
    y += point.y;
    z += point.z;
  }
  const count = PALM_LANDMARK_INDICES.length;
  return [x / count, y / count, z / count];
}

function spatialCost(previous: HandDetection, current: HandDetection): number {
  const previousWrist = previous.imageLandmarks[0];
  const currentWrist = current.imageLandmarks[0];
  if (previousWrist === undefined || currentWrist === undefined) {
    throw new Error("A hand detection must contain exactly 21 image landmarks");
  }
  return (
    squaredDistance(pointCoordinates(previousWrist), pointCoordinates(currentWrist)) +
    squaredDistance(palmCentroid(previous), palmCentroid(current))
  );
}

function assignmentCost(
  previous: HandDetection,
  current: HandDetection,
): readonly [number, number] {
  const spatial = spatialCost(previous, current);
  const handednessPenalty =
    previous.reportedHandedness === current.reportedHandedness
      ? 0
      : HAND_TRACKING_CONFIG.handednessDisagreementPenalty;
  return [spatial, spatial + handednessPenalty];
}

function compareNumber(left: number, right: number): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function compareString(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function compareDetections(left: HandDetection, right: HandDetection): number {
  const leftWrist = left.imageLandmarks[0];
  const rightWrist = right.imageLandmarks[0];
  if (leftWrist === undefined || rightWrist === undefined) {
    throw new Error("A hand detection must contain exactly 21 image landmarks");
  }
  const leftPalm = palmCentroid(left);
  const rightPalm = palmCentroid(right);
  const numericComparisons = [
    compareNumber(leftWrist.x, rightWrist.x),
    compareNumber(leftWrist.y, rightWrist.y),
    compareNumber(leftWrist.z, rightWrist.z),
    compareNumber(leftPalm[0], rightPalm[0]),
    compareNumber(leftPalm[1], rightPalm[1]),
    compareNumber(leftPalm[2], rightPalm[2]),
  ];
  for (const comparison of numericComparisons) {
    if (comparison !== 0) return comparison;
  }
  const handednessComparison = compareString(left.reportedHandedness, right.reportedHandedness);
  return handednessComparison === 0
    ? compareNumber(left.detectorIndex, right.detectorIndex)
    : handednessComparison;
}

function comparePairSignatures(
  left: readonly (readonly [number, number])[],
  right: readonly (readonly [number, number])[],
  detections: readonly HandDetection[],
): number {
  for (let index = 0; index < Math.min(left.length, right.length); index += 1) {
    const leftPair = left[index];
    const rightPair = right[index];
    if (leftPair === undefined || rightPair === undefined) continue;
    const slotComparison = compareNumber(leftPair[0], rightPair[0]);
    if (slotComparison !== 0) return slotComparison;
    const leftDetection = detections[leftPair[1]];
    const rightDetection = detections[rightPair[1]];
    if (leftDetection === undefined || rightDetection === undefined) {
      throw new Error("Tracking assignment referenced a missing detection");
    }
    const detectionComparison = compareDetections(leftDetection, rightDetection);
    if (detectionComparison !== 0) return detectionComparison;
  }
  return compareNumber(left.length, right.length);
}

function combinations(values: readonly number[], count: number): number[][] {
  if (count === 0) return [[]];
  const output: number[][] = [];
  values.forEach((value, index) => {
    for (const remainder of combinations(values.slice(index + 1), count - 1)) {
      output.push([value, ...remainder]);
    }
  });
  return output;
}

function permutations(values: readonly number[], count: number): number[][] {
  if (count === 0) return [[]];
  const output: number[][] = [];
  values.forEach((value, index) => {
    const remaining = [...values.slice(0, index), ...values.slice(index + 1)];
    for (const remainder of permutations(remaining, count - 1)) {
      output.push([value, ...remainder]);
    }
  });
  return output;
}

function enumerateAssignments(
  previous: TrackedDetections,
  detections: readonly HandDetection[],
): Assignment[] {
  const activeSlots = [0, 1].filter((index) => previous[index] !== null);
  const positions = detections.map((_, index) => index);
  const assignments: Assignment[] = [];
  const maximumPairs = Math.min(activeSlots.length, detections.length);

  for (let pairCount = 0; pairCount <= maximumPairs; pairCount += 1) {
    for (const slots of combinations(activeSlots, pairCount)) {
      for (const orderedPositions of permutations(positions, pairCount)) {
        const pairs = slots.map((slot, index) => [slot, orderedPositions[index] ?? -1] as const);
        let cost = 0;
        let valid = true;
        for (const [slot, position] of pairs) {
          const prior = previous[slot];
          const current = detections[position];
          if (prior === null || prior === undefined || current === undefined) {
            valid = false;
            break;
          }
          const [spatial, total] = assignmentCost(prior, current);
          if (spatial > HAND_TRACKING_CONFIG.maxSpatialCost) {
            valid = false;
            break;
          }
          cost += total;
        }
        if (valid) assignments.push({ pairs, cost });
      }
    }
  }
  return assignments;
}

function unambiguousPairs(
  assignments: readonly Assignment[],
  detections: readonly HandDetection[],
): readonly (readonly [number, number])[] {
  const maximumPairCount = Math.max(...assignments.map(({ pairs }) => pairs.length));
  const fullest = assignments.filter(({ pairs }) => pairs.length === maximumPairCount);
  fullest.sort(
    (left, right) =>
      compareNumber(left.cost, right.cost) ||
      comparePairSignatures(left.pairs, right.pairs, detections),
  );
  const best = fullest[0];
  if (best === undefined) return [];
  const competitive = fullest.filter(
    ({ cost }) => cost <= best.cost + HAND_TRACKING_CONFIG.ambiguityMargin,
  );
  return best.pairs.filter(([slot, position]) =>
    competitive.every(({ pairs }) =>
      pairs.some(([candidateSlot, candidatePosition]) => {
        return slot === candidateSlot && position === candidatePosition;
      }),
    ),
  );
}

function fallbackPairs(
  previous: TrackedDetections,
  detections: readonly HandDetection[],
  availableSlots: readonly number[],
  detectionPositions: readonly number[],
): readonly (readonly [number, number])[] {
  const candidates: Array<{
    overwrittenSlots: number;
    cost: number;
    pairs: readonly (readonly [number, number])[];
  }> = [];
  for (const slots of combinations(availableSlots, detectionPositions.length)) {
    for (const orderedPositions of permutations(detectionPositions, detectionPositions.length)) {
      const pairs = slots.map((slot, index) => [slot, orderedPositions[index] ?? -1] as const);
      let overwrittenSlots = 0;
      let cost = 0;
      for (const [slot, position] of pairs) {
        const prior = previous[slot];
        const current = detections[position];
        if (prior !== null && prior !== undefined && current !== undefined) {
          overwrittenSlots += 1;
          cost += assignmentCost(prior, current)[1];
        }
      }
      candidates.push({ overwrittenSlots, cost, pairs });
    }
  }
  candidates.sort(
    (left, right) =>
      compareNumber(left.overwrittenSlots, right.overwrittenSlots) ||
      compareNumber(left.cost, right.cost) ||
      comparePairSignatures(left.pairs, right.pairs, detections),
  );
  return candidates[0]?.pairs ?? [];
}

export class HandIdentityTracker {
  private previous: TrackedDetections = [null, null];

  reset(): void {
    this.previous = [null, null];
  }

  track(detections: readonly HandDetection[]): TrackedDetections {
    if (detections.length > 2) {
      throw new Error("HandIdentityTracker accepts at most two detections per frame");
    }
    const detectorIndices = detections.map(({ detectorIndex }) => detectorIndex);
    if (new Set(detectorIndices).size !== detectorIndices.length) {
      throw new Error("detectorIndex values must be unique within a frame");
    }
    if (detections.length === 0) return [null, null];

    const activeSlots = [0, 1].filter((index) => this.previous[index] !== null);
    if (activeSlots.length === 0) {
      const ordered = [...detections].sort(compareDetections);
      const initialized: TrackedDetections = [ordered[0] ?? null, ordered[1] ?? null];
      this.previous = initialized;
      return initialized;
    }

    const pairs = unambiguousPairs(enumerateAssignments(this.previous, detections), detections);
    const tracked: [HandDetection | null, HandDetection | null] = [null, null];
    const nextPrevious: [HandDetection | null, HandDetection | null] = [...this.previous];
    const usedSlots = new Set<number>();
    const usedPositions = new Set<number>();

    for (const [slot, position] of pairs) {
      const detection = detections[position];
      if (detection === undefined || (slot !== 0 && slot !== 1)) continue;
      tracked[slot] = detection;
      nextPrevious[slot] = detection;
      usedSlots.add(slot);
      usedPositions.add(position);
    }

    const fallbackPositions = detections
      .map((_, index) => index)
      .filter((index) => !usedPositions.has(index));
    const fallbackSlots = [0, 1].filter((index) => !usedSlots.has(index));
    for (const [slot, position] of fallbackPairs(
      this.previous,
      detections,
      fallbackSlots,
      fallbackPositions,
    )) {
      const detection = detections[position];
      if (detection === undefined || (slot !== 0 && slot !== 1)) continue;
      tracked[slot] = detection;
      nextPrevious[slot] = detection;
    }

    this.previous = nextPrevious;
    return tracked;
  }
}
