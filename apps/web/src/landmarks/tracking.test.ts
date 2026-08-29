import { describe, expect, it } from "vitest";

import type { LandmarkPoint, ReportedHandedness } from "./protocol";
import { HandIdentityTracker, type HandDetection } from "./tracking";

function point(x: number): LandmarkPoint {
  return { x, y: 0.5, z: 0, visibility: null, presence: null };
}

function hand(
  detectorIndex: number,
  x: number,
  reportedHandedness: ReportedHandedness,
): HandDetection {
  const landmarks = Array.from({ length: 21 }, () => point(x));
  return {
    detectorIndex,
    imageLandmarks: landmarks,
    worldLandmarks: landmarks,
    reportedHandedness,
    handednessConfidence: 0.9,
  };
}

function indices(result: ReturnType<HandIdentityTracker["track"]>): Array<number | null> {
  return result.map((detection) => detection?.detectorIndex ?? null);
}

describe("HandIdentityTracker", () => {
  it("initializes from geometry and keeps identities when detector order reverses", () => {
    const tracker = new HandIdentityTracker();

    expect(indices(tracker.track([hand(3, 0.8, "right"), hand(9, 0.2, "left")]))).toEqual([9, 3]);
    expect(indices(tracker.track([hand(0, 0.78, "right"), hand(1, 0.22, "left")]))).toEqual([1, 0]);
  });

  it("uses handedness to resolve otherwise equal spatial assignments", () => {
    const tracker = new HandIdentityTracker();
    tracker.track([hand(0, 0.4, "left"), hand(1, 0.6, "right")]);

    expect(indices(tracker.track([hand(0, 0.5, "right"), hand(1, 0.5, "left")]))).toEqual([1, 0]);
  });

  it("falls back deterministically when continuity is ambiguous", () => {
    const tracker = new HandIdentityTracker();
    tracker.track([hand(0, 0.25, "left"), hand(1, 0.75, "left")]);

    expect(indices(tracker.track([hand(7, 0.5, "left"), hand(4, 0.5, "left")]))).toEqual([4, 7]);
    expect(indices(tracker.track([hand(7, 0.7, "left"), hand(4, 0.3, "left")]))).toEqual([4, 7]);
  });

  it("retains identity across an empty frame and forgets it only on reset", () => {
    const tracker = new HandIdentityTracker();
    tracker.track([hand(0, 0.2, "left"), hand(1, 0.8, "right")]);

    expect(indices(tracker.track([]))).toEqual([null, null]);
    expect(indices(tracker.track([hand(5, 0.78, "right"), hand(6, 0.22, "left")]))).toEqual([6, 5]);

    tracker.reset();
    expect(indices(tracker.track([hand(9, 0.9, "left"), hand(3, 0.1, "right")]))).toEqual([3, 9]);
  });
});
