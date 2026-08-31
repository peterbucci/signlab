import { describe, expect, it } from "vitest";

import config from "../../../../configs/evaluation/candidate-event-detector-v1.json";
import fixture from "../../../../tests/fixtures/public/events/candidate-event-stream-v1.json";
import {
  absentHandSlots,
  type HandSlot,
  type HandSlotId,
  type LandmarkPoint,
} from "../landmarks/protocol";
import {
  CandidateEventDetector,
  CandidateObservationProjector,
  createCandidateEventDetector,
  type CandidateEvent,
  type CandidateObservation,
  type CandidateSourceFrame,
} from "./candidateEvents";

const SEGMENTER_SHA256 = "sha256:0443badf68d34347a00096682cf049b6f49b5253c12e47bf61b068a597aa162d";
const QUALITY_SHA256 = "sha256:680b0904e1cc5d8e03119032e92920a3a0185917a600c4293323b7925da9a545";

const detector = () => new CandidateEventDetector(config, SEGMENTER_SHA256);
const observation = (
  timestampUs: number,
  motionQ: number,
  handPresent = true,
  qualityOk = true,
): CandidateObservation => ({ timestampUs, handPresent, qualityOk, motionQ });

function activePrefix(candidate: CandidateEventDetector): void {
  [
    observation(0, 0),
    observation(50_000, 1_000_000),
    observation(100_000, 1_000_000),
    observation(150_000, 1_000_000),
    observation(200_000, 1_000_000),
  ].forEach((row) => expect(candidate.push(row)).toBeNull());
  expect(candidate.state).toBe("recording");
}

const point = (x: number, y = 0): LandmarkPoint => ({
  x,
  y,
  z: 0,
  visibility: null,
  presence: null,
});

function hand(slotId: HandSlotId, x: number, confidence = 0.9): HandSlot {
  const landmarks = Array.from({ length: 21 }, () => point(x));
  return {
    slotId,
    present: true,
    detectorIndex: slotId === "hand_0" ? 0 : 1,
    trackingId: slotId,
    handedness: slotId === "hand_0" ? "left" : "right",
    handednessConfidence: confidence,
    imageLandmarks: landmarks,
    worldLandmarks: landmarks,
  };
}

function frame(
  relativeTimestampUs: number,
  first: HandSlot | null,
  second: HandSlot | null = null,
  valid = true,
): CandidateSourceFrame {
  const absent = absentHandSlots();
  return {
    relativeTimestampUs,
    valid,
    hands: [first ?? absent[0], second ?? absent[1]],
  };
}

describe("browser candidate-event parity", () => {
  it("matches every Python transition and event on the shared fixture", () => {
    const candidate = detector();
    const transitions: Array<{ frame_index: number; state: string }> = [];
    const events: CandidateEvent[] = [];
    let prior = candidate.state;

    fixture.observations.forEach((row, frameIndex) => {
      const event = candidate.push({
        timestampUs: row.timestamp_us,
        handPresent: row.hand_present,
        qualityOk: row.quality_ok,
        motionQ: row.motion_q,
      });
      if (candidate.state !== prior) {
        transitions.push({ frame_index: frameIndex, state: candidate.state });
        prior = candidate.state;
      }
      if (event !== null) events.push(event);
    });

    expect(candidate.finish()).toBeNull();
    expect(transitions).toEqual(fixture.expected_transitions);
    expect(events).toEqual([
      {
        firstFrameIndex: 5,
        lastFrameIndex: 26,
        firstTimestampUs: 250_000,
        lastTimestampUs: 1_250_000,
        terminationReason: "settled",
        configSha256: SEGMENTER_SHA256,
      },
      {
        firstFrameIndex: 36,
        lastFrameIndex: 41,
        firstTimestampUs: 1_750_000,
        lastTimestampUs: 2_000_000,
        terminationReason: "signal_gap",
        configSha256: SEGMENTER_SHA256,
      },
    ]);
    expect(events.every(Object.isFrozen)).toBe(true);
  });

  it("projects hands with Python palm arithmetic and preserves only valid baselines", () => {
    const projector = new CandidateObservationProjector();
    expect(projector.project(frame(0, hand("hand_0", 0), hand("hand_1", 0)))).toEqual(
      observation(0, 0),
    );
    expect(projector.project(frame(25_000, hand("hand_0", 10), null, false))).toEqual(
      observation(25_000, 0, false, false),
    );
    expect(
      projector.project(frame(50_000, hand("hand_0", 0.01, 0.01), hand("hand_1", 0.02, 1))),
    ).toEqual(observation(50_000, 400_000));
    expect(projector.project(frame(75_000, null))).toEqual(observation(75_000, 0, false));
    expect(projector.project(frame(100_000, hand("hand_0", 0.5)))).toEqual(observation(100_000, 0));
    expect(() => projector.project(frame(100_000, null))).toThrow(
      "candidate.events.timestamps_not_increasing",
    );
    projector.reset();
    expect(projector.project(frame(0, hand("hand_0", 0.0000005))).motionQ).toBe(0);
    expect(projector.project(frame(1_000_000, hand("hand_0", 0.0000015))).motionQ).toBe(2);
  });

  it("suppresses static hands and nonzero motion below the start threshold", () => {
    const quiet = detector();
    [0, 100_000, 200_000, 100_000, 0].forEach((motionQ, index) => {
      expect(quiet.push(observation(index * 50_000, motionQ))).toBeNull();
    });
    expect(quiet.state).toBe("inactive");
    expect(quiet.finish()).toBeNull();
  });

  it("honors inclusive gaps, recovers from a short pause, and fails closed on excess", () => {
    const inclusive = detector();
    activePrefix(inclusive);
    expect(inclusive.push(observation(225_000, 0, true, false))).toBeNull();
    expect(inclusive.push(observation(250_000, 0, true, false))).toBeNull();
    expect(inclusive.push(observation(300_000, 1_000_000))).toBeNull();
    expect(inclusive.state).toBe("recording");

    [350_000, 400_000, 450_000, 500_000].forEach((timestamp) =>
      expect(inclusive.push(observation(timestamp, 0))).toBeNull(),
    );
    expect(inclusive.state).toBe("finalizing");
    expect(inclusive.push(observation(550_000, 1_000_000))).toBeNull();
    expect(inclusive.state).toBe("recording");

    const exceeded = detector();
    activePrefix(exceeded);
    expect(exceeded.push(observation(225_000, 0, true, false))).toBeNull();
    expect(exceeded.push(observation(250_000, 0, true, false))).toBeNull();
    expect(exceeded.push(observation(275_000, 0, true, false))).toMatchObject({
      lastTimestampUs: 200_000,
      terminationReason: "signal_gap",
    });
    expect(exceeded.state).toBe("cooldown");

    const timed = detector();
    activePrefix(timed);
    expect(timed.push(observation(300_001, 1_000_000))).toMatchObject({
      lastTimestampUs: 200_000,
      terminationReason: "signal_gap",
    });
  });

  it("enforces ordering, stream end, and the exact maximum-duration deadline", () => {
    const ordered = detector();
    expect(ordered.push(observation(0, 0, false))).toBeNull();
    expect(() => ordered.push(observation(0, 0, false))).toThrow(
      "candidate.events.timestamps_not_increasing",
    );

    const ended = detector();
    activePrefix(ended);
    expect(ended.finish()).toMatchObject({
      firstTimestampUs: 0,
      lastTimestampUs: 200_000,
      terminationReason: "stream_end",
    });
    expect(ended.finish()).toBeNull();
    expect(() => ended.push(observation(250_000, 1_000_000))).toThrow(
      "candidate.events.stream_finished",
    );

    const trailingQuiet = detector();
    activePrefix(trailingQuiet);
    [250_000, 300_000, 350_000, 400_000].forEach((timestamp) =>
      expect(trailingQuiet.push(observation(timestamp, 0))).toBeNull(),
    );
    expect(trailingQuiet.finish()).toMatchObject({
      lastTimestampUs: 350_000,
      terminationReason: "stream_end",
    });

    const tooShort = detector();
    [
      observation(0, 0),
      observation(50_000, 1_000_000),
      observation(100_000, 1_000_000),
      observation(150_000, 1_000_000),
    ].forEach((row) => expect(tooShort.push(row)).toBeNull());
    expect(tooShort.push(observation(175_000, 0, true, false))).toBeNull();
    expect(tooShort.push(observation(200_000, 0, true, false))).toBeNull();
    expect(tooShort.push(observation(225_000, 0, true, false))).toBeNull();
    expect(tooShort.state).toBe("cooldown");

    const bounded = detector();
    activePrefix(bounded);
    for (let timestamp = 300_000; timestamp < 4_000_000; timestamp += 100_000) {
      expect(bounded.push(observation(timestamp, 1_000_000))).toBeNull();
    }
    expect(bounded.push(observation(4_000_000, 1_000_000))).toMatchObject({
      lastTimestampUs: 4_000_000,
      terminationReason: "max_duration",
    });
  });

  it("constructs only from the exact segmenter in a verified bundle", async () => {
    const bundle = (segmenter: unknown, segmenterSha256 = SEGMENTER_SHA256) =>
      ({
        manifest: {
          components: {
            segmenter_sha256: segmenterSha256,
            quality_policy_sha256: QUALITY_SHA256,
          },
        },
        bytesByRole: { segmenter: new Blob([JSON.stringify(segmenter)]) },
      }) as unknown as Parameters<typeof createCandidateEventDetector>[0];
    await expect(createCandidateEventDetector(bundle(config))).resolves.toMatchObject({
      state: "inactive",
      configSha256: SEGMENTER_SHA256,
    });
    await expect(
      createCandidateEventDetector(bundle({ ...config, start_motion_q: 1 })),
    ).rejects.toThrow("candidate.events.invalid_config");
    await expect(
      createCandidateEventDetector(bundle(config, `sha256:${"0".repeat(64)}`)),
    ).rejects.toThrow("candidate.events.incompatible_config");
  });
});
