import candidateEventConfig from "../../../../configs/evaluation/candidate-event-detector-v1.json";

import type { LandmarkFrameMessage, LandmarkPoint } from "../landmarks/protocol";
import type { VerifiedModelBundle } from "../modelBundle/modelBundleSession";

const PALM_LANDMARK_INDICES = [0, 5, 9, 17] as const;
const COORDINATE_QUANTIZATION = 1_000_000;
const PPM = 1_000_000;
const CURRENT_SEGMENTER_SHA256 =
  "sha256:0443badf68d34347a00096682cf049b6f49b5253c12e47bf61b068a597aa162d";

export type CandidateEventState = "inactive" | "arming" | "recording" | "finalizing" | "cooldown";
export type CandidateTerminationReason = "settled" | "signal_gap" | "max_duration" | "stream_end";

export interface CandidateObservation {
  readonly timestampUs: number;
  readonly handPresent: boolean;
  readonly qualityOk: boolean;
  readonly motionQ: number;
}

export interface CandidateEvent {
  readonly firstFrameIndex: number;
  readonly lastFrameIndex: number;
  readonly firstTimestampUs: number;
  readonly lastTimestampUs: number;
  readonly terminationReason: CandidateTerminationReason;
  readonly configSha256: string;
}

export type CandidateSourceFrame = Pick<
  LandmarkFrameMessage,
  "relativeTimestampUs" | "valid" | "hands"
>;

type Boundary = readonly [frameIndex: number, timestampUs: number];
type CandidateEventBundle = Pick<VerifiedModelBundle, "manifest" | "bytesByRole">;

function fail(code: string): never {
  throw new Error(code);
}

const isSafeNonnegative = (value: unknown): value is number =>
  typeof value === "number" && Number.isSafeInteger(value) && value >= 0;

function validateConfig(value: unknown): Readonly<typeof candidateEventConfig> {
  if (JSON.stringify(value) !== JSON.stringify(candidateEventConfig)) {
    return fail("candidate.events.invalid_config");
  }
  return Object.freeze({ ...candidateEventConfig });
}

function validateObservation(value: CandidateObservation): void {
  if (
    !isSafeNonnegative(value.timestampUs) ||
    typeof value.handPresent !== "boolean" ||
    typeof value.qualityOk !== "boolean" ||
    !isSafeNonnegative(value.motionQ)
  ) {
    fail("candidate.events.invalid_observation");
  }
}

function roundHalfEven(value: number): number {
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

function palmCentroidQ(points: readonly LandmarkPoint[] | null): readonly [number, number] {
  if (points === null || points.length <= PALM_LANDMARK_INDICES.at(-1)!) {
    return fail("candidate.events.invalid_landmarks");
  }
  const coordinates = PALM_LANDMARK_INDICES.map((index) => points[index]);
  if (
    coordinates.some((point) => point === undefined || ![point.x, point.y].every(Number.isFinite))
  ) {
    return fail("candidate.events.invalid_landmarks");
  }
  const quantize = (axis: "x" | "y") =>
    roundHalfEven(
      (coordinates.reduce((total, point) => total + point![axis], 0) * COORDINATE_QUANTIZATION) / 4,
    );
  return [quantize("x"), quantize("y")];
}

/** Project landmark-worker frames into the runtime-neutral Python detector contract. */
export class CandidateObservationProjector {
  private previous: Array<{
    readonly point: readonly [number, number];
    readonly timestampUs: number;
  } | null> = [null, null];
  private lastTimestampUs: number | null = null;

  project(frame: CandidateSourceFrame): CandidateObservation {
    const timestampUs = frame.relativeTimestampUs;
    if (
      !isSafeNonnegative(timestampUs) ||
      (this.lastTimestampUs !== null && timestampUs <= this.lastTimestampUs)
    ) {
      return fail("candidate.events.timestamps_not_increasing");
    }
    this.lastTimestampUs = timestampUs;
    if (!frame.valid) {
      return Object.freeze({ timestampUs, handPresent: false, qualityOk: false, motionQ: 0 });
    }

    const speeds: number[] = [];
    frame.hands.forEach((hand, slotIndex) => {
      if (!hand.present) {
        this.previous[slotIndex] = null;
        return;
      }
      const point = palmCentroidQ(hand.imageLandmarks);
      const prior = this.previous[slotIndex];
      this.previous[slotIndex] = { point, timestampUs };
      if (prior === null || prior === undefined) return;
      const elapsedUs = timestampUs - prior.timestampUs;
      const distanceQ = Math.abs(point[0] - prior.point[0]) + Math.abs(point[1] - prior.point[1]);
      const speed = Math.floor((distanceQ * PPM + Math.floor(elapsedUs / 2)) / elapsedUs);
      if (!isSafeNonnegative(speed)) fail("candidate.events.invalid_motion");
      speeds.push(speed);
    });
    return Object.freeze({
      timestampUs,
      handPresent: frame.hands.some((hand) => hand.present),
      qualityOk: true,
      motionQ: Math.max(0, ...speeds),
    });
  }

  reset(): void {
    this.previous = [null, null];
    this.lastTimestampUs = null;
  }
}

/** Online parity port of the validated Python candidate-event state machine. */
export class CandidateEventDetector {
  readonly config: Readonly<typeof candidateEventConfig>;
  readonly configSha256: string;
  private stateValue: CandidateEventState = "inactive";
  private nextIndex = 0;
  private lastTimestampUs: number | null = null;
  private smoothedMotionQ: number | null = null;
  private preRoll: Boundary[] = [];
  private armSinceUs: number | null = null;
  private eventStart: Boundary | null = null;
  private lastUsable: Boundary | null = null;
  private lastActive: Boundary | null = null;
  private quietSinceUs: number | null = null;
  private gapFrames = 0;
  private cooldownQuietSinceUs: number | null = null;
  private finished = false;

  constructor(config: unknown, configSha256: string) {
    this.config = validateConfig(config);
    if (configSha256 !== CURRENT_SEGMENTER_SHA256) {
      fail("candidate.events.invalid_config_identity");
    }
    this.configSha256 = configSha256;
  }

  get state(): CandidateEventState {
    return this.stateValue;
  }

  push(observation: CandidateObservation): CandidateEvent | null {
    if (this.finished) fail("candidate.events.stream_finished");
    validateObservation(observation);
    if (this.lastTimestampUs !== null && observation.timestampUs <= this.lastTimestampUs) {
      fail("candidate.events.timestamps_not_increasing");
    }
    const index = this.nextIndex++;
    this.lastTimestampUs = observation.timestampUs;
    const usable = this.updateMotion(observation);
    if (this.stateValue === "inactive") this.consumeInactive(observation, index, usable);
    else if (this.stateValue === "arming") this.consumeArming(observation, index, usable);
    else if (this.stateValue === "recording" || this.stateValue === "finalizing") {
      return this.consumeActive(observation, index, usable);
    } else this.consumeCooldown(observation, usable);
    return null;
  }

  finish(): CandidateEvent | null {
    if (this.finished) return null;
    this.finished = true;
    let event: CandidateEvent | null = null;
    if (
      (this.stateValue === "recording" || this.stateValue === "finalizing") &&
      this.lastUsable !== null
    ) {
      const end = this.stateValue === "finalizing" ? this.lastActive : this.lastUsable;
      if (end !== null) event = this.buildEvent(end, "stream_end");
    }
    this.resetState("inactive", true);
    return event;
  }

  private updateMotion(observation: CandidateObservation): boolean {
    const usable =
      observation.handPresent &&
      observation.qualityOk &&
      observation.motionQ <= this.config.maximum_motion_q;
    if (usable) {
      const alpha = this.config.smoothing_alpha_ppm;
      this.smoothedMotionQ =
        this.smoothedMotionQ === null
          ? observation.motionQ
          : Math.floor(
              (alpha * observation.motionQ +
                (PPM - alpha) * this.smoothedMotionQ +
                Math.floor(PPM / 2)) /
                PPM,
            );
    }
    return usable;
  }

  private consumeInactive(observation: CandidateObservation, index: number, usable: boolean): void {
    if (!usable) {
      if (!observation.handPresent) {
        this.preRoll = [];
        this.lastUsable = null;
        this.smoothedMotionQ = null;
      }
      return;
    }
    const current: Boundary = [index, observation.timestampUs];
    if (
      this.lastUsable !== null &&
      observation.timestampUs - this.lastUsable[1] > this.config.maximum_gap_us
    ) {
      this.preRoll = [current];
      this.smoothedMotionQ = 0;
      this.lastUsable = current;
      return;
    }
    this.lastUsable = current;
    this.preRoll.push(current);
    while (observation.timestampUs - this.preRoll[0]![1] > this.config.pre_roll_us) {
      this.preRoll.shift();
    }
    if (this.motionAtLeast(this.config.start_motion_q)) {
      this.eventStart = this.preRoll[0]!;
      this.armSinceUs = observation.timestampUs;
      this.gapFrames = 0;
      this.stateValue = "arming";
    }
  }

  private consumeArming(observation: CandidateObservation, index: number, usable: boolean): void {
    if (!usable) {
      this.gapFrames += 1;
      if (this.hardGap(observation.timestampUs)) this.resetState("inactive", true);
      return;
    }
    if (this.hardGap(observation.timestampUs)) {
      this.resetState("inactive", true);
      return;
    }
    this.gapFrames = 0;
    if (this.motionAtMost(this.config.stop_motion_q)) {
      this.resetState("inactive", true);
      return;
    }
    if (this.armSinceUs === null || this.eventStart === null) {
      fail("candidate.events.internal_state");
    }
    this.lastUsable = [index, observation.timestampUs];
    if (observation.timestampUs - this.armSinceUs >= this.config.arming_duration_us) {
      this.lastActive = this.lastUsable;
      this.stateValue = "recording";
    }
  }

  private consumeActive(
    observation: CandidateObservation,
    index: number,
    usable: boolean,
  ): CandidateEvent | null {
    if (this.eventStart === null || this.lastUsable === null) {
      return fail("candidate.events.internal_state");
    }
    const deadlineUs = this.eventStart[1] + this.config.maximum_event_duration_us;
    if (!usable) {
      this.gapFrames += 1;
      if (this.hardGap(observation.timestampUs)) {
        return this.terminate(this.lastUsable, "signal_gap", observation, usable);
      }
      if (observation.timestampUs > deadlineUs) {
        return this.terminate(this.lastUsable, "max_duration", observation, usable);
      }
      return null;
    }
    if (this.hardGap(observation.timestampUs)) {
      return this.terminate(this.lastUsable, "signal_gap", observation, usable);
    }
    if (observation.timestampUs > deadlineUs) {
      return this.terminate(this.lastUsable, "max_duration", observation, usable);
    }
    this.gapFrames = 0;
    const current: Boundary = [index, observation.timestampUs];
    this.lastUsable = current;
    if (observation.timestampUs === deadlineUs) {
      return this.terminate(current, "max_duration", observation, usable);
    }
    if (this.stateValue === "recording") {
      if (this.motionAtMost(this.config.stop_motion_q)) {
        this.quietSinceUs = observation.timestampUs;
        this.stateValue = "finalizing";
      } else this.lastActive = current;
      return null;
    }
    if (this.quietSinceUs === null || this.lastActive === null) {
      return fail("candidate.events.internal_state");
    }
    if (observation.timestampUs - this.quietSinceUs >= this.config.finalization_duration_us) {
      return this.terminate(this.lastActive, "settled", observation, usable);
    }
    if (this.motionAtLeast(this.config.start_motion_q)) {
      this.lastActive = current;
      this.quietSinceUs = null;
      this.stateValue = "recording";
    }
    return null;
  }

  private consumeCooldown(observation: CandidateObservation, usable: boolean): void {
    const stableQuiet =
      observation.qualityOk &&
      (!observation.handPresent || (usable && this.motionAtMost(this.config.stop_motion_q)));
    if (!stableQuiet) {
      this.cooldownQuietSinceUs = null;
      return;
    }
    this.cooldownQuietSinceUs ??= observation.timestampUs;
    if (observation.timestampUs - this.cooldownQuietSinceUs >= this.config.cooldown_duration_us) {
      this.resetState("inactive", true);
    }
  }

  private hardGap(timestampUs: number): boolean {
    const reference = this.lastUsable ?? this.eventStart;
    return (
      reference === null ||
      this.gapFrames > this.config.maximum_gap_frames ||
      timestampUs - reference[1] > this.config.maximum_gap_us
    );
  }

  private motionAtLeast(threshold: number): boolean {
    return this.smoothedMotionQ !== null && this.smoothedMotionQ >= threshold;
  }

  private motionAtMost(threshold: number): boolean {
    return this.smoothedMotionQ !== null && this.smoothedMotionQ <= threshold;
  }

  private terminate(
    end: Boundary,
    reason: CandidateTerminationReason,
    observation: CandidateObservation,
    usable: boolean,
  ): CandidateEvent | null {
    const event = this.buildEvent(end, reason);
    this.resetState("cooldown");
    if (
      observation.qualityOk &&
      (!observation.handPresent || (usable && this.motionAtMost(this.config.stop_motion_q)))
    ) {
      this.cooldownQuietSinceUs = observation.timestampUs;
    }
    return event;
  }

  private buildEvent(end: Boundary, reason: CandidateTerminationReason): CandidateEvent | null {
    if (this.eventStart === null) return fail("candidate.events.internal_state");
    if (end[1] - this.eventStart[1] < this.config.minimum_event_duration_us) return null;
    return Object.freeze({
      firstFrameIndex: this.eventStart[0],
      lastFrameIndex: end[0],
      firstTimestampUs: this.eventStart[1],
      lastTimestampUs: end[1],
      terminationReason: reason,
      configSha256: this.configSha256,
    });
  }

  private resetState(state: CandidateEventState, resetMotion = false): void {
    this.stateValue = state;
    this.preRoll = [];
    this.armSinceUs = null;
    this.eventStart = null;
    this.lastUsable = null;
    this.lastActive = null;
    this.quietSinceUs = null;
    this.gapFrames = 0;
    this.cooldownQuietSinceUs = null;
    if (resetMotion) this.smoothedMotionQ = null;
  }
}

/** Create the detector only from the segmenter already verified by the bundle loader. */
export async function createCandidateEventDetector(
  bundle: CandidateEventBundle,
): Promise<CandidateEventDetector> {
  let value: unknown;
  try {
    value = JSON.parse(await bundle.bytesByRole.segmenter.text()) as unknown;
  } catch {
    return fail("candidate.events.invalid_config");
  }
  if (
    bundle.manifest.components.segmenter_sha256 !== CURRENT_SEGMENTER_SHA256 ||
    bundle.manifest.components.quality_policy_sha256 !== candidateEventConfig.quality_policy_sha256
  ) {
    return fail("candidate.events.incompatible_config");
  }
  return new CandidateEventDetector(value, CURRENT_SEGMENTER_SHA256);
}
