import type { ScoredCandidateDecision } from "./candidateDecision";
import type {
  CandidateFrame,
  CandidateQualityEvidence,
  SourceMirrorState,
} from "./candidatePreprocessing";

export const CANDIDATE_INFERENCE_PROTOCOL_VERSION = "signlab-candidate-inference-worker/1" as const;

export interface CandidateInferenceInput {
  readonly frames: readonly CandidateFrame[];
  readonly sourceMirrorState: SourceMirrorState;
  readonly quality: CandidateQualityEvidence;
}

export interface InitializeCandidateInference {
  readonly type: "initialize";
  readonly protocolVersion: typeof CANDIDATE_INFERENCE_PROTOCOL_VERSION;
  readonly bundle: { readonly id: string; readonly version: string };
  readonly modelBuffer: ArrayBuffer;
  readonly featurePlanBuffer: ArrayBuffer;
  readonly decisionPolicyBuffer: ArrayBuffer;
}

export interface ClassifyCandidate {
  readonly type: "classify";
  readonly protocolVersion: typeof CANDIDATE_INFERENCE_PROTOCOL_VERSION;
  readonly requestId: number;
  readonly input: CandidateInferenceInput;
}

export interface StopCandidateInference {
  readonly type: "stop";
  readonly protocolVersion: typeof CANDIDATE_INFERENCE_PROTOCOL_VERSION;
}

export type CandidateInferenceWorkerInput =
  InitializeCandidateInference | ClassifyCandidate | StopCandidateInference;

interface CandidateInferenceOutputBase {
  readonly protocolVersion: typeof CANDIDATE_INFERENCE_PROTOCOL_VERSION;
}

export interface CandidateInferenceReady extends CandidateInferenceOutputBase {
  readonly type: "ready";
  readonly bundle: { readonly id: string; readonly version: string };
  readonly backend: "wasm";
  readonly startupMs: number;
}

export interface CandidateInferenceTimings {
  readonly preprocessingMs: number;
  readonly inferenceMs: number;
  readonly decisionMs: number;
  readonly totalMs: number;
}

export interface CandidateInferenceResult
  extends CandidateInferenceOutputBase, ScoredCandidateDecision {
  readonly type: "result";
  readonly requestId: number;
  readonly bundle: { readonly id: string; readonly version: string };
  readonly backend: "wasm";
  readonly timings: CandidateInferenceTimings;
}

export type CandidateInferenceFailureCode =
  | "candidate.inference.initialization.failed"
  | "candidate.inference.input.invalid"
  | "candidate.inference.runtime.failed"
  | "candidate.inference.output.invalid"
  | "candidate.inference.protocol.invalid";

export interface CandidateInferenceFailure extends CandidateInferenceOutputBase {
  readonly type: "failure";
  readonly code: CandidateInferenceFailureCode;
  readonly requestId: number | null;
  readonly fatal: boolean;
}

export interface CandidateInferenceStopped extends CandidateInferenceOutputBase {
  readonly type: "stopped";
}

export type CandidateInferenceWorkerOutput =
  | CandidateInferenceReady
  | CandidateInferenceResult
  | CandidateInferenceFailure
  | CandidateInferenceStopped;
