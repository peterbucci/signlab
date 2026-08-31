import * as ort from "onnxruntime-web/wasm";

import decisionPolicy from "../../../../docs/reports/popsign-constructed-calibration-policy-v1.json";
import featurePlan from "../../../../src/signlab/resources/features/config/hand-local-64-1.default.json";
import { exactJson, scoreCandidate } from "./candidateDecision";
import {
  CANDIDATE_INFERENCE_PROTOCOL_VERSION,
  type CandidateInferenceFailureCode,
  type CandidateInferenceWorkerInput,
  type CandidateInferenceWorkerOutput,
  type ClassifyCandidate,
  type InitializeCandidateInference,
} from "./candidateInferenceProtocol";
import { preprocessCandidate } from "./candidatePreprocessing";

export interface CandidateInferenceEngine {
  run(values: Float32Array): Promise<unknown>;
  close(): Promise<void> | void;
}

type EngineFactory = (model: ArrayBuffer) => Promise<CandidateInferenceEngine>;

export const CANDIDATE_BROWSER_RUNTIME = Object.freeze({
  backend: "wasm",
  wasmThreads: 1,
} as const);

export async function createCandidateInferenceEngine(
  modelBuffer: ArrayBuffer,
): Promise<CandidateInferenceEngine> {
  ort.env.wasm.numThreads = CANDIDATE_BROWSER_RUNTIME.wasmThreads;
  ort.env.wasm.proxy = false;
  const session = await ort.InferenceSession.create(new Uint8Array(modelBuffer), {
    executionProviders: [CANDIDATE_BROWSER_RUNTIME.backend],
    executionMode: "sequential",
  });
  const input = session.inputMetadata.length === 1 ? session.inputMetadata[0] : undefined;
  const output = session.outputMetadata.length === 1 ? session.outputMetadata[0] : undefined;
  if (
    !input?.isTensor ||
    input.name !== "input" ||
    input.type !== "float32" ||
    input.shape.length !== 3 ||
    input.shape[0] !== 1 ||
    input.shape[1] !== 64 ||
    input.shape[2] !== 126 ||
    !output?.isTensor ||
    output.name !== "probabilities" ||
    output.type !== "float32" ||
    output.shape.length !== 2 ||
    output.shape[0] !== 1 ||
    output.shape[1] !== 6
  ) {
    await session.release();
    throw new Error("candidate.inference.model_contract.invalid");
  }
  return {
    async run(values) {
      if (values.length !== 64 * 126) {
        throw new Error("candidate.inference.tensor.invalid");
      }
      const results = await session.run({
        input: new ort.Tensor("float32", values, [1, 64, 126]),
      });
      const output = results.probabilities;
      if (
        output?.type !== "float32" ||
        output.dims.join() !== "1,6" ||
        !(output.data instanceof Float32Array) ||
        output.data.length !== 6
      ) {
        throw new Error("candidate.inference.output_contract.invalid");
      }
      return new Float32Array(output.data);
    },
    close: () => session.release(),
  };
}

function requireExactJson(buffer: ArrayBuffer, expected: unknown): void {
  try {
    const value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(buffer)) as unknown;
    if (!exactJson(value, expected)) throw new Error();
  } catch {
    throw new Error("candidate.inference.config.invalid");
  }
}

const safeId = (value: number) => Number.isSafeInteger(value) && value >= 0;
const elapsed = (start: number, end: number) => Math.max(0, end - start);

export class CandidateInferenceSession {
  private engine: CandidateInferenceEngine | null = null;
  private bundle: { readonly id: string; readonly version: string } | null = null;
  private initializing = false;
  private busy = false;
  private stopped = false;

  constructor(
    private readonly createEngine: EngineFactory,
    private readonly post: (message: CandidateInferenceWorkerOutput) => void,
    private readonly now: () => number = () => performance.now(),
  ) {}

  async handle(message: CandidateInferenceWorkerInput): Promise<void> {
    if (message.protocolVersion !== CANDIDATE_INFERENCE_PROTOCOL_VERSION) {
      this.fail("candidate.inference.protocol.invalid", null, true);
    } else if (message.type === "initialize") await this.initialize(message);
    else if (message.type === "classify") await this.classify(message);
    else if (message.type === "stop") await this.stop();
    else this.fail("candidate.inference.protocol.invalid", null, true);
  }

  private async initialize(message: InitializeCandidateInference): Promise<void> {
    if (
      this.stopped ||
      this.initializing ||
      this.engine !== null ||
      message.bundle.id.length === 0 ||
      message.bundle.version.length === 0
    ) {
      this.fail("candidate.inference.protocol.invalid", null, true);
      return;
    }
    this.initializing = true;
    const started = this.now();
    try {
      requireExactJson(message.featurePlanBuffer, featurePlan);
      requireExactJson(message.decisionPolicyBuffer, decisionPolicy);
      const engine = await this.createEngine(message.modelBuffer);
      if (this.stopped) {
        await engine.close();
        return;
      }
      this.engine = engine;
      this.bundle = Object.freeze({ ...message.bundle });
      this.post({
        type: "ready",
        protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
        bundle: this.bundle,
        backend: CANDIDATE_BROWSER_RUNTIME.backend,
        startupMs: elapsed(started, this.now()),
      });
    } catch {
      this.fail("candidate.inference.initialization.failed", null, true);
    } finally {
      this.initializing = false;
    }
  }

  private async classify(message: ClassifyCandidate): Promise<void> {
    if (!safeId(message.requestId) || !this.engine || !this.bundle || this.stopped) {
      this.fail("candidate.inference.protocol.invalid", message.requestId, true);
      return;
    }
    if (this.busy) {
      this.fail("candidate.inference.protocol.invalid", message.requestId, false);
      return;
    }
    this.busy = true;
    const started = this.now();
    try {
      let tensor: Float32Array;
      try {
        const { frames, sourceMirrorState, quality } = message.input;
        tensor = preprocessCandidate(frames, sourceMirrorState, quality, featurePlan).values;
      } catch {
        this.fail("candidate.inference.input.invalid", message.requestId, false);
        return;
      }
      const preprocessed = this.now();
      let probabilities: unknown;
      try {
        probabilities = await this.engine.run(tensor);
      } catch (error) {
        if (!this.stopped) {
          const invalidOutput =
            error instanceof Error &&
            error.message === "candidate.inference.output_contract.invalid";
          this.fail(
            invalidOutput
              ? "candidate.inference.output.invalid"
              : "candidate.inference.runtime.failed",
            message.requestId,
            false,
          );
        }
        return;
      }
      const inferred = this.now();
      const scored = scoreCandidate(probabilities, decisionPolicy);
      if (scored === null) {
        this.fail("candidate.inference.output.invalid", message.requestId, false);
        return;
      }
      const decided = this.now();
      if (this.stopped) return;
      this.post({
        type: "result",
        protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
        requestId: message.requestId,
        ...scored,
        bundle: this.bundle,
        backend: CANDIDATE_BROWSER_RUNTIME.backend,
        timings: {
          preprocessingMs: elapsed(started, preprocessed),
          inferenceMs: elapsed(preprocessed, inferred),
          decisionMs: elapsed(inferred, decided),
          totalMs: elapsed(started, decided),
        },
      });
    } finally {
      this.busy = false;
    }
  }

  private async stop(): Promise<void> {
    if (this.stopped) return;
    this.stopped = true;
    const engine = this.engine;
    this.engine = null;
    await Promise.resolve(engine?.close()).catch(() => undefined);
    this.post({
      type: "stopped",
      protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
    });
  }

  private fail(
    code: CandidateInferenceFailureCode,
    requestId: number | null,
    fatal: boolean,
  ): void {
    this.post({
      type: "failure",
      protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
      code,
      requestId: safeId(requestId ?? -1) ? requestId : null,
      fatal,
    });
  }
}
