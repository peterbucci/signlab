/// <reference types="node" />
// @vitest-environment node

import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import { describe, expect, it, vi } from "vitest";

import decisionPolicy from "../../../../docs/reports/popsign-constructed-calibration-policy-v1.json";
import featurePlan from "../../../../src/signlab/resources/features/config/hand-local-64-1.default.json";
import fixture from "../../../../tests/fixtures/public/parity/candidate-runtime-goldens-v1.json";
import {
  CANDIDATE_INFERENCE_PROTOCOL_VERSION,
  type CandidateInferenceFailure,
  type CandidateInferenceInput,
  type CandidateInferenceResult,
  type CandidateInferenceWorkerInput,
  type CandidateInferenceWorkerOutput,
  type InitializeCandidateInference,
} from "./candidateInferenceProtocol";
import {
  CANDIDATE_BROWSER_RUNTIME,
  CandidateInferenceSession,
  createCandidateInferenceEngine,
  type CandidateInferenceEngine,
} from "./candidateInferenceSession";

const repositoryRoot = resolve(process.cwd(), "../..");
const bundle = { id: "candidate-fixture", version: "1" } as const;
const source = fixture.preprocessingCases.find(({ id }) => id === "gap_unmirrored");
const runtime = fixture.onnx.cases.find(
  ({ preprocessingCaseId }) => preprocessingCaseId === "gap_unmirrored",
);
if (source === undefined || runtime === undefined) throw new Error("missing candidate golden");

const input = {
  frames: source.frames,
  sourceMirrorState: source.sourceMirrorState,
  quality: source.quality,
} as unknown as CandidateInferenceInput;

const encode = (value: unknown): ArrayBuffer =>
  new TextEncoder().encode(JSON.stringify(value)).buffer;
const initialize = (
  modelBuffer = new Uint8Array([1, 2, 3]).buffer,
): InitializeCandidateInference => ({
  type: "initialize",
  protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
  bundle,
  modelBuffer,
  featurePlanBuffer: encode(featurePlan),
  decisionPolicyBuffer: encode(Object.fromEntries(Object.entries(decisionPolicy).reverse())),
});
const classify = (requestId: number, candidateInput = input) => ({
  type: "classify" as const,
  protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
  requestId,
  input: candidateInput,
});
const expectedTensorQ = () => {
  const expected = source.expected as unknown as {
    readonly nonPaddingFrameCount: number;
    readonly valuesQ: readonly (readonly number[])[];
  };
  return expected.valuesQ.flat().concat(Array((64 - expected.nonPaddingFrameCount) * 126).fill(0));
};

describe("candidate inference session", () => {
  it("pins the browser release to one WASM thread", () => {
    expect(CANDIDATE_BROWSER_RUNTIME).toEqual({ backend: "wasm", wasmThreads: 1 });
  });

  it("reports an unknown message as a typed protocol failure", async () => {
    const messages: CandidateInferenceWorkerOutput[] = [];
    const session = new CandidateInferenceSession(vi.fn(), (message) => messages.push(message));

    await session.handle({
      type: "unknown",
      protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
    } as unknown as CandidateInferenceWorkerInput);

    expect(messages).toEqual([
      {
        type: "failure",
        protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
        code: "candidate.inference.protocol.invalid",
        requestId: null,
        fatal: true,
      },
    ]);
  });

  it("initializes exact configs, reuses one engine, scores golden inputs, and stops cleanly", async () => {
    const messages: CandidateInferenceWorkerOutput[] = [];
    const close = vi.fn();
    const tensors: Float32Array[] = [];
    const createEngine = vi.fn(async (modelBuffer: ArrayBuffer) => {
      const engine = await createCandidateInferenceEngine(modelBuffer);
      return {
        run: async (values: Float32Array) => {
          tensors.push(values);
          return engine.run(values);
        },
        close: async () => {
          close();
          await engine.close();
        },
      };
    });
    let clock = 0;
    const session = new CandidateInferenceSession(
      createEngine,
      (message) => messages.push(message),
      () => ++clock,
    );

    const model = Uint8Array.from(
      await readFile(resolve(repositoryRoot, fixture.resources.testModel.path)),
    ).buffer;
    await session.handle(initialize(model));
    await session.handle(classify(7));
    await session.handle(classify(8));
    await session.handle({
      type: "stop",
      protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
    });

    expect(messages[0]).toEqual({
      type: "ready",
      protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
      bundle,
      backend: "wasm",
      startupMs: 1,
    });
    const results = messages.filter(
      (message): message is CandidateInferenceResult => message.type === "result",
    );
    expect(results).toHaveLength(2);
    expect(results.map(({ requestId }) => requestId)).toEqual([7, 8]);
    expect(results[0]).toMatchObject({
      decision: { kind: "target", label: "hello" },
      reason: "accepted_target",
      bundle,
      backend: "wasm",
      timings: { preprocessingMs: 1, inferenceMs: 1, decisionMs: 1, totalMs: 3 },
    });
    expect(results[0]!.rankedScores.map(({ label }) => label)).toEqual([
      "hello",
      "yes",
      "please",
      "thank_you",
      "no",
      "other",
    ]);
    expect(Array.from(tensors[0]!, (value) => Math.round(value * 1_000_000))).toEqual(
      expectedTensorQ(),
    );
    expect(createEngine).toHaveBeenCalledTimes(1);
    expect(tensors).toHaveLength(2);
    expect(close).toHaveBeenCalledTimes(1);
    expect(messages.at(-1)).toEqual({
      type: "stopped",
      protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
    });
  });

  it("rejects changed configs before engine creation and sanitizes engine startup failures", async () => {
    for (const key of ["featurePlanBuffer", "decisionPolicyBuffer"] as const) {
      const messages: CandidateInferenceWorkerOutput[] = [];
      const engine: CandidateInferenceEngine = { run: vi.fn(), close: vi.fn() };
      const createEngine = vi.fn(() => Promise.resolve(engine));
      const session = new CandidateInferenceSession(createEngine, (message) =>
        messages.push(message),
      );

      await session.handle({ ...initialize(), [key]: encode({ changed: true }) });

      expect(createEngine).not.toHaveBeenCalled();
      expect(messages).toEqual([
        {
          type: "failure",
          protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
          code: "candidate.inference.initialization.failed",
          requestId: null,
          fatal: true,
        },
      ]);
    }

    const messages: CandidateInferenceWorkerOutput[] = [];
    const session = new CandidateInferenceSession(
      () => Promise.reject(new Error("secret model loader details")),
      (message) => messages.push(message),
    );
    await session.handle(initialize());
    expect(messages).toEqual([
      {
        type: "failure",
        protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
        code: "candidate.inference.initialization.failed",
        requestId: null,
        fatal: true,
      },
    ]);
    expect(JSON.stringify(messages)).not.toContain("secret");
  });

  it("reports sanitized input, runtime, and output failures", async () => {
    const cases: readonly {
      readonly requestId: number;
      readonly candidateInput: CandidateInferenceInput;
      readonly run: (values: Float32Array) => Promise<unknown>;
      readonly code: CandidateInferenceFailure["code"];
    }[] = [
      {
        requestId: 11,
        candidateInput: { ...input, frames: [] },
        run: () => Promise.resolve(new Float32Array(runtime.probabilities)),
        code: "candidate.inference.input.invalid",
      },
      {
        requestId: 12,
        candidateInput: input,
        run: () => Promise.reject(new Error("secret filesystem path and runtime details")),
        code: "candidate.inference.runtime.failed",
      },
      {
        requestId: 13,
        candidateInput: input,
        run: () => Promise.resolve(new Float32Array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])),
        code: "candidate.inference.output.invalid",
      },
    ];

    for (const failureCase of cases) {
      const messages: CandidateInferenceWorkerOutput[] = [];
      const session = new CandidateInferenceSession(
        () => Promise.resolve({ run: failureCase.run, close: vi.fn() }),
        (message) => messages.push(message),
      );
      await session.handle(initialize());
      messages.length = 0;

      await session.handle(classify(failureCase.requestId, failureCase.candidateInput));

      expect(messages).toEqual([
        {
          type: "failure",
          protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
          code: failureCase.code,
          requestId: failureCase.requestId,
          fatal: false,
        },
      ]);
      expect(JSON.stringify(messages)).not.toContain("secret");
    }
  });

  it("defensively rejects an overlapping classification without interrupting the first", async () => {
    const messages: CandidateInferenceWorkerOutput[] = [];
    let resolveRun: ((probabilities: Float32Array) => void) | undefined;
    const pending = new Promise<Float32Array>((resolve) => {
      resolveRun = resolve;
    });
    const session = new CandidateInferenceSession(
      () => Promise.resolve({ run: () => pending, close: vi.fn() }),
      (message) => messages.push(message),
    );
    await session.handle(initialize());

    const first = session.handle(classify(21));
    await session.handle(classify(22));
    expect(messages.at(-1)).toMatchObject({
      type: "failure",
      code: "candidate.inference.protocol.invalid",
      requestId: 22,
      fatal: false,
    });

    resolveRun?.(new Float32Array(runtime.probabilities));
    await first;
    expect(messages.at(-1)).toMatchObject({ type: "result", requestId: 21 });
  });
});
