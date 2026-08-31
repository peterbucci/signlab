import { describe, expect, it, vi } from "vitest";

import type { VerifiedModelBundle } from "../modelBundle/modelBundleSession";
import {
  CandidateInferenceWorkerClient,
  type CandidateInferenceClientEvent,
  type CandidateInferenceWorkerPort,
} from "./CandidateInferenceWorkerClient";
import {
  CANDIDATE_INFERENCE_PROTOCOL_VERSION,
  type CandidateInferenceInput,
  type CandidateInferenceWorkerInput,
  type CandidateInferenceWorkerOutput,
} from "./candidateInferenceProtocol";

class ManualWorkerPort implements CandidateInferenceWorkerPort {
  readonly posts: Array<{
    message: CandidateInferenceWorkerInput;
    transfer: readonly Transferable[];
  }> = [];
  readonly terminate = vi.fn();
  failNextPost = false;
  private messageListener: ((message: CandidateInferenceWorkerOutput) => void) | undefined;
  private errorListener: (() => void) | undefined;

  post(message: CandidateInferenceWorkerInput, transfer: readonly Transferable[] = []): void {
    if (this.failNextPost) {
      this.failNextPost = false;
      throw new Error("worker post failed");
    }
    this.posts.push({ message, transfer });
  }

  onMessage(listener: (message: CandidateInferenceWorkerOutput) => void): () => void {
    this.messageListener = listener;
    return () => {
      this.messageListener = undefined;
    };
  }

  onError(listener: () => void): () => void {
    this.errorListener = listener;
    return () => {
      this.errorListener = undefined;
    };
  }

  emit(message: CandidateInferenceWorkerOutput): void {
    this.messageListener?.(message);
  }

  fail(): void {
    this.errorListener?.();
  }
}

const bundle = (): VerifiedModelBundle =>
  ({
    id: "candidate-browser-model",
    version: "1.0.0",
    manifest: {},
    bytesByRole: {
      model: new Blob([new Uint8Array([1])]),
      feature_plan: new Blob([new Uint8Array([2, 3])]),
      decision_policy: new Blob([new Uint8Array([4, 5, 6])]),
    },
  }) as unknown as VerifiedModelBundle;

const input = {
  frames: [],
  sourceMirrorState: "not_mirrored",
  quality: {},
} as unknown as CandidateInferenceInput;

const ready = (): CandidateInferenceWorkerOutput => ({
  type: "ready",
  protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
  bundle: { id: "candidate-browser-model", version: "1.0.0" },
  backend: "wasm",
  startupMs: 5,
});

const result = (requestId: number): CandidateInferenceWorkerOutput => ({
  type: "result",
  protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
  requestId,
  decision: { kind: "target", label: "hello", confidence: 0.9 },
  reason: "accepted_target",
  rankedScores: [
    { label: "hello", confidence: 0.9 },
    { label: "no", confidence: 0.02 },
    { label: "please", confidence: 0.02 },
    { label: "thank_you", confidence: 0.02 },
    { label: "yes", confidence: 0.02 },
    { label: "other", confidence: 0.02 },
  ],
  bundle: { id: "candidate-browser-model", version: "1.0.0" },
  backend: "wasm",
  timings: { preprocessingMs: 1, inferenceMs: 2, decisionMs: 1, totalMs: 4 },
});

describe("CandidateInferenceWorkerClient", () => {
  it("transfers the verified runtime assets and permits exactly one request at a time", async () => {
    const port = new ManualWorkerPort();
    const events: CandidateInferenceClientEvent[] = [];
    const client = new CandidateInferenceWorkerClient(port, (event) => events.push(event));

    expect(() => client.classify(0, input)).toThrow("candidate.inference.client.not_ready");
    await client.initialize(bundle());
    const initialization = port.posts[0]?.message;
    expect(initialization).toMatchObject({
      type: "initialize",
      bundle: { id: "candidate-browser-model", version: "1.0.0" },
    });
    if (initialization?.type !== "initialize") throw new Error("expected initialization");
    expect([
      initialization.modelBuffer.byteLength,
      initialization.featurePlanBuffer.byteLength,
      initialization.decisionPolicyBuffer.byteLength,
    ]).toEqual([1, 2, 3]);
    expect(port.posts[0]?.transfer.map((buffer) => (buffer as ArrayBuffer).byteLength)).toEqual([
      1, 2, 3,
    ]);

    port.emit(ready());
    client.classify(7, input);
    expect(port.posts.at(-1)?.message).toMatchObject({ type: "classify", requestId: 7, input });
    expect(() => client.classify(8, input)).toThrow("candidate.inference.client.busy");

    port.emit(result(7));
    client.classify(8, input);
    port.emit({
      type: "failure",
      protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
      code: "candidate.inference.input.invalid",
      requestId: 8,
      fatal: false,
    });
    client.classify(9, input);
    expect(events.map(({ type }) => type)).toEqual(["ready", "result", "failure"]);

    client.close();
    expect(port.posts.at(-1)?.message.type).toBe("stop");
    port.emit({ type: "stopped", protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION });
    expect(port.terminate).toHaveBeenCalledOnce();
  });

  it("fails closed on an uncorrelated response or worker transport error", async () => {
    const port = new ManualWorkerPort();
    const events: CandidateInferenceClientEvent[] = [];
    const client = new CandidateInferenceWorkerClient(port, (event) => events.push(event));
    await client.initialize(bundle());
    port.emit(ready());
    client.classify(1, input);

    port.emit(result(2));

    expect(events.at(-1)).toEqual({
      type: "worker-transport-failure",
      code: "candidate.inference.transport.failed",
    });
    expect(port.terminate).toHaveBeenCalledOnce();
    port.fail();
    expect(events.filter(({ type }) => type === "worker-transport-failure")).toHaveLength(1);
    expect(() => client.classify(3, input)).toThrow("candidate.inference.client.not_ready");
  });

  it("reports a sanitized transport failure when stop cannot be posted", () => {
    const port = new ManualWorkerPort();
    const events: CandidateInferenceClientEvent[] = [];
    const client = new CandidateInferenceWorkerClient(port, (event) => events.push(event));
    port.failNextPost = true;

    client.close();

    expect(events).toEqual([
      {
        type: "worker-transport-failure",
        code: "candidate.inference.transport.failed",
      },
    ]);
    expect(port.terminate).toHaveBeenCalledOnce();
  });

  it.each(["transport", "stopped"] as const)(
    "releases the worker before a throwing consumer callback on %s",
    (terminal) => {
      const port = new ManualWorkerPort();
      new CandidateInferenceWorkerClient(port, () => {
        throw new Error("consumer callback failed");
      });
      const trigger = () => {
        if (terminal === "transport") port.fail();
        else
          port.emit({
            type: "stopped",
            protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
          });
      };

      expect(trigger).toThrow("consumer callback failed");
      expect(port.terminate).toHaveBeenCalledOnce();
      expect(trigger).not.toThrow();
    },
  );
});
