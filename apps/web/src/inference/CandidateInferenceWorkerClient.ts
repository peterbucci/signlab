import type { VerifiedModelBundle } from "../modelBundle/modelBundleSession";
import {
  CANDIDATE_INFERENCE_PROTOCOL_VERSION,
  type CandidateInferenceInput,
  type CandidateInferenceWorkerInput,
  type CandidateInferenceWorkerOutput,
} from "./candidateInferenceProtocol";

export interface CandidateInferenceWorkerPort {
  post(message: CandidateInferenceWorkerInput, transfer?: readonly Transferable[]): void;
  onMessage(listener: (message: CandidateInferenceWorkerOutput) => void): () => void;
  onError(listener: () => void): () => void;
  terminate(): void;
}

export type CandidateInferenceClientEvent =
  | CandidateInferenceWorkerOutput
  | {
      readonly type: "worker-transport-failure";
      readonly code: "candidate.inference.transport.failed";
    };

class BrowserCandidateInferenceWorkerPort implements CandidateInferenceWorkerPort {
  private readonly worker = new Worker(new URL("./candidateInference.worker.ts", import.meta.url), {
    name: "signlab-candidate-inference",
    type: "module",
  });

  post(message: CandidateInferenceWorkerInput, transfer: readonly Transferable[] = []): void {
    this.worker.postMessage(message, [...transfer]);
  }

  onMessage(listener: (message: CandidateInferenceWorkerOutput) => void): () => void {
    const handler = (event: MessageEvent<CandidateInferenceWorkerOutput>) => listener(event.data);
    this.worker.addEventListener("message", handler);
    return () => this.worker.removeEventListener("message", handler);
  }

  onError(listener: () => void): () => void {
    this.worker.addEventListener("error", listener);
    return () => this.worker.removeEventListener("error", listener);
  }

  terminate(): void {
    this.worker.terminate();
  }
}

export function createCandidateInferenceWorkerClient(
  onEvent: (event: CandidateInferenceClientEvent) => void,
): CandidateInferenceWorkerClient {
  return new CandidateInferenceWorkerClient(new BrowserCandidateInferenceWorkerPort(), onEvent);
}

export class CandidateInferenceWorkerClient {
  private initialized = false;
  private ready = false;
  private closed = false;
  private inFlightRequestId: number | null = null;
  private readonly removeMessageListener: () => void;
  private readonly removeErrorListener: () => void;

  constructor(
    private readonly port: CandidateInferenceWorkerPort,
    private readonly onEvent: (event: CandidateInferenceClientEvent) => void,
  ) {
    this.removeMessageListener = port.onMessage((message) => this.handle(message));
    this.removeErrorListener = port.onError(() => this.failTransport());
  }

  async initialize(bundle: VerifiedModelBundle): Promise<void> {
    if (this.initialized || this.closed) throw new Error("candidate.inference.client.closed");
    this.initialized = true;
    try {
      const [modelBuffer, featurePlanBuffer, decisionPolicyBuffer] = await Promise.all([
        bundle.bytesByRole.model.arrayBuffer(),
        bundle.bytesByRole.feature_plan.arrayBuffer(),
        bundle.bytesByRole.decision_policy.arrayBuffer(),
      ]);
      if (this.closed) return;
      this.port.post(
        {
          type: "initialize",
          protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
          bundle: { id: bundle.id, version: bundle.version },
          modelBuffer,
          featurePlanBuffer,
          decisionPolicyBuffer,
        },
        [modelBuffer, featurePlanBuffer, decisionPolicyBuffer],
      );
    } catch {
      this.failTransport();
    }
  }

  classify(requestId: number, input: CandidateInferenceInput): void {
    if (!this.ready || this.closed) throw new Error("candidate.inference.client.not_ready");
    if (this.inFlightRequestId !== null) throw new Error("candidate.inference.client.busy");
    this.inFlightRequestId = requestId;
    try {
      this.port.post({
        type: "classify",
        protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
        requestId,
        input,
      });
    } catch {
      this.failTransport();
    }
  }

  close(): void {
    if (this.closed) return;
    this.ready = false;
    try {
      this.port.post({ type: "stop", protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION });
    } catch {
      this.failTransport();
    }
  }

  private handle(message: CandidateInferenceWorkerOutput): void {
    if (message.protocolVersion !== CANDIDATE_INFERENCE_PROTOCOL_VERSION) {
      this.failTransport();
      return;
    }
    if (message.type === "ready") this.ready = true;
    else if (message.type === "result" || (message.type === "failure" && !message.fatal)) {
      if (message.requestId !== this.inFlightRequestId) return this.failTransport();
      this.inFlightRequestId = null;
    }
    if ((message.type === "failure" && message.fatal) || message.type === "stopped")
      this.finalize();
    this.onEvent(message);
  }

  private failTransport(): void {
    if (this.closed) return;
    this.finalize();
    this.onEvent({
      type: "worker-transport-failure",
      code: "candidate.inference.transport.failed",
    });
  }

  private finalize(): void {
    if (this.closed) return;
    this.closed = true;
    this.ready = false;
    this.inFlightRequestId = null;
    this.removeMessageListener();
    this.removeErrorListener();
    this.port.terminate();
  }
}
