import type {
  CandidateInferenceWorkerInput,
  CandidateInferenceWorkerOutput,
} from "./candidateInferenceProtocol";
import {
  CandidateInferenceSession,
  createCandidateInferenceEngine,
} from "./candidateInferenceSession";

interface CandidateInferenceWorkerScope {
  onmessage: ((event: MessageEvent<CandidateInferenceWorkerInput>) => void) | null;
  postMessage(message: CandidateInferenceWorkerOutput): void;
  close(): void;
}

const workerScope = self as unknown as CandidateInferenceWorkerScope;
const session = new CandidateInferenceSession(createCandidateInferenceEngine, (message) => {
  workerScope.postMessage(message);
});

workerScope.onmessage = (event) => {
  const stopAfterHandling = event.data.type === "stop";
  void session.handle(event.data).finally(() => {
    if (stopAfterHandling) workerScope.close();
  });
};
