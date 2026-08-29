import { createMediaPipeDetector } from "./mediapipeRuntime";
import type { LandmarkWorkerInputMessage, LandmarkWorkerOutputMessage } from "./protocol";
import { LandmarkWorkerSession } from "./workerSession";

interface LandmarkWorkerScope {
  onmessage: ((event: MessageEvent<LandmarkWorkerInputMessage>) => void) | null;
  postMessage(message: LandmarkWorkerOutputMessage): void;
  close(): void;
}

const workerScope = self as unknown as LandmarkWorkerScope;
const session = new LandmarkWorkerSession(createMediaPipeDetector, (message) => {
  workerScope.postMessage(message);
});

workerScope.onmessage = (event) => {
  const stopAfterHandling = event.data.type === "stop";
  void session.handle(event.data).finally(() => {
    if (stopAfterHandling) workerScope.close();
  });
};
