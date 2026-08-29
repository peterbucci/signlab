import {
  LANDMARK_WORKER_PROTOCOL_VERSION,
  type LandmarkWorkerInputMessage,
  type LandmarkWorkerOutputMessage,
} from "./protocol";

export interface LandmarkWorkerPort {
  post(message: LandmarkWorkerInputMessage, transfer?: readonly Transferable[]): void;
  onMessage(listener: (message: LandmarkWorkerOutputMessage) => void): () => void;
  onError(listener: () => void): () => void;
  terminate(): void;
}

export interface FrameDroppedEvent {
  readonly type: "frame-dropped";
  readonly frameId: number;
  readonly droppedFrames: number;
}

export interface WorkerTransportFailureEvent {
  readonly type: "worker-transport-failure";
  readonly code: "extraction.worker.transport.failed";
  readonly failureCount: number;
}

export type LandmarkClientEvent =
  LandmarkWorkerOutputMessage | FrameDroppedEvent | WorkerTransportFailureEvent;

export interface LandmarkWorkerMetrics {
  readonly droppedFrames: number;
  readonly failureEvents: number;
  readonly inFlightFrames: 0 | 1;
  readonly pendingFrames: 0 | 1;
}

interface PendingFrame {
  readonly frameId: number;
  readonly relativeTimestampUs: number;
  readonly frame: ImageBitmap;
}

class BrowserLandmarkWorkerPort implements LandmarkWorkerPort {
  private readonly worker = new Worker(new URL("./landmark.worker.ts", import.meta.url), {
    name: "signlab-landmarks",
    type: "module",
  });

  post(message: LandmarkWorkerInputMessage, transfer: readonly Transferable[] = []): void {
    this.worker.postMessage(message, [...transfer]);
  }

  onMessage(listener: (message: LandmarkWorkerOutputMessage) => void): () => void {
    const handler = (event: MessageEvent<LandmarkWorkerOutputMessage>) => {
      listener(event.data);
    };
    this.worker.addEventListener("message", handler);
    return () => {
      this.worker.removeEventListener("message", handler);
    };
  }

  onError(listener: () => void): () => void {
    const handler = () => {
      listener();
    };
    this.worker.addEventListener("error", handler);
    return () => {
      this.worker.removeEventListener("error", handler);
    };
  }

  terminate(): void {
    this.worker.terminate();
  }
}

export function supportsLandmarkWorkerRuntime(): boolean {
  const simdProbe = new Uint8Array([
    0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253,
    15, 253, 98, 11,
  ]);
  return (
    typeof Worker !== "undefined" &&
    typeof ImageBitmap !== "undefined" &&
    typeof OffscreenCanvas !== "undefined" &&
    typeof WebAssembly !== "undefined" &&
    typeof WebAssembly.validate === "function" &&
    WebAssembly.validate(simdProbe)
  );
}

export function createLandmarkWorkerClient(
  onEvent: (event: LandmarkClientEvent) => void,
): LandmarkWorkerClient {
  return new LandmarkWorkerClient(new BrowserLandmarkWorkerPort(), onEvent);
}

export class LandmarkWorkerClient {
  private initialized = false;
  private ready = false;
  private acceptingFrames = true;
  private failed = false;
  private nextFrameId = 0;
  private originTimestampUs: number | undefined;
  private previousCaptureTimestampUs = -1;
  private previousTaskTimestampMs = -1;
  private inFlightFrameId: number | null = null;
  private pendingFrame: PendingFrame | null = null;
  private droppedFrames = 0;
  private failureEvents = 0;
  private closePromise: Promise<void> | undefined;
  private finishClose: (() => void) | undefined;
  private closeTimer: ReturnType<typeof setTimeout> | undefined;
  private finalized = false;
  private readonly removeMessageListener: () => void;
  private readonly removeErrorListener: () => void;

  constructor(
    private readonly port: LandmarkWorkerPort,
    private readonly onEvent: (event: LandmarkClientEvent) => void,
  ) {
    this.removeMessageListener = port.onMessage((message) => {
      this.handleMessage(message);
    });
    this.removeErrorListener = port.onError(() => {
      this.failTransport();
    });
  }

  initialize(handModelBuffer: ArrayBuffer, poseModelBuffer: ArrayBuffer): void {
    if (this.initialized || !this.acceptingFrames || this.failed) {
      throw new Error("The landmark worker can only be initialized once");
    }
    this.initialized = true;
    try {
      this.port.post(
        {
          type: "initialize",
          protocolVersion: LANDMARK_WORKER_PROTOCOL_VERSION,
          handModelBuffer,
          poseModelBuffer,
        },
        [handModelBuffer, poseModelBuffer],
      );
    } catch {
      this.failTransport();
    }
  }

  submitFrame(frame: ImageBitmap, captureTimestampMs: number): number {
    if (!this.initialized || !this.acceptingFrames || this.failed) {
      frame.close();
      throw new Error("The landmark worker is not accepting frames");
    }
    if (!Number.isFinite(captureTimestampMs) || captureTimestampMs < 0) {
      frame.close();
      throw new Error("captureTimestampMs must be finite and non-negative");
    }
    const captureTimestampUs = Math.round(captureTimestampMs * 1_000);
    if (
      !Number.isSafeInteger(captureTimestampUs) ||
      captureTimestampUs < this.previousCaptureTimestampUs
    ) {
      frame.close();
      throw new Error("Frame capture timestamps must be safe and monotonic");
    }
    this.originTimestampUs ??= captureTimestampUs;
    const relativeTimestampUs = captureTimestampUs - this.originTimestampUs;
    this.previousCaptureTimestampUs = captureTimestampUs;

    if (!Number.isSafeInteger(this.nextFrameId)) {
      frame.close();
      throw new Error("The landmark worker exhausted its frame identifiers");
    }

    const pending: PendingFrame = {
      frameId: this.nextFrameId,
      relativeTimestampUs,
      frame,
    };
    this.nextFrameId += 1;

    if (!this.ready || this.inFlightFrameId !== null) {
      this.replacePending(pending);
    } else {
      this.send(pending);
    }
    return pending.frameId;
  }

  metrics(): LandmarkWorkerMetrics {
    return {
      droppedFrames: this.droppedFrames,
      failureEvents: this.failureEvents,
      inFlightFrames: this.inFlightFrameId === null ? 0 : 1,
      pendingFrames: this.pendingFrame === null ? 0 : 1,
    };
  }

  close(timeoutMs = 2_000): Promise<void> {
    if (this.closePromise !== undefined) return this.closePromise;
    if (this.finalized) {
      this.closePromise = Promise.resolve();
      return this.closePromise;
    }
    this.acceptingFrames = false;
    this.releasePending();
    this.closePromise = new Promise<void>((resolve) => {
      this.finishClose = resolve;
    });
    if (!this.initialized || this.failed) {
      this.finalizeClose();
      return this.closePromise;
    }
    try {
      this.port.post({
        type: "stop",
        protocolVersion: LANDMARK_WORKER_PROTOCOL_VERSION,
      });
    } catch {
      this.failTransport();
      return this.closePromise;
    }
    this.closeTimer = setTimeout(() => {
      this.finalizeClose();
    }, timeoutMs);
    return this.closePromise;
  }

  private replacePending(frame: PendingFrame): void {
    const replaced = this.pendingFrame;
    this.pendingFrame = frame;
    if (replaced !== null) {
      replaced.frame.close();
      this.droppedFrames += 1;
      this.onEvent({
        type: "frame-dropped",
        frameId: replaced.frameId,
        droppedFrames: this.droppedFrames,
      });
    }
  }

  private send(pending: PendingFrame): void {
    this.pendingFrame = null;
    this.inFlightFrameId = pending.frameId;
    const taskTimestampMs = Math.max(
      Math.floor(pending.relativeTimestampUs / 1_000),
      this.previousTaskTimestampMs + 1,
    );
    this.previousTaskTimestampMs = taskTimestampMs;
    try {
      this.port.post(
        {
          type: "process-frame",
          protocolVersion: LANDMARK_WORKER_PROTOCOL_VERSION,
          frameId: pending.frameId,
          relativeTimestampUs: pending.relativeTimestampUs,
          taskTimestampMs,
          frame: pending.frame,
        },
        [pending.frame],
      );
    } catch {
      this.inFlightFrameId = null;
      pending.frame.close();
      this.failTransport();
    }
  }

  private flushPending(): void {
    if (
      !this.ready ||
      !this.acceptingFrames ||
      this.inFlightFrameId !== null ||
      this.pendingFrame === null
    ) {
      return;
    }
    this.send(this.pendingFrame);
  }

  private handleMessage(message: LandmarkWorkerOutputMessage): void {
    if (message.protocolVersion !== LANDMARK_WORKER_PROTOCOL_VERSION) {
      this.failTransport();
      return;
    }
    this.failureEvents = Math.max(this.failureEvents, message.failureCount);

    if (message.type === "ready") {
      this.ready = true;
      this.flushPending();
      this.onEvent(message);
      return;
    }
    if (message.type === "frame") {
      if (message.frameId !== this.inFlightFrameId) {
        this.failTransport();
        return;
      }
      this.inFlightFrameId = null;
      this.flushPending();
      this.onEvent(message);
      return;
    }
    if (message.type === "failure") {
      this.failed = true;
      this.ready = false;
      this.acceptingFrames = false;
      this.inFlightFrameId = null;
      this.releasePending();
      this.finalizeClose();
      this.onEvent(message);
      return;
    }
    this.finalizeClose();
    this.onEvent(message);
  }

  private releasePending(): void {
    this.pendingFrame?.frame.close();
    this.pendingFrame = null;
  }

  private failTransport(): void {
    if (this.failed) return;
    this.failed = true;
    this.ready = false;
    this.acceptingFrames = false;
    this.inFlightFrameId = null;
    this.failureEvents += 1;
    this.releasePending();
    this.finalizeClose();
    this.onEvent({
      type: "worker-transport-failure",
      code: "extraction.worker.transport.failed",
      failureCount: this.failureEvents,
    });
  }

  private finalizeClose(): void {
    if (this.finalized) return;
    this.finalized = true;
    this.acceptingFrames = false;
    this.ready = false;
    this.inFlightFrameId = null;
    this.releasePending();
    if (this.closeTimer !== undefined) clearTimeout(this.closeTimer);
    this.closeTimer = undefined;
    this.removeMessageListener();
    this.removeErrorListener();
    this.port.terminate();
    this.finishClose?.();
    this.finishClose = undefined;
  }
}
