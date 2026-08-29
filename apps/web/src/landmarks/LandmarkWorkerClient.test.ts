import { describe, expect, it, vi } from "vitest";

import {
  LandmarkWorkerClient,
  type LandmarkClientEvent,
  type LandmarkWorkerPort,
} from "./LandmarkWorkerClient";
import {
  LANDMARK_WORKER_PROTOCOL_VERSION,
  absentBodyAnchors,
  absentHandSlots,
  type LandmarkWorkerInputMessage,
  type LandmarkWorkerOutputMessage,
  type ProcessLandmarkFrame,
} from "./protocol";

class ManualWorkerPort implements LandmarkWorkerPort {
  readonly posts: Array<{
    message: LandmarkWorkerInputMessage;
    transfer: readonly Transferable[];
  }> = [];
  readonly terminate = vi.fn();
  private messageListener: ((message: LandmarkWorkerOutputMessage) => void) | undefined;
  private errorListener: (() => void) | undefined;

  post(message: LandmarkWorkerInputMessage, transfer: readonly Transferable[] = []): void {
    this.posts.push({ message, transfer });
  }

  onMessage(listener: (message: LandmarkWorkerOutputMessage) => void): () => void {
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

  emit(message: LandmarkWorkerOutputMessage): void {
    this.messageListener?.(message);
  }

  fail(): void {
    this.errorListener?.();
  }
}

interface TestFrame {
  readonly bitmap: ImageBitmap;
  readonly close: ReturnType<typeof vi.fn>;
}

function frame(): TestFrame {
  const close = vi.fn();
  return { bitmap: { close } as unknown as ImageBitmap, close };
}

function ready(): LandmarkWorkerOutputMessage {
  return {
    type: "ready",
    protocolVersion: LANDMARK_WORKER_PROTOCOL_VERSION,
    startupMs: 5,
    failureCount: 0,
  };
}

function completedFrame(message: ProcessLandmarkFrame): LandmarkWorkerOutputMessage {
  return {
    type: "frame",
    protocolVersion: LANDMARK_WORKER_PROTOCOL_VERSION,
    frameId: message.frameId,
    relativeTimestampUs: message.relativeTimestampUs,
    taskTimestampMs: message.taskTimestampMs,
    valid: true,
    invalidReason: null,
    failureCode: null,
    hands: absentHandSlots(),
    bodyAnchors: absentBodyAnchors(),
    processingMs: 4,
    failureCount: 0,
  };
}

function processMessages(port: ManualWorkerPort): ProcessLandmarkFrame[] {
  return port.posts
    .map(({ message }) => message)
    .filter((message): message is ProcessLandmarkFrame => message.type === "process-frame");
}

describe("LandmarkWorkerClient", () => {
  it("keeps only one in-flight frame and the newest waiting frame under load", async () => {
    const port = new ManualWorkerPort();
    const events: LandmarkClientEvent[] = [];
    const client = new LandmarkWorkerClient(port, (event) => events.push(event));
    const handModel = new ArrayBuffer(2);
    const poseModel = new ArrayBuffer(3);
    client.initialize(handModel, poseModel);
    port.emit(ready());

    const frames = Array.from({ length: 10_000 }, frame);
    let maximumRetained = 0;
    for (const current of frames) {
      client.submitFrame(current.bitmap, 100);
      const metrics = client.metrics();
      maximumRetained = Math.max(maximumRetained, metrics.inFlightFrames + metrics.pendingFrames);
    }

    expect(port.posts[0]).toMatchObject({
      message: { type: "initialize", handModelBuffer: handModel, poseModelBuffer: poseModel },
      transfer: [handModel, poseModel],
    });
    expect(processMessages(port)).toHaveLength(1);
    expect(client.metrics()).toEqual({
      droppedFrames: 9_998,
      failureEvents: 0,
      inFlightFrames: 1,
      pendingFrames: 1,
    });
    expect(maximumRetained).toBe(2);

    const first = processMessages(port)[0];
    expect(first).toMatchObject({ frameId: 0, relativeTimestampUs: 0, taskTimestampMs: 0 });
    if (first === undefined) throw new Error("expected the first frame message");
    port.emit(completedFrame(first));

    const sent = processMessages(port);
    expect(sent).toHaveLength(2);
    expect(sent[1]).toMatchObject({
      frameId: 9_999,
      relativeTimestampUs: 0,
      taskTimestampMs: 1,
    });
    expect(frames[0]?.close).not.toHaveBeenCalled();
    expect(frames[9_999]?.close).not.toHaveBeenCalled();
    expect(frames.slice(1, -1).every(({ close }) => close.mock.calls.length === 1)).toBe(true);

    const dropped = events.filter(({ type }) => type === "frame-dropped");
    expect(dropped).toHaveLength(9_998);
    expect(dropped.at(-1)).toEqual({
      type: "frame-dropped",
      frameId: 9_998,
      droppedFrames: 9_998,
    });
    expect(JSON.stringify(dropped)).not.toContain("bitmap");

    const latest = sent[1];
    if (latest === undefined) throw new Error("expected the newest frame message");
    port.emit(completedFrame(latest));
    const closing = client.close();
    port.emit({
      type: "stopped",
      protocolVersion: LANDMARK_WORKER_PROTOCOL_VERSION,
      failureCount: 0,
    });
    await closing;
    expect(port.terminate).toHaveBeenCalledTimes(1);
  });

  it("closes a waiting frame and terminates when the worker transport fails", async () => {
    const port = new ManualWorkerPort();
    const events: LandmarkClientEvent[] = [];
    const client = new LandmarkWorkerClient(port, (event) => events.push(event));
    client.initialize(new ArrayBuffer(1), new ArrayBuffer(1));
    const waiting = frame();
    client.submitFrame(waiting.bitmap, 0);

    port.fail();

    expect(waiting.close).toHaveBeenCalledTimes(1);
    expect(events.at(-1)).toEqual({
      type: "worker-transport-failure",
      code: "extraction.worker.transport.failed",
      failureCount: 1,
    });
    expect(port.terminate).toHaveBeenCalledTimes(1);
    await expect(client.close()).resolves.toBeUndefined();
    expect(port.terminate).toHaveBeenCalledTimes(1);
  });

  it("makes an early stopped message terminal and clears retained-frame metrics", async () => {
    const port = new ManualWorkerPort();
    const client = new LandmarkWorkerClient(port, vi.fn());
    client.initialize(new ArrayBuffer(1), new ArrayBuffer(1));
    port.emit(ready());
    client.submitFrame(frame().bitmap, 0);

    port.emit({
      type: "stopped",
      protocolVersion: LANDMARK_WORKER_PROTOCOL_VERSION,
      failureCount: 0,
    });

    expect(client.metrics()).toMatchObject({ inFlightFrames: 0, pendingFrames: 0 });
    const rejected = frame();
    expect(() => client.submitFrame(rejected.bitmap, 1)).toThrow(
      "The landmark worker is not accepting frames",
    );
    expect(rejected.close).toHaveBeenCalledTimes(1);
    expect(port.terminate).toHaveBeenCalledTimes(1);
    await expect(client.close()).resolves.toBeUndefined();
  });
});
