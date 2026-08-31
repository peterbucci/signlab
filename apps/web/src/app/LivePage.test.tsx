import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { StrictMode } from "react";
import { describe, expect, it, vi } from "vitest";

import type { CameraEnvironment } from "../camera/useCameraSession";
import {
  CANDIDATE_INFERENCE_PROTOCOL_VERSION,
  type CandidateInferenceResult,
} from "../inference/candidateInferenceProtocol";
import type { LiveRecognitionSnapshot } from "../live/liveRecognitionSession";
import {
  type ModelBundleSession,
  type ModelBundleStatus,
  type VerifiedModelBundle,
} from "../modelBundle/modelBundleSession";
import { LivePage } from "./routes";

class TestDocument extends EventTarget {
  hidden = false;
}

class TestMediaDevices extends EventTarget {
  getUserMedia = vi.fn<(constraints: MediaStreamConstraints) => Promise<MediaStream>>();
  enumerateDevices = vi.fn<() => Promise<MediaDeviceInfo[]>>();
}

interface TestTrack extends EventTarget {
  enabled: boolean;
  getSettings: () => MediaTrackSettings;
  stop: ReturnType<typeof vi.fn>;
}

function cameraDevice(deviceId: string, label: string): MediaDeviceInfo {
  return {
    deviceId,
    groupId: "group",
    kind: "videoinput",
    label,
    toJSON: () => ({}),
  };
}

function cameraStream(deviceId: string) {
  const track = new EventTarget() as TestTrack;
  track.enabled = true;
  track.getSettings = () => ({ deviceId });
  track.stop = vi.fn();
  const mediaTrack = track as unknown as MediaStreamTrack;
  const stream = {
    getTracks: () => [mediaTrack],
    getVideoTracks: () => [mediaTrack],
  } as unknown as MediaStream;
  return { stream, track };
}

function cameraEnvironment(mediaDevices?: TestMediaDevices, secure = true) {
  const pageDocument = new TestDocument();
  const environment: CameraEnvironment = {
    document: pageDocument,
    isSecureContext: secure,
    mediaDevices,
  };
  return { environment, pageDocument };
}

async function startCamera() {
  fireEvent.click(screen.getByRole("button", { name: "Start camera" }));
  await screen.findByText(
    "Camera is on. Raw video stays on this page and is not saved or uploaded.",
  );
}

describe("consent-first camera preview", () => {
  it("waits for explicit consent and keeps the preview out of transport and storage APIs", async () => {
    const devices = new TestMediaDevices();
    const { stream, track } = cameraStream("front");
    devices.getUserMedia.mockResolvedValue(stream);
    devices.enumerateDevices.mockResolvedValue([cameraDevice("front", "Front camera")]);
    const { environment } = cameraEnvironment(devices);
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const xhrSpy = vi.spyOn(XMLHttpRequest.prototype, "send");
    const storageSpy = vi.spyOn(Storage.prototype, "setItem");
    const webSocketSpy = vi.fn();
    const sendBeaconSpy = vi.fn();
    vi.stubGlobal("WebSocket", webSocketSpy);
    vi.stubGlobal(
      "navigator",
      Object.create(navigator, {
        sendBeacon: { value: sendBeaconSpy },
      }) as Navigator,
    );

    render(
      <StrictMode>
        <LivePage cameraEnvironment={environment} />
      </StrictMode>,
    );

    expect(devices.getUserMedia).not.toHaveBeenCalled();
    expect(
      screen.getByText(/asks for permission only after you select Start camera/),
    ).toBeVisible();
    expect(screen.getByRole("heading", { name: "How to try it" })).toBeVisible();
    expect(
      screen.getByText(/Choose one prompt: Hello, No, Please, Thank you, or Yes/),
    ).toBeVisible();
    expect(screen.getByText(/even light facing you, not behind you/)).toBeVisible();
    expect(screen.getByRole("link", { name: "Read its limitations." })).toHaveAttribute(
      "href",
      "#/limitations",
    );
    expect(screen.getByRole("status", { name: "Model bundle status" })).toHaveTextContent(
      "No model bundle is configured.",
    );

    await startCamera();

    expect(devices.getUserMedia).toHaveBeenCalledWith({
      audio: false,
      video: { facingMode: { ideal: "user" } },
    });
    const video = screen.getByLabelText("Local camera preview");
    await waitFor(() => expect(video).toHaveProperty("srcObject", stream));
    expect(video).toHaveClass("is-mirrored");

    fireEvent.click(screen.getByRole("checkbox", { name: "Mirror preview" }));
    expect(video).not.toHaveClass("is-mirrored");

    fireEvent.click(screen.getByRole("button", { name: "Pause preview" }));
    expect(track.enabled).toBe(false);
    expect(screen.getByText(/Stop the camera to release it completely/)).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Resume preview" }));
    expect(track.enabled).toBe(true);

    fireEvent.click(screen.getByRole("button", { name: "Stop camera" }));
    expect(track.stop).toHaveBeenCalledOnce();
    expect(screen.queryByLabelText("Local camera preview")).not.toBeInTheDocument();
    expect(fetchSpy).not.toHaveBeenCalled();
    expect(xhrSpy).not.toHaveBeenCalled();
    expect(webSocketSpy).not.toHaveBeenCalled();
    expect(sendBeaconSpy).not.toHaveBeenCalled();
    expect(storageSpy).not.toHaveBeenCalled();
  });

  it("acquires a replacement before releasing the selected camera", async () => {
    const devices = new TestMediaDevices();
    const first = cameraStream("front");
    const second = cameraStream("back");
    let resolveReplacement!: (stream: MediaStream) => void;
    const replacement = new Promise<MediaStream>((resolve) => {
      resolveReplacement = resolve;
    });
    devices.getUserMedia.mockResolvedValueOnce(first.stream).mockReturnValueOnce(replacement);
    devices.enumerateDevices.mockResolvedValue([
      cameraDevice("front", "Front camera"),
      cameraDevice("back", "Back camera"),
    ]);
    const { environment } = cameraEnvironment(devices);

    render(<LivePage cameraEnvironment={environment} />);
    await startCamera();
    const selector = await screen.findByRole("combobox", { name: "Camera" });

    fireEvent.click(screen.getByRole("button", { name: "Pause preview" }));
    expect(selector).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Resume preview" }));
    expect(selector).toBeEnabled();

    fireEvent.change(selector, { target: { value: "back" } });
    expect(devices.getUserMedia).toHaveBeenLastCalledWith({
      audio: false,
      video: { deviceId: { exact: "back" } },
    });
    expect(first.track.stop).not.toHaveBeenCalled();

    await act(async () => {
      resolveReplacement(second.stream);
      await replacement;
    });

    expect(first.track.stop).toHaveBeenCalledOnce();
    expect(screen.getByLabelText("Local camera preview")).toHaveProperty(
      "srcObject",
      second.stream,
    );
  });

  it("keeps the current camera when a replacement cannot start", async () => {
    const devices = new TestMediaDevices();
    const current = cameraStream("front");
    const failure = new Error("busy");
    failure.name = "NotReadableError";
    devices.getUserMedia.mockResolvedValueOnce(current.stream).mockRejectedValueOnce(failure);
    devices.enumerateDevices.mockResolvedValue([
      cameraDevice("front", "Front camera"),
      cameraDevice("back", "Back camera"),
    ]);
    const { environment } = cameraEnvironment(devices);

    render(<LivePage cameraEnvironment={environment} />);
    await startCamera();
    const selector = await screen.findByRole("combobox", { name: "Camera" });
    fireEvent.change(selector, { target: { value: "back" } });

    expect(await screen.findByText(/previous camera is still active/)).toBeVisible();
    expect(current.track.stop).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Local camera preview")).toHaveProperty(
      "srcObject",
      current.stream,
    );
    expect(selector).toHaveValue("front");
  });

  it("releases the camera when the page is hidden or unmounted", async () => {
    const devices = new TestMediaDevices();
    const first = cameraStream("front");
    devices.getUserMedia.mockResolvedValue(first.stream);
    devices.enumerateDevices.mockResolvedValue([]);
    const { environment, pageDocument } = cameraEnvironment(devices);
    const view = render(<LivePage cameraEnvironment={environment} />);
    await startCamera();

    pageDocument.hidden = true;
    act(() => {
      pageDocument.dispatchEvent(new Event("visibilitychange"));
    });

    expect(first.track.stop).toHaveBeenCalledOnce();
    expect(screen.getByText(/page was hidden/)).toBeVisible();

    pageDocument.hidden = false;
    const second = cameraStream("front");
    devices.getUserMedia.mockResolvedValue(second.stream);
    await startCamera();
    view.unmount();
    expect(second.track.stop).toHaveBeenCalledOnce();
  });

  it("reports external interruption and actionable access failures", async () => {
    const devices = new TestMediaDevices();
    const active = cameraStream("front");
    devices.getUserMedia.mockResolvedValue(active.stream);
    devices.enumerateDevices.mockResolvedValue([]);
    const { environment } = cameraEnvironment(devices);
    const view = render(<LivePage cameraEnvironment={environment} />);
    await startCamera();

    act(() => {
      active.track.dispatchEvent(new Event("ended"));
    });
    expect(screen.getByText(/camera stopped unexpectedly/)).toBeVisible();
    expect(screen.getByRole("button", { name: "Start camera" })).toBeVisible();
    expect(active.track.stop).toHaveBeenCalledOnce();

    view.unmount();
    const deniedDevices = new TestMediaDevices();
    const denied = new Error("denied");
    denied.name = "NotAllowedError";
    deniedDevices.getUserMedia.mockRejectedValue(denied);
    deniedDevices.enumerateDevices.mockResolvedValue([]);
    render(<LivePage cameraEnvironment={cameraEnvironment(deniedDevices).environment} />);
    fireEvent.click(screen.getByRole("button", { name: "Start camera" }));
    expect(await screen.findByText(/permission was denied/)).toBeVisible();
  });

  it("explains insecure and unavailable environments without requesting access", () => {
    const insecureDevices = new TestMediaDevices();
    const insecure = cameraEnvironment(insecureDevices, false).environment;
    const view = render(<LivePage cameraEnvironment={insecure} />);

    expect(screen.getByText(/requires HTTPS or a local development address/)).toBeVisible();
    expect(screen.queryByRole("button", { name: "Start camera" })).not.toBeInTheDocument();
    expect(insecureDevices.getUserMedia).not.toHaveBeenCalled();

    view.unmount();
    render(<LivePage cameraEnvironment={cameraEnvironment().environment} />);
    expect(screen.getByText(/browser does not provide camera access/)).toBeVisible();
    expect(screen.queryByRole("button", { name: "Start camera" })).not.toBeInTheDocument();
  });

  it("releases a stream that arrives after unmount", async () => {
    const devices = new TestMediaDevices();
    const late = cameraStream("front");
    let resolvePermission!: (stream: MediaStream) => void;
    const permission = new Promise<MediaStream>((resolve) => {
      resolvePermission = resolve;
    });
    devices.getUserMedia.mockReturnValue(permission);
    devices.enumerateDevices.mockResolvedValue([]);
    const { environment } = cameraEnvironment(devices);
    const view = render(<LivePage cameraEnvironment={environment} />);
    fireEvent.click(screen.getByRole("button", { name: "Start camera" }));

    view.unmount();
    await act(async () => {
      resolvePermission(late.stream);
      await permission;
    });

    await waitFor(() => expect(late.track.stop).toHaveBeenCalledOnce());
  });

  it("releases a pending stream if the page becomes hidden", async () => {
    const devices = new TestMediaDevices();
    const late = cameraStream("front");
    let resolvePermission!: (stream: MediaStream) => void;
    const permission = new Promise<MediaStream>((resolve) => {
      resolvePermission = resolve;
    });
    devices.getUserMedia.mockReturnValue(permission);
    devices.enumerateDevices.mockResolvedValue([]);
    const { environment, pageDocument } = cameraEnvironment(devices);
    render(<LivePage cameraEnvironment={environment} />);
    fireEvent.click(screen.getByRole("button", { name: "Start camera" }));

    pageDocument.hidden = true;
    act(() => {
      pageDocument.dispatchEvent(new Event("visibilitychange"));
    });
    expect(screen.getByText(/page was hidden/)).toBeVisible();

    await act(async () => {
      resolvePermission(late.stream);
      await permission;
    });

    await waitFor(() => expect(late.track.stop).toHaveBeenCalledOnce());
    expect(screen.queryByLabelText("Local camera preview")).not.toBeInTheDocument();
  });
});

describe("model bundle status", () => {
  it.each([
    [
      {
        phase: "ready",
        active: { id: "candidate_bundle", version: "1.2.3" },
      } satisfies ModelBundleStatus,
      false,
      "Verified model bundle candidate_bundle version 1.2.3 is ready.",
    ],
    [
      {
        phase: "error",
        active: { id: "previous_bundle", version: "1.0.0" },
        failureReason: "A model bundle file failed its integrity check.",
      } satisfies ModelBundleStatus,
      true,
      "A model bundle file failed its integrity check. previous_bundle version 1.0.0 remains active.",
    ],
  ])("shows a concise %s state", async (result, rejects, message) => {
    const load: ModelBundleSession["load"] = (_url, onStatus) => {
      onStatus?.(result);
      return rejects
        ? Promise.reject(new Error("simulated load failure"))
        : Promise.resolve({} as VerifiedModelBundle);
    };
    const loader: Pick<ModelBundleSession, "load" | "status"> = {
      status: { phase: "idle", active: null },
      load: vi.fn(load),
    };

    render(<LivePage modelBundleUrl="https://example.test/bundle/" modelBundleSession={loader} />);

    expect(await screen.findByText(message)).toBeVisible();
    expect(loader.load).toHaveBeenCalledOnce();
  });

  it("retries a blocked bundle load without reloading the page", async () => {
    const bundle = { id: "browser-candidate", version: "1.0.0" } as VerifiedModelBundle;
    let attempt = 0;
    const load: ModelBundleSession["load"] = (_url, onStatus) => {
      attempt += 1;
      if (attempt === 1) {
        onStatus?.({
          phase: "error",
          active: null,
          failureReason: "The model bundle could not be downloaded.",
        });
        return Promise.reject(new Error("simulated download failure"));
      }
      onStatus?.({ phase: "ready", active: { id: bundle.id, version: bundle.version } });
      return Promise.resolve(bundle);
    };
    const loader: Pick<ModelBundleSession, "load" | "status"> = {
      status: { phase: "idle", active: null },
      load: vi.fn(load),
    };

    render(<LivePage modelBundleUrl="https://example.test/bundle/" modelBundleSession={loader} />);
    fireEvent.click(await screen.findByRole("button", { name: "Retry setup" }));

    await waitFor(() => expect(loader.load).toHaveBeenCalledTimes(2));
    expect(await screen.findByText(/Verified model bundle browser-candidate/)).toBeVisible();
  });
});

const resultCases = [
  {
    title: "Hello",
    decision: { kind: "target", label: "hello", confidence: 0.82 } as const,
    reason: "accepted_target" as const,
  },
  {
    title: "Other movement",
    decision: { kind: "other", label: "other", confidence: 0.74 } as const,
    reason: "accepted_other" as const,
  },
  {
    title: "No confident match",
    decision: { kind: "abstain" } as const,
    reason: "below_threshold" as const,
  },
];
type TestLiveRuntime = NonNullable<Parameters<typeof LivePage>[0]["liveRuntime"]>;
type TestLiveSessionFactory = TestLiveRuntime["createSession"];

function inferenceResult(
  decision: (typeof resultCases)[number]["decision"],
  reason: (typeof resultCases)[number]["reason"],
): CandidateInferenceResult {
  return {
    type: "result",
    protocolVersion: CANDIDATE_INFERENCE_PROTOCOL_VERSION,
    requestId: 0,
    bundle: { id: "browser-candidate", version: "1.0.0" },
    backend: "wasm",
    decision,
    reason,
    rankedScores: [
      { label: "hello", confidence: 0.82 },
      { label: "other", confidence: 0.12 },
      { label: "yes", confidence: 0.06 },
    ],
    timings: { preprocessingMs: 2, inferenceMs: 5, decisionMs: 1, totalMs: 8 },
  };
}

describe("live on-device result", () => {
  it.each(resultCases)(
    "keeps a $title event result stable",
    async ({ title, decision, reason }) => {
      const devices = new TestMediaDevices();
      const active = cameraStream("front");
      devices.getUserMedia.mockResolvedValue(active.stream);
      devices.enumerateDevices.mockResolvedValue([]);
      const bundle = { id: "browser-candidate", version: "1.0.0" } as VerifiedModelBundle;
      const load: ModelBundleSession["load"] = (_url, onStatus) => {
        onStatus?.({ phase: "ready", active: { id: bundle.id, version: bundle.version } });
        return Promise.resolve(bundle);
      };
      const loader: Pick<ModelBundleSession, "load" | "status"> = {
        status: { phase: "idle", active: null },
        load: vi.fn(load),
      };
      let emit: (snapshot: LiveRecognitionSnapshot) => void = () => undefined;
      const close = vi.fn(() => Promise.resolve());
      const frameSource = {
        request: vi.fn(() => 7),
        cancel: vi.fn(),
        capture: vi.fn(() => Promise.resolve({ close: vi.fn() } as unknown as ImageBitmap)),
      };
      const liveRuntime: TestLiveRuntime = {
        loadAssets: () =>
          Promise.resolve({
            handModelBuffer: new ArrayBuffer(1),
            poseModelBuffer: new ArrayBuffer(1),
          }),
        createSession: (_bundle, _buffers, onState) => {
          emit = onState;
          return {
            initialize: () => {
              onState({ phase: "ready", stableResult: null, failureCode: null });
              return Promise.resolve();
            },
            submitFrame: vi.fn(),
            close,
          };
        },
        ...frameSource,
      };

      render(
        <LivePage
          cameraEnvironment={cameraEnvironment(devices).environment}
          modelBundleUrl="https://example.test/bundle/"
          modelBundleSession={loader}
          liveRuntime={liveRuntime}
        />,
      );

      await screen.findByText(/Verified model bundle browser-candidate/);
      await startCamera();
      await screen.findByText(/Models are ready/);
      act(() => {
        emit({
          phase: "recording",
          stableResult: inferenceResult(decision, reason),
          failureCode: null,
          diagnostics: {
            detectorState: "recording",
            landmarkState: "usable",
            detectedHands: 2,
            droppedFrames: 3,
            backend: "wasm",
            bundle: { id: bundle.id, version: bundle.version },
          },
        });
      });

      const resultCard = screen.getByRole("region", { name: "Latest recognition result" });
      expect(within(resultCard).getByText(title, { selector: "strong" })).toBeVisible();
      expect(within(resultCard).getByRole("list", { name: "Top calibrated scores" })).toBeVisible();
      expect(screen.getByText("8 ms")).toBeVisible();
      expect(within(resultCard).getByText("WASM")).toBeVisible();
      expect(within(resultCard).getByText("browser-candidate 1.0.0")).toBeVisible();
      expect(within(resultCard).getByText(/stronger model matches, not guarantees/)).toBeVisible();
      fireEvent.click(screen.getByText("Session diagnostics"));
      const diagnostics = screen.getByLabelText("Session diagnostics");
      expect(within(diagnostics).getAllByText("Gesture in progress")).toHaveLength(2);
      expect(within(diagnostics).getByText("2 hands detected")).toBeVisible();
      expect(within(diagnostics).getByText("3")).toBeVisible();
      expect(within(diagnostics).getByText("WASM")).toBeVisible();
      expect(within(diagnostics).getByText("browser-candidate 1.0.0")).toBeVisible();
      fireEvent.click(screen.getByRole("button", { name: "Stop camera" }));
      await waitFor(() => expect(close).toHaveBeenCalledOnce());
      expect(frameSource.cancel).toHaveBeenCalledWith(expect.any(HTMLVideoElement), 7);
    },
  );

  it("submits one captured bitmap and closes a late bitmap after stop", async () => {
    const devices = new TestMediaDevices();
    devices.getUserMedia.mockResolvedValue(cameraStream("front").stream);
    devices.enumerateDevices.mockResolvedValue([]);
    const bundle = { id: "browser-candidate", version: "1.0.0" } as VerifiedModelBundle;
    const loader: Pick<ModelBundleSession, "load" | "status"> = {
      status: { phase: "ready", active: { id: bundle.id, version: bundle.version } },
      load: vi.fn(() => Promise.resolve(bundle)),
    };
    const callbacks: Array<(timestampMs: number) => void> = [];
    const firstBitmap = { close: vi.fn() } as unknown as ImageBitmap;
    const lateClose = vi.fn();
    const lateBitmap = { close: lateClose } as unknown as ImageBitmap;
    let resolveLate!: (bitmap: ImageBitmap) => void;
    const lateCapture = new Promise<ImageBitmap>((resolve) => {
      resolveLate = resolve;
    });
    const submitFrame = vi.fn();
    const close = vi.fn(() => Promise.resolve());
    const capture = vi.fn().mockResolvedValueOnce(firstBitmap).mockReturnValueOnce(lateCapture);
    const liveRuntime: TestLiveRuntime = {
      loadAssets: () =>
        Promise.resolve({
          handModelBuffer: new ArrayBuffer(1),
          poseModelBuffer: new ArrayBuffer(1),
        }),
      createSession: () => ({ initialize: () => Promise.resolve(), submitFrame, close }),
      request: vi.fn((_video: HTMLVideoElement, callback: (timestampMs: number) => void) => {
        callbacks.push(callback);
        return callbacks.length;
      }),
      cancel: vi.fn(),
      capture,
    };

    render(
      <LivePage
        cameraEnvironment={cameraEnvironment(devices).environment}
        modelBundleUrl="https://example.test/bundle/"
        modelBundleSession={loader}
        liveRuntime={liveRuntime}
      />,
    );

    await startCamera();
    await waitFor(() => expect(callbacks).toHaveLength(1));
    act(() => callbacks[0]!(1_250));
    await waitFor(() => expect(submitFrame).toHaveBeenCalledWith(firstBitmap, 1_250));
    await waitFor(() => expect(callbacks).toHaveLength(2));
    act(() => callbacks[1]!(1_500));
    await waitFor(() => expect(capture).toHaveBeenCalledTimes(2));
    fireEvent.click(screen.getByRole("button", { name: "Stop camera" }));
    await waitFor(() => expect(close).toHaveBeenCalledOnce());
    await act(async () => {
      resolveLate(lateBitmap);
      await lateCapture;
    });

    expect(lateClose).toHaveBeenCalledOnce();
    expect(submitFrame).toHaveBeenCalledOnce();
  });

  it("keeps the last diagnostics when frame capture fails", async () => {
    const devices = new TestMediaDevices();
    devices.getUserMedia.mockResolvedValue(cameraStream("front").stream);
    devices.enumerateDevices.mockResolvedValue([]);
    const bundle = { id: "browser-candidate", version: "1.0.0" } as VerifiedModelBundle;
    const callbacks: Array<(timestampMs: number) => void> = [];
    const liveRuntime: TestLiveRuntime = {
      loadAssets: () =>
        Promise.resolve({
          handModelBuffer: new ArrayBuffer(1),
          poseModelBuffer: new ArrayBuffer(1),
        }),
      createSession: (_bundle, _buffers, onState) => ({
        initialize: () => {
          onState({
            phase: "ready",
            stableResult: null,
            failureCode: null,
            diagnostics: {
              detectorState: "inactive",
              landmarkState: "usable",
              detectedHands: 1,
              droppedFrames: 4,
              backend: "wasm",
              bundle: { id: bundle.id, version: bundle.version },
            },
          });
          return Promise.resolve();
        },
        submitFrame: vi.fn(),
        close: vi.fn(() => Promise.resolve()),
      }),
      request: vi.fn((_video: HTMLVideoElement, callback: (timestampMs: number) => void) => {
        callbacks.push(callback);
        return callbacks.length;
      }),
      cancel: vi.fn(),
      capture: vi.fn(() => Promise.reject(new Error("simulated capture failure"))),
    };

    render(
      <LivePage
        cameraEnvironment={cameraEnvironment(devices).environment}
        modelBundleUrl="https://example.test/bundle/"
        modelBundleSession={{
          status: { phase: "ready", active: { id: bundle.id, version: bundle.version } },
          load: vi.fn(() => Promise.resolve(bundle)),
        }}
        liveRuntime={liveRuntime}
      />,
    );

    await startCamera();
    await waitFor(() => expect(callbacks).toHaveLength(1));
    act(() => callbacks[0]!(1_250));
    await screen.findByRole("button", { name: "Retry setup" });
    fireEvent.click(screen.getByText("Session diagnostics"));
    const diagnostics = screen.getByLabelText("Session diagnostics");
    expect(within(diagnostics).getByText("Inactive")).toBeVisible();
    expect(within(diagnostics).getByText("1 hand detected")).toBeVisible();
    expect(within(diagnostics).getByText("4")).toBeVisible();
  });

  it("recreates a failed runtime when the user retries", async () => {
    const devices = new TestMediaDevices();
    devices.getUserMedia.mockResolvedValue(cameraStream("front").stream);
    devices.enumerateDevices.mockResolvedValue([]);
    const bundle = { id: "browser-candidate", version: "1.0.0" } as VerifiedModelBundle;
    const load: ModelBundleSession["load"] = () => Promise.resolve(bundle);
    const loader: Pick<ModelBundleSession, "load" | "status"> = {
      status: { phase: "ready", active: { id: bundle.id, version: bundle.version } },
      load: vi.fn(load),
    };
    const createFailedSession: TestLiveSessionFactory = (_bundle, _buffers, onState) => ({
      initialize: () => {
        onState({ phase: "failed", stableResult: null, failureCode: "test.failure" });
        return Promise.resolve();
      },
      submitFrame: vi.fn(),
      close: vi.fn(() => Promise.resolve()),
    });
    const factory = vi.fn(createFailedSession);

    render(
      <LivePage
        cameraEnvironment={cameraEnvironment(devices).environment}
        modelBundleUrl="https://example.test/bundle/"
        modelBundleSession={loader}
        liveRuntime={{
          loadAssets: () =>
            Promise.resolve({
              handModelBuffer: new ArrayBuffer(1),
              poseModelBuffer: new ArrayBuffer(1),
            }),
          createSession: factory,
          request: () => 1,
          cancel: vi.fn(),
          capture: vi.fn(),
        }}
      />,
    );

    await startCamera();
    const retry = await screen.findByRole("button", { name: "Retry setup" });
    fireEvent.click(retry);
    await waitFor(() => expect(factory).toHaveBeenCalledTimes(2));
    fireEvent.click(screen.getByRole("button", { name: "Stop camera" }));
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "Retry setup" })).not.toBeInTheDocument(),
    );
  });
});
