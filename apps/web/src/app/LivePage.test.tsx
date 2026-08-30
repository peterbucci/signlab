import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { StrictMode } from "react";
import { describe, expect, it, vi } from "vitest";

import type { CameraEnvironment } from "../camera/useCameraSession";
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

    await startCamera();

    expect(devices.getUserMedia).toHaveBeenCalledWith({
      audio: false,
      video: { facingMode: { ideal: "user" } },
    });
    const video = screen.getByLabelText("Local camera preview");
    expect(video).toHaveProperty("srcObject", stream);
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
