import { useCallback, useEffect, useRef, useState } from "react";

export type CameraStatus =
  | "idle"
  | "requesting"
  | "active"
  | "paused"
  | "switching"
  | "denied"
  | "unavailable"
  | "insecure"
  | "interrupted"
  | "error";

export interface CameraDevice {
  readonly deviceId: string;
  readonly label: string;
}

export interface CameraEnvironment {
  readonly document: Pick<Document, "hidden" | "addEventListener" | "removeEventListener">;
  readonly isSecureContext: boolean;
  readonly mediaDevices: Pick<MediaDevices, "getUserMedia" | "enumerateDevices"> | undefined;
}

export interface CameraSessionState {
  readonly status: CameraStatus;
  readonly message: string;
  readonly stream: MediaStream | null;
  readonly devices: readonly CameraDevice[];
  readonly selectedDeviceId: string | null;
}

function browserCameraEnvironment(): CameraEnvironment {
  return {
    document,
    isSecureContext: globalThis.isSecureContext === true,
    mediaDevices: navigator.mediaDevices,
  };
}

function initialState(environment: CameraEnvironment): CameraSessionState {
  if (!environment.isSecureContext) {
    return {
      status: "insecure",
      message: "Camera access requires HTTPS or a local development address.",
      stream: null,
      devices: [],
      selectedDeviceId: null,
    };
  }
  if (environment.mediaDevices?.getUserMedia === undefined) {
    return {
      status: "unavailable",
      message: "This browser does not provide camera access.",
      stream: null,
      devices: [],
      selectedDeviceId: null,
    };
  }
  return {
    status: "idle",
    message: "Camera is off. Nothing is being captured.",
    stream: null,
    devices: [],
    selectedDeviceId: null,
  };
}

function cameraError(error: unknown): Pick<CameraSessionState, "status" | "message"> {
  const name =
    typeof error === "object" && error !== null && "name" in error
      ? String(error.name)
      : "UnknownError";
  if (name === "NotAllowedError" || name === "SecurityError") {
    return {
      status: "denied",
      message:
        "Camera permission was denied. Allow camera access in site settings, then try again.",
    };
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return {
      status: "unavailable",
      message: "No usable camera was found. Connect or select another camera, then try again.",
    };
  }
  if (name === "NotReadableError" || name === "AbortError") {
    return {
      status: "error",
      message: "The camera could not start. Close other camera apps and try again.",
    };
  }
  return {
    status: "error",
    message: "The camera could not start. Check the device and try again.",
  };
}

function videoConstraints(deviceId?: string): MediaStreamConstraints {
  return {
    audio: false,
    video:
      deviceId === undefined
        ? { facingMode: { ideal: "user" } }
        : { deviceId: { exact: deviceId } },
  };
}

export function useCameraSession(environment?: CameraEnvironment) {
  const [cameraEnvironment] = useState(() => environment ?? browserCameraEnvironment());
  const [state, setState] = useState(() => initialState(cameraEnvironment));
  const streamRef = useRef<MediaStream | null>(null);
  const operationRef = useRef(0);
  const mountedRef = useRef(true);
  const trackEndHandlersRef = useRef(new Map<MediaStreamTrack, EventListener>());

  const releaseStream = useCallback((stream: MediaStream) => {
    for (const track of stream.getTracks()) {
      const handler = trackEndHandlersRef.current.get(track);
      if (handler !== undefined) track.removeEventListener("ended", handler);
      trackEndHandlersRef.current.delete(track);
      track.stop();
    }
  }, []);

  const refreshDevices = useCallback(async () => {
    const mediaDevices = cameraEnvironment.mediaDevices;
    if (mediaDevices === undefined) return;
    try {
      const devices = (await mediaDevices.enumerateDevices())
        .filter(({ kind }) => kind === "videoinput")
        .map(({ deviceId, label }, index) => ({
          deviceId,
          label: label || `Camera ${index + 1}`,
        }));
      if (mountedRef.current) setState((current) => ({ ...current, devices }));
    } catch {
      // Device labels are helpful but not required to keep an active stream usable.
    }
  }, [cameraEnvironment]);

  const commitStream = useCallback(
    (stream: MediaStream, requestedDeviceId: string | undefined, operation: number) => {
      if (!mountedRef.current || operation !== operationRef.current) {
        releaseStream(stream);
        return;
      }
      const videoTrack = stream.getVideoTracks()[0];
      if (videoTrack === undefined) {
        releaseStream(stream);
        const previous = streamRef.current;
        streamRef.current = null;
        if (previous !== null) releaseStream(previous);
        setState((current) => ({
          ...current,
          status: "unavailable",
          message: "The selected device did not provide a video track.",
          stream: null,
          selectedDeviceId: null,
        }));
        return;
      }

      const ended = () => {
        if (streamRef.current !== stream) return;
        streamRef.current = null;
        releaseStream(stream);
        if (mountedRef.current) {
          setState((current) => ({
            ...current,
            status: "interrupted",
            message: "The camera stopped unexpectedly. Reconnect it or start the camera again.",
            stream: null,
          }));
        }
      };
      for (const track of stream.getTracks()) {
        trackEndHandlersRef.current.set(track, ended);
        track.addEventListener("ended", ended, { once: true });
      }

      const previous = streamRef.current;
      streamRef.current = stream;
      if (previous !== null && previous !== stream) releaseStream(previous);
      const selectedDeviceId = videoTrack.getSettings().deviceId ?? requestedDeviceId ?? null;
      setState((current) => ({
        ...current,
        status: "active",
        message: "Camera is on. Raw video stays on this page and is not saved or uploaded.",
        stream,
        selectedDeviceId,
      }));
      void refreshDevices();
    },
    [refreshDevices, releaseStream],
  );

  const requestStream = useCallback(
    async (deviceId?: string, switching = false) => {
      const current = cameraEnvironment;
      if (!current.isSecureContext) {
        setState((value) => ({
          ...value,
          status: "insecure",
          message: "Camera access requires HTTPS or a local development address.",
        }));
        return;
      }
      if (current.mediaDevices?.getUserMedia === undefined) {
        setState((value) => ({
          ...value,
          status: "unavailable",
          message: "This browser does not provide camera access.",
        }));
        return;
      }

      const operation = operationRef.current + 1;
      operationRef.current = operation;
      setState((value) => ({
        ...value,
        status: switching ? "switching" : "requesting",
        message: switching
          ? "Switching cameras…"
          : "Waiting for you to choose whether to allow camera access…",
      }));
      try {
        const stream = await current.mediaDevices.getUserMedia(videoConstraints(deviceId));
        commitStream(stream, deviceId, operation);
      } catch (error) {
        if (!mountedRef.current || operation !== operationRef.current) return;
        const failure = cameraError(error);
        if (switching && streamRef.current !== null) {
          setState((value) => ({
            ...value,
            status: "active",
            message: `${failure.message} The previous camera is still active.`,
          }));
        } else {
          setState((value) => ({ ...value, ...failure, stream: null }));
        }
      }
    },
    [cameraEnvironment, commitStream],
  );

  const start = useCallback(() => requestStream(), [requestStream]);

  const switchCamera = useCallback(
    (deviceId: string) => {
      if (state.status !== "active" || deviceId === state.selectedDeviceId) {
        return Promise.resolve();
      }
      return requestStream(deviceId, streamRef.current !== null);
    },
    [requestStream, state.selectedDeviceId, state.status],
  );

  const pause = useCallback(() => {
    const stream = streamRef.current;
    if (stream === null || state.status !== "active") return;
    for (const track of stream.getVideoTracks()) track.enabled = false;
    setState((current) => ({
      ...current,
      status: "paused",
      message: "Preview is paused. Stop the camera to release it completely.",
    }));
  }, [state.status]);

  const resume = useCallback(() => {
    const stream = streamRef.current;
    if (stream === null || state.status !== "paused") return;
    for (const track of stream.getVideoTracks()) track.enabled = true;
    setState((current) => ({
      ...current,
      status: "active",
      message: "Camera is on. Raw video stays on this page and is not saved or uploaded.",
    }));
  }, [state.status]);

  const stop = useCallback(
    (message = "Camera is off. Nothing is being captured.") => {
      operationRef.current += 1;
      const stream = streamRef.current;
      streamRef.current = null;
      if (stream !== null) releaseStream(stream);
      if (mountedRef.current) {
        setState((current) => ({
          ...current,
          status: "idle",
          message,
          stream: null,
          selectedDeviceId: null,
        }));
      }
    },
    [releaseStream],
  );

  useEffect(() => {
    mountedRef.current = true;
    const current = cameraEnvironment;
    const handleVisibility = () => {
      if (!current.document.hidden) return;
      operationRef.current += 1;
      const stream = streamRef.current;
      streamRef.current = null;
      if (stream !== null) releaseStream(stream);
      setState((cameraState) => {
        const wasStarting =
          cameraState.status === "requesting" || cameraState.status === "switching";
        if (stream === null && !wasStarting) return cameraState;
        return {
          ...cameraState,
          status: "idle",
          message:
            "Camera stopped because this page was hidden. Start it again when you are ready.",
          stream: null,
          selectedDeviceId: null,
        };
      });
    };
    current.document.addEventListener("visibilitychange", handleVisibility);
    return () => {
      mountedRef.current = false;
      operationRef.current += 1;
      current.document.removeEventListener("visibilitychange", handleVisibility);
      const stream = streamRef.current;
      streamRef.current = null;
      if (stream !== null) releaseStream(stream);
    };
  }, [cameraEnvironment, releaseStream]);

  const canRequest =
    cameraEnvironment.isSecureContext && cameraEnvironment.mediaDevices?.getUserMedia !== undefined;

  return { state, canRequest, start, pause, resume, stop, switchCamera } as const;
}
