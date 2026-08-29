import { supportsLandmarkWorkerRuntime } from "../landmarks/LandmarkWorkerClient";

export interface StaticPageDefinition {
  path: string;
  label: string;
  eyebrow: string;
  heading: string;
  summary: string;
  status: string;
  detailsTitle: string;
  details: readonly string[];
  note?: string;
}

const landmarkRuntimeSupport = supportsLandmarkWorkerRuntime()
  ? "This browser exposes the Worker, ImageBitmap, OffscreenCanvas, WebAssembly, and SIMD capabilities needed by the later demo."
  : "This browser does not expose every Worker, ImageBitmap, OffscreenCanvas, WebAssembly, and SIMD capability needed by the later demo.";

export const staticPages = [
  {
    path: "/live",
    label: "Live demo",
    eyebrow: "Browser demo",
    heading: "Live recognition",
    summary:
      "This page will eventually recognize one completed gesture event at a time from five project prompts: Hello, No, Please, Thank you, and Yes.",
    status:
      "Landmark extraction is implemented off the UI thread but is not active on this page yet. No camera permission is requested, no frame is captured, and no model bundle is loaded.",
    detailsTitle: "Current worker boundary",
    details: [
      "Once supplied with already-verified model bytes, MediaPipe initializes once inside a worker.",
      "The worker returns typed hand and body landmarks and keeps only the newest waiting frame.",
      "Processed or replaced frames are released, and image data is not logged.",
      landmarkRuntimeSupport,
      "Camera controls and gesture recognition remain separate follow-up work.",
    ],
    note: "These English labels are project prompts, not a validated claim about ASL vocabulary.",
  },
  {
    path: "/replay",
    label: "Replay",
    eyebrow: "Repeatable testing",
    heading: "Deterministic replay",
    summary:
      "Replay will run an approved recording or landmark fixture through the same browser pipeline as the camera. This makes deployment behavior repeatable without recording new video.",
    status:
      "The replay harness is not connected yet. This route does not read files or run inference.",
    detailsTitle: "Why replay matters",
    details: [
      "The same input should produce the same result.",
      "Browser behavior can be compared with Python behavior.",
      "Tests can run without requesting camera access.",
    ],
  },
  {
    path: "/results",
    label: "Results",
    eyebrow: "Reviewed evidence only",
    heading: "Research results",
    summary:
      "Only reviewed signer-held-out results tied to a frozen dataset, split, code revision, and model bundle will be published here.",
    status: "There is no promoted model or approved headline accuracy result yet.",
    detailsTitle: "Future evidence",
    details: [
      "Macro-F1 and per-class behavior",
      "Uncertainty and abstention",
      "False activations in continuous video",
      "Browser inference latency",
    ],
  },
  {
    path: "/methodology",
    label: "Methodology",
    eyebrow: "Traceable research",
    heading: "How the research is built",
    summary:
      "SignLab separates data preparation, feature extraction, training, evaluation, model export, and browser inference so every published claim can be traced to exact evidence.",
    status:
      "Dataset and model cards will appear after their review gates pass. Repository documentation is the current source of truth.",
    detailsTitle: "Evidence chain",
    details: [
      "Licensed or explicitly authorized source data",
      "Frozen signer-separated train, validation, and test splits",
      "Versioned features, configuration, and model bundle",
      "Reproducible evaluation and browser parity checks",
    ],
  },
  {
    path: "/feedback",
    label: "Feedback",
    eyebrow: "Local by default",
    heading: "Feedback stays local by default",
    summary:
      "The completed demo will let users review a prediction and choose whether to save a correction locally.",
    status:
      "Feedback collection is not active. This scaffold stores no feedback and sends nothing anywhere.",
    detailsTitle: "Planned safeguards",
    details: [
      "Nothing is saved without a clear user action.",
      "Local corrections remain separate from research training data.",
      "Any future export must be explicit and reviewable.",
    ],
  },
  {
    path: "/privacy",
    label: "Privacy",
    eyebrow: "On-device by design",
    heading: "Designed for on-device processing",
    summary:
      "The completed browser demo is intended to process camera frames on the user’s device. Raw video will not be uploaded or stored by default.",
    status:
      "This scaffold does not request camera access, store feedback, run inference, or include analytics.",
    detailsTitle: "Current privacy facts",
    details: [
      "This page loads only static application files.",
      "There is no application server or user account.",
      "Future camera and storage behavior must pass separate review gates.",
    ],
  },
  {
    path: "/limitations",
    label: "Limitations",
    eyebrow: "Claim boundary",
    heading: "What SignLab does—and does not—claim",
    summary:
      "SignLab is a research prototype for recognizing isolated performances of five predefined hand gestures within continuous webcam video. No sign-language or translation capability is claimed.",
    status:
      "No approved model performance result exists yet, and the browser inference path is not connected.",
    detailsTitle: "Known limitations",
    details: [
      "Five predefined prompts only—not an open vocabulary.",
      "Isolated events—not sentences or conversation.",
      "Inactive, other, and abstain have different meanings.",
      "English labels do not establish validated ASL vocabulary.",
      "This is not a production accessibility or safety tool.",
    ],
  },
] as const satisfies readonly StaticPageDefinition[];

export const navigationItems = [
  { path: "/", label: "Overview" },
  ...staticPages.map(({ path, label }) => ({ path, label })),
] as const;
