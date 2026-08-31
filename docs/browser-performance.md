# Browser performance evidence

SignLab measures the release build on one named machine without collecting a
person's video, landmarks, device identifiers, or local paths. From
`apps/web`, run:

```text
npm run benchmark:reference
```

The command verifies the exact ignored candidate bundle, downloads and verifies
the two public hash-pinned MediaPipe task files when needed, builds the production
site, and drives its real workers in Chromium. It installs the matching pinned
Chromium build on first use. A non-default ignored bundle can be selected with
`-- --bundle-root=<directory>`.

It writes the sanitized JSON and short Markdown evidence in `docs/reports/`.
The committed report names the browser, operating system, CPU, logical CPU
count, memory, source commit, fixture identities, sample counts, and limitations.

## What the measurements mean

- Cold and warm startup run in the same browser context. Cold begins with empty
  browser and Cache Storage caches; warm preserves both. Startup runs from page
  navigation through both production workers reporting ready.
- Landmark FPS is completed MediaPipe frames divided by a fixed six-second
  window. Drop rate is production-client drops divided by completed plus dropped
  frames during that window. The boundary snapshot excludes at most the one
  in-flight and one pending frame guaranteed by the bounded queue.
- Event inference p50 and p95 use nearest-rank percentiles over 50 sequential
  worker-reported totals after five warmups. Each total covers preprocessing,
  ONNX Runtime Web inference, and the decision policy—not camera-to-result time.
- UI long tasks are Chromium main-thread tasks of at least 50 ms. Worker work is
  outside that measurement.
- Memory is reported only through the credible browser API. It is currently
  unavailable because the intentionally single-threaded release is not
  cross-origin isolated.

The visual input is Chromium's generated 640×480, 20 FPS fake-camera video. It
contains no person, keeping the run private and repeatable, but it is easier than
normal hand-containing video and is not a real-world FPS guarantee. Inference
uses the public `gap_unmirrored` fixture already used for parity testing.

## CI budgets

Every web build checks raw static, JavaScript, largest-JavaScript, and WASM byte
ceilings in `apps/web/benchmark/budgets.v1.json`. Tests also lock the nearest-rank
calculation, report path sanitization, the existing one-in-flight/one-pending
landmark queue, and the one-thread WASM runtime. The pinned MediaPipe task sizes
are checked against their public lock file.

Reference-machine wall-clock timings never fail shared CI. They establish a
reviewed baseline; a future WebGPU or threading experiment needs its own small,
evidence-driven story rather than expanding this runner.
