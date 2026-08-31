# Browser reference baseline v1

> Reference-machine evidence only; timings are not cross-device guarantees or CI thresholds.

- Browser: Chromium 151.0.7922.34
- Machine: win32 10.0.26200 x64, Intel(R) Core(TM) i9-14900HX, 32 logical CPUs, 32 GiB RAM
- Toolchain: Node 24.20.0, npm 11.19.0
- Runtime: WASM, one thread
- Fixture: Chromium's generated no-person 640×480 camera; 50 public synthetic candidate events after 5 warmups
- Percentiles: nearest-rank over preprocessing + ONNX + decision time

| Run | Startup ms | Landmark FPS | Dropped | Inference p50 ms | Inference p95 ms | UI long tasks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold cache | 2327.2 | 17.8 | 11.6% | 0.5 | 0.8 | 1 |
| Warm cache | 981.2 | 20.1 | 0.0% | 0.5 | 0.9 | 1 |

Memory: unavailable because this release does not enable cross-origin isolation.

Static assets: 26761032 bytes; candidate bundle: 151324 bytes; MediaPipe tasks: 13596851 bytes.

## Limits

- One Windows reference machine and Chromium version only.
- The no-person landmark workload is easier than hand-containing video.
- The fixed-window snapshot excludes at most two frames retained by the bounded queue.
- Inference excludes capture, event detection, worker transfer, and rendering.
- Long Tasks covers the main thread, not worker computation.
- Wall-clock values are reviewed evidence, never CI thresholds.
