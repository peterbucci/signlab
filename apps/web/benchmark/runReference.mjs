import { existsSync } from "node:fs";
import { cp, mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import { cpus, platform, release, totalmem } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn, spawnSync } from "node:child_process";
import { chromium } from "playwright";
import { assertSanitized, assertStaticBudgets, loadBudgets, nearestRank, sha256, summarizeStaticAssets } from "./releaseMetrics.mjs";

const appRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = resolve(appRoot, "../..");
const distRoot = join(appRoot, "dist");
const option = (name, fallback) => process.argv.find((item) => item.startsWith(`${name}=`))?.slice(name.length + 1) ?? fallback;
const bundleRoot = resolve(appRoot, option("--bundle-root", "../../artifacts/story37-parity-bundle"));
const taskLockPath = join(repositoryRoot, "src/signlab/resources/extraction/models/mediapipe-tasks-1.0.1.lock.json");
const fixturePath = join(repositoryRoot, "tests/fixtures/public/parity/candidate-runtime-goldens-v1.json");
const wait = (milliseconds) => new Promise((done) => setTimeout(done, milliseconds));
const parse = async (path) => JSON.parse(await readFile(path, "utf8"));

function runNode(arguments_, environment = process.env) {
  const run = spawnSync(process.execPath, arguments_, { cwd: appRoot, env: environment, stdio: "inherit" });
  if (run.status !== 0) throw new Error("benchmark.command.failed");
}

async function prepare(budgets) {
  const manifestBytes = await readFile(join(bundleRoot, "manifest.json"));
  const manifest = JSON.parse(manifestBytes);
  let candidateBytes = manifestBytes.length;
  for (const asset of manifest.assets) candidateBytes += (await stat(join(bundleRoot, asset.locator.path))).size;
  const modelSha256 = manifest.assets.find(({ role }) => role === "model")?.sha256;
  if (sha256(manifestBytes) !== budgets.candidateBundle.manifestSha256 || candidateBytes !== budgets.candidateBundle.expectedBytes ||
      candidateBytes > budgets.candidateBundle.maximumBytes || modelSha256 !== budgets.candidateBundle.modelSha256) {
    throw new Error("benchmark.bundle.invalid");
  }
  runNode([process.env.npm_execpath, "run", "build"], { ...process.env,
    VITE_SIGNLAB_MODEL_BUNDLE_URL: "/models/candidate/",
    VITE_SIGNLAB_TRUSTED_MODEL_MANIFEST_SHA256: budgets.candidateBundle.manifestSha256 });
  const staticAssets = await summarizeStaticAssets(distRoot);
  assertStaticBudgets(staticAssets, budgets.staticAssets);
  await cp(bundleRoot, join(distRoot, "models/candidate"), { recursive: true });
  const taskLock = await parse(taskLockPath);
  const taskRoot = join(distRoot, "models/mediapipe");
  await mkdir(taskRoot, { recursive: true });
  for (const task of taskLock.tasks) {
    const response = await fetch(task.source_url);
    const bytes = Buffer.from(await response.arrayBuffer());
    if (!response.ok || bytes.length !== task.size_bytes || sha256(bytes) !== task.sha256) throw new Error("benchmark.task.invalid");
    await writeFile(join(taskRoot, task.filename), bytes);
  }
  return { candidateBytes, manifestBytes, staticAssets, taskLock };
}

async function preview() {
  const port = 4176;
  const server = spawn(process.execPath, [join(appRoot, "node_modules/vite/bin/vite.js"), "preview", "--host", "127.0.0.1", "--port", `${port}`, "--strictPort"], { cwd: appRoot, stdio: "ignore" });
  for (let attempt = 0; attempt < 50; attempt += 1) {
    try { if ((await fetch(`http://127.0.0.1:${port}`)).ok) return { origin: `http://127.0.0.1:${port}`, server }; } catch {}
    if (server.exitCode !== null) break;
    await wait(100);
  }
  server.kill();
  throw new Error("benchmark.preview.failed");
}

async function measureInference(page, workerUrl, input) {
  const run = await page.evaluate(async ({ workerUrl, input }) => {
    const manifest = await (await fetch("/models/candidate/manifest.json")).json();
    const pathFor = (role) => manifest.assets.find((asset) => asset.role === role).locator.path;
    const buffers = await Promise.all(["model", "feature_plan", "decision_policy"].map(async (role) =>
      (await fetch(`/models/candidate/${pathFor(role)}`)).arrayBuffer()));
    const worker = new Worker(workerUrl, { type: "module" });
    const receive = (type, requestId) => new Promise((resolve, reject) => {
      const handler = ({ data }) => {
        if (data.type !== "failure" && (data.type !== type || data.requestId !== requestId)) return;
        worker.removeEventListener("message", handler);
        data.type === "failure" ? reject(new Error(data.code)) : resolve(data);
      };
      worker.addEventListener("message", handler);
    });
    const protocolVersion = "signlab-candidate-inference-worker/1";
    const ready = receive("ready", undefined);
    worker.postMessage({ type: "initialize", protocolVersion, bundle: { id: manifest.bundle_id, version: manifest.version },
      modelBuffer: buffers[0], featurePlanBuffer: buffers[1], decisionPolicyBuffer: buffers[2] }, buffers);
    const initialized = await ready;
    const measured = [];
    for (let requestId = 0; requestId < 55; requestId += 1) {
      const result = receive("result", requestId);
      worker.postMessage({ type: "classify", protocolVersion, requestId, input });
      const message = await result;
      if (requestId >= 5) measured.push(message.timings.totalMs);
    }
    worker.postMessage({ type: "stop", protocolVersion });
    return { backend: initialized.backend, measured };
  }, { workerUrl, input });
  return { backend: run.backend, fixture: "public_candidate_runtime_gap_unmirrored", warmups: 5, samples: 50,
    p50Ms: nearestRank(run.measured, 0.5), p95Ms: nearestRank(run.measured, 0.95) };
}

async function measurePage(context, origin, workerUrl, input) {
  const page = await context.newPage();
  await page.goto(`${origin}/#/live`);
  await page.getByRole("button", { name: "Start camera" }).click();
  await page.getByText("Watching for the start of a gesture.").waitFor({ timeout: 120000 });
  const startupMs = await page.evaluate(() => performance.now());
  const counters = () => page.evaluate(() => Object.fromEntries([...document.querySelectorAll("dt")]
    .map((node) => [node.textContent, Number(node.nextElementSibling?.textContent)])));
  const start = await counters();
  const startedAt = performance.now();
  await wait(6000);
  const elapsedSeconds = (performance.now() - startedAt) / 1000;
  const end = await counters();
  await page.getByRole("button", { name: "Stop camera" }).click();
  const inference = await measureInference(page, workerUrl, input);
  const longTasks = await page.evaluate(() => globalThis.__signlabLongTasks);
  const processedFrames = end["Processed frames"] - start["Processed frames"];
  const droppedFrames = end["Dropped frames"] - start["Dropped frames"];
  await page.close();
  return { startupMs, landmarks: { fixture: "chromium_fake_camera_640x480_20fps_no_person", windowSeconds: elapsedSeconds,
      processedFrames, droppedFrames, processedFps: processedFrames / elapsedSeconds,
      dropRate: droppedFrames / Math.max(1, processedFrames + droppedFrames) }, inference,
    uiLongTasks: { count: longTasks.length, totalMs: longTasks.reduce((sum, value) => sum + value, 0),
      maximumMs: Math.max(0, ...longTasks) },
    memory: { status: "unavailable", reason: "cross_origin_isolation_not_enabled" } };
}

function markdown(report) {
  const row = (name, run) => `| ${name} | ${run.startupMs.toFixed(1)} | ${run.landmarks.processedFps.toFixed(1)} | ${(run.landmarks.dropRate * 100).toFixed(1)}% | ${run.inference.p50Ms.toFixed(1)} | ${run.inference.p95Ms.toFixed(1)} | ${run.uiLongTasks.count} |`;
  return `# Browser reference baseline v1\n\n> Reference-machine evidence only; timings are not cross-device guarantees or CI thresholds.\n\n- Browser: ${report.environment.browser}\n- Machine: ${report.environment.os}, ${report.environment.cpu}, ${report.environment.logicalCpuCount} logical CPUs, ${report.environment.memoryGiB} GiB RAM\n- Runtime: WASM, one thread\n- Fixture: Chromium's generated no-person 640×480 camera; 50 public synthetic candidate events after 5 warmups\n- Percentiles: nearest-rank over preprocessing + ONNX + decision time\n\n| Run | Startup ms | Landmark FPS | Dropped | Inference p50 ms | Inference p95 ms | UI long tasks |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n${row("Cold cache", report.measurements.cold)}\n${row("Warm cache", report.measurements.warm)}\n\nMemory: unavailable because this release does not enable cross-origin isolation.\n\nStatic assets: ${report.assets.static.totalBytes} bytes; candidate bundle: ${report.assets.candidate.totalBytes} bytes; MediaPipe tasks: ${report.assets.mediapipe.totalBytes} bytes.\n\n## Limits\n\n${report.limitations.map((item) => `- ${item}`).join("\n")}\n`;
}

const budgets = await loadBudgets();
const assets = await prepare(budgets);
if (!existsSync(chromium.executablePath())) runNode([join(appRoot, "node_modules/playwright/cli.js"), "install", "chromium"]);
const { origin, server } = await preview();
const workerName = (await readdir(join(distRoot, "assets"))).find((name) => /^candidateInference\.worker-.*\.js$/u.test(name));
if (!workerName) throw new Error("benchmark.worker.missing");
const fixtureBytes = await readFile(fixturePath);
const fixture = JSON.parse(fixtureBytes);
const source = fixture.preprocessingCases.find(({ id }) => id === "gap_unmirrored");
const input = { frames: source.frames, sourceMirrorState: source.sourceMirrorState, quality: source.quality };
const browser = await chromium.launch({ headless: true,
  args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"] });
const context = await browser.newContext({ permissions: ["camera"] });
await context.addInitScript(() => {
  globalThis.__signlabLongTasks = [];
  new PerformanceObserver((list) => globalThis.__signlabLongTasks.push(...list.getEntries().map(({ duration }) => duration)))
    .observe({ type: "longtask", buffered: true });
});
try {
  const cold = await measurePage(context, origin, `${origin}/assets/${workerName}`, input);
  const warm = await measurePage(context, origin, `${origin}/assets/${workerName}`, input);
  if ([cold, warm].some((run) => run.inference.backend !== budgets.runtime.backend)) throw new Error("benchmark.runtime.invalid");
  const taskBytes = assets.taskLock.tasks.reduce((sum, task) => sum + task.size_bytes, 0);
  const report = { format: "signlab-browser-reference/1", recordedAt: new Date().toISOString(),
    sourceCommit: spawnSync("git", ["rev-parse", "HEAD"], { cwd: repositoryRoot, encoding: "utf8" }).stdout.trim(),
    environment: { browser: `Chromium ${browser.version()}`, os: `${platform()} ${release()} ${process.arch}`,
      cpu: cpus()[0].model, logicalCpuCount: cpus().length, memoryGiB: Math.round(totalmem() / 2 ** 30) },
    runtime: budgets.runtime,
    fixtures: { landmarks: { id: "chromium_fake_camera_v1", personData: false },
      inference: { id: source.id, sha256: sha256(fixtureBytes) } },
    percentileMethod: "nearest_rank", measurements: { cold, warm },
    assets: { static: assets.staticAssets,
      candidate: { totalBytes: assets.candidateBytes, semanticSha256: budgets.candidateBundle.semanticSha256,
        manifestSha256: sha256(assets.manifestBytes), modelSha256: budgets.candidateBundle.modelSha256 },
      mediapipe: { totalBytes: taskBytes, tasks: assets.taskLock.tasks.map(({ filename, size_bytes, sha256 }) =>
        ({ filename, sizeBytes: size_bytes, sha256 })) } },
    limitations: ["One Windows reference machine and Chromium version only.", "The no-person landmark workload is easier than hand-containing video.",
      "The fixed-window snapshot excludes at most two frames retained by the bounded queue.",
      "Inference excludes capture, event detection, worker transfer, and rendering.",
      "Long Tasks covers the main thread, not worker computation.",
      "Wall-clock values are reviewed evidence, never CI thresholds."] };
  assertSanitized(report);
  const reportRoot = join(repositoryRoot, "docs/reports");
  await writeFile(join(reportRoot, "browser-reference-v1.json"), `${JSON.stringify(report)}\n`);
  await writeFile(join(reportRoot, "browser-reference-v1.md"), markdown(report));
  process.stdout.write("Browser reference reports written.\n");
} finally {
  await context.close();
  await browser.close();
  server.kill();
}
