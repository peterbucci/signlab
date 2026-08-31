import { createHash } from "node:crypto";
import { readFile, readdir, stat } from "node:fs/promises";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const benchmarkRoot = dirname(fileURLToPath(import.meta.url));
const appRoot = resolve(benchmarkRoot, "..");
const repositoryRoot = resolve(appRoot, "../..");

export const sha256 = (bytes) => `sha256:${createHash("sha256").update(bytes).digest("hex")}`;

export function nearestRank(values, percentile) {
  if (values.length === 0 || percentile <= 0 || percentile > 1 || !values.every(Number.isFinite)) {
    throw new Error("benchmark.metrics.invalid_sample");
  }
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.ceil(percentile * ordered.length) - 1];
}

async function filesUnder(root) {
  const output = [];
  for (const entry of await readdir(root, { withFileTypes: true })) {
    const path = join(root, entry.name);
    if (entry.isDirectory()) output.push(...(await filesUnder(path)));
    else if (entry.isFile()) output.push({ path, bytes: (await stat(path)).size });
  }
  return output;
}

export async function summarizeStaticAssets(root) {
  const files = await filesUnder(root);
  const javascript = files.filter(({ path }) => extname(path) === ".js");
  const wasm = files.filter(({ path }) => extname(path) === ".wasm");
  const sum = (items) => items.reduce((total, item) => total + item.bytes, 0);
  return {
    fileCount: files.length,
    totalBytes: sum(files),
    javascriptBytes: sum(javascript),
    largestJavaScriptBytes: Math.max(0, ...javascript.map(({ bytes }) => bytes)),
    wasmBytes: sum(wasm),
  };
}

export function assertStaticBudgets(summary, limits) {
  for (const [metric, maximum] of Object.entries(limits)) {
    const key = metric.replace(/Maximum$/, "");
    if (!Number.isSafeInteger(summary[key]) || summary[key] > maximum) {
      throw new Error(`benchmark.budget.${key}_exceeded`);
    }
  }
}

export function assertSanitized(value) {
  const serialized = JSON.stringify(value);
  if (/file:|[A-Za-z]:\\|\/(?:home|Users)\//u.test(serialized)) {
    throw new Error("benchmark.report.private_path");
  }
}

export async function loadBudgets() {
  return JSON.parse(await readFile(join(benchmarkRoot, "budgets.v1.json"), "utf8"));
}

async function check(distRoot) {
  const budgets = await loadBudgets();
  const summary = await summarizeStaticAssets(distRoot);
  assertStaticBudgets(summary, budgets.staticAssets);
  const taskLock = JSON.parse(
    await readFile(
      join(
        repositoryRoot,
        "src/signlab/resources/extraction/models/mediapipe-tasks-1.0.1.lock.json",
      ),
      "utf8",
    ),
  );
  const taskBytes = taskLock.tasks.reduce((total, task) => total + task.size_bytes, 0);
  if (
    taskBytes !== budgets.mediapipeTasks.expectedBytes ||
    taskBytes > budgets.mediapipeTasks.maximumBytes
  ) {
    throw new Error("benchmark.budget.mediapipe_tasks_invalid");
  }
  process.stdout.write("Release asset budgets passed.\n");
}

if (process.argv[2] === "--check") await check(resolve(appRoot, process.argv[3] ?? "dist"));
