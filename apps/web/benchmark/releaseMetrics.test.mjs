// @vitest-environment node

import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  assertSanitized,
  assertStaticBudgets,
  nearestRank,
  summarizeFrameWindow,
  summarizeStaticAssets,
} from "./releaseMetrics.mjs";

const temporaryRoots = [];
afterEach(async () =>
  Promise.all(temporaryRoots.splice(0).map((root) => rm(root, { recursive: true }))),
);

describe("release benchmark metrics", () => {
  it("uses the documented nearest-rank percentile", () => {
    const sample = [9, 1, 8, 2, 7, 3, 6, 4, 5, 10];
    expect(nearestRank(sample, 0.5)).toBe(5);
    expect(nearestRank(sample, 0.95)).toBe(10);
    expect(() => nearestRank([], 0.5)).toThrow("benchmark.metrics.invalid_sample");
  });

  it("calculates the fixed-window frame and long-task metrics", () => {
    expect(summarizeFrameWindow(120, 3, 6, [52, 68])).toEqual({
      processedFps: 20,
      dropRate: 3 / 123,
      uiLongTasks: { count: 2, totalMs: 120, maximumMs: 68 },
    });
    expect(() => summarizeFrameWindow(1, 0, 0, [])).toThrow("benchmark.metrics.invalid_window");
  });

  it("summarizes raw build bytes and rejects a crossed ceiling", async () => {
    const root = await mkdtemp(join(tmpdir(), "signlab-release-metrics-"));
    temporaryRoots.push(root);
    await mkdir(join(root, "assets"));
    await writeFile(join(root, "index.html"), "1234");
    await writeFile(join(root, "assets", "app.js"), "12345");
    await writeFile(join(root, "assets", "runtime.wasm"), "123456");
    const summary = await summarizeStaticAssets(root);
    expect(summary).toEqual({
      fileCount: 3,
      totalBytes: 15,
      javascriptBytes: 5,
      largestJavaScriptBytes: 5,
      wasmBytes: 6,
    });
    expect(() => assertStaticBudgets(summary, { totalBytesMaximum: 15 })).not.toThrow();
    expect(() => assertStaticBudgets(summary, { javascriptBytesMaximum: 4 })).toThrow(
      "benchmark.budget.javascriptBytes_exceeded",
    );
  });

  it("rejects local paths before evidence is written", () => {
    expect(() => assertSanitized({ result: "portable" })).not.toThrow();
    expect(() =>
      assertSanitized({ path: ["C:", "Users", "person", "model.onnx"].join("\\") }),
    ).toThrow("benchmark.report.private_path");
    expect(() => assertSanitized({ path: ["", "home", "person", "model.onnx"].join("/") })).toThrow(
      "benchmark.report.private_path",
    );
  });
});
