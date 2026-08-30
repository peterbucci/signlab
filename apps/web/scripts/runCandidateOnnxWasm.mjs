import { readFile, writeFile } from "node:fs/promises";

import * as ort from "onnxruntime-web";

const [requestPath, outputPath] = process.argv.slice(2);
if (requestPath === undefined || outputPath === undefined) {
  throw new Error("usage: runCandidateOnnxWasm.mjs REQUEST OUTPUT");
}

const request = JSON.parse(await readFile(requestPath, "utf8"));
if (
  typeof request.modelPath !== "string" ||
  request.inputName !== "input" ||
  request.outputName !== "probabilities" ||
  !Array.isArray(request.cases) ||
  request.cases.length === 0
) {
  throw new Error("invalid request");
}

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
const model = new Uint8Array(await readFile(request.modelPath));
const session = await ort.InferenceSession.create(model, {
  executionProviders: ["wasm"],
});
if (
  session.inputNames.length !== 1 ||
  session.inputNames[0] !== request.inputName ||
  session.outputNames.length !== 1 ||
  session.outputNames[0] !== request.outputName
) {
  throw new Error("invalid model contract");
}

const results = [];
for (const item of request.cases) {
  if (
    typeof item.alias !== "string" ||
    !Array.isArray(item.values) ||
    item.values.length !== 64 * 126 ||
    item.values.some((value) => typeof value !== "number" || !Number.isFinite(value))
  ) {
    throw new Error("invalid case");
  }
  const input = new ort.Tensor("float32", Float32Array.from(item.values), [1, 64, 126]);
  const outputs = await session.run({ [request.inputName]: input });
  const output = outputs[request.outputName];
  if (output === undefined || output.type !== "float32" || output.size !== 6) {
    throw new Error("invalid model output");
  }
  results.push({ alias: item.alias, probabilities: Array.from(output.data) });
}

await writeFile(
  outputPath,
  `${JSON.stringify({ executionProvider: "wasm", wasmThreads: 1, cases: results })}\n`,
  {
    encoding: "utf8",
    flag: "wx",
  },
);
