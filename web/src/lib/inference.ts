// The /wasm entry excludes the WebGPU (jsep) runtime: 14 MB instead of 29 MB, and it
// keeps every emitted asset comfortably under Cloudflare Pages' 25 MiB per-file limit.
import * as ort from "onnxruntime-web/wasm";
import { SIZE } from "./raster";

export type Prediction = { index: number; label: string; prob: number };

export type ModelInfo = {
  classes: string[];
  valTop1: number | null;
  size: number;
};

export class ModelMissingError extends Error {}

export type SketchModel = ModelInfo & {
  predict: (input: Float32Array) => Promise<{ preds: Prediction[]; ms: number }>;
  provider: string;
};

function softmaxTopK(logits: Float32Array, k: number): { index: number; prob: number }[] {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];

  const exps = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max);
    exps[i] = e;
    sum += e;
  }

  return Array.from(exps.keys())
    .sort((a, b) => exps[b] - exps[a])
    .slice(0, k)
    .map((i) => ({ index: i, prob: exps[i] / sum }));
}

let runtimeConfigured = false;

function configureRuntime() {
  if (runtimeConfigured) return;
  // Single-threaded WASM on purpose: no COOP/COEP headers required (Cloudflare Pages
  // friendly), one ~14 MB artifact, and a 1.6 M-parameter net at 64x64 runs in ~2 ms.
  // wasmPaths is left to Vite: the /wasm entry resolves its own .wasm asset URL, so the
  // file is hashed and cached with the rest of the bundle.
  ort.env.wasm.numThreads = 1;
  ort.env.logLevel = "error";
  runtimeConfigured = true;
}

export async function loadSketchModel(onStatus?: (m: string) => void): Promise<SketchModel> {
  configureRuntime();

  onStatus?.("reading labels");
  const labelsRes = await fetch("/model/labels.json");
  // Vite's dev server answers missing files with index.html, so status alone is not enough.
  const labelsType = labelsRes.headers.get("content-type") ?? "";
  if (!labelsRes.ok || !labelsType.includes("json")) {
    throw new ModelMissingError("model/labels.json is missing");
  }
  const labels = (await labelsRes.json()) as {
    classes: string[];
    size: number;
    valTop1: number | null;
  };

  const modelRes = await fetch("/model/model.onnx");
  if (!modelRes.ok) throw new ModelMissingError("model/model.onnx is missing");

  // Stream the weights so the wait is visible instead of a frozen "downloading model…".
  const total = Number(modelRes.headers.get("content-length")) || 0;
  const chunks: Uint8Array[] = [];
  let received = 0;
  let lastPaint = 0;
  const report = (force = false) => {
    if (!onStatus) return;
    const now = performance.now();
    if (!force && now - lastPaint < 200) return;
    lastPaint = now;
    const mb = (n: number) => (n / 1048576).toFixed(1);
    onStatus(
      total > 0
        ? `downloading model ${mb(received)} / ${mb(total)} MB`
        : `downloading model ${mb(received)} MB`,
    );
  };
  report(true);

  if (modelRes.body) {
    const reader = modelRes.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        chunks.push(value);
        received += value.length;
        report();
      }
    }
  } else {
    const buf = new Uint8Array(await modelRes.arrayBuffer());
    chunks.push(buf);
    received = buf.length;
  }

  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) {
    bytes.set(c, offset);
    offset += c.length;
  }
  // A 14 MB binary that starts with "<!doctype" is an HTML error page, not a model.
  if (bytes.length < 1024 || (bytes[0] === 0x3c && bytes[1] === 0x21)) {
    throw new ModelMissingError("model/model.onnx is missing");
  }

  onStatus?.("warming up");
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const dim = labels.size ?? SIZE;

  // One dummy pass: allocates the arenas so the first real stroke is not slow.
  await session.run({
    [inputName]: new ort.Tensor("float32", new Float32Array(dim * dim), [1, 1, dim, dim]),
  });

  const predict = async (input: Float32Array) => {
    const t0 = performance.now();
    const out = await session.run({
      [inputName]: new ort.Tensor("float32", input, [1, 1, dim, dim]),
    });
    const ms = performance.now() - t0;
    const logits = out[outputName].data as Float32Array;
    const preds = softmaxTopK(logits, 5).map((p) => ({
      ...p,
      label: labels.classes[p.index] ?? "?",
    }));
    return { preds, ms };
  };

  return {
    classes: labels.classes,
    size: dim,
    valTop1: labels.valTop1 ?? null,
    provider: "wasm",
    predict,
  };
}
