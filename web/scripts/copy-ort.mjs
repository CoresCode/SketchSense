/**
 * onnxruntime-web ships several WASM runtimes inside node_modules (plain, jsep, jspi,
 * asyncify -- ~61 MB in total). Vite will not copy any of them for us.
 *
 * We only need the plain SIMD build for a single-threaded WASM session, so we publish
 * exactly one file (~14 MB) to public/ort/ and point ort.env.wasm.wasmPaths at it.
 * Anything else that a previous install left behind is pruned so a deploy never trips
 * Cloudflare's 25 MiB per-file limit.
 *
 * Runs automatically after `npm install`.
 */
import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync, unlinkSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const dist = join(root, "node_modules", "onnxruntime-web", "dist");
const out = join(root, "public", "ort");

if (!existsSync(dist)) {
  console.warn("[ort] node_modules/onnxruntime-web not installed yet - skipping");
  process.exit(0);
}

// Preferred artefact, newest naming first. The fallback keeps this working if ORT renames.
const preferred = "ort-wasm-simd-threaded.wasm";
const rejected = /(jsep|jspi|asyncify|training)/;

const candidates = readdirSync(dist).filter(
  (f) => f.endsWith(".wasm") && !rejected.test(f),
);
const chosen = candidates.includes(preferred) ? [preferred] : candidates.slice(0, 1);

if (chosen.length === 0) {
  console.warn("[ort] no usable .wasm found in", dist);
  process.exit(0);
}

mkdirSync(out, { recursive: true });

// prune previously published runtimes that are no longer wanted
for (const existing of readdirSync(out)) {
  if (existing.endsWith(".wasm") && !chosen.includes(existing)) {
    unlinkSync(join(out, existing));
    console.log(`[ort] pruned stale ${existing}`);
  }
}

let bytes = 0;
for (const file of chosen) {
  copyFileSync(join(dist, file), join(out, file));
  bytes += statSync(join(dist, file)).size;
}

console.log(
  `[ort] published ${chosen.join(", ")} (${(bytes / 1e6).toFixed(1)} MB) -> public/ort`,
);
