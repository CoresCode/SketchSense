/**
 * Reproducible screenshots for the README.
 *
 * The browser I drive through the agent tooling cannot capture pixels, so this script
 * launches headless Chrome with a debugging port, draws on the canvas with real input
 * events over CDP, and screenshots the result. Every image in docs/ can be regenerated:
 *
 *   node tools/capture.mjs --url http://localhost:5174 --out docs/shots/house.png --scene house
 *
 * Requires nothing beyond Node 18+ (global fetch + WebSocket) and a local Chrome.
 */

import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawn } from "node:child_process";

const CHROME =
  process.env.CHROME_PATH ?? "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";

// Accepts both --key=value and --key value.
function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith("--")) continue;
    const eq = token.indexOf("=");
    if (eq > -1) {
      out[token.slice(2, eq)] = token.slice(eq + 1);
    } else {
      out[token.slice(2)] = argv[++i];
    }
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));

const url = args.url ?? "http://localhost:5173";
const out = args.out ?? "docs/shots/sketchsense.png";
const scene = args.scene ?? "house";
const width = Number(args.width ?? 1440);
const height = Number(args.height ?? 1200);
const port = Number(args.port ?? 9333);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function cdpTarget() {
  for (let i = 0; i < 40; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/json/list`);
      const list = await res.json();
      const page = list.find((t) => t.type === "page");
      if (page) return page;
    } catch {
      /* not up yet */
    }
    await sleep(250);
  }
  throw new Error("Chrome debugging endpoint never came up");
}

function connect(wsUrl) {
  const ws = new WebSocket(wsUrl);
  let id = 0;
  const pending = new Map();
  ws.addEventListener("message", (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id);
      pending.delete(msg.id);
      msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result);
    }
  });
  const ready = new Promise((resolve, reject) => {
    ws.addEventListener("open", resolve, { once: true });
    ws.addEventListener("error", reject, { once: true });
  });
  const send = (method, params = {}) =>
    new Promise((resolve, reject) => {
      const msgId = ++id;
      pending.set(msgId, { resolve, reject });
      ws.send(JSON.stringify({ id: msgId, method, params }));
    });
  return { ws, ready, send };
}

const SCENES = {
  // a house: square body plus triangular roof, drawn the way a person would
  house: [
    [[0.32, 0.60], [0.68, 0.60]],
    [[0.68, 0.60], [0.68, 0.86]],
    [[0.68, 0.86], [0.32, 0.86]],
    [[0.32, 0.86], [0.32, 0.60]],
    [[0.26, 0.60], [0.50, 0.34], [0.74, 0.60]],
  ],
  sun: [
    Array.from({ length: 30 }, (_, i) => {
      const t = (i / 29) * Math.PI * 2;
      return [0.5 + 0.14 * Math.cos(t), 0.5 + 0.14 * Math.sin(t)];
    }),
    ...Array.from({ length: 8 }, (_, i) => {
      const t = (i / 8) * Math.PI * 2;
      return [
        [0.5 + 0.22 * Math.cos(t), 0.5 + 0.22 * Math.sin(t)],
        [0.5 + 0.34 * Math.cos(t), 0.5 + 0.34 * Math.sin(t)],
      ];
    }),
  ],
  fish: [
    [[0.28, 0.5], [0.36, 0.4], [0.5, 0.36], [0.64, 0.42], [0.68, 0.5], [0.62, 0.58], [0.5, 0.63], [0.36, 0.59], [0.28, 0.5]],
    [[0.28, 0.5], [0.14, 0.38], [0.16, 0.5], [0.14, 0.62], [0.28, 0.5]],
    [[0.56, 0.46], [0.56, 0.46]],
  ],
};

function interpolate(stroke, perSegment = 12) {
  const pts = [];
  for (let i = 1; i < stroke.length; i++) {
    const [x1, y1] = stroke[i - 1];
    const [x2, y2] = stroke[i];
    const n = Math.max(2, perSegment);
    for (let k = 0; k < n; k++) {
      pts.push([x1 + ((x2 - x1) * k) / (n - 1), y1 + ((y2 - y1) * k) / (n - 1)]);
    }
  }
  return pts;
}

async function main() {
  const profile = mkdtempSync(join(tmpdir(), "sketchsense-shot-"));
  const chrome = spawn(
    CHROME,
    [
      "--headless=new",
      "--disable-gpu",
      "--hide-scrollbars",
      "--no-first-run",
      "--no-default-browser-check",
      `--user-data-dir=${profile}`,
      `--window-size=${width},${height}`,
      `--remote-debugging-port=${port}`,
      url,
    ],
    { stdio: "ignore" },
  );

  try {
    const target = await cdpTarget();
    const { ws, ready, send } = connect(target.webSocketDebuggerUrl);
    await ready;

    await send("Page.enable");
    await send("Runtime.enable");
    await send("Emulation.setDeviceMetricsOverride", {
      width,
      height,
      deviceScaleFactor: 1,
      mobile: false,
    });

    // Wait for the app shell, then for the ONNX session. Polling the DOM is more reliable
    // than chasing load events when the page boots a 6 MB model and a 14 MB wasm runtime.
    let shell = false;
    for (let i = 0; i < 80; i++) {
      const r = await send("Runtime.evaluate", {
        expression: `({ title: document.title, canvas: !!document.querySelector('canvas[aria-label="Drawing canvas"]') })`,
        returnByValue: true,
      });
      if (r.result.value?.canvas) {
        shell = true;
        break;
      }
      await sleep(250);
    }
    if (!shell) {
      const diag = await send("Runtime.evaluate", {
        expression: `({ title: document.title, url: location.href, body: document.body ? document.body.innerText.slice(0,200) : "no body" })`,
        returnByValue: true,
      });
      throw new Error(`app never rendered: ${JSON.stringify(diag.result.value)}`);
    }

    let ok = false;
    for (let i = 0; i < 100; i++) {
      const r = await send("Runtime.evaluate", {
        expression: `document.body.innerText.includes("top-1")`,
        returnByValue: true,
      });
      if (r.result.value === true) {
        ok = true;
        break;
      }
      await sleep(250);
    }
    if (!ok) console.warn("[capture] model never reported ready; capturing anyway");

    const box = await send("Runtime.evaluate", {
      expression: `(() => {
        const c = document.querySelector('canvas[aria-label="Drawing canvas"]');
        if (!c) return null;
        const r = c.getBoundingClientRect();
        return { x: r.left, y: r.top, w: r.width, h: r.height };
      })()`,
      returnByValue: true,
    });
    const rect = box.result.value;
    if (!rect) throw new Error("canvas not found - is the dev server running?");

    const strokes = SCENES[scene] ?? SCENES.house;
    for (const stroke of strokes) {
      const pts = interpolate(stroke);
      const first = pts[0];
      await send("Input.dispatchMouseEvent", {
        type: "mousePressed",
        x: rect.x + first[0] * rect.w,
        y: rect.y + first[1] * rect.h,
        button: "left",
        clickCount: 1,
        buttons: 1,
      });
      for (const [px, py] of pts.slice(1)) {
        await send("Input.dispatchMouseEvent", {
          type: "mouseMoved",
          x: rect.x + px * rect.w,
          y: rect.y + py * rect.h,
          button: "left",
          buttons: 1,
        });
        await sleep(8);
      }
      const last = pts[pts.length - 1];
      await send("Input.dispatchMouseEvent", {
        type: "mouseReleased",
        x: rect.x + last[0] * rect.w,
        y: rect.y + last[1] * rect.h,
        button: "left",
        clickCount: 1,
        buttons: 0,
      });
      await sleep(120);
    }

    await sleep(900);

    const readout = await send("Runtime.evaluate", {
      expression: `(() => {
        const t = document.body.innerText;
        const at = t.indexOf("MODEL'S GUESS");
        return at < 0 ? t.slice(0, 200) : t.slice(at, at + 120);
      })()`,
      returnByValue: true,
    });
    console.log("[capture] prediction:", readout.result.value.replace(/\\n/g, " | "));

    const shot = await send("Page.captureScreenshot", { format: "png" });
    writeFileSync(out, Buffer.from(shot.data, "base64"));
    console.log(`[capture] wrote ${out}`);

    ws.close();
  } finally {
    // kill the whole tree: headless Chrome spawns renderer children that outlive kill()
    if (process.platform === "win32") {
      spawn("taskkill", ["/PID", String(chrome.pid), "/T", "/F"], { stdio: "ignore" });
    } else {
      chrome.kill();
    }
    await sleep(400);
  }
}

main().catch((err) => {
  console.error("[capture] failed:", err.message);
  process.exit(1);
});
