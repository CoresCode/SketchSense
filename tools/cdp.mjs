/**
 * Small Chrome DevTools Protocol helper shared by the screenshot and QA scripts.
 *
 * Node 18+ only: global fetch and WebSocket, no dependencies. Chrome is launched headless
 * with a throwaway profile so repeated runs never share cookies or cache.
 */

import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawn } from "node:child_process";

export const CHROME =
  process.env.CHROME_PATH ?? "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Accepts both --key=value and --key value (and bare flags). */
export function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!token.startsWith("--")) continue;
    const eq = token.indexOf("=");
    if (eq > -1) out[token.slice(2, eq)] = token.slice(eq + 1);
    else if (argv[i + 1] === undefined || argv[i + 1].startsWith("--")) out[token.slice(2)] = true;
    else out[token.slice(2)] = argv[++i];
  }
  return out;
}

async function cdpTarget(port) {
  for (let i = 0; i < 60; i++) {
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

/**
 * Launch headless Chrome, navigate, and return the small API the scripts need.
 * `onEvent` receives every CDP event (used by the QA script to catch console errors).
 */
export async function openBrowser({
  url,
  width = 1440,
  height = 1200,
  port = 9333,
  initScript,
  onEvent,
} = {}) {
  const profile = mkdtempSync(join(tmpdir(), "sketchsense-cdp-"));
  const chrome = spawn(
    CHROME,
    [
      "--headless=new",
      "--disable-gpu",
      "--no-first-run",
      "--no-default-browser-check",
      "--hide-scrollbars",
      `--user-data-dir=${profile}`,
      `--window-size=${width},${height}`,
      `--remote-debugging-port=${port}`,
      "about:blank",
    ],
    { stdio: "ignore" },
  );

  const target = await cdpTarget(port);
  const { ws, ready, send } = connect(target.webSocketDebuggerUrl);
  await ready;

  ws.addEventListener("message", (ev) => {
    if (!onEvent) return;
    const msg = JSON.parse(ev.data);
    if (msg.method) onEvent(msg);
  });

  await send("Page.enable");
  await send("Runtime.enable");
  await send("Network.enable");
  await send("Log.enable");
  await send("Emulation.setDeviceMetricsOverride", {
    width,
    height,
    deviceScaleFactor: 1,
    mobile: false,
  });
  if (initScript) await send("Page.addScriptToEvaluateOnNewDocument", { source: initScript });
  await send("Page.navigate", { url });

  const evaluate = async (expression) => {
    const r = await send("Runtime.evaluate", {
      expression,
      returnByValue: true,
      awaitPromise: true,
    });
    if (r.exceptionDetails) {
      throw new Error(`evaluate failed: ${r.exceptionDetails.text ?? "unknown"}`);
    }
    return r.result.value;
  };

  const bodyText = () => evaluate("document.body.innerText");

  const waitFor = async (predicate, { timeout = 20000, interval = 100 } = {}) => {
    const started = Date.now();
    while (Date.now() - started < timeout) {
      if (await predicate()) return true;
      await sleep(interval);
    }
    return false;
  };

  /** The app enables the game tabs only once the ONNX session is ready. Works at any width. */
  const modelReady = () =>
    evaluate(`(() => [...document.querySelectorAll("button")].some(
      (el) => el.innerText.trim() === "Speed round" && !el.disabled,
    ))()`);

  const waitForModel = (timeout = 40000) => waitFor(modelReady, { timeout });

  const clickByText = (text) =>
    evaluate(`(() => {
      const want = ${JSON.stringify(text.toLowerCase())};
      const all = [...document.querySelectorAll("button")];
      const b = all.find((el) => el.innerText.trim().toLowerCase() === want)
        ?? all.find((el) => el.innerText.trim().toLowerCase().startsWith(want));
      if (!b) return false;
      // Clicking a disabled button is a no-op, so claiming success would hide a real stall.
      if (b.disabled) return false;
      b.click();
      return true;
    })()`);

  const canvasBox = (label = "Drawing canvas") =>
    evaluate(`(() => {
      const c = document.querySelector('canvas[aria-label=${JSON.stringify(label)}]');
      if (!c) return null;
      const r = c.getBoundingClientRect();
      return { x: r.left, y: r.top, w: r.width, h: r.height };
    })()`);

  /** Draw one stroke through real input events, in 0..1 coordinates of the canvas. */
  const drawStroke = async (box, pts, perPointDelay = 12) => {
    const at = ([px, py]) => ({ x: box.x + px * box.w, y: box.y + py * box.h });
    const first = at(pts[0]);
    await send("Input.dispatchMouseEvent", {
      type: "mousePressed",
      x: first.x,
      y: first.y,
      button: "left",
      clickCount: 1,
      buttons: 1,
    });
    for (const p of pts.slice(1)) {
      const q = at(p);
      await send("Input.dispatchMouseEvent", {
        type: "mouseMoved",
        x: q.x,
        y: q.y,
        button: "left",
        buttons: 1,
      });
      await sleep(perPointDelay);
    }
    const last = at(pts[pts.length - 1]);
    await send("Input.dispatchMouseEvent", {
      type: "mouseReleased",
      x: last.x,
      y: last.y,
      button: "left",
      clickCount: 1,
      buttons: 0,
    });
  };

  const screenshot = async (path) => {
    const shot = await send("Page.captureScreenshot", { format: "png" });
    writeFileSync(path, Buffer.from(shot.data, "base64"));
    return path;
  };

  const setViewport = (w, h) =>
    send("Emulation.setDeviceMetricsOverride", {
      width: w,
      height: h,
      deviceScaleFactor: 1,
      mobile: false,
    });

  const typeInto = async (selector, text) => {
    const focused = await evaluate(`(() => {
      const el = document.querySelector(${JSON.stringify(selector)});
      if (!el) return false;
      el.focus();
      el.select?.();
      return document.activeElement === el;
    })()`);
    if (!focused) return false;
    await send("Input.insertText", { text });
    return true;
  };

  const close = async () => {
    try {
      ws.close();
    } catch {
      /* already closed */
    }
    if (process.platform === "win32") {
      spawn("taskkill", ["/PID", String(chrome.pid), "/T", "/F"], { stdio: "ignore" });
    } else {
      chrome.kill();
    }
    await sleep(300);
  };

  return {
    send,
    evaluate,
    bodyText,
    waitFor,
    modelReady,
    waitForModel,
    clickByText,
    canvasBox,
    drawStroke,
    screenshot,
    setViewport,
    typeInto,
    close,
  };
}

/** A tiny pass/fail reporter shared by the QA scripts. */
export function reporter() {
  const failures = [];
  return {
    check(label, ok, detail = "") {
      console.log(`${ok ? "PASS" : "FAIL"}  ${label}${detail ? `  (${detail})` : ""}`);
      if (!ok) failures.push(label);
      return ok;
    },
    get failures() {
      return failures;
    },
    finish() {
      if (failures.length) {
        console.error(`\n${failures.length} check(s) failed:`);
        for (const f of failures) console.error(`  - ${f}`);
        process.exit(1);
      }
      console.log("\nall checks passed");
    },
  };
}
