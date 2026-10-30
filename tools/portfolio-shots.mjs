/**
 * Portfolio screenshots: three real frames of the app, driven with real mouse and keyboard
 * input over CDP - nothing is mocked, stitched or retouched.
 *
 *   1. free-draw.png   a house drawn by hand, the live top-5 naming it
 *   2. speed-round.png a finished smiley face scoring before the clock runs out
 *   3. ai-draws.png    the network's sketch, the correct guess, and what the model thought
 *
 * Each frame gets its own browser profile. Sharing one page would carry the speed round's
 * confetti into the AI-draw shot, which is real but reads as a rendering bug.
 *
 * The AI-draw frame needs the target class in advance so the script can type the right
 * answer, so `Math.random` is pinned to 0.365: class order is sketch-samples.json order, and
 * floor(0.365 * 100) is index 36, which is "fish". The speed round pins it to "smiley face"
 * the same way the regression test does.
 *
 *   node tools/portfolio-shots.mjs http://127.0.0.1:4173 docs/portfolio 1440 1210 2
 */

import { mkdirSync, readFileSync } from "node:fs";
import { openBrowser, sleep } from "./cdp.mjs";

const url = process.argv[2] ?? "http://127.0.0.1:4173";
const outDir = process.argv[3] ?? "docs/portfolio";
const W = Number(process.argv[4] ?? 1440);
const H = Number(process.argv[5] ?? 1210);
const SCALE = Number(process.argv[6] ?? 2);
const ONLY = process.argv[7] ?? "all"; // free-draw | speed-round | ai-draws | all

mkdirSync(outDir, { recursive: true });

// ---- what to draw --------------------------------------------------------------
// Free draw uses "house" by default: a shape everyone recognises, so the 95.5% next to it
// needs no explanation. SCENE=fish|sun swaps it.
const SCENES = {
  house: [
    [[0.32, 0.6], [0.68, 0.6]],
    [[0.68, 0.6], [0.68, 0.86]],
    [[0.68, 0.86], [0.32, 0.86]],
    [[0.32, 0.86], [0.32, 0.6]],
    [[0.26, 0.6], [0.5, 0.34], [0.74, 0.6]],
  ],
  fish: [
    [[0.28, 0.5], [0.36, 0.4], [0.5, 0.36], [0.64, 0.42], [0.68, 0.5], [0.62, 0.58], [0.5, 0.63], [0.36, 0.59], [0.28, 0.5]],
    [[0.28, 0.5], [0.14, 0.38], [0.16, 0.5], [0.14, 0.62], [0.28, 0.5]],
    [[0.56, 0.46], [0.56, 0.46]],
  ],
  sun: [
    Array.from({ length: 34 }, (_, i) => {
      const t = (i / 33) * Math.PI * 2;
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
};
const scene = SCENES[process.env.SCENE ?? "house"] ?? SCENES.house;

const circle = Array.from({ length: 40 }, (_, i) => {
  const t = (i / 39) * Math.PI * 2;
  return [0.5 + 0.2 * Math.cos(t), 0.46 + 0.19 * Math.sin(t)];
});
const mouth = Array.from({ length: 14 }, (_, i) => {
  const t = (i / 13) * Math.PI;
  return [0.5 - 0.13 * Math.cos(t), 0.5 + 0.11 * Math.sin(t)];
});

function densify(stroke, per = 8) {
  if (stroke.length === 1) return stroke;
  const pts = [];
  for (let i = 1; i < stroke.length; i++) {
    const [x1, y1] = stroke[i - 1];
    const [x2, y2] = stroke[i];
    for (let k = 0; k < per; k++) {
      pts.push([x1 + ((x2 - x1) * k) / (per - 1), y1 + ((y2 - y1) * k) / (per - 1)]);
    }
  }
  return pts;
}

// The speed round picks its prompt from game-classes.json, so the value to pin is just
// "whatever lands in the smiley-face bucket" - computed, not guessed.
const pool = JSON.parse(
  readFileSync(new URL("../web/public/model/game-classes.json", import.meta.url), "utf8"),
).classes;
const SMILEY_R = (pool.indexOf("smiley face") + 0.5) / pool.length;

/** Fresh browser, fixed 2x viewport, model warm, then hand over to `body`. */
async function shot(name, body) {
  const b = await openBrowser({ url, width: W, height: H, port: 9336 });
  try {
    // Retina output: the CSS layout stays WxH, the PNG comes out at 2x.
    await b.send("Emulation.setDeviceMetricsOverride", {
      width: W,
      height: H,
      deviceScaleFactor: SCALE,
      mobile: false,
    });
    if (!(await b.waitForModel(60000))) throw new Error(`${name}: model never loaded`);
    await body(b);
    await b.screenshot(`${outDir}/${name}.png`);
    console.log(`[shots] ${name}.png`);
  } finally {
    await b.close();
  }
}

// ---- 1. free draw ---------------------------------------------------------------
if (ONLY === "all" || ONLY === "free-draw") await shot("free-draw", async (b) => {
  const box = await b.canvasBox();
  for (const stroke of scene) await b.drawStroke(box, densify(stroke), 6);
  await b.waitFor(async () => /\d\d\.\d%/.test(await b.bodyText()), { timeout: 15000 });
  await sleep(900); // let the confidence bars finish their transition
  const top5 = await b.evaluate(
    `(() => document.body.innerText.split("\\n").filter((l) => /%$/.test(l.trim())).slice(0, 5).join(" | "))()`,
  );
  console.log("[shots] top-5:", top5);
});

// ---- 2. speed round -------------------------------------------------------------
if (ONLY === "all" || ONLY === "speed-round") await shot("speed-round", async (b) => {
  await b.evaluate(`Math.random = () => ${SMILEY_R};`);
  if (!(await b.clickByText("Speed round"))) throw new Error('no "Speed round" tab');
  await sleep(800);
  if (!(await b.clickByText("start"))) throw new Error('no "Start" button');
  await sleep(600);
  const prompt = (await b.bodyText()).toLowerCase();
  if (!prompt.includes("smiley face")) throw new Error(`prompt is not smiley face: ${prompt.slice(0, 120)}`);

  const box = await b.canvasBox();
  await b.drawStroke(box, circle, 8);
  await sleep(180);
  await b.drawStroke(box, [[0.42, 0.37], [0.42, 0.37]], 0); // left eye
  await sleep(180);
  await b.drawStroke(box, [[0.58, 0.37], [0.58, 0.37]], 0); // right eye
  await sleep(220);
  await b.drawStroke(box, mouth, 8);

  const scored = await b.waitFor(
    async () => (await b.bodyText()).toLowerCase().includes("got it!"),
    { timeout: 12000 },
  );
  if (!scored) throw new Error("speed round never scored");
  await sleep(450); // let the points land and the progress bar settle
});

// ---- 3. AI draws ----------------------------------------------------------------
if (ONLY === "all" || ONLY === "ai-draws") await shot("ai-draws", async (b) => {
  // 0.365 picks "fish" out of the 100-class sample pool at index 36.
  await b.evaluate(`Math.random = () => 0.365;`);
  if (!(await b.clickByText("AI draws"))) throw new Error('no "AI draws" tab');
  await sleep(1200); // sketch-samples.json + the idle card
  if (!(await b.clickByText("easy"))) throw new Error('no "easy" button');

  const playing = await b.waitFor(
    async () => (await b.bodyText()).toLowerCase().includes("what is it?"),
    { timeout: 25000 },
  );
  if (!playing) throw new Error("the replay never finished");
  await sleep(300);

  await b.typeInto('input[aria-label="Your guess"]', "fish");
  await sleep(400);
  const submitted = await b.evaluate(`(() => {
    const btn = document.querySelector('button[type="submit"]');
    if (!btn) return false;
    btn.click();
    return true;
  })()`);
  if (!submitted) throw new Error("could not submit the guess");

  const revealed = await b.waitFor(
    async () => (await b.bodyText()).toLowerCase().includes("yes —"),
    { timeout: 8000 },
  );
  if (!revealed) throw new Error("the correct guess was not confirmed");
  const detail = await b.evaluate(
    `(() => { const t = document.body.innerText; const i = t.indexOf("yes —"); return t.slice(Math.max(0, i - 40), i + 90).replace(/\\n/g, " | "); })()`,
  );
  console.log("[shots] aidraw says:", detail);
  // The confetti burst is real, but 1.2 s in it has fallen clear of the answer text while
  // the celebration is still on screen.
  await sleep(1200);
});
