/**
 * Whole-site smoke test: loads the real page in headless Chrome, plays all three modes with
 * real input, and fails on console errors, 4xx/5xx requests, or wrong model output.
 *
 *   node tools/qa.mjs --url http://127.0.0.1:5180
 *
 * It is deliberately not a unit test: the point is to catch the things a unit test cannot —
 * a model that loads but predicts nothing, a tab that throws on mount, a layout that scrolls
 * sideways on a phone.
 */

import { openBrowser, parseArgs, reporter, sleep } from "./cdp.mjs";

const args = parseArgs(process.argv.slice(2));
const url = args.url ?? "http://127.0.0.1:5180";
const shot = args.out ?? "docs/shots/mobile.png";

const { check, finish } = reporter();

const consoleErrors = [];
const badRequests = [];

function watch(msg) {
  if (msg.method === "Runtime.consoleAPICalled" && ["error", "warning"].includes(msg.params.type)) {
    const text = msg.params.args
      .map((a) => a.value ?? a.description ?? "")
      .join(" ")
      .trim();
    if (text) consoleErrors.push(`console.${msg.params.type}: ${text}`);
  }
  if (msg.method === "Log.entryAdded" && msg.params.entry.level === "error") {
    const entry = msg.params.entry;
    consoleErrors.push(`log: ${entry.text}${entry.url ? ` (${entry.url})` : ""}`);
  }
  if (msg.method === "Network.responseReceived") {
    const r = msg.params.response;
    if (r.status >= 400) badRequests.push(`${r.status} ${r.url}`);
  }
  if (msg.method === "Network.loadingFailed" && !msg.params.canceled) {
    badRequests.push(`failed ${msg.params.errorText}`);
  }
}

const HOUSE = [
  [[0.32, 0.6], [0.68, 0.6]],
  [[0.68, 0.6], [0.68, 0.86]],
  [[0.68, 0.86], [0.32, 0.86]],
  [[0.32, 0.86], [0.32, 0.6]],
  [[0.26, 0.6], [0.5, 0.34], [0.74, 0.6]],
];

/** Interpolate so a two-point line becomes a smooth drag like a real finger. */
function interleave(stroke, per = 10) {
  const pts = [];
  for (let i = 1; i < stroke.length; i++) {
    const [x1, y1] = stroke[i - 1];
    const [x2, y2] = stroke[i];
    for (let k = 0; k < per; k++) {
      pts.push([x1 + ((x2 - x1) * k) / (per - 1), y1 + ((y2 - y1) * k) / (per - 1)]);
    }
  }
  return pts.length ? pts : stroke;
}

async function main() {
  const b = await openBrowser({ url, port: 9335, onEvent: watch });

  try {
    // ---- landing + model ---------------------------------------------------------------
    const loaded = await b.waitForModel(40000);
    check("model loads", loaded);

    const text = await b.bodyText();
    check("header shows the class count", /\b100 classes\b/.test(text));
    check("header shows a measured accuracy", /\d+\.\d+% top-1/.test(text));
    check("the model reports in-browser inference", text.includes("in-browser"));

    const listCount = await b.evaluate(`(() => {
      const d = document.querySelector("details");
      if (!d) return -1;
      d.open = true;
      return d.querySelectorAll("span").length;
    })()`);
    check("the class list holds every class", listCount === 100, String(listCount));
    await b.evaluate(`document.querySelector("details").open = false`);

    // ---- free draw ---------------------------------------------------------------------
    const box = await b.canvasBox();
    check("free-draw canvas is present", !!box);
    if (!box) throw new Error("no canvas");

    for (const stroke of HOUSE) await b.drawStroke(box, interleave(stroke), 6);
    await sleep(900);

    const predText = await b.bodyText();
    const m = predText.match(/MODEL'S GUESS\s*\n\s*([\d.]+) ms\s*\n\s*([^\n]+)\s*\n\s*([<\d.]+%)/);
    check("a prediction is shown", !!m, m ? `${m[2]} ${m[3]}` : "no match");
    if (m) {
      const ms = Number(m[1]);
      check("top-1 is the thing that was drawn", m[2].trim().toLowerCase() === "house", m[2]);
      check("top-1 confidence is high", parseFloat(m[3]) > 50, m[3]);
      check("inference is fast", ms > 0 && ms < 100, `${ms} ms`);
    }

    const altRows = (predText.match(/MODEL'S GUESS[\s\S]*?Softmax over the trained classes/) ?? [""])[0]
      .split("\n")
      .filter((l) => /^[<\d.]+%$/.test(l.trim())).length;
    check("top-5 is rendered", altRows === 5, `${altRows} rows`);

    // ---- speed round -------------------------------------------------------------------
    check('switched to "Speed round"', await b.clickByText("Speed round"));
    await sleep(400);
    check("started the speed round", await b.clickByText("start"));
    await sleep(500);

    const speedText = await b.bodyText();
    const prompt = speedText.match(/DRAW A\s*\n\s*([^\n]+)/i);
    check("speed round shows a prompt", !!prompt, prompt ? prompt[1] : "none");
    const t1 = speedText.match(/([\d.]+)s left/);
    await sleep(1200);
    const t2 = (await b.bodyText()).match(/([\d.]+)s left/);
    check("the clock runs", !!t1 && !!t2 && Number(t2[1]) < Number(t1[1]), `${t1?.[1]} -> ${t2?.[1]}`);

    // ---- AI draws ----------------------------------------------------------------------
    check('switched to "AI draws"', await b.clickByText("AI draws"));
    await sleep(700);
    check(
      "AI-draw card explains the mode",
      (await b.bodyText()).includes("AI draws, you guess"),
    );

    check("picked the easy tier", await b.clickByText("easy"));
    await sleep(3600); // let the stroke animation play
    const playing = await b.bodyText();
    check("a sketch is drawn for the player", playing.includes("GUESS THE DRAWING"));
    check("a guess input is offered", playing.includes("type your answer"));

    const typed = await b.typeInto('input[placeholder*="bicycle"]', "house");
    check("typing into the guess box works", typed);
    const fieldValue = await b.evaluate(
      `document.querySelector('input[placeholder*="bicycle"]')?.value ?? ""`,
    );
    check("the guess box holds the text", fieldValue === "house", fieldValue);

    check("submitted a guess", await b.clickByText("guess"));

    const wrongFeedback = await b.waitFor(
      async () => (await b.bodyText()).includes("keep going"),
      { timeout: 3000, interval: 100 },
    );
    check("a wrong guess says so instead of clearing silently", wrongFeedback);
    const cleared = await b.evaluate(
      `document.querySelector('input[placeholder*="bicycle"]')?.value ?? ""`,
    );
    check("the input clears for a retry", cleared === "", cleared);

    const revealed = await b.waitFor(async () => (await b.bodyText()).includes("it was"), {
      timeout: 26000,
      interval: 200,
    });
    check("the round reveals the answer when the clock runs out", revealed);

    // ---- mobile layout -----------------------------------------------------------------
    await b.setViewport(390, 844);
    await b.send("Page.navigate", { url });
    const mobileReady = await b.waitForModel(40000);
    check("model loads on a phone-sized viewport", mobileReady);
    await sleep(400);

    const overflow = await b.evaluate(`(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      innerWidth: window.innerWidth,
    }))()`);
    check(
      "no horizontal scroll on a 390 px phone",
      overflow.scrollWidth <= overflow.innerWidth + 1,
      `${overflow.scrollWidth} vs ${overflow.innerWidth}`,
    );

    const tabsVisible = await b.evaluate(`(() => {
      const labels = ["Free draw", "Speed round", "AI draws"];
      return labels.every((l) =>
        [...document.querySelectorAll("button")].some((b) => b.innerText.trim() === l),
      );
    })()`);
    check("all three modes reachable on mobile", tabsVisible);

    // the prediction panel is below the fold on a phone, so a compact strip must exist
    const mobileBox = await b.canvasBox();
    if (mobileBox) {
      for (const stroke of HOUSE) await b.drawStroke(mobileBox, interleave(stroke), 5);
      await sleep(900);
    }
    const strip = await b.evaluate(`(() => {
      const el = document.querySelector('[data-testid="mobile-prediction"]');
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return { text: el.innerText, visible: r.width > 0 && r.height > 0 };
    })()`);
    check(
      "phone shows the guess next to the canvas",
      !!strip && strip.visible,
      strip?.text?.replace(/\n/g, " | "),
    );
    check(
      "the phone strip names the drawing",
      !!strip && strip.text.toLowerCase().includes("house"),
      strip?.text?.replace(/\n/g, " | "),
    );

    await b.screenshot(shot);
    console.log(`[qa] wrote ${shot}`);

    // ---- console / network -------------------------------------------------------------
    const noisy = [...new Set(consoleErrors)];
    check("no console errors or warnings", noisy.length === 0, noisy.join(" | ").slice(0, 220));
    const bad = [...new Set(badRequests)];
    check("no failed requests", bad.length === 0, bad.join(" | ").slice(0, 220));
  } finally {
    await b.close();
  }

  finish();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
