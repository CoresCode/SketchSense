/**
 * Regression test for the speed round's "don't guess while I'm still drawing" rule.
 *
 * The bug: drawing a circle and two dots for "smiley face" scored a hit immediately, before
 * the mouth existed, so the player never got to finish. A guess now only counts once the ink
 * has been still for a moment, and the UI says so while it waits.
 *
 * This drives a real headless Chrome with real input events, forces the prompt to be
 * "smiley face" (Math.random is pinned before the app boots), draws the face the way a person
 * would, and asserts:
 *
 *   - a half-drawn face does not score
 *   - the pen lifting does not score on the same frame
 *   - the score does land, but only after the settle window
 *
 *   node tools/test_speed_round.mjs --url http://127.0.0.1:5180
 */

import { openBrowser, parseArgs, reporter, sleep } from "./cdp.mjs";

const args = parseArgs(process.argv.slice(2));
const url = args.url ?? "http://127.0.0.1:5180";

// labels.json is alphabetical; "smiley face" is index 79. Pinning Math.random to the middle
// of that bucket makes pickTarget() choose it on the first attempt.
const RANDOM = (79 + 0.5) / 100;

const { check, finish } = reporter();

async function main() {
  const b = await openBrowser({
    url,
    port: Number(args.port ?? 9334),
    initScript: `Math.random = () => ${RANDOM};`,
  });

  try {
    const status = () =>
      b.evaluate(`(() => {
        const t = document.body.innerText;
        if (t.includes("got it!")) return "hit";
        if (t.includes("confirming")) return "confirming";
        if (t.includes("waiting for a guess")) return "waiting";
        if (t.includes("missed")) return "missed";
        return "unknown";
      })()`);

    const loaded = await b.waitFor(async () => (await b.bodyText()).includes("top-1"), {
      timeout: 30000,
    });
    check("model loads in the browser", loaded);

    check('opened the "Speed round" tab', await b.clickByText("Speed round"));
    await sleep(500);
    check("started a round", await b.clickByText("start"));
    await sleep(500);

    check(
      'prompt is "smiley face"',
      (await b.bodyText()).toLowerCase().includes("smiley face"),
    );

    const box = await b.canvasBox();
    check("found the drawing canvas", !!box);
    if (!box) throw new Error("cannot continue without the canvas");

    // ---- still drawing: circle, then two eyes, with human-sized gaps -----------------
    const circle = Array.from({ length: 36 }, (_, i) => {
      const t = (i / 35) * Math.PI * 2;
      return [0.5 + 0.2 * Math.cos(t), 0.46 + 0.19 * Math.sin(t)];
    });
    const mouth = Array.from({ length: 12 }, (_, i) => {
      const t = (i / 11) * Math.PI;
      return [0.5 - 0.13 * Math.cos(t), 0.5 + 0.11 * Math.sin(t)];
    });

    await b.drawStroke(box, circle);
    await sleep(200);
    await b.drawStroke(box, [[0.42, 0.37], [0.42, 0.37]], 0); // left eye
    await sleep(220);
    await b.drawStroke(box, [[0.58, 0.37], [0.58, 0.37]], 0); // right eye

    const midStatus = await status();
    check("no score while the face is still missing its mouth", midStatus !== "hit", midStatus);
    check(
      "status explains that it is waiting",
      midStatus === "waiting" || midStatus === "confirming",
      midStatus,
    );

    // ---- finish the drawing and measure when the point is awarded --------------------
    await b.drawStroke(box, mouth);
    const drawnAt = Date.now();

    const afterInk = await status();
    check("does not score on the frame the pen lifts", afterInk !== "hit", afterInk);

    let hitAt = null;
    let sawConfirming = false;
    for (let i = 0; i < 50; i++) {
      await sleep(100);
      const s = await status();
      if (s === "confirming") sawConfirming = true;
      if (s === "hit") {
        hitAt = Date.now() - drawnAt;
        break;
      }
    }

    check("scores once the drawing is finished and still", hitAt !== null, `${hitAt} ms`);
    check("waited for the settle window", hitAt !== null && hitAt >= 1000, `${hitAt} ms`);
    check("showed the confirming state while waiting", sawConfirming);

    const score = await b.evaluate(`(() => {
      const text = document.body.innerText;
      const m = text.match(/SCORE\\s*\\n?\\s*(\\d+)/i);
      return m ? Number(m[1]) : null;
    })()`);
    check("awarded points", typeof score === "number" && score > 0, String(score));
  } finally {
    await b.close();
  }

  finish();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
