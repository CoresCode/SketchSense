/**
 * Slow, end-to-end game test: plays a whole speed round and a whole AI-draw game by simply
 * letting every clock expire, then checks the final screens and the "play again" path.
 *
 * Everything else in tools/ avoids this because it takes ~4 minutes. This is the one test
 * that proves the last round actually ends, the score is honest, and replay resets state.
 *
 *   node tools/test_full_rounds.mjs --url http://127.0.0.1:5180
 */

import { openBrowser, parseArgs, reporter, sleep } from "./cdp.mjs";

const args = parseArgs(process.argv.slice(2));
const url = args.url ?? "http://127.0.0.1:5180";

const { check, finish } = reporter();

const FINAL = /final score/i;

async function playToEnd(b, { label, roundMs, rounds, doneText }) {
  const started = Date.now();
  let sawFinal = false;
  const budget = rounds * (roundMs + 4000) + 8000;

  while (Date.now() - started < budget) {
    if (FINAL.test(await b.bodyText())) {
      sawFinal = true;
      break;
    }
    await sleep(500);
  }
  const elapsed = Math.round((Date.now() - started) / 1000);
  check(`${label}: game reaches the final screen`, sawFinal, `${elapsed}s`);
  if (!sawFinal) {
    const tail = (await b.bodyText()).replace(/\n+/g, " | ").slice(0, 240);
    console.log(`  last state: ${tail}`);
    return false;
  }

  const text = await b.bodyText();
  check(`${label}: summary counts the rounds`, new RegExp(doneText, "i").test(text), doneText);
  return true;
}

async function main() {
  const b = await openBrowser({ url, port: 9336 });

  try {
    await b.waitForModel(40000);

    // ---- speed round: five timeouts ---------------------------------------------------
    check('switched to "Speed round"', await b.clickByText("Speed round"));
    await sleep(400);
    check("started the speed round", await b.clickByText("start"));

    const speedDone = await playToEnd(b, {
      label: "speed round",
      roundMs: 20000,
      rounds: 5,
      doneText: "0 of 5 recognised",
    });

    if (speedDone) {
      check("replay button is offered", await b.clickByText("play again"));
      await sleep(700);
      const replay = await b.bodyText();
      check("replay restarts at round 1", replay.includes("1/5"), replay.replace(/\n+/g, " | ").slice(0, 60));
      check("replay clears the final screen", !FINAL.test(replay));
      check("replay resets the clock", /\d+\.\ds left/.test(replay));
    }

    // ---- AI draws: five timeouts -------------------------------------------------------
    check('switched to "AI draws"', await b.clickByText("AI draws"));
    await sleep(700);
    check("picked the easy tier", await b.clickByText("easy"));

    const aiDone = await playToEnd(b, {
      label: "AI draws",
      roundMs: 22000,
      rounds: 5,
      doneText: "0 of 5 read correctly",
    });

    if (aiDone) {
      check("AI-draw replay is offered", await b.clickByText("play again"));
      await sleep(700);
      const replay = await b.bodyText();
      check("AI-draw replay restarts at round 1", replay.includes("1/5"), replay.replace(/\n+/g, " | ").slice(0, 60));
      check("AI-draw replay clears the final screen", !FINAL.test(replay));
    }
  } finally {
    await b.close();
  }

  finish();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
