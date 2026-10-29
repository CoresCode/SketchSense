import { useCallback, useEffect, useRef, useState } from "react";
import confetti from "canvas-confetti";
import type { Prediction } from "../lib/inference";
import type { Stroke } from "../lib/raster";

export type RoundResult = { target: string; hit: boolean; seconds: number };

type Props = {
  classes: string[];
  preds: Prediction[];
  strokes: Stroke[];
  onClear: () => void;
  onExit: () => void;
};

const ROUNDS = 5;
const SECONDS = 20;
const HIT_THRESHOLD = 0.35;
// A confident prediction is not the same as a finished drawing: a circle and two dots is
// already "smiley face" at 97%, and scoring that instantly stops the player before they can
// draw the mouth. So a guess only counts once the ink has been still for a moment.
const SETTLE_MS = 1200;
const MIN_STROKES = 2;

type Phase = "idle" | "playing" | "feedback" | "done";

function pickTarget(classes: string[], avoid: string | null): string {
  if (classes.length === 0) return "cat";
  for (let i = 0; i < 12; i++) {
    const c = classes[Math.floor(Math.random() * classes.length)];
    if (c !== avoid) return c;
  }
  return classes[0];
}

export default function SpeedRound({ classes, preds, strokes, onClear, onExit }: Props) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [round, setRound] = useState(1);
  const [target, setTarget] = useState<string>("cat");
  const [timeLeft, setTimeLeft] = useState(SECONDS);
  const [score, setScore] = useState(0);
  const [results, setResults] = useState<RoundResult[]>([]);
  const [lastHit, setLastHit] = useState<boolean | null>(null);
  const [copied, setCopied] = useState(false);
  const [confirming, setConfirming] = useState(false);

  const resolvedRef = useRef(false);
  const advanceTimer = useRef<number | null>(null);
  const settleTimer = useRef<number | null>(null);
  const targetRef = useRef(target);
  targetRef.current = target;
  const predsRef = useRef(preds);
  predsRef.current = preds;

  const cancelSettle = useCallback(() => {
    if (settleTimer.current) {
      window.clearTimeout(settleTimer.current);
      settleTimer.current = null;
    }
  }, []);

  const startRound = useCallback(
    (index: number, avoid: string | null) => {
      resolvedRef.current = false;
      cancelSettle();
      setConfirming(false);
      setLastHit(null);
      setRound(index);
      setTimeLeft(SECONDS);
      setTarget(pickTarget(classes, avoid));
      setPhase("playing");
      onClear();
    },
    [classes, cancelSettle, onClear],
  );

  const resolve = useCallback(
    (hit: boolean, remaining: number) => {
      if (resolvedRef.current) return;
      resolvedRef.current = true;
      cancelSettle();
      setConfirming(false);
      setPhase("feedback");
      setLastHit(hit);
      setResults((prev) => [
        ...prev,
        { target: targetRef.current, hit, seconds: Math.round(SECONDS - remaining) },
      ]);
      if (hit) {
        setScore((s) => s + 100 + Math.round(remaining) * 5);
        confetti({
          particleCount: 90,
          spread: 70,
          origin: { y: 0.62 },
          colors: ["#8b5cf6", "#22d3ee", "#f5f3ff", "#fbbf24"],
          disableForReducedMotion: true,
        });
      }
    },
    [],
  );

  const timeLeftRef = useRef(timeLeft);
  timeLeftRef.current = timeLeft;

  // tick
  useEffect(() => {
    if (phase !== "playing") return;
    const started = performance.now();
    const id = window.setInterval(() => {
      const elapsed = (performance.now() - started) / 1000;
      const left = Math.max(0, SECONDS - elapsed);
      setTimeLeft(left);
      if (left <= 0) {
        window.clearInterval(id);
        resolve(false, 0);
      }
    }, 100);
    return () => window.clearInterval(id);
  }, [phase, round, resolve]);

  // hit detection: the guess has to survive the player putting the pen down, not just a
  // single frame. Otherwise the round ends the moment a partial drawing is recognisable.
  useEffect(() => {
    if (phase !== "playing") return;
    const top = preds[0];
    const strokeCount = strokes.filter((s) => s.length > 0).length;
    const matches =
      !!top && top.label === target && top.prob >= HIT_THRESHOLD && strokeCount >= MIN_STROKES;

    cancelSettle();
    if (!matches) {
      setConfirming(false);
      return;
    }

    setConfirming(true);
    settleTimer.current = window.setTimeout(() => {
      settleTimer.current = null;
      setConfirming(false);
      // re-read the latest prediction: the guard must still be true when it fires
      const now = predsRef.current[0];
      if (now && now.label === targetRef.current && now.prob >= HIT_THRESHOLD) {
        resolve(true, timeLeftRef.current);
      }
    }, SETTLE_MS);

    return cancelSettle;
  }, [preds, strokes, phase, target, resolve, cancelSettle]);

  // advance after feedback
  useEffect(() => {
    if (phase !== "feedback") return;
    advanceTimer.current = window.setTimeout(() => {
      if (round >= ROUNDS) {
        setPhase("done");
      } else {
        startRound(round + 1, target);
      }
    }, 1600);
    return () => {
      if (advanceTimer.current) window.clearTimeout(advanceTimer.current);
    };
  }, [phase, round, startRound, target]);

  const playAgain = () => {
    setScore(0);
    setResults([]);
    setCopied(false);
    startRound(1, null);
  };

  const shareText = () => {
    const hits = results.filter((r) => r.hit).length;
    return `SketchSense speed round: ${hits}/${results.length} guessed, ${score} points 🎨 (drew against a browser CNN)`;
  };

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(shareText());
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  if (phase === "idle") {
    return (
      <div className="glass rounded-2xl p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="font-display text-lg font-semibold text-white">Speed round</h3>
            <p className="mt-1 text-sm text-white/50">
              {ROUNDS} prompts, {SECONDS}s each. Draw the thing — the model only calls it once you
              stop, so a half-finished doodle gets a moment to become a doodle.
            </p>
          </div>
          <button
            onClick={() => startRound(1, null)}
            className="shrink-0 rounded-xl bg-gradient-to-r from-brand to-cyan-glow px-4 py-2 text-sm font-semibold text-ink-950 transition hover:brightness-110"
          >
            Start
          </button>
        </div>
        <div className="mt-4 grid grid-cols-3 gap-2 text-center">
          {[
            ["Rounds", String(ROUNDS)],
            ["Per round", `${SECONDS}s`],
            ["Bonus", "faster = more"],
          ].map(([k, v]) => (
            <div key={k} className="rounded-xl border border-white/8 bg-white/3 px-2 py-2.5">
              <div className="font-mono text-sm text-white/80">{v}</div>
              <div className="mt-0.5 text-[10px] tracking-wide text-white/35 uppercase">{k}</div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="relative">
      <div className="glass overflow-hidden rounded-2xl">
        <div className="flex items-center justify-between gap-3 px-4 py-3">
          <div className="flex items-center gap-3">
            <span className="rounded-lg bg-white/8 px-2 py-1 font-mono text-xs text-white/50">
              {round}/{ROUNDS}
            </span>
            <div>
              <p className="text-[10px] tracking-widest text-white/35 uppercase">Draw a</p>
              <p className="font-display text-xl leading-tight font-bold text-white capitalize">
                {target}
              </p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-[10px] tracking-widest text-white/35 uppercase">Score</p>
            <p className="font-mono text-xl font-semibold text-white">{score}</p>
          </div>
        </div>

        <div className="h-1 w-full bg-white/8">
          <div
            className="h-full origin-left bg-gradient-to-r from-brand to-cyan-glow transition-[width] duration-100 ease-linear"
            style={{ width: `${(timeLeft / SECONDS) * 100}%` }}
          />
        </div>

        <div className="flex items-center justify-between px-4 py-2 text-xs">
          <span className="font-mono text-white/50">{timeLeft.toFixed(1)}s left</span>
          <span
            className={
              lastHit === true
                ? "font-medium text-emerald-400"
                : lastHit === false
                  ? "font-medium text-rose-400"
                  : confirming
                    ? "font-medium text-cyan-300"
                    : "text-white/30"
            }
          >
            {lastHit === true
              ? "got it!"
              : lastHit === false
                ? "missed"
                : confirming
                  ? "hold still — confirming…"
                  : "waiting for a guess…"}
          </span>
        </div>
      </div>

      {phase === "done" && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink-950/80 p-4 backdrop-blur-sm">
          <div className="glass animate-rise w-full max-w-sm rounded-2xl p-5 text-center">
            <p className="text-[10px] tracking-widest text-white/40 uppercase">Final score</p>
            <p className="font-display my-1 text-5xl font-bold text-gradient">{score}</p>
            <p className="text-sm text-white/55">
              {results.filter((r) => r.hit).length} of {results.length} recognised
            </p>

            <div className="mt-4 flex flex-wrap justify-center gap-1.5">
              {results.map((r, i) => (
                <span
                  key={i}
                  className={`rounded-md px-2 py-1 text-[11px] capitalize ${
                    r.hit ? "bg-emerald-500/15 text-emerald-300" : "bg-rose-500/15 text-rose-300"
                  }`}
                  title={r.hit ? `guessed in ${r.seconds}s` : "not guessed"}
                >
                  {r.target}
                </span>
              ))}
            </div>

            <div className="mt-5 flex gap-2">
              <button
                onClick={copy}
                className="flex-1 rounded-xl border border-white/12 bg-white/5 px-3 py-2 text-sm font-medium text-white/80 transition hover:bg-white/10"
              >
                {copied ? "Copied!" : "Copy result"}
              </button>
              <button
                onClick={playAgain}
                className="flex-1 rounded-xl bg-gradient-to-r from-brand to-cyan-glow px-3 py-2 text-sm font-semibold text-ink-950 transition hover:brightness-110"
              >
                Play again
              </button>
            </div>
            <button
              onClick={onExit}
              className="mt-3 text-xs text-white/35 underline-offset-4 hover:text-white/60 hover:underline"
            >
              back to free draw
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
