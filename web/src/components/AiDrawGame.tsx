import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import confetti from "canvas-confetti";
import { rasterize, type Stroke } from "../lib/raster";
import type { SketchModel } from "../lib/inference";

type FlatStroke = number[]; // [x0, y0, x1, y1, …]
type Sample = FlatStroke[]; // one drawing, a few polylines
type SampleClass = { slug: string; name: string; easy: Sample[]; hard: Sample[] };
type SampleFile = { canvas: number; classes: SampleClass[] };
type Difficulty = "easy" | "hard";

type Props = {
  model: SketchModel | null;
  onExit: () => void;
};

const ROUNDS = 5;
const SECONDS = 22;
const LOGICAL = 400;
const SRC = 256; // the exporter writes normalised coordinates in 0..255
const PAPER = "#fffaf0";
const INK = "#12161c";

const STYLE: Record<Difficulty, { width: number; opacity: number; msPerPoint: number }> = {
  easy: { width: 11, opacity: 1, msPerPoint: 20 },
  hard: { width: 8, opacity: 0.62, msPerPoint: 38 },
};

type Phase = "idle" | "playing" | "feedback" | "done";

type Result = { name: string; hit: boolean; seconds: number; modelGuess: string | null; modelProb: number | null };

function toStrokes(flat: Sample): Stroke[] {
  return flat.map((s) => {
    const stroke: Stroke = [];
    for (let i = 0; i + 1 < s.length; i += 2) stroke.push({ x: s[i], y: s[i + 1] });
    return stroke;
  });
}

/** Hard mode wobbles the line so it reads like a hurried human hand, not a plotter. */
function humanise(strokes: Stroke[], amount: number): Stroke[] {
  if (amount <= 0) return strokes;
  return strokes.map((stroke) =>
    stroke.map((p, i) => {
      if (i === 0) return p;
      const w = Math.sin(i * 2.7 + stroke.length) * amount;
      return { x: p.x + w, y: p.y - w * 0.7 };
    }),
  );
}

const normalise = (s: string) =>
  s
    .toLowerCase()
    .replace(/[^a-z ]/g, "")
    .replace(/\s+/g, " ")
    .trim();

function editDistance(a: string, b: string): number {
  if (a === b) return 0;
  const prev = new Array(b.length + 1).fill(0).map((_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    let diag = prev[0];
    prev[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const tmp = prev[j];
      prev[j] = Math.min(prev[j] + 1, prev[j - 1] + 1, diag + (a[i - 1] === b[j - 1] ? 0 : 1));
      diag = tmp;
    }
  }
  return prev[b.length];
}

function matches(guess: string, target: string): boolean {
  const g = normalise(guess);
  const t = normalise(target);
  if (!g) return false;
  if (g === t) return true;
  // one typo is forgiven on longer words, so "flamengo" still counts
  if (t.length >= 5 && editDistance(g, t) <= 1) return true;
  return false;
}

export default function AiDrawGame({ model, onExit }: Props) {
  const [samples, setSamples] = useState<SampleFile | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [difficulty, setDifficulty] = useState<Difficulty>("easy");
  const [phase, setPhase] = useState<Phase>("idle");
  const [round, setRound] = useState(1);
  const [target, setTarget] = useState("");
  const [timeLeft, setTimeLeft] = useState(SECONDS);
  const [drawing, setDrawing] = useState(false);
  const [score, setScore] = useState(0);
  const [guess, setGuess] = useState("");
  const [results, setResults] = useState<Result[]>([]);
  const [last, setLast] = useState<Result | null>(null);
  const [copied, setCopied] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const strokesRef = useRef<Stroke[]>([]);
  const rafRef = useRef(0);
  const resolvedRef = useRef(false);
  const advanceRef = useRef(0);
  const targetRef = useRef("");
  const modelRef = useRef(model);
  modelRef.current = model;

  const classes = samples?.classes ?? [];

  useEffect(() => {
    let cancelled = false;
    fetch("/model/sketch-samples.json")
      .then((res) => {
        const type = res.headers.get("content-type") ?? "";
        if (!res.ok || !type.includes("json")) throw new Error("sketch-samples.json is missing");
        return res.json() as Promise<SampleFile>;
      })
      .then((data) => {
        if (!cancelled) setSamples(data);
      })
      .catch((err: unknown) => {
        if (!cancelled) setLoadError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const suggestions = useMemo(() => {
    const q = normalise(guess);
    if (q.length < 2) return [];
    return classes
      .map((c) => c.name)
      .filter((name) => normalise(name).startsWith(q) || normalise(name).includes(q))
      .slice(0, 6);
  }, [guess, classes]);

  const paint = useCallback((revealed: number, style: { width: number; opacity: number }) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== LOGICAL * dpr) {
      canvas.width = LOGICAL * dpr;
      canvas.height = LOGICAL * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = PAPER;
    ctx.fillRect(0, 0, LOGICAL, LOGICAL);

    const s = LOGICAL / SRC;
    ctx.strokeStyle = INK;
    ctx.fillStyle = INK;
    ctx.globalAlpha = style.opacity;
    ctx.lineWidth = style.width;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    let remaining = revealed;
    for (const stroke of strokesRef.current) {
      if (remaining <= 0) break;
      const take = Math.min(stroke.length, remaining);
      remaining -= take;
      if (take === 1) {
        ctx.beginPath();
        ctx.arc(stroke[0].x * s, stroke[0].y * s, style.width / 2, 0, Math.PI * 2);
        ctx.fill();
        continue;
      }
      ctx.beginPath();
      ctx.moveTo(stroke[0].x * s, stroke[0].y * s);
      for (let i = 1; i < take - 1; i++) {
        const mx = ((stroke[i].x + stroke[i + 1].x) / 2) * s;
        const my = ((stroke[i].y + stroke[i + 1].y) / 2) * s;
        ctx.quadraticCurveTo(stroke[i].x * s, stroke[i].y * s, mx, my);
      }
      const tail = stroke[take - 1];
      ctx.lineTo(tail.x * s, tail.y * s);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }, []);

  const resolve = useCallback((hit: boolean, remaining: number, name: string) => {
    if (resolvedRef.current) return;
    resolvedRef.current = true;
    setDrawing(false);
    window.cancelAnimationFrame(rafRef.current);

    const row: Result = {
      name,
      hit,
      seconds: Math.round(SECONDS - remaining),
      modelGuess: null,
      modelProb: null,
    };
    setLast(row);
    setResults((prev) => [...prev, row]);
    setPhase("feedback");

    if (hit) {
      setScore((s) => s + 100 + Math.round(remaining) * 5);
      confetti({
        particleCount: 100,
        spread: 80,
        origin: { y: 0.6 },
        colors: ["#8b5cf6", "#22d3ee", "#f5f3ff", "#fbbf24"],
        disableForReducedMotion: true,
      });
    }

    // show what the model itself made of the drawing it just watched
    const mdl = modelRef.current;
    if (mdl && strokesRef.current.length > 0) {
      mdl
        .predict(rasterize(strokesRef.current))
        .then(({ preds }) => {
          const top = preds[0];
          if (!top) return;
          setLast((cur) =>
            cur && cur.name === name ? { ...cur, modelGuess: top.label, modelProb: top.prob } : cur,
          );
        })
        .catch(() => {
          /* the reveal is a nice-to-have, never worth surfacing an error */
        });
    }
  }, []);

  const startRound = useCallback(
    (index: number, avoid: string | null, diff: Difficulty) => {
      const pool = (samples?.classes ?? []).filter((c) => c[diff].length > 0);
      if (pool.length === 0) return;

      let chosen = pool[Math.floor(Math.random() * pool.length)];
      for (let i = 0; i < 8 && chosen.name === avoid; i++) {
        chosen = pool[Math.floor(Math.random() * pool.length)];
      }
      const raw = chosen[diff][Math.floor(Math.random() * chosen[diff].length)];
      const base = toStrokes(raw);
      strokesRef.current = diff === "hard" ? humanise(base, 2.2) : base;

      resolvedRef.current = false;
      setLast(null);
      setGuess("");
      setRound(index);
      setTarget(chosen.name);
      targetRef.current = chosen.name;
      setTimeLeft(SECONDS);
      setPhase("playing");
      setDrawing(true);

      const style = STYLE[diff];
      const total = strokesRef.current.reduce((n, s) => n + s.length, 0);
      const started = performance.now();
      paint(1, style);

      const tick = () => {
        if (resolvedRef.current) return;
        const revealed = Math.min(total, Math.floor((performance.now() - started) / style.msPerPoint) + 1);
        paint(revealed, style);
        if (revealed < total) {
          rafRef.current = requestAnimationFrame(tick);
        } else {
          setDrawing(false);
        }
      };
      window.cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(tick);
    },
    [paint, samples],
  );

  // countdown
  useEffect(() => {
    if (phase !== "playing") return;
    const started = performance.now();
    const id = window.setInterval(() => {
      const left = Math.max(0, SECONDS - (performance.now() - started) / 1000);
      setTimeLeft(left);
      if (left <= 0) {
        window.clearInterval(id);
        resolve(false, 0, targetRef.current);
      }
    }, 100);
    return () => window.clearInterval(id);
  }, [phase, round, resolve]);

  // advance after feedback
  useEffect(() => {
    if (phase !== "feedback") return;
    advanceRef.current = window.setTimeout(() => {
      if (round >= ROUNDS) setPhase("done");
      else startRound(round + 1, targetRef.current, difficulty);
    }, 2000);
    return () => window.clearTimeout(advanceRef.current);
  }, [phase, round, difficulty, startRound]);

  useEffect(() => () => window.cancelAnimationFrame(rafRef.current), []);

  const submit = () => {
    if (phase !== "playing" || resolvedRef.current || !guess.trim()) return;
    if (matches(guess, target)) resolve(true, timeLeft, target);
    else setGuess("");
  };

  const start = (diff: Difficulty) => {
    setDifficulty(diff);
    setScore(0);
    setResults([]);
    setCopied(false);
    startRound(1, null, diff);
  };

  const shareText = () => {
    const hits = results.filter((r) => r.hit).length;
    return `SketchSense: I guessed ${hits}/${results.length} of the network's doodles, ${score} points 🎨`;
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

  if (loadError) {
    return (
      <div className="glass rounded-2xl p-5">
        <h3 className="font-display text-lg font-semibold text-white">AI draws</h3>
        <p className="mt-2 text-sm text-rose-300">{loadError}</p>
        <p className="mt-1 font-mono text-[11px] text-white/40">run: python ml/export_samples.py</p>
      </div>
    );
  }

  if (phase === "idle") {
    return (
      <div className="glass rounded-2xl p-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="font-display text-lg font-semibold text-white">AI draws, you guess</h3>
            <p className="mt-1 max-w-md text-sm text-white/50">
              The network sketches one of the {classes.length || 100} things it knows, one stroke at a
              time. Name it before the clock runs out.
            </p>
          </div>
        </div>

        <div className="mt-4 grid gap-2 sm:grid-cols-2">
          <button
            onClick={() => start("easy")}
            disabled={classes.length === 0}
            className="group rounded-xl border border-white/10 bg-white/5 p-3 text-left transition hover:border-white/25 hover:bg-white/8 disabled:opacity-40"
          >
            <p className="font-display text-sm font-semibold text-white">Easy</p>
            <p className="mt-0.5 text-[11px] leading-relaxed text-white/45">
              Clean, confident drawings — the ones the model is sure about.
            </p>
          </button>
          <button
            onClick={() => start("hard")}
            disabled={classes.length === 0}
            className="group rounded-xl border border-white/10 bg-white/5 p-3 text-left transition hover:border-white/25 hover:bg-white/8 disabled:opacity-40"
          >
            <p className="font-display text-sm font-semibold text-white">Hard</p>
            <p className="mt-0.5 text-[11px] leading-relaxed text-white/45">
              Scruffy, ambiguous scribbles — the ones that nearly fooled it.
            </p>
          </button>
        </div>

        <div className="mt-3 grid grid-cols-3 gap-2 text-center">
          {[
            ["Rounds", String(ROUNDS)],
            ["Per round", `${SECONDS}s`],
            ["Guess", "type it"],
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
    <div className="grid gap-4 lg:grid-cols-[560px_minmax(0,1fr)]">
      <div className="space-y-3">
        <div className="glass rounded-2xl p-4 sm:p-5">
          <div className="relative mx-auto w-full">
            <canvas
              ref={canvasRef}
              width={LOGICAL}
              height={LOGICAL}
              className="aspect-square w-full rounded-2xl bg-[#fffaf0] shadow-[0_24px_60px_-20px_rgba(0,0,0,0.9),inset_0_0_0_1px_rgba(0,0,0,0.08)]"
              aria-label="The network's drawing"
            />
          </div>
          <div className="mx-auto mt-3 flex w-full items-center justify-between">
            <p className="text-xs text-white/40">
              {drawing ? "drawing…" : phase === "playing" ? "what is it?" : "revealed"}
            </p>
            <span className="rounded-lg bg-white/8 px-2 py-1 font-mono text-[11px] text-white/50">
              {difficulty}
            </span>
          </div>
        </div>
      </div>

      <div className="space-y-3">
        <div className="glass overflow-hidden rounded-2xl">
          <div className="flex items-center justify-between gap-3 px-4 py-3">
            <div className="flex items-center gap-3">
              <span className="rounded-lg bg-white/8 px-2 py-1 font-mono text-xs text-white/50">
                {round}/{ROUNDS}
              </span>
              <div>
                <p className="text-[10px] tracking-widest text-white/35 uppercase">Guess the drawing</p>
                <p className="font-display text-base leading-tight font-semibold text-white">
                  {phase === "feedback" ? last?.name : "type your answer"}
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

          <div className="p-4">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                submit();
              }}
              className="relative"
            >
              <input
                value={guess}
                onChange={(e) => setGuess(e.target.value)}
                disabled={phase !== "playing"}
                autoFocus
                placeholder={phase === "playing" ? "cat, bicycle, flamingo…" : "…"}
                aria-label="Your guess"
                className="w-full rounded-xl border border-white/12 bg-white/5 px-3 py-2.5 text-sm text-white placeholder:text-white/25 focus:border-brand/60 focus:outline-none disabled:opacity-50"
              />
              {suggestions.length > 0 && phase === "playing" && (
                <div className="absolute z-20 mt-1 w-full overflow-hidden rounded-xl border border-white/10 bg-ink-950/95 backdrop-blur">
                  {suggestions.map((name) => (
                    <button
                      key={name}
                      type="button"
                      onClick={() => {
                        setGuess(name);
                        if (matches(name, target)) resolve(true, timeLeft, target);
                      }}
                      className="block w-full px-3 py-1.5 text-left text-xs text-white/70 capitalize hover:bg-white/10"
                    >
                      {name}
                    </button>
                  ))}
                </div>
              )}
              <button
                type="submit"
                disabled={phase !== "playing"}
                className="mt-2 w-full rounded-xl bg-gradient-to-r from-brand to-cyan-glow px-3 py-2 text-sm font-semibold text-ink-950 transition hover:brightness-110 disabled:opacity-40"
              >
                Guess
              </button>
            </form>

            <p className="mt-2 text-center font-mono text-[11px] text-white/35">
              {timeLeft.toFixed(1)}s left
            </p>

            {phase === "feedback" && last && (
              <div className="mt-3 rounded-xl border border-white/8 bg-white/3 p-3 text-center">
                <p
                  className={`font-display text-sm font-semibold ${last.hit ? "text-emerald-300" : "text-rose-300"}`}
                >
                  {last.hit ? `yes — ${last.name}` : `it was ${last.name}`}
                </p>
                {last.modelGuess && (
                  <p className="mt-1 text-[11px] text-white/40">
                    the model said{" "}
                    <span className="text-white/70">{last.modelGuess}</span>{" "}
                    {last.modelProb != null && `(${(last.modelProb * 100).toFixed(0)}%)`}
                  </p>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="glass rounded-2xl px-4 py-3">
          <p className="text-[10px] tracking-widest text-white/35 uppercase">How the difficulty works</p>
          <p className="mt-1 text-[11px] leading-relaxed text-white/45">
            Every candidate sketch is scored by the model against its own label.{" "}
            <span className="text-white/70">Easy</span> uses the ones it is most certain about;{" "}
            <span className="text-white/70">Hard</span> uses sketches from the ambiguous middle of that
            ranking, drawn with a wobblier hand.
          </p>
        </div>
      </div>

      {phase === "done" && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink-950/80 p-4 backdrop-blur-sm">
          <div className="glass animate-rise w-full max-w-sm rounded-2xl p-5 text-center">
            <p className="text-[10px] tracking-widest text-white/40 uppercase">Final score</p>
            <p className="font-display my-1 text-5xl font-bold text-gradient">{score}</p>
            <p className="text-sm text-white/55">
              {results.filter((r) => r.hit).length} of {results.length} read correctly
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
                  {r.name}
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
                onClick={() => start(difficulty)}
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
