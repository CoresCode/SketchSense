import { useCallback, useEffect, useMemo, useState } from "react";
import DrawCanvas from "./components/DrawCanvas";
import PredictionBars from "./components/PredictionBars";
import SpeedRound from "./components/SpeedRound";
import { hasInk, rasterize, rasterToDataUrl, type Stroke } from "./lib/raster";
import {
  loadSketchModel,
  ModelMissingError,
  type Prediction,
  type SketchModel,
} from "./lib/inference";

type ModelState =
  | { status: "loading"; message: string }
  | { status: "ready"; model: SketchModel }
  | { status: "missing" }
  | { status: "error"; message: string };

export default function App() {
  const [modelState, setModelState] = useState<ModelState>({ status: "loading", message: "starting" });
  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const [preds, setPreds] = useState<Prediction[]>([]);
  const [ms, setMs] = useState<number | null>(null);
  const [mode, setMode] = useState<"free" | "speed">("free");
  const [resetSignal, setResetSignal] = useState(0);

  useEffect(() => {
    let cancelled = false;
    loadSketchModel((message) => {
      if (!cancelled) setModelState({ status: "loading", message });
    })
      .then((model) => {
        if (!cancelled) setModelState({ status: "ready", model });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (err instanceof ModelMissingError) setModelState({ status: "missing" });
        else setModelState({ status: "error", message: err instanceof Error ? err.message : String(err) });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const grid = useMemo(() => rasterize(strokes), [strokes]);
  const ink = hasInk(strokes);

  useEffect(() => {
    if (modelState.status !== "ready") return;
    if (!ink) {
      setPreds([]);
      setMs(null);
      return;
    }
    let cancelled = false;
    const id = window.setTimeout(() => {
      modelState.model
        .predict(grid)
        .then(({ preds: p, ms: t }) => {
          if (cancelled) return;
          setPreds(p);
          setMs(t);
        })
        .catch(() => {
          /* a dropped frame of inference is not worth surfacing */
        });
    }, 25);
    return () => {
      cancelled = true;
      window.clearTimeout(id);
    };
  }, [grid, ink, modelState]);

  const clearCanvas = useCallback(() => setResetSignal((n) => n + 1), []);

  const ready = modelState.status === "ready";
  const model = modelState.status === "ready" ? modelState.model : null;

  return (
    <div className="min-h-screen">
      <header className="mx-auto flex max-w-6xl items-center justify-between px-4 py-5">
        <div className="flex items-center gap-2.5">
          <span className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-brand to-cyan-glow text-lg shadow-lg shadow-brand/20">
            ✏️
          </span>
          <div>
            <p className="font-display text-sm font-semibold tracking-tight text-white">SketchSense</p>
            <p className="text-[11px] text-white/35">draw → CNN → answer</p>
          </div>
        </div>

        <div className="hidden items-center gap-2 sm:flex">
          {model && (
            <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 font-mono text-[11px] text-white/50">
              {model.classes.length} classes
            </span>
          )}
          {model?.valTop1 != null && (
            <span className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 font-mono text-[11px] text-emerald-300">
              {(model.valTop1 * 100).toFixed(1)}% top-1
            </span>
          )}
          <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 font-mono text-[11px] text-white/50">
            in-browser
          </span>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 pb-16">
        <section className="mb-6 max-w-2xl">
          <h1 className="font-display text-4xl leading-[1.05] font-bold tracking-tight text-white sm:text-5xl">
            Draw it.
            <br />
            <span className="text-gradient">We'll name it.</span>
          </h1>
          <p className="mt-3 max-w-xl text-sm leading-relaxed text-white/50 sm:text-base">
            A small convolutional network trained from scratch on Google QuickDraw. It runs on your
            device through WebAssembly — no server, no upload, no API key.
          </p>
        </section>

        <div className="mb-4 inline-flex rounded-xl border border-white/10 bg-white/5 p-1">
          {(
            [
              ["free", "Free draw"],
              ["speed", "Speed round"],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              onClick={() => setMode(key)}
              disabled={!ready && key === "speed"}
              className={`rounded-lg px-4 py-1.5 text-sm font-medium transition ${
                mode === key
                  ? "bg-white text-ink-950 shadow"
                  : "text-white/55 hover:text-white disabled:opacity-40"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="grid gap-4 lg:grid-cols-[560px_minmax(0,1fr)]">
          <div className="space-y-3">
            {mode === "speed" && model && (
              <SpeedRound
                classes={model.classes}
                preds={preds}
                onClear={clearCanvas}
                onExit={() => {
                  setMode("free");
                  clearCanvas();
                }}
              />
            )}

            <div className="glass rounded-2xl p-4 sm:p-5">
              <DrawCanvas
                onChange={setStrokes}
                resetSignal={resetSignal}
                disabled={!ready}
                hint={ready && !ink ? "draw anything" : undefined}
              />

              {model && (
                <details className="mt-4 border-t border-white/8 pt-3">
                  <summary className="cursor-pointer text-xs text-white/40 transition hover:text-white/70">
                    {model.classes.length} things it knows — see the full list
                  </summary>
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {model.classes.map((c) => (
                      <span
                        key={c}
                        className="rounded-md border border-white/8 bg-white/4 px-2 py-0.5 text-[11px] text-white/45 capitalize"
                      >
                        {c}
                      </span>
                    ))}
                  </div>
                </details>
              )}
            </div>
          </div>

          <div className="space-y-4">
            <div className="glass min-h-[300px] rounded-2xl p-4 sm:p-5 lg:sticky lg:top-5">
              {modelState.status === "missing" ? (
                <div className="py-10 text-center">
                  <p className="text-sm font-medium text-white/70">No model yet</p>
                  <p className="mx-auto mt-2 max-w-xs text-xs leading-relaxed text-white/40">
                    Train it once, then reload this page:
                  </p>
                  <pre className="mx-auto mt-3 w-fit rounded-lg bg-black/40 px-3 py-2 text-left font-mono text-[11px] leading-relaxed text-white/50">
{`cd ml
..\\.venv\\Scripts\\python.exe train.py
..\\.venv\\Scripts\\python.exe export.py
..\\.venv\\Scripts\\python.exe report.py`}
                  </pre>
                </div>
              ) : modelState.status === "error" ? (
                <div className="py-10 text-center">
                  <p className="text-sm font-medium text-rose-300">Model failed to load</p>
                  <p className="mt-2 font-mono text-xs break-words text-white/40">{modelState.message}</p>
                </div>
              ) : (
                <PredictionBars preds={preds} ms={ms} hasInk={ink} ready={ready} />
              )}
            </div>

            {ready && ink && (
              <div className="glass hidden rounded-2xl p-4 lg:block">
                <p className="mb-2 text-[10px] tracking-widest text-white/35 uppercase">
                  What the net sees · 64×64
                </p>
                <div className="flex items-center gap-3">
                  <img
                    src={rasterToDataUrl(grid)}
                    alt="model input"
                    width={64}
                    height={64}
                    className="h-16 w-16 rounded-md border border-white/10 [image-rendering:pixelated]"
                  />
                  <p className="text-[11px] leading-relaxed text-white/35">
                    Same tensor the Python trainer produced: bbox-centred, uniformly scaled, box-filtered.
                  </p>
                </div>
              </div>
            )}

            {modelState.status === "loading" && (
              <div className="glass rounded-2xl px-4 py-3">
                <p className="font-mono text-[11px] text-white/40">{modelState.message}…</p>
              </div>
            )}
          </div>
        </div>

        <section className="mt-10 grid gap-3 sm:grid-cols-3">
          {[
            [
              "Trained, not borrowed",
              model
                ? `${model.classes.length} QuickDraw categories, residual CNN, ~1.6M parameters.`
                : "Residual CNN trained from raw QuickDraw strokes.",
            ],
            [
              "Measured honestly",
              model?.valTop1 != null
                ? `${(model.valTop1 * 100).toFixed(1)}% top-1 on a held-out split, not on training data.`
                : "Top-1 / top-5 on a held-out split.",
            ],
            [
              "Zero backend",
              ms !== null
                ? `ONNX Runtime Web, ${ms.toFixed(1)} ms per guess on this device.`
                : "ONNX Runtime Web, a few ms per guess.",
            ],
          ].map(([title, body]) => (
            <div key={title} className="glass rounded-2xl p-4">
              <p className="font-display text-sm font-semibold text-white/90">{title}</p>
              <p className="mt-1 text-xs leading-relaxed text-white/45">{body}</p>
            </div>
          ))}
        </section>

        <footer className="mt-10 flex flex-col items-start justify-between gap-2 border-t border-white/8 pt-5 text-xs text-white/30 sm:flex-row sm:items-center">
          <p>
            Built by{" "}
            <a href="https://anuragyadavcodes.com" className="text-white/55 hover:text-white">
              Anurag Yadav
            </a>{" "}
            · AIML, final year diploma
          </p>
          <p className="font-mono">ONNX · React · PyTorch · QuickDraw</p>
        </footer>
      </main>
    </div>
  );
}
