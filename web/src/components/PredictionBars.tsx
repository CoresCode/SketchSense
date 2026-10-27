import type { Prediction } from "../lib/inference";

type Props = {
  preds: Prediction[];
  ms: number | null;
  hasInk: boolean;
  ready: boolean;
};

const pct = (p: number) => (p >= 0.995 ? "99%" : p < 0.001 ? "<0.1%" : `${(p * 100).toFixed(1)}%`);

export default function PredictionBars({ preds, ms, hasInk, ready }: Props) {
  const top = preds[0];

  return (
    <div className="flex h-full flex-col">
      <div className="mb-4 flex items-baseline justify-between">
        <h2 className="font-display text-sm font-semibold tracking-wide text-white/50 uppercase">
          Model's guess
        </h2>
        <span className="font-mono text-[11px] text-white/35">
          {ms !== null ? `${ms.toFixed(1)} ms` : "—"}
        </span>
      </div>

      {!ready && (
        <div className="flex flex-1 items-center justify-center rounded-xl border border-dashed border-white/10 p-6 text-center">
          <p className="text-sm text-white/40">Model is loading…</p>
        </div>
      )}

      {ready && !hasInk && (
        <div className="flex flex-1 flex-col items-center justify-center rounded-xl border border-dashed border-white/10 p-6 text-center">
          <div className="mb-3 text-3xl">✏️</div>
          <p className="text-sm font-medium text-white/60">Nothing drawn yet</p>
          <p className="mt-1 text-xs text-white/35">
            The net re-runs a few milliseconds after every stroke.
          </p>
        </div>
      )}

      {ready && hasInk && top && (
        <div className="space-y-5">
          <div key={top.label + top.index} className="animate-rise">
            <div className="flex items-end justify-between gap-3">
              <span className="font-display text-3xl leading-none font-bold text-gradient capitalize">
                {top.label}
              </span>
              <span className="font-mono text-lg text-white/70">{pct(top.prob)}</span>
            </div>
            <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/8">
              <div
                className="h-full rounded-full bg-gradient-to-r from-brand to-cyan-glow transition-[width] duration-200 ease-out"
                style={{ width: `${Math.max(2, top.prob * 100)}%` }}
              />
            </div>
          </div>

          <div className="space-y-2.5">
            {preds.slice(1).map((p) => (
              <div key={p.index} className="group">
                <div className="mb-1 flex items-baseline justify-between text-sm">
                  <span className="text-white/55 capitalize">{p.label}</span>
                  <span className="font-mono text-[11px] text-white/35">{pct(p.prob)}</span>
                </div>
                <div className="h-1 overflow-hidden rounded-full bg-white/6">
                  <div
                    className="h-full rounded-full bg-white/25 transition-[width] duration-200 ease-out group-hover:bg-white/40"
                    style={{ width: `${Math.max(1, p.prob * 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="mt-auto pt-6">
        <p className="text-[11px] leading-relaxed text-white/25">
          Softmax over the trained classes. Top-1 is what the game checks; the rest is here so you
          can see when the model is torn between two shapes.
        </p>
      </div>
    </div>
  );
}
