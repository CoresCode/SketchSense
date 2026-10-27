import { StrictMode, useMemo } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { rasterize, SIZE, type Stroke } from "./lib/raster";
import fixtures from "./lib/__fixtures__/raster.fixtures.json";

type RawSketch = { label: string; strokes: number[][][]; expected: string };

function toStrokes(raw: number[][][]): Stroke[] {
  return raw.map((stroke) => stroke[0].map((x, i) => ({ x, y: stroke[1][i] ?? 0 })));
}

function drawGrid(grid: Uint8Array, size = SIZE) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(size, size);
  for (let i = 0; i < grid.length; i++) {
    img.data[i * 4] = grid[i];
    img.data[i * 4 + 1] = grid[i];
    img.data[i * 4 + 2] = grid[i];
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  canvas.style.width = "96px";
  canvas.style.height = "96px";
  canvas.style.imageRendering = "pixelated";
  return canvas.toDataURL();
}

type Row = {
  label: string;
  meanAbs: number;
  maxAbs: number;
  within8: number;
  inkAgreement: number;
  expectedUrl: string;
  gotUrl: string;
  ok: boolean;
};

function compare(sample: RawSketch): Row {
  const expected = Uint8Array.from(atob(sample.expected), (c) => c.charCodeAt(0));
  const gotFloat = rasterize(toStrokes(sample.strokes));
  const got = new Uint8Array(gotFloat.length);
  for (let i = 0; i < gotFloat.length; i++) got[i] = Math.round(Math.min(1, gotFloat[i]) * 255);

  let sum = 0;
  let max = 0;
  let within8 = 0;
  let inkBoth = 0;
  let inkEither = 0;
  for (let i = 0; i < expected.length; i++) {
    const d = Math.abs(expected[i] - got[i]);
    sum += d;
    if (d > max) max = d;
    if (d <= 8) within8++;
    const e = expected[i] > 32;
    const g = got[i] > 32;
    if (e && g) inkBoth++;
    if (e || g) inkEither++;
  }
  const meanAbs = sum / expected.length;
  const inkAgreement = inkEither === 0 ? 1 : inkBoth / inkEither;
  return {
    label: sample.label,
    meanAbs,
    maxAbs: max,
    within8: within8 / expected.length,
    inkAgreement,
    expectedUrl: drawGrid(expected),
    gotUrl: drawGrid(got),
    // Pixel-exact is impossible (PIL vs canvas line rasterisation differ by design),
    // so the bar is: same shape, same ink coverage, no structural disagreement.
    ok: meanAbs < 4 && inkAgreement > 0.93,
  };
}

function Report() {
  const rows = useMemo(() => (fixtures.samples as RawSketch[]).map(compare), []);
  const passed = rows.filter((r) => r.ok).length;
  const worstMean = Math.max(...rows.map((r) => r.meanAbs));
  const worstInk = Math.min(...rows.map((r) => r.inkAgreement));

  return (
    <div className="mx-auto max-w-4xl px-4 py-10">
      <p className="text-[10px] tracking-widest text-white/35 uppercase">Build-time verification</p>
      <h1 className="font-display mt-1 text-3xl font-bold text-white">Rasteriser parity</h1>
      <p className="mt-2 max-w-2xl text-sm leading-relaxed text-white/50">
        <code className="rounded bg-white/8 px-1">ml/raster.py</code> and{" "}
        <code className="rounded bg-white/8 px-1">web/src/lib/raster.ts</code> must turn the same
        strokes into the same tensor, otherwise the browser is subtly out of distribution. These are
        24 real QuickDraw sketches rendered by both implementations.
      </p>

      <div className="mt-5 grid gap-3 sm:grid-cols-3">
        {[
          ["Passed", `${passed}/${rows.length}`, passed === rows.length ? "text-emerald-300" : "text-amber-300"],
          ["Worst mean |Δ|", `${worstMean.toFixed(2)} / 255`, "text-white/80"],
          ["Worst ink overlap", `${(worstInk * 100).toFixed(1)}%`, "text-white/80"],
        ].map(([k, v, cls]) => (
          <div key={k} className="glass rounded-2xl p-4">
            <div className="text-[10px] tracking-widest text-white/35 uppercase">{k}</div>
            <div className={`font-mono mt-1 text-xl ${cls}`}>{v}</div>
          </div>
        ))}
      </div>

      <div className="glass mt-4 overflow-hidden rounded-2xl">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-white/8 text-[10px] tracking-widest text-white/35 uppercase">
            <tr>
              <th className="px-4 py-3">Python</th>
              <th className="px-4 py-3">Browser</th>
              <th className="px-4 py-3">Class</th>
              <th className="px-4 py-3 text-right">mean |Δ|</th>
              <th className="px-4 py-3 text-right">≤8 levels</th>
              <th className="px-4 py-3 text-right">ink ∩</th>
              <th className="px-4 py-3 text-right">ok</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-b border-white/5 last:border-0">
                <td className="px-4 py-2">
                  <img src={r.expectedUrl} alt="" className="h-12 w-12 rounded" />
                </td>
                <td className="px-4 py-2">
                  <img src={r.gotUrl} alt="" className="h-12 w-12 rounded" />
                </td>
                <td className="px-4 py-2 capitalize">{r.label.replace(/_/g, " ")}</td>
                <td className="px-4 py-2 text-right font-mono text-white/60">{r.meanAbs.toFixed(2)}</td>
                <td className="px-4 py-2 text-right font-mono text-white/60">
                  {(r.within8 * 100).toFixed(1)}%
                </td>
                <td className="px-4 py-2 text-right font-mono text-white/60">
                  {(r.inkAgreement * 100).toFixed(1)}%
                </td>
                <td className="px-4 py-2 text-right">
                  <span className={r.ok ? "text-emerald-400" : "text-rose-400"}>
                    {r.ok ? "pass" : "fail"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-4 text-xs leading-relaxed text-white/30">
        Absolute equality is not expected: PIL and canvas antialias line strokes differently. The
        pass bar is mean error under 4/255 and over 93% ink overlap, which is well inside the model's
        tolerance. Fixtures are regenerated by <code className="text-white/50">ml/fixtures.py</code>.
      </p>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Report />
  </StrictMode>,
);
