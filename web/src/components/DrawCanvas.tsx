import { useCallback, useEffect, useRef } from "react";
import type { Point, Stroke } from "../lib/raster";

type Props = {
  onChange: (strokes: Stroke[]) => void;
  resetSignal: number;
  disabled?: boolean;
  hint?: string;
};

const LOGICAL = 400; // the coordinate space strokes are stored in
const PEN = 10; // ~= 6px at the 256 canvas the model sees, scaled up
const PAPER = "#fffaf0";
const INK = "#12161c";

export default function DrawCanvas({ onChange, resetSignal, disabled, hint }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const strokesRef = useRef<Stroke[]>([]);
  const drawingRef = useRef(false);
  const lastEmitRef = useRef(0);
  const resetSeenRef = useRef(resetSignal);

  const paint = useCallback(() => {
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

    ctx.strokeStyle = INK;
    ctx.fillStyle = INK;
    ctx.lineWidth = PEN;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    for (const stroke of strokesRef.current) {
      if (stroke.length === 0) continue;
      if (stroke.length === 1) {
        ctx.beginPath();
        ctx.arc(stroke[0].x, stroke[0].y, PEN / 2, 0, Math.PI * 2);
        ctx.fill();
        continue;
      }
      // Midpoint quadratics: cheap smoothing that also makes mouse input feel less angular.
      ctx.beginPath();
      ctx.moveTo(stroke[0].x, stroke[0].y);
      for (let i = 1; i < stroke.length - 1; i++) {
        const midX = (stroke[i].x + stroke[i + 1].x) / 2;
        const midY = (stroke[i].y + stroke[i + 1].y) / 2;
        ctx.quadraticCurveTo(stroke[i].x, stroke[i].y, midX, midY);
      }
      const last = stroke[stroke.length - 1];
      ctx.lineTo(last.x, last.y);
      ctx.stroke();
    }
  }, []);

  const emit = useCallback(
    (throttle = false) => {
      const now = performance.now();
      if (throttle && now - lastEmitRef.current < 60) return;
      lastEmitRef.current = now;
      onChange(strokesRef.current.map((s) => s.slice()));
    },
    [onChange],
  );

  useEffect(() => {
    paint();
  }, [paint]);

  useEffect(() => {
    if (resetSignal !== resetSeenRef.current) {
      resetSeenRef.current = resetSignal;
      strokesRef.current = [];
      drawingRef.current = false;
      paint();
      onChange([]);
    }
  }, [resetSignal, paint, onChange]);

  const toLocal = (e: React.PointerEvent<HTMLCanvasElement>): Point => {
    const rect = e.currentTarget.getBoundingClientRect();
    const scale = LOGICAL / rect.width;
    return {
      x: (e.clientX - rect.left) * scale,
      y: (e.clientY - rect.top) * scale,
    };
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (disabled) return;
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      // some browsers reject capture for already-inactive pointers; drawing still works
    }
    drawingRef.current = true;
    strokesRef.current.push([toLocal(e)]);
    paint();
    emit();
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawingRef.current || disabled) return;
    const stroke = strokesRef.current[strokesRef.current.length - 1];
    if (!stroke) return;
    const p = toLocal(e);

    // Drop points that are too close together: fewer points, same visual line,
    // and the bbox normalisation stays stable while drawing.
    const prev = stroke[stroke.length - 1];
    if (Math.hypot(p.x - prev.x, p.y - prev.y) < 1.5) return;

    stroke.push(p);
    paint();
    emit(true);
  };

  const endStroke = () => {
    if (!drawingRef.current) return;
    drawingRef.current = false;
    emit();
  };

  const undo = () => {
    strokesRef.current.pop();
    paint();
    emit();
  };

  const clear = () => {
    strokesRef.current = [];
    paint();
    emit();
  };

  return (
    <div className="relative">
      <div className="mx-auto w-full">
        <canvas
          ref={canvasRef}
          width={LOGICAL}
          height={LOGICAL}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endStroke}
          onPointerCancel={endStroke}
          onPointerLeave={endStroke}
          className="aspect-square w-full touch-none rounded-2xl bg-[#fffaf0] shadow-[0_24px_60px_-20px_rgba(0,0,0,0.9),inset_0_0_0_1px_rgba(0,0,0,0.08)]"
          style={{ cursor: disabled ? "not-allowed" : "crosshair" }}
          aria-label="Drawing canvas"
        />
      </div>

      {hint && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <span className="rounded-full bg-black/5 px-4 py-2 text-sm font-medium text-black/30">{hint}</span>
        </div>
      )}

      <div className="mx-auto mt-3 flex w-full max-w-[540px] items-center justify-between">
        <p className="text-xs text-white/40">Draw with mouse, finger or stylus</p>
        <div className="flex gap-2">
          <button
            onClick={undo}
            disabled={disabled}
            className="rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-white/70 transition hover:bg-white/10 disabled:opacity-40"
          >
            Undo
          </button>
          <button
            onClick={clear}
            disabled={disabled}
            className="rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-white/70 transition hover:bg-white/10 disabled:opacity-40"
          >
            Clear
          </button>
        </div>
      </div>
    </div>
  );
}
