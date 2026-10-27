/**
 * The browser half of the training/inference contract.
 *
 * ml/raster.py does exactly this, and ml/fixtures.py asserts the two agree:
 *   1. drop strokes with fewer than 2 points
 *   2. bounding box over every remaining point
 *   3. translate the box centre to the canvas centre
 *   4. scale so the longest side spans CANVAS - 2*PAD
 *   5. draw at 256x256, then box-filter 4x4 down to 64x64
 *
 * Step 3-4 are why a tiny drawing in the corner of a big canvas still gets read
 * correctly: geometry is normalised, not just scaled.
 */

export const SIZE = 64;
export const CANVAS = 256;
export const PAD = 18;
export const LINE_WIDTH = 6;
export const SUPERSAMPLE = CANVAS / SIZE;

export type Point = { x: number; y: number };
export type Stroke = Point[];

let srcCanvas: HTMLCanvasElement | null = null;
let srcCtx: CanvasRenderingContext2D | null = null;

function ensureScratch() {
  if (!srcCanvas) {
    srcCanvas = document.createElement("canvas");
    srcCanvas.width = CANVAS;
    srcCanvas.height = CANVAS;
    srcCtx = srcCanvas.getContext("2d", { willReadFrequently: true });
  }
  return { canvas: srcCanvas!, ctx: srcCtx! };
}

export function hasInk(strokes: Stroke[]): boolean {
  return strokes.some((s) => s.length >= 1);
}

/** Strokes (any coordinate space) -> 64x64 grayscale, row-major, 0..1. */
export function rasterize(strokes: Stroke[]): Float32Array {
  const out = new Float32Array(SIZE * SIZE);
  // A tap is a single point: ml/raster.py draws it as a dot, so we must too, otherwise
  // eyes and punctuation silently disappear from the browser's version of the input.
  const usable = strokes.filter((s) => s.length >= 1);
  if (usable.length === 0) return out;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const stroke of usable) {
    for (const p of stroke) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }
  }

  const extent = Math.max(maxX - minX, maxY - minY);
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const scale = extent < 1e-3 ? 1 : (CANVAS - 2 * PAD) / extent;

  const { ctx } = ensureScratch();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, CANVAS, CANVAS);

  ctx.strokeStyle = "#fff";
  ctx.fillStyle = "#fff";
  ctx.lineWidth = LINE_WIDTH;
  // PIL's ImageDraw.line has flat caps and rounded joints; match it, otherwise the
  // browser is gently out of distribution at every stroke ending.
  ctx.lineCap = "butt";
  ctx.lineJoin = "round";

  for (const stroke of usable) {
    if (stroke.length === 1) {
      const x = (stroke[0].x - cx) * scale + CANVAS / 2;
      const y = (stroke[0].y - cy) * scale + CANVAS / 2;
      ctx.beginPath();
      ctx.arc(x, y, LINE_WIDTH / 2, 0, Math.PI * 2);
      ctx.fill();
      continue;
    }
    ctx.beginPath();
    for (let i = 0; i < stroke.length; i++) {
      const x = (stroke[i].x - cx) * scale + CANVAS / 2;
      const y = (stroke[i].y - cy) * scale + CANVAS / 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  const { data } = ctx.getImageData(0, 0, CANVAS, CANVAS);
  const s = SUPERSAMPLE;
  const area = s * s;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      let sum = 0;
      for (let dy = 0; dy < s; dy++) {
        for (let dx = 0; dx < s; dx++) {
          const px = (y * s + dy) * CANVAS + (x * s + dx);
          sum += data[px * 4]; // red channel == luminance, canvas is greyscale
        }
      }
      out[y * SIZE + x] = sum / area / 255;
    }
  }
  return out;
}

/** Debug helper: draw the model's view so a human can sanity-check the tensor. */
export function rasterToDataUrl(grid: Float32Array, size = SIZE): string {
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d")!;
  const img = ctx.createImageData(size, size);
  for (let i = 0; i < grid.length; i++) {
    const v = Math.round(Math.min(1, Math.max(0, grid[i])) * 255);
    img.data[i * 4] = v;
    img.data[i * 4 + 1] = v;
    img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c.toDataURL();
}
