"""Stroke -> bitmap rasterizer.

This file is the *contract* between training and the browser: `web/src/lib/raster.ts`
implements the same four steps, so a drawing on the canvas produces the same tensor
the model was trained on.

Pipeline per sketch:
  1. gather every point, compute the bounding box
  2. translate so the box centre hits the canvas centre
  3. scale so the longest side spans (canvas - 2*pad)
  4. draw at SUPERSAMPLE x resolution, then box-filter down to `size`

Normalizing geometry (instead of just dividing 0-255 by the canvas) is what lets a
user draw a tiny cat in the corner and still get a correct prediction.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

SUPERSAMPLE = 4
STROKE_WIDTH = 6  # in supersampled pixels
PAD = 18  # in supersampled pixels


def normalise_strokes(
    strokes: Sequence[Sequence[Sequence[float]]], canvas: int = 256, pad: int = PAD
) -> list[np.ndarray]:
    """Return stroke polylines in canvas coordinates, or [] if nothing is drawable."""
    lines: list[np.ndarray] = []
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        xs = np.asarray(stroke[0], dtype=np.float32)
        ys = np.asarray(stroke[1], dtype=np.float32)
        if xs.size == 0:
            continue
        lines.append(np.stack([xs, ys], axis=1))
    if not lines:
        return []

    pts = np.concatenate(lines, axis=0)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    extent = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    centre = (lo + hi) / 2.0

    if extent < 1e-3:  # a single dot: blow it up to a visible blob
        scale = 1.0
        extent = 1.0
    else:
        scale = (canvas - 2 * pad) / extent

    out = []
    for line in lines:
        moved = (line - centre) * scale + canvas / 2.0
        out.append(moved)
    return out


def rasterize(
    strokes: Sequence[Sequence[Sequence[float]]],
    size: int = 64,
    canvas: int = 256,
    pad: int = PAD,
    line_width: int = STROKE_WIDTH,
) -> np.ndarray:
    """Sketch strokes -> float32 array (size, size) in [0, 1], white strokes on black.

    The final resize uses BOX (exact 4x4 area average) rather than LANCZOS on purpose:
    the browser version in web/src/lib/raster.ts can reproduce a box filter exactly,
    so training and inference see the same tensor. A windowed-sinc resize cannot be
    matched that precisely in canvas.
    """
    img = Image.new("L", (canvas, canvas), 0)
    lines = normalise_strokes(strokes, canvas=canvas, pad=pad)

    if not lines:
        return np.zeros((size, size), dtype=np.float32)

    draw = ImageDraw.Draw(img)
    for line in lines:
        pts = [tuple(map(float, p)) for p in line]
        if len(pts) == 1:
            x, y = pts[0]
            draw.ellipse([x - line_width / 2, y - line_width / 2, x + line_width / 2, y + line_width / 2], fill=255)
        else:
            draw.line(pts, fill=255, width=line_width, joint="curve")

    small = img.resize((size, size), Image.BOX)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    return arr


def rasterize_uint8(strokes, size: int = 64) -> np.ndarray:
    """Same as rasterize() but returns uint8 (0-255) for compact dataset storage."""
    return (rasterize(strokes, size=size) * 255.0).round().astype(np.uint8)
