"""Freeze real QuickDraw sketches plus the exact tensor `raster.py` produces.

The browser has its own rasteriser (web/src/lib/raster.ts). "Looks about right" is not
good enough when training/inference mismatch silently costs accuracy, so we ship 24 real
sketches with their expected 64x64 grids and let web/raster-check.html assert the two
implementations agree.

Run this after ml/prepare.py, then open the built /raster-check.html page.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from raster import CANVAS, LINE_WIDTH, PAD, rasterize_uint8

HERE = Path(__file__).parent
REPO = HERE.parent
RAW = HERE / "data" / "raw"
OUT = REPO / "web" / "src" / "lib" / "__fixtures__" / "raster.fixtures.json"


def read_first_drawings(path: Path, count: int) -> list[list]:
    out: list[list] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(out) >= count:
                break
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line)["drawing"])
            except (ValueError, KeyError):
                continue
    return out


def main() -> None:
    files = sorted(RAW.glob("*.ndjson"))
    if not files:
        raise SystemExit(f"no sketches in {RAW}; run download.py first")

    # spread the samples across the alphabet so we cover every drawing style
    step = max(1, len(files) // 12)
    chosen = files[::step][:12]

    samples = []
    for path in chosen:
        for drawing in read_first_drawings(path, 2):
            grid = rasterize_uint8(drawing, size=64)
            if int(grid.max()) == 0:
                continue
            samples.append(
                {
                    "label": path.stem,
                    "strokes": drawing,
                    "expected": base64.b64encode(grid.tobytes()).decode("ascii"),
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "size": 64,
        "canvas": CANVAS,
        "pad": PAD,
        "lineWidth": LINE_WIDTH,
        "source": "ml/raster.py (PIL, BOX downsample)",
        "samples": samples,
    }
    OUT.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {len(samples)} fixtures -> {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
