"""Export a handful of real QuickDraw sketches per class as animation data for the
"AI draws, you guess" game.

The difficult part of a guessing game is picking drawings that are *fair*. The
classifier decides: it scores every candidate against its own true label, and

  easy -> the sketches it is most confident about (clean, prototypical)
  hard -> sketches in the 20th-35th confidence percentile, i.e. the ambiguous
          middle. The very worst sketches are excluded because they are not
          ambiguous, they are just unrecognisable.

Run:  ..\\.venv\\Scripts\\python.exe export_samples.py
      ..\\.venv\\Scripts\\python.exe export_samples.py --ckpt artifacts\\run2\\best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import build
from prepare import read_drawings
from raster import normalise_strokes, rasterize_uint8

HERE = Path(__file__).parent
REPO = HERE.parent

MAX_POINTS = 24  # per stroke, keeps the JSON small and the animation cheap
MAX_STROKES = 7
MIN_POINTS = 8
MAX_TOTAL_POINTS = 260


def resample(line: np.ndarray, limit: int = MAX_POINTS) -> list[int]:
    """Uniformly thin a polyline, always keeping the first and last point."""
    n = len(line)
    if n > limit:
        idx = np.linspace(0, n - 1, limit).round().astype(int)
        line = line[idx]
    flat: list[int] = []
    for x, y in line:
        flat.append(int(round(float(np.clip(x, 0, 255)))))
        flat.append(int(round(float(np.clip(y, 0, 255)))))
    return flat


def candidate(path: Path, scan: int) -> list[dict]:
    """Read the first `scan` drawings and keep the ones that are worth animating."""
    out: list[dict] = []
    for drawing in read_drawings(path, scan):
        lines = normalise_strokes(drawing)
        if not (2 <= len(lines) <= MAX_STROKES):
            continue
        total = sum(len(line) for line in lines)
        if not (MIN_POINTS <= total <= MAX_TOTAL_POINTS):
            continue
        out.append({"lines": lines, "strokes": len(lines), "points": total})
        if len(out) >= scan // 2:
            break
    return out


@torch.no_grad()
def score(model, class_index: int, rows: list[dict], device: torch.device) -> None:
    """Attach the model's probability for the true class to each candidate."""
    batch = 64
    for start in range(0, len(rows), batch):
        part = rows[start : start + batch]
        arr = np.stack([rasterize_uint8([line.T.tolist() for line in r["lines"]]) for r in part])
        x = torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(1).to(device)
        x = x.to(memory_format=torch.channels_last)
        probs = torch.softmax(model(x), dim=1)[:, class_index]
        for row, p in zip(part, probs.tolist(), strict=True):
            row["conf"] = float(p)


def heuristic(rows: list[dict]) -> None:
    """Fallback ordering when the checkpoint does not know this class yet."""
    for row in rows:
        # no model available: aim for a readable, middle-of-the-road sketch rather
        # than the simplest one (the simplest is often an unguessable squiggle)
        row["conf"] = -abs(row["strokes"] - 3.5) - abs(row["points"] - 80) / 120.0


def pick(rows: list[dict], per: int) -> tuple[list[dict], list[dict]]:
    rows = sorted(rows, key=lambda r: r["conf"], reverse=True)
    if not rows:
        return [], []
    # A one-stroke squiggle can still be "confident" and is still unguessable,
    # so the easy tier asks for at least two strokes.
    ranked = [r for r in rows if r["strokes"] >= 2] or rows
    easy = ranked[:per]
    lo = int(len(rows) * 0.20)
    hi = max(lo + per, int(len(rows) * 0.40))
    hard = rows[lo:hi][:per]
    return easy, hard


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=HERE / "data" / "raw")
    ap.add_argument("--processed", type=Path, default=HERE / "data" / "processed")
    ap.add_argument("--ckpt", type=Path, default=HERE / "artifacts" / "best.pt")
    ap.add_argument("--out", type=Path, default=REPO / "web" / "public" / "model" / "sketch-samples.json")
    ap.add_argument("--per", type=int, default=6, help="sketches kept per difficulty tier")
    ap.add_argument("--scan", type=int, default=600, help="candidates to examine per class")
    args = ap.parse_args()

    manifest = json.loads((args.processed / "manifest.json").read_text(encoding="utf-8"))
    slugs: list[str] = manifest["classes"]
    names: dict[str, str] = manifest.get("names", {})

    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    known: dict[str, int] = {}
    if args.ckpt.exists():
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        model = build(
            len(ckpt["classes"]),
            widths=tuple(ckpt.get("widths", (24, 48, 96, 192))),
            stem_stride=ckpt.get("stem_stride", 1),
        ).to(device).eval()
        model.load_state_dict(ckpt["model"])
        known = {slug: i for i, slug in enumerate(ckpt["classes"])}

    out: list[dict] = []
    for n, slug in enumerate(slugs, 1):
        src = args.raw / f"{slug}.ndjson"
        if not src.exists():
            print(f"  {slug}: no raw file, skipped", flush=True)
            continue
        rows = candidate(src, args.scan)
        if not rows:
            print(f"  {slug}: no usable sketches, skipped", flush=True)
            continue
        if model is not None and slug in known:
            score(model, known[slug], rows, device)
        else:
            heuristic(rows)
        easy, hard = pick(rows, args.per)
        out.append(
            {
                "slug": slug,
                "name": names.get(slug, slug.replace("_", " ")),
                "easy": [[resample(line) for line in r["lines"]] for r in easy],
                "hard": [[resample(line) for line in r["lines"]] for r in hard],
            }
        )
        if n % 10 == 0:
            print(f"  {n}/{len(slugs)} classes", flush=True)

    payload = {"canvas": 256, "classes": out}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    mb = args.out.stat().st_size / 1e6
    print(f"\n{len(out)} classes -> {args.out} ({mb:.2f} MB)")
    if model is None:
        print("note: no checkpoint found, ranking used the heuristic")


if __name__ == "__main__":
    main()
