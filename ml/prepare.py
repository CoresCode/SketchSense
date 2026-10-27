"""Rasterize downloaded NDJSON sketches into per-class uint8 arrays.

Output: ml/data/processed/<class>.npy  (N, 64, 64) uint8 + ml/data/processed/manifest.json

Parallelised across classes (not across sketches) because that keeps the sample
order inside a class deterministic, which makes the train/val split reproducible.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from raster import rasterize_uint8


def read_drawings(path: Path, limit: int):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                return
            line = line.strip()
            if not line:
                continue
            import json as _json

            try:
                yield _json.loads(line)["drawing"]
            except (ValueError, KeyError):
                continue


def process_class(job: tuple[Path, int, Path]) -> tuple[str, int]:
    src, limit, out_dir = job
    slug = src.stem
    dest = out_dir / f"{slug}.npy"
    if dest.exists():
        arr = np.load(dest, mmap_mode="r")
        return slug, arr.shape[0]

    rows = [rasterize_uint8(d, size=64) for d in read_drawings(src, limit)]
    if not rows:
        return slug, 0
    arr = np.stack(rows, axis=0)
    # Guard against a class with very few usable sketches
    if arr.shape[0] < 200:
        return slug, 0
    np.save(dest, arr)
    return slug, int(arr.shape[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path(__file__).parent / "data" / "raw")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "data" / "processed")
    ap.add_argument("--limit", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sources = sorted(args.raw.glob("*.ndjson"))
    if not sources:
        raise SystemExit(f"no .ndjson files in {args.raw} -- run download.py first")

    jobs = [(p, args.limit, args.out) for p in sources]
    started = time.time()
    counts: dict[str, int] = {}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for slug, n in pool.map(process_class, jobs):
            counts[slug] = n
            flag = "" if n else "  <-- skipped (too few sketches)"
            print(f"{slug:<20} {n:>6}{flag}", flush=True)

    keep = sorted(k for k, v in counts.items() if v)
    manifest = {
        "size": 64,
        "classes": keep,
        "counts": {k: counts[k] for k in keep},
        "total": int(sum(counts[k] for k in keep)),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n{len(keep)} classes, {manifest['total']:,} sketches in {(time.time()-started)/60:.1f} min")


if __name__ == "__main__":
    main()
