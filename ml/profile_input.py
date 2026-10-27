"""Time the host-side half of a training step: gather, convert, transfer.

Once the GPU work got 4x cheaper, this stopped being hidden behind it.

Run:  ..\\.venv\\Scripts\\python.exe profile_input.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
PROCESSED = HERE / "data" / "processed"
BATCH = 256
REPS = 40


def main() -> None:
    manifest = json.loads((PROCESSED / "manifest.json").read_text(encoding="utf-8"))
    total, size = int(manifest["total"]), int(manifest["size"])
    X = np.empty((total, size, size), dtype=np.uint8)
    cursor = 0
    for slug in manifest["classes"]:
        arr = np.load(PROCESSED / f"{slug}.npy")
        X[cursor : cursor + arr.shape[0]] = arr
        cursor += arr.shape[0]
    X = X[:cursor]
    y = np.random.randint(0, len(manifest["classes"]), total).astype(np.int16)
    print(f"{len(X):,} images, {X.nbytes / 2**30:.1f} GiB uint8 host copy\n", flush=True)

    dev = torch.device("cuda")
    rng = np.random.default_rng(0)

    def bench(label: str, fn) -> None:
        for _ in range(3):
            fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(REPS):
            fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - start) / REPS * 1000
        print(f"{label:<44} {ms:6.1f} ms/step", flush=True)

    idx = rng.integers(0, len(X), BATCH)
    bench("gather X[idx] (uint8)", lambda: X[idx])
    bench("gather + astype(float32) + /255", lambda: X[idx].astype(np.float32) / 255.0)
    bench("gather -> torch -> h2d float32 (current)", lambda: torch.from_numpy(
        X[idx].astype(np.float32) / 255.0).unsqueeze(1).to(dev, non_blocking=True))
    bench("gather -> torch uint8 -> h2d (4x less traffic)", lambda: torch.from_numpy(
        X[idx]).unsqueeze(1).to(dev, non_blocking=True).float().div_(255.0))
    bench("labels y[idx].astype(int64) -> h2d", lambda: torch.from_numpy(
        y[idx].astype(np.int64)).to(dev))


if __name__ == "__main__":
    main()
