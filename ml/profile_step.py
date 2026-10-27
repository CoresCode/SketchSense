"""Break one training step into its pieces: augmentation, forward, backward.

bench.py and sweep_arch.py measured the network alone; training also runs an
affine warp on every batch, which those numbers never included.

Run:  ..\\.venv\\Scripts\\python.exe profile_step.py
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from model import build
from train import affine_augment

BATCH = 256
REPS = 15


def main() -> None:
    dev = torch.device("cuda")
    model = build(100, stem_stride=2).to(dev).to(memory_format=torch.channels_last)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    x = torch.randn(BATCH, 1, 64, 64, device=dev).to(memory_format=torch.channels_last)
    y = torch.randint(0, 100, (BATCH,), device=dev)

    def timeit(label: str, fn) -> float:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(REPS):
            fn()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - start) / REPS * 1000
        print(f"{label:<40} {ms:6.1f} ms/step", flush=True)
        return ms

    aug = timeit("affine_augment only", lambda: affine_augment(x))

    def fwd_bwd() -> None:
        with torch.amp.autocast("cuda"):
            loss = crit(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    net = timeit("forward + backward + step only", fwd_bwd)

    def full() -> None:
        xb = affine_augment(x).to(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda"):
            loss = crit(model(xb), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    both = timeit("augment + forward + backward", full)

    def nearest() -> None:
        xb = affine_augment(x, mode="nearest").to(memory_format=torch.channels_last)
        with torch.amp.autocast("cuda"):
            loss = crit(model(xb), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    near = timeit("augment (nearest) + forward + backward", nearest)

    print(f"\naugmentation share of a step: {aug / both * 100:.0f}%")
    print(f"nearest would save: {both - near:.1f} ms/step")


if __name__ == "__main__":
    main()
