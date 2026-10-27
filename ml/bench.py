"""Measure real training throughput on this machine, at a few batch sizes.

Used to size the training run (epochs x images) instead of guessing. Run:

    ..\\.venv\\Scripts\\python.exe bench.py
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from model import build, count_params


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    device = torch.device("cuda")
    model = build(84).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    print(f"{torch.cuda.get_device_name(0)} · {count_params(model)/1e6:.2f}M params")
    print(f"torch {torch.__version__} · cuda {torch.version.cuda}\n")

    for batch in (128, 256, 512):
        x = torch.randn(batch, 1, 64, 64, device=device).to(memory_format=torch.channels_last)
        y = torch.randint(0, 84, (batch,), device=device)

        for _ in range(3):  # warm up cudnn kernels and allocator
            with torch.amp.autocast("cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        start = time.perf_counter()
        steps = 10
        for _ in range(steps):
            with torch.amp.autocast("cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        per_step = (time.perf_counter() - start) / steps
        ips = batch / per_step
        print(
            f"batch {batch:>3}  {per_step*1000:6.1f} ms/step  {ips:7.0f} img/s"
            f"  -> 1,000,000 images = {1_000_000/ips/60:.1f} min"
        )


if __name__ == "__main__":
    main()
