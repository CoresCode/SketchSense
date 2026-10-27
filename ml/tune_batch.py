"""Batch-size tuning for the stem_stride=2 architecture.

Run:  ..\\.venv\\Scripts\\python.exe tune_batch.py
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from model import build, count_params


def main() -> None:
    dev = torch.device("cuda")
    model = build(100, stem_stride=2).to(dev).to(memory_format=torch.channels_last)
    print(f"{count_params(model)/1e6:.2f}M params", flush=True)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    for batch in (128, 256, 512, 1024):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randn(batch, 1, 64, 64, device=dev).to(memory_format=torch.channels_last)
        y = torch.randint(0, 100, (batch,), device=dev)

        def step() -> None:
            with torch.amp.autocast("cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        for _ in range(3):
            step()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(12):
            step()
        torch.cuda.synchronize()
        per_step = (time.perf_counter() - start) / 12
        ips = batch / per_step
        print(f"batch {batch:>4}  {per_step*1000:6.1f} ms/step  {ips:6.0f} img/s", flush=True)
        del opt, x, y
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
