"""Throughput sweep over widths / stem stride, so the architecture is chosen by
measurement instead of by guessing.

Run:  ..\\.venv\\Scripts\\python.exe sweep_arch.py
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from model import build, count_params

CONFIGS = (
    ("current   24/48/96/192 stem1", (24, 48, 96, 192), 1),
    ("stem2     24/48/96/192 stem2", (24, 48, 96, 192), 2),
    ("wide      32/64/128/256 stem1", (32, 64, 128, 256), 1),
    ("wide+stem2 32/64/128/256 stem2", (32, 64, 128, 256), 2),
    ("wider     40/80/160/320 stem2", (40, 80, 160, 320), 2),
    ("slim      16/32/64/128  stem2", (16, 32, 64, 128), 2),
)

BATCH = 256
STEPS = 10


def main() -> None:
    dev = torch.device("cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    print(f"{torch.cuda.get_device_name(0)}  bf16 {torch.cuda.is_bf16_supported()}\n", flush=True)

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    for label, widths, stride in CONFIGS:
        model = build(84, widths=widths, stem_stride=stride).to(dev)
        model = model.to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randn(BATCH, 1, 64, 64, device=dev).to(memory_format=torch.channels_last)
        y = torch.randint(0, 84, (BATCH,), device=dev)

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
        for _ in range(STEPS):
            step()
        torch.cuda.synchronize()
        per_step = (time.perf_counter() - start) / STEPS
        ips = BATCH / per_step
        print(
            f"{label:<32} {count_params(model)/1e6:5.2f}M  {per_step*1000:6.1f} ms/step"
            f"  {ips:6.0f} img/s   1.2M imgs/epoch = {1_200_000/ips/60:4.1f} min",
            flush=True,
        )
        del model, opt, x, y
        torch.cuda.empty_cache()

    print(f"\nclock: {torch.cuda.clock_rate() if hasattr(torch.cuda, 'clock_rate') else 'n/a'}")


if __name__ == "__main__":
    main()
