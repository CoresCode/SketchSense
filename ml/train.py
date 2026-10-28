"""Train SketchNet on the rasterized QuickDraw subset.

Everything is held in RAM as uint8 (a 1M x 64 x 64 dataset is ~4 GB, your box has
23 GB), batches are converted to float on the GPU, and augmentation happens on the
GPU too -- so the dataloader never becomes the bottleneck.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import build, count_params

HERE = Path(__file__).parent


def load_processed(processed: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    classes = manifest["classes"]
    arrays, labels = [], []
    total = int(manifest["total"])
    X = np.empty((total, manifest["size"], manifest["size"]), dtype=np.uint8)
    cursor = 0
    for i, slug in enumerate(classes):
        arr = np.load(processed / f"{slug}.npy")
        n = arr.shape[0]
        X[cursor : cursor + n] = arr
        labels.append(np.full(n, i, dtype=np.int16))
        cursor += n
    if cursor != total:
        X = X[:cursor]
    return X, np.concatenate(labels), manifest


def stratified_split(y: np.ndarray, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_val = max(150, int(len(idx) * val_frac))
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    return np.concatenate(train_parts), np.concatenate(val_parts)


def affine_augment(x: torch.Tensor, max_rot: float = 12.0, max_scale: float = 0.12,
                   max_shift: float = 0.08, mode: str = "bilinear") -> torch.Tensor:
    """Random rotate/scale/translate, done on GPU with a single grid_sample."""
    b = x.shape[0]
    dev = x.device
    ang = (torch.rand(b, device=dev) * 2 - 1) * math.radians(max_rot)
    scale = 1.0 + (torch.rand(b, device=dev) * 2 - 1) * max_scale
    tx = (torch.rand(b, device=dev) * 2 - 1) * max_shift
    ty = (torch.rand(b, device=dev) * 2 - 1) * max_shift
    cos = torch.cos(ang) * scale
    sin = torch.sin(ang) * scale
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros", mode=mode)


def to_batch(X: np.ndarray, idx: np.ndarray, device: torch.device) -> torch.Tensor:
    # Send uint8 and convert on the card: 4x less PCIe traffic than float32, and the
    # division is exact either way, so the model sees identical values.
    x = torch.from_numpy(X[idx]).to(device, non_blocking=True).unsqueeze(1)
    return x.float().div_(255.0)


@torch.no_grad()
def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, idx: np.ndarray,
             device: torch.device, chunk: int = 512) -> tuple[float, float]:
    model.eval()
    top1 = top5 = 0
    for start in range(0, len(idx), chunk):
        part = idx[start : start + chunk]
        xb = to_batch(X, part, device)
        logits = model(xb)
        k = min(5, logits.shape[1])
        _, top = logits.topk(k, dim=1)
        target = torch.from_numpy(y[part].astype(np.int64)).to(device)
        hit1 = (top[:, 0] == target)
        hit5 = (top == target.unsqueeze(1)).any(dim=1)
        top1 += int(hit1.sum())
        top5 += int(hit5.sum())
    return top1 / len(idx), top5 / len(idx)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", type=Path, default=HERE / "data" / "processed")
    ap.add_argument("--out", type=Path, default=HERE / "artifacts")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--val-frac", type=float, default=0.06)
    ap.add_argument("--subset", type=float, default=1.0, help="fraction of train set per epoch (fast experiments)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--run-name", type=str, default="baseline")
    ap.add_argument("--widths", type=str, default="24,48,96,192", help="channel widths, e.g. 24,48,96,192")
    ap.add_argument("--stem-stride", type=int, default=2, help="2 halves the resolution up front (4x faster)")
    ap.add_argument("--init-from", type=Path, default=None, help="warm-start weights from a checkpoint")
    args = ap.parse_args()

    widths = tuple(int(w) for w in args.widths.split(","))
    if len(widths) != 4:
        raise SystemExit("--widths needs exactly 4 comma-separated ints")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    t0 = time.time()
    print(f"loading {args.processed} ...", flush=True)
    X, y, manifest = load_processed(args.processed)
    classes = manifest["classes"]
    print(f"{len(X):,} sketches / {len(classes)} classes, {time.time()-t0:.1f}s", flush=True)

    train_idx, val_idx = stratified_split(y, args.val_frac, args.seed)
    print(f"train {len(train_idx):,}  val {len(val_idx):,}", flush=True)

    model = build(len(classes), widths=widths, stem_stride=args.stem_stride).to(device)
    if args.init_from is not None:
        seed_ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(seed_ckpt["model"])
        print(
            f"warm start from {args.init_from.name} (epoch {seed_ckpt.get('epoch')},"
            f" val top1 {seed_ckpt.get('val_top1')})",
            flush=True,
        )
    model = model.to(memory_format=torch.channels_last)
    print(f"model params: {count_params(model)/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_idx) * args.subset / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    args.out.mkdir(parents=True, exist_ok=True)
    best = 0.0
    history: list[dict] = []
    run_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(train_idx)
        if args.subset < 1.0:
            order = order[: int(len(order) * args.subset)]
        running = correct = seen = 0.0
        # Accumulate the running stats on the card. Reading them every step forces a
        # full device sync, which stops the CPU from preparing the next batch while
        # the GPU is still busy -- about 20% of wall time on a small model.
        running_t = torch.zeros((), device=device)
        correct_t = torch.zeros((), device=device)
        ep_start = time.time()

        for step in range(steps_per_epoch):
            idx = order[step * args.batch : (step + 1) * args.batch]
            if len(idx) < 2:
                continue
            xb = to_batch(X, idx, device)
            yb = torch.from_numpy(y[idx].astype(np.int64)).to(device)
            xb = affine_augment(xb).to(memory_format=torch.channels_last)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            running_t += loss.detach() * len(idx)
            correct_t += (logits.detach().argmax(1) == yb).sum()
            seen += len(idx)

            if step % 100 == 0 or step == steps_per_epoch - 1:
                running = float(running_t)
                correct = int(correct_t)
                done = (step + 1) / steps_per_epoch
                per_step = (time.time() - ep_start) / (step + 1)
                print(
                    f"  ep{epoch:02d} {done*100:5.1f}%  loss {running/seen:.3f}  acc {correct/seen:.3f}"
                    f"  {per_step*1000:.0f}ms/step  eta {per_step*(steps_per_epoch-step-1)/60:.1f}m",
                    flush=True,
                )

        running = float(running_t)
        correct = int(correct_t)
        val_top1, val_top5 = evaluate(model, X, y, val_idx, device)
        row = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "train_acc": correct / max(seen, 1),
            "val_top1": val_top1,
            "val_top5": val_top5,
            "lr": sched.get_last_lr()[0],
            "seconds": round(time.time() - ep_start, 1),
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}  train {row['train_acc']:.3f}  val top1 {val_top1:.3f}  top5 {val_top5:.3f}"
            f"  ({row['seconds']}s)",
            flush=True,
        )

        if val_top1 > best:
            best = val_top1
            torch.save(
                {
                    "model": model.state_dict(),
                    "classes": classes,
                    "names": [manifest.get("names", {}).get(c, c.replace("_", " ")) for c in classes],
                    "size": manifest["size"],
                    "val_top1": val_top1,
                    "val_top5": val_top5,
                    "epoch": epoch,
                    "widths": list(widths),
                    "stem_stride": args.stem_stride,
                    "params": count_params(model),
                },
                args.out / "best.pt",
            )
            print(f"  saved new best ({val_top1:.4f})", flush=True)

        (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    metrics = {
        "run_name": args.run_name,
        "best_val_top1": best,
        "epochs": args.epochs,
        "params": count_params(model),
        "widths": list(widths),
        "stem_stride": args.stem_stride,
        "classes": len(classes),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "batch": args.batch,
        "lr": args.lr,
        "train_minutes": round((time.time() - run_start) / 60, 1),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "history": history,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nbest val top1 {best:.4f} after {(time.time()-run_start)/60:.1f} min -> {args.out/'best.pt'}")


if __name__ == "__main__":
    main()
