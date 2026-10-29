"""Post-training evaluation: per-class accuracy, confusion pairs, and README numbers.

Runs the same stratified split as train.py (same seed), so these numbers are the ones the
model never saw. Writes:
  docs/metrics.md
  docs/confusion_matrix.png
and regenerates the stats block in README.md, between the `stats:start` / `stats:end`
markers. Regenerating (instead of substituting one-shot placeholders) means swapping in a
better checkpoint later just means running this again.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from model import build
from train import load_processed, stratified_split, to_batch

HERE = Path(__file__).parent
REPO = HERE.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=HERE / "artifacts" / "best.pt")
    ap.add_argument("--processed", type=Path, default=HERE / "data" / "processed")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val-frac", type=float, default=0.06)
    ap.add_argument("--docs", type=Path, default=REPO / "docs")
    ap.add_argument(
        "--prompt-min-acc",
        type=float,
        default=0.8,
        help="classes at or above this held-out accuracy seed the speed-round prompts",
    )
    ap.add_argument(
        "--latency-ms",
        type=float,
        default=10,
        help="measured single-thread WASM inference time to quote in the README",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    classes: list[str] = ckpt["classes"]
    display: list[str] = ckpt.get("names") or [c.replace("_", " ") for c in classes]
    model = build(
        len(classes),
        widths=tuple(ckpt.get("widths", (24, 48, 96, 192))),
        stem_stride=ckpt.get("stem_stride", 1),
    ).to(device).eval()
    model.load_state_dict(ckpt["model"])

    X, y, manifest = load_processed(args.processed)
    _, val_idx = stratified_split(y, args.val_frac, args.seed)

    preds = np.empty(len(val_idx), dtype=np.int16)
    chunk = 512
    with torch.no_grad():
        for start in range(0, len(val_idx), chunk):
            part = val_idx[start : start + chunk]
            logits = model(to_batch(X, part, device))
            preds[start : start + chunk] = logits.argmax(1).cpu().numpy()

    y_val = y[val_idx]
    n = len(classes)
    matrix = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_val, preds):
        matrix[t, p] += 1

    per_class = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    correct = int((preds == y_val).sum())
    top1 = correct / len(y_val)

    # top confusion pairs
    off = matrix.copy()
    np.fill_diagonal(off, 0)
    flat = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1][:40], off.shape))[0]
    pairs = []
    for t, p in flat:
        if off[t, p] < 5:
            continue
        pairs.append((display[t], display[p], int(off[t, p])))
        if len(pairs) >= 10:
            break

    # confusion matrix image (row-normalised, log-ish scaling; every 5th class labelled
    # so 100 categories stay readable instead of turning the axes into a black smear)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=160)
    fig.patch.set_facecolor("#0b0f18")
    ax.set_facecolor("#0b0f18")
    im = ax.imshow(norm, cmap="magma", vmin=0, vmax=max(0.25, float(np.percentile(norm, 99.7))))
    ax.set_title(f"SketchSense · row-normalised confusion ({n} classes, val top-1 {top1:.1%})",
                 color="#e6e9f0", fontsize=11)
    ax.set_xlabel("predicted", color="#8b93a7")
    ax.set_ylabel("true", color="#8b93a7")
    step = 5 if n > 40 else 1
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([display[i] for i in ticks], rotation=90, fontsize=7)
    ax.set_yticklabels([display[i] for i in ticks], fontsize=7)
    ax.tick_params(colors="#8b93a7", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(colors="#8b93a7", labelsize=7)
    fig.tight_layout()
    args.docs.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.docs / "confusion_matrix.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    order = np.argsort(per_class)
    lines = [
        "# Evaluation",
        "",
        f"Model: `{n}` classes, {ckpt['params']/1e6:.2f} M parameters, best epoch {ckpt['epoch']}.",
        f"Held-out split: **{len(y_val):,} sketches** ({args.val_frac:.0%} per class, seed {args.seed}), "
        "never used for gradients.",
        "",
        f"* **Top-1 accuracy: {top1:.2%}**",
        f"* Top-5 accuracy: {ckpt.get('val_top5', float('nan')):.2%}",
        "",
        "## Most confused pairs",
        "",
        "| true | predicted as | count |",
        "|---|---|---|",
    ]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in pairs]
    lines += [
        "",
        "## Weakest classes",
        "",
        "| class | accuracy | support |",
        "|---|---|---|",
    ]
    for i in order[:12]:
        lines.append(f"| {display[i]} | {per_class[i]:.1%} | {int(matrix[i].sum())} |")

    lines += ["", "## Strongest classes", "", "| class | accuracy | support |", "|---|---|---|"]
    for i in order[::-1][:12]:
        lines.append(f"| {display[i]} | {per_class[i]:.1%} | {int(matrix[i].sum())} |")
    lines += [
        "",
        "![confusion matrix](confusion_matrix.png)",
        "",
        "The weak end is almost entirely shape-similar pairs (small round things, long thin things, "
        "animal faces). That is the expected failure mode at 64x64 greyscale and it is why the "
        "speed round requires 35% confidence rather than a bare argmax.",
    ]
    (args.docs / "metrics.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- prompt pool for the speed round ----------------------------------
    # Some classes are simply not drawable in twenty seconds (Mount Rushmore) or are the
    # model's blind spots. Prompting with those turns the game into a quiz about the
    # dataset, so the round draws its prompts from the classes that hold up on held-out
    # data. Written next to the model so the browser can fetch it with the weights.
    good = [display[i] for i in range(n) if per_class[i] >= args.prompt_min_acc]
    pool = {"min_accuracy": args.prompt_min_acc, "classes": good}
    web_model = REPO / "web" / "public" / "model"
    if web_model.exists():
        (web_model / "game-classes.json").write_text(json.dumps(pool, indent=2), encoding="utf-8")
        print(f"speed-round prompt pool: {len(good)}/{n} classes at >={args.prompt_min_acc:.0%}")

    # ---- regenerate the README stats block ---------------------------------
    # The block sits between the stats:start / stats:end markers, so re-running this after
    # swapping in a better checkpoint just rewrites the numbers instead of leaving stale ones.
    onnx = REPO / "web" / "public" / "model" / "model.onnx"
    model_mb = onnx.stat().st_size / 1e6 if onnx.exists() else float("nan")
    block = "\n".join(
        [
            "<!-- stats:start -->",
            "<!-- regenerated by `python ml/report.py --ckpt <best.pt>`; do not edit by hand -->",
            "| | |",
            "|---|---|",
            f"| Classes | **{n}** QuickDraw categories |",
            f"| Model | 4-stage residual CNN, **{ckpt['params'] / 1e6:.2f} M parameters** |",
            f"| Held-out top-1 | **{top1:.1%}** |",
            f"| Held-out top-5 | **{ckpt.get('val_top5', float('nan')):.1%}** |",
            f"| Inference | **~{args.latency_ms:g} ms** per guess (WASM, single thread, this laptop) |",
            f"| Model file | **{model_mb:.1f} MB** fp32 ONNX |",
            "| Backend | none |",
            "<!-- stats:end -->",
        ]
    )
    readme_path = REPO / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    pattern = re.compile(r"<!-- stats:start -->.*?<!-- stats:end -->", re.S)
    if pattern.search(readme):
        readme = pattern.sub(lambda _: block, readme)
        readme_path.write_text(readme, encoding="utf-8")
    else:
        print("warning: README has no stats:start/stats:end markers; left untouched")

    print(f"top-1 {top1:.4f} on {len(y_val):,} held-out sketches -> docs/metrics.md")
    print("top confusion pairs:", ", ".join(f"{a}->{b}" for a, b, _ in pairs[:5]))
    print("README placeholders left:", re.findall(r"\{\{\w+\}\}", readme))


if __name__ == "__main__":
    main()
