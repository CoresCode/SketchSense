"""Post-training evaluation: per-class accuracy, confusion pairs, and README numbers.

Runs the same stratified split as train.py (same seed), so these numbers are the ones the
model never saw. Writes:
  docs/metrics.md
  docs/confusion_matrix.png
and fills the {{placeholders}} in README.md.
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

    # confusion matrix image (row-normalised, log-ish scaling so 84 classes stay legible)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 9), dpi=140)
    fig.patch.set_facecolor("#0b0f18")
    ax.set_facecolor("#0b0f18")
    im = ax.imshow(norm, cmap="magma", vmin=0, vmax=max(0.25, float(np.percentile(norm, 99.7))))
    ax.set_title(f"SketchSense · row-normalised confusion ({n} classes, val top-1 {top1:.1%})",
                 color="#e6e9f0", fontsize=11)
    ax.set_xlabel("predicted", color="#8b93a7")
    ax.set_ylabel("true", color="#8b93a7")
    ax.tick_params(colors="#8b93a7", labelsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(colors="#8b93a7", labelsize=6)
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

    # ---- fill README placeholders -----------------------------------------
    report = json.loads((HERE / "artifacts" / "export_report.json").read_text(encoding="utf-8")) \
        if (HERE / "artifacts" / "export_report.json").exists() else {}
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    subs = {
        "val_top1": f"{top1 * 100:.1f}%",
        "val_top5": f"{ckpt.get('val_top5', 0) * 100:.1f}%",
        "model_mb": f"{report.get('onnx_mb', 0):.1f}",
        "classes": str(n),
    }
    for key, value in subs.items():
        readme = readme.replace("{{" + key + "}}", value)
    (REPO / "README.md").write_text(readme, encoding="utf-8")

    print(f"top-1 {top1:.4f} on {len(y_val):,} held-out sketches -> docs/metrics.md")
    print("top confusion pairs:", ", ".join(f"{a}->{b}" for a, b, _ in pairs[:5]))
    print("README placeholders left:", re.findall(r"\{\{\w+\}\}", readme))


if __name__ == "__main__":
    main()
