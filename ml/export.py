"""Export best.pt -> ONNX and prove the exported graph matches PyTorch.

The parity check is the important part: it catches ops that silently change
behaviour in ONNX (BN folding, dynamic shapes, resize alignment) before they ship
to the browser.

Writes:
  ml/artifacts/sketchnet.onnx
  web/public/model/model.onnx
  web/public/model/labels.json
  ml/artifacts/export_report.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from model import build

HERE = Path(__file__).parent
REPO = HERE.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=HERE / "artifacts" / "best.pt")
    ap.add_argument("--processed", type=Path, default=HERE / "data" / "processed")
    ap.add_argument("--out", type=Path, default=HERE / "artifacts" / "sketchnet.onnx")
    ap.add_argument("--web", type=Path, default=REPO / "web" / "public" / "model")
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes: list[str] = ckpt["classes"]
    size: int = ckpt["size"]
    model = build(len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval()
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())

    dummy = torch.randn(1, 1, size, size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["drawing"],
        output_names=["logits"],
        dynamic_axes={"drawing": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_mb = args.out.stat().st_size / 1e6
    print(f"exported {args.out} ({onnx_mb:.2f} MB, opset {args.opset})")

    # ---- parity check -------------------------------------------------------
    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    manifest = json.loads((args.processed / "manifest.json").read_text(encoding="utf-8"))
    batch = []
    for slug in manifest["classes"][: min(len(manifest["classes"]), 8)]:
        arr = np.load(args.processed / f"{slug}.npy", mmap_mode="r")[: args.samples // 8]
        batch.append(np.asarray(arr))
    x = np.concatenate(batch).astype(np.float32) / 255.0
    x = x[:, None, :, :]

    with torch.no_grad():
        torch_logits = model(torch.from_numpy(x)).numpy()
    ort_logits = sess.run(None, {"drawing": x})[0]

    max_abs = float(np.abs(torch_logits - ort_logits).max())
    agree_top1 = float((torch_logits.argmax(1) == ort_logits.argmax(1)).mean())
    # cosine similarity per sample, averaged
    a = torch_logits / np.linalg.norm(torch_logits, axis=1, keepdims=True)
    b = ort_logits / np.linalg.norm(ort_logits, axis=1, keepdims=True)
    cos = float((a * b).sum(axis=1).mean())

    report = {
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "onnx_mb": round(onnx_mb, 2),
        "opset": args.opset,
        "batch_norm_folded": has_bn,
        "samples_checked": int(x.shape[0]),
        "max_abs_logit_diff": max_abs,
        "top1_agreement": agree_top1,
        "mean_cosine": cos,
        "classes": len(classes),
        "val_top1": ckpt.get("val_top1"),
        "val_top5": ckpt.get("val_top5"),
    }
    (HERE / "artifacts" / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if agree_top1 < 0.999 or max_abs > 1e-2:
        raise SystemExit("parity check FAILED - do not ship this graph")

    # ---- ship it -----------------------------------------------------------
    args.web.mkdir(parents=True, exist_ok=True)
    (args.web / "model.onnx").write_bytes(args.out.read_bytes())
    labels = {"size": size, "classes": classes, "valTop1": ckpt.get("val_top1")}
    (args.web / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(f"copied model + labels to {args.web}")


if __name__ == "__main__":
    main()
