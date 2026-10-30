"""Quantise the exported ONNX model to int8 and prove the accuracy cost is small.

The shipped fp32 graph is 6.4 MB. On a slow connection that is the biggest single thing
between a visitor and a working page - once brotli-compressed, the ONNX Runtime wasm is only
3.4 MB, so the weights outweigh the runtime. Cutting them is worth more than any amount of
prefetching, because a bandwidth-limited link pays for total bytes, not for overlap.

Two modes, because they are not equally good on a convnet this small:

  static   QDQ int8 with calibrated activation ranges. Much better than dynamic here -
           dynamic quantisation re-derives activation scales per call and cost 1.1 pp.
  dynamic  weights only. Kept for comparison; it was the 1.1 pp loser.

Calibration samples come from the training split, never the held-out split, so the reported
accuracy stays honest.

    python quantize.py --mode static --onnx artifacts/sketchnet.onnx --out artifacts/sketchnet.int8.onnx

Exits non-zero if held-out top-1 drops by more than --max-drop, so it cannot silently ship a
worse model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

from train import load_processed, stratified_split


def to_input(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return (X[idx].astype(np.float32) / 255.0)[:, None, :, :]


def run(onnx_path: Path, X: np.ndarray, y: np.ndarray, idx: np.ndarray, chunk: int = 256):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.log_severity_level = 3
    session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    top1 = top5 = 0
    for start in range(0, len(idx), chunk):
        part = idx[start : start + chunk]
        logits = session.run(None, {input_name: to_input(X, part)})[0]
        order = np.argsort(logits, axis=1)[:, ::-1][:, :5]
        truth = y[part]
        top1 += int((order[:, 0] == truth).sum())
        top5 += int((order == truth[:, None]).any(axis=1).sum())
    return top1 / len(idx), top5 / len(idx)


class Batches(CalibrationDataReader):
    """Feeds real sketches to the calibrator, one batch at a time."""

    def __init__(self, X: np.ndarray, idx: np.ndarray, name: str, batch: int = 32):
        self.X = X
        self.name = name
        self.batch = batch
        self.idx = idx
        self.pos = 0

    def get_next(self):
        if self.pos >= len(self.idx):
            return None
        part = self.idx[self.pos : self.pos + self.batch]
        self.pos += self.batch
        return {self.name: to_input(self.X, part)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, default=Path("artifacts/sketchnet.onnx"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/sketchnet.int8.onnx"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--mode", choices=["static", "dynamic"], default="static")
    ap.add_argument("--calib", type=int, default=1024, help="calibration sketches (training split)")
    ap.add_argument("--val-frac", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--limit", type=int, default=0, help="evaluate a subset (0 = all)")
    ap.add_argument("--max-drop", type=float, default=0.003,
                    help="reject if top-1 falls by more than this (fraction)")
    args = ap.parse_args()

    X, y, _ = load_processed(args.processed)
    train_idx, val_idx = stratified_split(y, args.val_frac, args.seed)
    if args.limit:
        val_idx = val_idx[:: max(1, len(val_idx) // args.limit)][: args.limit]

    print(f"fp32   evaluating on {len(val_idx):,} held-out sketches ...")
    fp1, fp5 = run(args.onnx, X, y, val_idx)
    print(f"fp32   top-1 {fp1:.4f}  top-5 {fp5:.4f}  ({args.onnx.stat().st_size / 1048576:.2f} MB)")

    input_name = ort.InferenceSession(
        str(args.onnx), providers=["CPUExecutionProvider"]
    ).get_inputs()[0].name

    if args.mode == "static":
        rng = np.random.default_rng(args.seed)
        calib_idx = rng.choice(train_idx, size=min(args.calib, len(train_idx)), replace=False)
        print(f"calibrating on {len(calib_idx):,} training sketches ...")
        quantize_static(
            str(args.onnx),
            str(args.out),
            Batches(X, calib_idx, input_name),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=CalibrationMethod.MinMax,
        )
    else:
        print("quantising (dynamic) ...")
        quantize_dynamic(
            str(args.onnx), str(args.out), weight_type=QuantType.QInt8,
            per_channel=True, reduce_range=False,
        )

    print("int8   evaluating ...")
    q1, q5 = run(args.out, X, y, val_idx)
    size = args.out.stat().st_size
    print(f"int8   top-1 {q1:.4f}  top-5 {q5:.4f}  ({size / 1048576:.2f} MB)")

    drop = fp1 - q1
    saved = 1 - size / args.onnx.stat().st_size
    print(f"\ntop-1 drop {drop * 100:+.2f} pp   size {saved * 100:.0f}% smaller")
    verdict = "ACCEPT" if drop <= args.max_drop else "REJECT"
    print(f"{verdict} (max drop allowed {args.max_drop * 100:.1f} pp)")

    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "fp32_top1": fp1, "fp32_top5": fp5,
                "int8_top1": q1, "int8_top5": q5,
                "drop_pp": drop * 100,
                "bytes": size,
                "saved_pct": saved * 100,
                "verdict": verdict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if verdict == "REJECT":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
