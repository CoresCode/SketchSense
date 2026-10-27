# SketchSense

**Draw a doodle, a convolutional network names it — in your browser, with no server.**

SketchSense is a small image classifier trained from scratch on Google's
[QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset) dataset and exported to
ONNX so it runs locally in the page via ONNX Runtime Web (WASM). Nothing is uploaded, there
is no API key, and hosting is a static folder.

<!-- numbers are updated by `python ml/report.py` after each training run -->

| | |
|---|---|
| Classes | `84` QuickDraw categories |
| Model | 4-stage residual CNN, **1.6 M parameters** |
| Held-out top-1 | **{{val_top1}}** |
| Held-out top-5 | **{{val_top5}}** |
| Inference | **{{latency}} ms** per guess (WASM, single thread) |
| Model file | **{{model_mb}} MB** fp32 ONNX |
| Backend | none |

![SketchSense](docs/screenshot.png)

---

## Why this exists

I wanted a portfolio project where the model is *actually mine* and the thing is *actually
fun to use*. Doodle classification is a good fit: the data is free and enormous, the task is
visually obvious (so a demo needs no explanation), and the compute is small enough that the
whole pipeline — data, training, export, deployment — fits in one repo and runs on a laptop.

## How it works

```
QuickDraw NDJSON (raw pen strokes)
        │  ml/download.py    stream + early-stop at 12k sketches/class
        ▼
ml/data/raw/*.ndjson
        │  ml/prepare.py     rasterise 64x64, bbox-normalised
        ▼
ml/data/processed/*.npy     uint8 (N, 64, 64)
        │  ml/train.py       AMP, OneCycle, GPU affine augmentation
        ▼
ml/artifacts/best.pt
        │  ml/export.py      ONNX opset 17 + parity gate vs PyTorch
        ▼
web/public/model/model.onnx ──► onnxruntime-web (WASM) ──► canvas UI
```

The browser replicates the Python preprocessing exactly (`web/src/lib/raster.ts` mirrors
`ml/raster.py`), because a silent preprocessing mismatch is the most common way a model that
scores well offline looks broken in production.

## The parts worth reading

**1. Preprocessing is a contract, and it is tested.**
Both rasterisers: drop strokes shorter than 2 points → take the bounding box → centre it →
scale the longest side to `canvas - 2*pad` → draw at 256×256 → box-filter 4×4 down to 64×64.
`ml/fixtures.py` freezes 24 real sketches with the exact tensor Python produced, and
`/raster-check.html` asserts the browser produces the same shape.

**2. The export cannot silently ship a broken graph.**
`ml/export.py` runs 512 validation images through both PyTorch and ONNX Runtime and exits
non-zero unless top-1 agreement is 100% and max logit drift is < 0.01. BatchNorm folding and
resize alignments are exactly the kind of thing that changes behaviour during export.

**3. Inference runs where the user is.**
Single-threaded WASM was chosen over WebGPU on purpose: one ~15 MB artifact instead of a
jsep build, no COOP/COEP headers, and a 1.6 M-parameter net at 64×64 is fast enough that the
GPU would be wasted. The whole site is static files.

**4. It is measured, not vibed.**
Top-1 / top-5 on a stratified held-out split, a confusion-matrix export, and per-class
accuracy in `docs/metrics.md`. The failure cases are the interesting part.

## Run it

```powershell
# model (already committed, but this is how it is produced)
cd ml
..\.venv\Scripts\python.exe download.py --max-sketches 12000 --workers 4
..\.venv\Scripts\python.exe prepare.py --limit 12000
..\.venv\Scripts\python.exe train.py --epochs 12 --batch 256
..\.venv\Scripts\python.exe export.py
..\.venv\Scripts\python.exe fixtures.py

# front end
cd ..\web
npm install
npm run dev      # http://127.0.0.1:5180
npm run build    # static output in dist/
```

The repo ships `web/public/model/model.onnx`, so `npm run dev` works without retraining.

## Repo layout

```
ml/      data pipeline, model, training, export, parity fixtures
web/     Vite + React + TS + Tailwind UI, ONNX Runtime Web inference
docs/    plan, metrics, screenshots
```

## Limitations

* 64×64 greyscale is the ceiling: fine detail (a whisker vs a spider leg) is lost, which shows
  up in the confusion pairs.
* QuickDraw is line art drawn by thousands of people; photos of real objects are out of
  distribution and the model is not claimed to handle them.
* No temporal/state modelling of stroke order — two visually identical doodles drawn in a
  different order look the same to the model. That is a deliberate scope cut, not an oversight.

## Credits

Dataset: [Google QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset), released
under CC BY 4.0. Model, pipeline and UI by [Anurag Yadav](https://anuragyadavcodes.com).
