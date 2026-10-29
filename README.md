# SketchSense

**Draw a doodle, a convolutional network names it — in your browser, with no server.**

SketchSense is a small image classifier trained from scratch on Google's
[QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset) dataset and exported to
ONNX so it runs locally in the page via ONNX Runtime Web (WASM). Nothing is uploaded, there
is no API key, and hosting is a static folder.

<!-- numbers are updated by `python ml/report.py` after each training run -->

| | |
|---|---|
| Classes | `{{classes}}` QuickDraw categories |
| Model | 4-stage residual CNN, **1.6 M parameters** |
| Held-out top-1 | **{{val_top1}}** |
| Held-out top-5 | **{{val_top5}}** |
| Inference | **{{latency}} ms** per guess (WASM, single thread) |
| Model file | **{{model_mb}} MB** fp32 ONNX |
| Backend | none |

![SketchSense](docs/screenshot.png)

---

## Three ways to play

| Mode | What it does |
|---|---|
| **Free draw** | Draw anything; the top-5 updates live as you draw, with confidence bars and a measured inference time. |
| **Speed round** | Fixed time per doodle, points for speed, and the guess only counts above 35% confidence *and* after the pen has been still for 1.2 s — a half-finished doodle gets a chance to become a doodle. |
| **AI draws** | The model plays the other side: real QuickDraw sketches are replayed stroke by stroke and you type what you think it is. |

**AI draws** deserves a note on honesty: it is not a generative model. The sketches are real
human drawings from the held-out data, animated in stroke order. What *is* model-driven is the
difficulty — every candidate is scored by the classifier against its own label, so **easy** uses
the sketches it is most certain about, and **hard** samples the ambiguous 20th–35th percentile
and adds a wobble so the line looks like a hurried human hand.

Guessing is free text with autocomplete and one-typo tolerance, not multiple choice, so it does
not turn into a 4-way coin flip.

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

**5. Speed came from looking, not guessing.**
The first version trained at 990 images/s. Giving the stem convolution stride 2 — the input
stays 64×64, the stem just halves resolution once, up front — took it to **4,033 images/s
(4.07×)** for roughly a point of accuracy at epoch 1. Batch size was swept as well (128, 256,
512, 1024); 256 won.

Two smaller wins followed, both from profiling rather than intuition: the running loss and
accuracy now accumulate on the GPU and are read once per 100 steps instead of every step
(a per-step read forces a device sync), and batches cross PCIe as `uint8` and are converted on
the card — 4× less traffic. Host-side step prep went from 6.0 ms to 0.9 ms in isolation.
The scripts that produced these numbers are in `ml/sweep_arch.py`, `ml/tune_batch.py`,
`ml/profile_input.py` and `ml/profile_step.py`.

## Run it

```powershell
# model (already committed, but this is how it is produced)
cd ml
..\.venv\Scripts\python.exe download.py --max-sketches 12000 --workers 4
..\.venv\Scripts\python.exe prepare.py --limit 12000

# 20 epochs from scratch, then a short low-LR warm restart for the last few points
..\.venv\Scripts\python.exe train.py --epochs 20 --batch 256 --lr 0.002 --stem-stride 2 --out artifacts\run2
..\.venv\Scripts\python.exe train.py --epochs 6 --batch 256 --lr 0.0004 --stem-stride 2 `
    --init-from artifacts\run2\best.pt --out artifacts\run2b

..\.venv\Scripts\python.exe export.py --ckpt artifacts\run2b\best.pt   # ONNX + parity gate
..\.venv\Scripts\python.exe export_samples.py --ckpt artifacts\run2b\best.pt  # AI-draw sketches
..\.venv\Scripts\python.exe report.py --ckpt artifacts\run2b\best.pt   # metrics + README numbers
..\.venv\Scripts\python.exe fixtures.py

# front end
cd ..\web
npm install
npm run dev      # http://127.0.0.1:5180
npm run build    # static output in dist/
```

The repo ships `web/public/model/model.onnx`, so `npm run dev` works without retraining.

## Test it like a user, not like a unit

`tools/qa.mjs` drives headless Chrome through all three modes with real mouse and keyboard
input, then fails on console errors, failed requests, a wrong top-1, slow inference, or a page
that scrolls sideways at 390 px. `tools/test_speed_round.mjs` is the focused regression test
for the scoring rule above: it draws a smiley face in the order a person would and asserts
nothing scores until the mouth is on and the pen has been down for the settle window.

```powershell
npm run typecheck                       # in web/
node tools/qa.mjs --url http://127.0.0.1:5180
node tools/test_speed_round.mjs --url http://127.0.0.1:5180
```

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
* 100 of QuickDraw's 345 categories are trained. Adding more is a data change, not a code
  change: edit `ml/classes.txt`, rerun the pipeline, retrain, re-export. The browser reads
  whatever `labels.json` and `model.onnx` it is given.

## Credits

Dataset: [Google QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset), released
under CC BY 4.0. Model, pipeline and UI by [Anurag Yadav](https://anuragyadavcodes.com).
