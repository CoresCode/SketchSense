# SketchSense — build notes

Notes from the week this was built. Kept short and factual; the README has the pitch.

## What it is

Draw a doodle, a CNN names it live, in the tab. The model is trained from raw Google
QuickDraw strokes, exported to ONNX, and runs in the page through ONNX Runtime Web. There is
no backend: a static folder is the whole deployment.

## Decisions and why

| Decision | Reason |
|---|---|
| **Single-threaded WASM**, not WebGPU | One ~15 MB runtime artifact instead of a WebGPU build, no COOP/COEP headers, works on any static host. A 1.6 M-parameter net at 64×64 runs in ~12 ms; the GPU would be wasted. |
| **64×64 greyscale**, bbox-normalised | Matches the shape of the training data (line art). Small enough that 1.2 M sketches fit on a laptop and the browser tensor is 4 KB. |
| **Stride-2 stem** | Keeps the 64×64 input contract, halves resolution once at the input instead of paying for it in every block. 4.07× faster training. |
| **100 of 345 QuickDraw classes** | Enough variety for the game to feel open-ended without making the confusion matrix a blur. More is a data change (`ml/classes.txt`), not a code change. |
| **AI-draw replays real sketches** | A generative model is a different project. Replaying held-out human sketches is honest, tiny (0.37 MB), and the *difficulty* is still model-driven: candidates are ranked by the classifier's own confidence. |
| **Free-text guessing** with autocomplete | Four buttons would make the game a coin flip. One-typo tolerance keeps it fair without a full spell corrector. |

## Numbers that drove the work

| Measurement | Result |
|---|---|
| Dataset | 1,200,000 sketches, 100 classes, 12k each; 4.6 GiB as `uint8` |
| Throughput, stride-1 stem | 990 images/s |
| Throughput, stride-2 stem | 4,033 images/s (**4.07×**) |
| Epoch time (RTX 5060 laptop, batch 256) | ~325 s |
| Held-out top-1 / top-5 | see README and `docs/metrics.md` |
| Browser inference | ~12 ms, single-threaded WASM |
| Shipped artifacts | model 6.4 MB, samples 0.37 MB, ORT wasm ~15 MB |

## Things that were wrong at first

* **"The GPU is starving."** It was not. My own profiling scripts were competing for the GPU
  and inflating step time by ~65%. The training loop did have a real inefficiency (a
  per-step device sync), but it was worth a few percent, not a rewrite.
* **`taco` is not a QuickDraw category** — the download 404'd. Replaced with `cactus`.
* **A confident sketch is not always a guessable sketch.** The first easy tier picked
  one-stroke squiggles the model was sure about and no human could name. Easy now requires at
  least two strokes; hard deliberately samples the ambiguous middle (20th–35th percentile),
  not the worst sketches, which are simply unrecognisable.

## Repo map

```
ml/      download.py raster.py prepare.py model.py train.py export.py
         export_samples.py report.py fixtures.py + profiling scripts
web/     Vite + React + TS + Tailwind UI, ONNX Runtime Web inference
docs/    metrics, confusion matrix, screenshots, deploy notes
tools/   capture.mjs (headless screenshots), wait_training.ps1
```

## Commands

```powershell
cd ml
..\.venv\Scripts\python.exe download.py --max-sketches 12000 --workers 4
..\.venv\Scripts\python.exe prepare.py --limit 12000
..\.venv\Scripts\python.exe train.py --epochs 20 --batch 256 --lr 0.002 --stem-stride 2 --out artifacts\run2
..\.venv\Scripts\python.exe train.py --epochs 6 --batch 256 --lr 0.0004 --stem-stride 2 `
    --init-from artifacts\run2\best.pt --out artifacts\run2b
..\.venv\Scripts\python.exe export.py --ckpt artifacts\run2b\best.pt
..\.venv\Scripts\python.exe export_samples.py --ckpt artifacts\run2b\best.pt
..\.venv\Scripts\python.exe report.py --ckpt artifacts\run2b\best.pt
cd ..\web ; npm install ; npm run dev
```
