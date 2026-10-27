# SketchSense -- working notes

## The pitch
Draw something on the page; a convolutional net guesses it live. Because the model is
exported to ONNX and runs **in the browser** (WebGPU, WASM fallback), there is no backend:
inference is a few milliseconds, hosting is a static site, and it works offline.

## Why this is more than a tutorial
* trained from raw Google QuickDraw strokes (own rasteriser, own preprocessing contract)
* bbox-normalised raster pipeline shared verbatim between Python and TypeScript, verified
  by fixtures, so training and browser see the same tensor
* measured, not vibes: top-1 / top-5 on a held-out split, confusion pairs, latency and
  model size recorded in README
* ONNX export gated on a numerical parity check against PyTorch

## Repo layout
```
ml/      download.py raster.py prepare.py model.py train.py export.py
web/     Vite + React + TS + Tailwind front end, onnxruntime-web inference
docs/    plan, metrics, screenshots
```

## 5-day plan
| Day | Goal | Definition of done |
|---|---|---|
| 1 | pipeline + smoke train + full dataset download | 3-class train runs, 84-class data on disk |
| 2 | real training run, ONNX export + parity | val top-1 measured, model.onnx under ~6 MB |
| 3 | canvas UI, live top-5, shared rasteriser | drawing in browser matches Python tensor |
| 4 | game mode: 10s rounds, scorecards, polish | playable end to end, mobile-friendly |
| 5 | Cloudflare Pages deploy, README with numbers, demo GIF | public link on the dev site |

## Commands
```powershell
cd ml
..\.venv\Scripts\python.exe download.py --max-sketches 12000 --workers 4
..\.venv\Scripts\python.exe prepare.py --limit 12000
..\.venv\Scripts\python.exe train.py --epochs 12 --batch 256
..\.venv\Scripts\python.exe export.py
cd ..\web ; npm install ; npm run dev
```
