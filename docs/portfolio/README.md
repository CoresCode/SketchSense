# Portfolio screenshots

Real captures of the deployed app — driven with real mouse and keyboard input through
headless Chrome, not mockups and not composited. Regenerate any of them with:

```powershell
cd web && npm run preview                 # or npm run dev
node tools/portfolio-shots.mjs http://127.0.0.1:4173 docs/portfolio 1440 1210 2
node tools/portfolio-shots.mjs http://127.0.0.1:4173 docs/portfolio 1440 1210 2 free-draw
```

| file | what it shows |
|---|---|
| `free-draw.png` | A house drawn by hand, the live top-5 naming it "House" at 95.5%, 3.7 ms per guess |
| `speed-round.png` | A finished smiley face scoring 180 with 15.9 s left, confetti mid-burst |
| `ai-draws.png` | The network's own sketch of a fish, the typed guess, "yes — fish", and the model's own 99% call |

Originals are 2880×2420 (1440×1210 CSS at 2x). `web/` holds 1600 px PNG and WebP versions
for a page — the WebP files are 50–60 KB each.

The speed round's prompt is pinned to "smiley face" and the AI-draw target to "fish" so the
script knows what to draw and what to type; nothing else about the UI is forced. Everything
else — scores, confidences, timings, the 64×64 thumbnails — is whatever the model actually
produced at capture time.

If the model changes, recapture rather than reusing these: the confidence numbers are real
and they should stay real.
