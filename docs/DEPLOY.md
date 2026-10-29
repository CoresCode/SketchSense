# Deploying SketchSense

The whole site is static: HTML, JS, one 6.4 MB ONNX file and the ONNX Runtime wasm. No server,
no environment variables, no cold starts. That is a direct consequence of running inference
in the browser — it is why a free tier is genuinely enough here.

## 1. Build

```powershell
cd web
npm install          # vite, react, onnxruntime-web
npm run build        # -> web/dist (~21 MB total)
```

Sanity check before uploading:

```powershell
Get-ChildItem dist -Recurse -File |
  Sort-Object Length -Descending |
  Select-Object -First 5 @{n='MB';e={[math]::Round($_.Length/1MB,2)}},Name
```

Files must stay under Cloudflare Pages' **25 MiB per file** limit. The wasm runtime is the
biggest artefact (~15 MB depending on the ORT version); the model is 6.4 MB.

## 2. Publish

**Option A — drag and drop (fastest, no CLI login)**

Cloudflare dashboard → **Workers & Pages** → **Create** → **Pages** → **Upload assets** →
project name `sketchsense` → drop the **contents of `dist/`** → Deploy.
You get `https://sketchsense.pages.dev`.

**Option B — wrangler (repeatable)**

```powershell
npx wrangler pages deploy dist --project-name sketchsense
```

Wrangler opens a browser once for auth; after that every deploy is one command.

## 3. Link previews

`web/public/og.png` (1200x630, built from the hero shot by `ml/og.py`) is what Slack,
Discord and LinkedIn show for the link. `web/index.html` points at it with a relative path,
which works for the page itself; scrapers want an absolute URL, so once the final hostname is
known change the two `content="/og.png"` values to `https://<host>/og.png`.

## 4. Custom domain

In the Pages project → **Custom domains** → add `sketch` (or `sketchsense`) under
`anuragyadavcodes.com`. Because the domain is already on Cloudflare, the CNAME is created
automatically. HTTPS is issued within a minute or two.

Then link it from the portfolio site:

```html
<a href="https://sketch.anuragyadavcodes.com">SketchSense — draw it, a CNN names it</a>
```

## 5. Caching

`web/public/_headers` is copied into `dist/` and controls caching on Pages:

* `/model/*` — short-lived, because retraining replaces the file at the same path
* everything else — default (hashed assets get immutable caching from Cloudflare anyway)

## Notes

* `dist/` is gitignored. The committed source of truth for the model is
  `web/public/model/{model.onnx,labels.json,game-classes.json}`, produced by `ml/export.py`
  and `ml/report.py`.
* `onnxruntime-web/wasm` resolves and bundles its own `.wasm` file, so the runtime lands in
  `dist/assets/` with a content hash. There is no separate copy to maintain — an earlier
  version copied the same 14 MB into `public/ort/` as well, which just doubled the upload.
* Pages serves `.wasm` with `application/wasm`, so `WebAssembly.instantiateStreaming` works
  without extra configuration.
