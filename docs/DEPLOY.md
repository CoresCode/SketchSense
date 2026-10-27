# Deploying SketchSense

The whole site is static: HTML, JS, one 4 MB ONNX file and the ONNX Runtime wasm. No server,
no environment variables, no cold starts. That is a direct consequence of running inference
in the browser — it is why a free tier is genuinely enough here.

## 1. Build

```powershell
cd web
npm install          # also publishes the ORT runtime to public/ort via postinstall
npm run build        # -> web/dist
```

Sanity check before uploading:

```powershell
Get-ChildItem dist -Recurse -File |
  Sort-Object Length -Descending |
  Select-Object -First 5 @{n='MB';e={[math]::Round($_.Length/1MB,2)}},Name
```

Files must stay under Cloudflare Pages' **25 MiB per file** limit. The wasm runtime is the
biggest artefact (~10-20 MB depending on the ORT version); the model is a few MB.

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

## 3. Custom domain

In the Pages project → **Custom domains** → add `sketch` (or `sketchsense`) under
`anuragyadavcodes.com`. Because the domain is already on Cloudflare, the CNAME is created
automatically. HTTPS is issued within a minute or two.

Then link it from the portfolio site:

```html
<a href="https://sketch.anuragyadavcodes.com">SketchSense — draw it, a CNN names it</a>
```

## 4. Caching

`web/public/_headers` is copied into `dist/` and controls caching on Pages:

* `/ort/*` — immutable for a year (the runtime only changes when ORT is upgraded)
* `/model/*` — short-lived, because retraining replaces the file at the same path
* everything else — default

## Notes

* `dist/` is gitignored. The committed source of truth for the model is
  `web/public/model/{model.onnx,labels.json}`, produced by `ml/export.py`.
* If a deploy shows a blank page, check the console for a 404 on
  `/ort/ort-wasm-simd-threaded.wasm` — that means `postinstall` did not run, so run
  `node scripts/copy-ort.mjs` manually.
* Pages serves `.wasm` with `application/wasm`, so `WebAssembly.instantiateStreaming` works
  without extra configuration.
