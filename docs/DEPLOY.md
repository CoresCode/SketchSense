# Deploying SketchSense

The whole site is static: HTML, JS, one 1.6 MB ONNX file and the ONNX Runtime wasm. No server,
no environment variables, no cold starts. That is a direct consequence of running inference in
the browser — it is why a free tier is genuinely enough here.

**Cost: £0.** Cloudflare Pages' free plan includes unlimited static sites, unlimited bandwidth
and unlimited custom domains. There is no card on file, no trial, no "free for 12 months".

## 0. Make a Cloudflare account (once, ~2 minutes)

1. Go to <https://dash.cloudflare.com/sign-up>.
2. Email + password, then click the verification link they send.
3. That is it — you land on the dashboard with no sites. You do **not** need to add a domain or
   a payment method to publish a `*.pages.dev` site.

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
biggest artefact (14.2 MB); the model is 1.6 MB.

## 2. Publish (drag and drop, no CLI)

1. Dashboard → **Workers & Pages** (left sidebar) → **Create** → **Pages** tab →
   **Upload assets**.
2. Project name: `sketchsense` → **Create project**.
3. Drag the **contents** of `web/dist` into the box — i.e. `index.html`, `assets/`, `model/`,
   `favicon.svg`, `_headers`, `og.png`, `raster-check.html`. Drag the *contents*, not the
   `dist` folder itself, or the site will live at `/dist/` and 404 at the root.
4. **Deploy site**. First build takes ~10 seconds.
5. You get `https://sketchsense.pages.dev` (or `sketchsense-abc.pages.dev` if the name is
   taken). Open it, draw a house, check it guesses "House".

To update later: same project → **Create new deployment** → drag the new `dist` contents.
Every deploy is kept, so a bad one is one click to roll back.

## 3. Link previews

`web/public/og.png` (1200x630, built from the hero shot by `ml/og.py`) is what Slack, Discord
and LinkedIn show for the link. `web/index.html` currently points at it with a relative path,
which is enough for the page itself; scrapers want an absolute URL, so once the hostname is
known change the two `content="/og.png"` values to `https://<host>/og.png` and redeploy.

## 4. Custom domain (free, if you already own the name)

A subdomain of a domain you already own costs nothing on Pages — Cloudflare does not charge for
custom domains on the free plan. The only thing that ever costs money is *registering* the
domain, which is already done for `corescode.com`.

1. Add the domain to Cloudflare (once): **Add a site** → `corescode.com` → Free plan →
   Cloudflare gives you two nameservers → set them at your registrar → wait for "Active".
2. In the Pages project → **Custom domains** → **Set up a custom domain** →
   `sketch.corescode.com` → **Activate**.
3. Because the zone is already on Cloudflare, the CNAME is created for you. HTTPS is issued
   within a minute or two.

If the domain is registered *through* Cloudflare already, step 1 is skipped entirely.

Then link it from the portfolio site:

```html
<a href="https://sketch.corescode.com">SketchSense — draw it, a CNN names it</a>
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
* Alternative for repeat deploys: `npx wrangler pages deploy dist --project-name sketchsense`
  after a one-time `npx wrangler login`. Same result, one command.
