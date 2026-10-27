import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    target: "es2022",
    // onnxruntime-web ships a ~25 MB wasm bundle; keep it as its own chunk so the
    // React shell paints before the runtime has finished downloading.
    rollupOptions: {
      input: {
        main: "index.html",
        check: "raster-check.html",
      },
      output: {
        manualChunks: {
          ort: ["onnxruntime-web"],
        },
      },
    },
  },
  server: {
    port: 5173,
  },
});
