import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    target: "es2022",
    rollupOptions: {
      input: {
        main: "index.html",
        check: "raster-check.html",
      },
    },
  },
  server: {
    // Bind IPv4 explicitly: on Windows, Vite's default `localhost` resolves to [::1] only,
    // which breaks tools that dial 127.0.0.1 (headless Chrome, curl, phones on the LAN).
    host: true,
    port: 5180,
  },
});
