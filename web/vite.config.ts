/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// ビルド成果物は Python パッケージの static/ に出力する(wheel に同梱。利用者の環境に Node は要らない)。
// base: "./" … 相対パスで読み込む(外部 URL・絶対 URL を作らない)。
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: {
    outDir: "../src/yayoi_ocr/app/static",
    emptyOutDir: true,
    sourcemap: false,
    assetsInlineLimit: 0,
  },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
  },
});
