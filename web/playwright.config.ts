import { defineConfig } from "@playwright/test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// E2E: Python のサーバを起動して Chromium で操作する(WebView2 は Chromium ベース)。
// 開発・E2E 専用に、ポートとトークンを固定する(通常の起動はランダム)。
const PORT = 8765;
export const TOKEN = "e2e-token-0123456789abcdef0123456789abcdef";
const dataDir = mkdtempSync(join(tmpdir(), "yayoi-e2e-"));

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    // ローカルで事前インストールの Chromium を使う場合は PW_CHROMIUM にパスを指定する
    launchOptions: process.env.PW_CHROMIUM ? { executablePath: process.env.PW_CHROMIUM } : {},
  },
  webServer: {
    command: `python -m yayoi_ocr.app --no-window --dev --port ${PORT} --token ${TOKEN} --data-dir "${dataDir}"`,
    url: `http://127.0.0.1:${PORT}/`,
    timeout: 60_000,
    reuseExistingServer: false,
  },
});
