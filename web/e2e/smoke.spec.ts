import { expect, test } from "./fixtures";

const TOKEN = "e2e-token-0123456789abcdef0123456789abcdef";

test("opens with the token in the fragment, connects, and hides the token", async ({ page }) => {
  const urls: string[] = [];
  page.on("request", (r) => urls.push(r.url()));
  await page.goto(`/#token=${TOKEN}`);
  await expect(page.getByTestId("connection")).toContainText("接続 OK");
  // フラグメントは URL から消える。トークンは storage / cookie に残らない
  expect(page.url()).not.toContain(TOKEN);
  const stored = await page.evaluate(() => JSON.stringify([localStorage, sessionStorage, document.cookie]));
  expect(stored).not.toContain(TOKEN);
  // 外部への通信が一切無い
  for (const u of urls) expect(u.startsWith("http://127.0.0.1:8765/") || u.startsWith("data:")).toBeTruthy();
  // 書き手のプロファイルは既定の「共通」だけ
  await expect(page.getByLabel("書き手のプロファイル")).toContainText("共通");
});

test("shows an error when the token is wrong", async ({ page }) => {
  await page.goto("/#token=wrong-token");
  await expect(page.getByTestId("connection")).toContainText("接続できません (401)");
});

test("shows an error when opened without a token", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("connection")).toContainText("トークンがありません");
});

test("serves the security headers and never CORS headers", async ({ request }) => {
  const res = await request.get("/");
  const h = res.headers();
  expect(h["content-security-policy"]).toContain("default-src 'self'");
  expect(h["x-frame-options"]).toBe("DENY");
  expect(Object.keys(h).filter((k) => k.startsWith("access-control-"))).toEqual([]);
  expect((await request.get("/api/health")).status()).toBe(401);
});
