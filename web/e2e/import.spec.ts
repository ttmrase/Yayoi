import { type APIRequestContext, type Page } from "@playwright/test";
import { expect, test } from "./fixtures";

const TOKEN = "e2e-token-0123456789abcdef0123456789abcdef";
const AUTH = { Authorization: `Bearer ${TOKEN}` };

// 1x1 の白い PNG(実データではない。fake バックエンドは画像の中身を読まない)
const PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
  "base64",
);
const png = (name: string) => ({ name, mimeType: "image/png", buffer: PNG });

async function fake(request: APIRequestContext, body: { delay?: number; fail_next?: number }) {
  const res = await request.post("/api/dev/fake", { headers: AUTH, data: body });
  expect(res.ok()).toBeTruthy();
}

async function openJobs(page: Page) {
  await page.goto(`/?import=${Date.now()}#token=${TOKEN}`);
  await expect(page.getByTestId("connection")).toContainText("接続 OK");
}

test.afterEach(async ({ request }) => {
  await fake(request, {}); // 他のテストに待ち時間・失敗を持ち越さない
});

test("⑨ ファイルを取り込むと、認識中の表示を経て確認画面で開ける(複数ファイルは番号の順)", async ({ page, request }) => {
  await fake(request, { delay: 1.2 });
  await openJobs(page);
  const stamp = `取込${Date.now() % 100000}`;
  await page.getByLabel("ジョブ名").fill(stamp);
  // 選んだ順は scan10, scan2 → ページは scan2, scan10 の順(自然順)
  await page.getByTestId("file-input").setInputFiles([png("scan10.png"), png("scan2.png")]);
  const row = page.locator("tr", { hasText: stamp });
  await expect(row).toBeVisible();
  await expect(row.getByTestId(/job-status-/)).toContainText("認識中");
  await expect(row.getByRole("button")).toBeDisabled(); // 1 ページ目ができるまで開けない
  await expect(row.getByTestId(/job-status-/)).toContainText("確認待ち", { timeout: 20_000 });
  await expect(row.getByRole("button")).toBeEnabled();
  await row.getByRole("button").click();
  await expect(page.getByTestId("review")).toBeVisible();
  await expect(page.getByTestId("page-no")).toContainText("ページ 1 / 2");
  // 認識結果の 1 行目が出ている(年を継承した日付は「展開済み」のバッジが付く)
  await expect(page.getByTestId("cell-1-0")).toBeVisible();
  await expect(page.getByTestId("badge-1-0")).toContainText("展開済み");
});

test("⑩ 取り込めないファイルは理由を出し、取り込めるものだけでジョブを作る", async ({ page }) => {
  await openJobs(page);
  const stamp = `混在${Date.now() % 100000}`;
  await page.getByLabel("ジョブ名").fill(stamp);
  // accept 属性は選択ダイアログの絞り込みだけなので、ドロップ相当の経路(setInputFiles)では通る
  await page.getByTestId("file-input").setInputFiles([
    png("ok.png"),
    { name: "memo.txt", mimeType: "text/plain", buffer: Buffer.from("hello") },
    { name: "broken.png", mimeType: "image/png", buffer: Buffer.from("not an image") },
  ]);
  const failures = page.getByTestId("import-failures");
  await expect(failures).toContainText("memo.txt: 対応していない形式です");
  await expect(failures).toContainText("broken.png: 読み込めないファイルです");
  const row = page.locator("tr", { hasText: stamp });
  await expect(row.getByTestId(/job-status-/)).toContainText("確認待ち", { timeout: 20_000 });
  await expect(row.locator("td").nth(1)).toHaveText("1"); // 取り込めた 1 ページだけ
});

test("⑪ 認識に失敗したページは理由と再試行を出し、再試行で確認できるようになる", async ({ page, request }) => {
  await fake(request, { fail_next: 1 });
  await openJobs(page);
  const stamp = `失敗${Date.now() % 100000}`;
  await page.getByLabel("ジョブ名").fill(stamp);
  await page.getByTestId("file-input").setInputFiles([png("a.png")]);
  const row = page.locator("tr", { hasText: stamp });
  await expect(row.getByTestId(/job-status-/)).toContainText("失敗 1", { timeout: 20_000 });
  await row.getByRole("button").click();
  await expect(page.getByTestId("review-pending")).toContainText("認識に失敗しました");
  await page.getByTestId("retry-page").click();
  await expect(page.getByTestId("review")).toBeVisible({ timeout: 20_000 });
  await expect(page.getByTestId("page-no")).toContainText("ページ 1 / 1");
});
