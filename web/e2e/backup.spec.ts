import { readFileSync } from "node:fs";
import AxeBuilder from "@axe-core/playwright";
import { type APIRequestContext, type Page } from "@playwright/test";
import { expect, test } from "./fixtures";

// バックアップと復元(設定画面)。保存は「名前を付けて保存」(ここでは Playwright のダウンロードイベント)。
// 復元の入れ替えは次の起動の最初なので、ここでは「準備ができた」まで確かめて取り消す(共有サーバのデータは変えない)。
const TOKEN = "e2e-token-0123456789abcdef0123456789abcdef";
const AUTH = { Authorization: `Bearer ${TOKEN}` };

// ファイルのパスにテスト名(日本語)が入ると、ヘッドレスの Chromium が選択を取りこぼすことがあるので、中身を渡す
const zipPayload = (buffer: Buffer) => ({ name: "backup.zip", mimeType: "application/zip", buffer });

const tab = (page: Page, name: string) => page.getByRole("navigation", { name: "画面" }).getByRole("button", { name });

async function openSettings(page: Page) {
  await page.goto(`/?bk=${Date.now()}#token=${TOKEN}`);
  await expect(page.getByTestId("connection")).toContainText("接続 OK");
  await tab(page, "設定").click();
  await expect(page.getByTestId("backup")).toBeVisible();
}

async function confirmedPage(request: APIRequestContext, seed: number) {
  const res = await request.post("/api/dev/jobs", {
    headers: AUTH,
    data: { pages: 1, seed, fill_rows: 6, error_rate: 0, review_rate: 0, ambiguous_rows: [] },
  });
  expect(res.ok()).toBeTruthy();
  const job = (await res.json()) as { pages: { id: number }[] };
  const c = await request.post(`/api/pages/${job.pages[0].id}/confirm`, { headers: AUTH, data: {} });
  expect(c.ok()).toBeTruthy();
}

test.afterEach(async ({ request }) => {
  await request.post("/api/backup/cancel", { headers: AUTH, data: {} });
  await request.delete("/api/backup/file", { headers: AUTH });
});

test("㉟ バックアップを作成して保存でき、その zip を選ぶと検査の要約が出て、復元の準備まで進み、取り消せる", async ({ page, request }, testInfo) => {
  await confirmedPage(request, 3501);
  await openSettings(page);
  await page.getByTestId("backup-create").click();
  await expect(page.getByTestId("backup-created")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("backup-created-summary")).toContainText("確定済み");
  await expect(page.getByTestId("backup-created-summary")).toContainText("元画像は含まない");

  // 保存(ダウンロード)
  const [download] = await Promise.all([page.waitForEvent("download"), page.getByTestId("backup-save").click()]);
  expect(download.suggestedFilename()).toMatch(/^yayoi-backup-\d{8}-\d{6}\.zip$/);
  const saved = testInfo.outputPath("backup.zip");
  await download.saveAs(saved);
  expect(readFileSync(saved).subarray(0, 2).toString()).toBe("PK");

  // 復元: 作ったバックアップを選ぶ → 検査 → 要約 → 確認 → 準備ができた
  await page.getByTestId("backup-discard").click();
  await expect(page.getByTestId("backup-created")).toHaveCount(0);
  await page.getByTestId("backup-file").setInputFiles(zipPayload(readFileSync(saved)));
  await expect(page.getByTestId("backup-incoming")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("backup-incoming-summary")).toContainText("確定済み");
  await page.getByTestId("backup-restore").click();
  await expect(page.getByTestId("backup-confirm")).toContainText("置き換わります");
  await page.getByTestId("backup-restore-go").click();
  await expect(page.getByTestId("backup-pending")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("backup-pending")).toContainText("開き直す");
  // 取り消す
  await page.getByTestId("backup-cancel-restore").click();
  await expect(page.getByTestId("backup-pending")).toHaveCount(0);
});

test("㊱ zip でないファイル・書き換えられた zip は、理由つきで断られ、何も残らない", async ({ page, request }, testInfo) => {
  await openSettings(page);
  const junk = testInfo.outputPath("junk.zip");
  const fs = await import("node:fs");
  fs.writeFileSync(junk, Buffer.from("PK\u0003\u0004 this is not really a zip".repeat(20)));
  await page.getByTestId("backup-file").setInputFiles(zipPayload(readFileSync(junk)));
  await expect(page.getByTestId("backup-error")).toContainText("zip として読めません", { timeout: 30_000 });
  await expect(page.getByTestId("backup-incoming")).toHaveCount(0);
  const st = (await (await request.get("/api/backup", { headers: AUTH })).json()) as { incoming: unknown; pending: unknown };
  expect(st.incoming).toBeNull();
  expect(st.pending).toBeNull();

  // 作成したバックアップの 1 バイトを書き換える → 検査で断られる
  await confirmedPage(request, 3502);
  await openSettings(page);
  await page.getByTestId("backup-create").click();
  await expect(page.getByTestId("backup-created")).toBeVisible({ timeout: 30_000 });
  const [download] = await Promise.all([page.waitForEvent("download"), page.getByTestId("backup-save").click()]);
  const saved = testInfo.outputPath("good.zip");
  await download.saveAs(saved);
  const buf = readFileSync(saved);
  // 先頭のファイル(DB)の圧縮データの中の 1 バイトを反転する(zip の CRC / ハッシュで必ず検出される場所。
  // ファイルの中ほどだと、日時などの無関係なバイトに当たることがある)
  const nameLen = buf.readUInt16LE(26);
  const extraLen = buf.readUInt16LE(28);
  buf[30 + nameLen + extraLen + 8] ^= 0xff;
  const bad = testInfo.outputPath("tampered.zip");
  fs.writeFileSync(bad, buf);
  await page.getByTestId("backup-discard").click();
  await page.getByTestId("backup-file").setInputFiles(zipPayload(readFileSync(bad)));
  await expect(page.getByTestId("backup-error")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("backup-incoming")).toHaveCount(0);
});

test("㊲ バックアップと復元の欄に、アクセシビリティ上の重大な違反がない", async ({ page, request }) => {
  await confirmedPage(request, 3503);
  await openSettings(page);
  await page.getByTestId("backup-create").click();
  await expect(page.getByTestId("backup-created")).toBeVisible({ timeout: 30_000 });
  const res = await new AxeBuilder({ page }).include('[data-testid="backup"]').analyze();
  const serious = res.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
  expect(serious, JSON.stringify(serious.map((v) => v.id))).toEqual([]);
});
