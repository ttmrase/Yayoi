import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

const TOKEN = "e2e-token-0123456789abcdef0123456789abcdef";
const AUTH = { Authorization: `Bearer ${TOKEN}` };

const PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
  "base64",
);

interface ApiCell {
  id: number;
  row: number;
  col: number;
  kind: string;
  review: boolean;
  passed: boolean;
  reasons: string[];
}

async function fake(request: APIRequestContext, body: Record<string, unknown>) {
  const res = await request.post("/api/dev/fake", { headers: AUTH, data: body });
  expect(res.ok()).toBeTruthy();
}

async function apiPage(request: APIRequestContext, id: number) {
  const res = await request.get(`/api/pages/${id}`, { headers: AUTH });
  return (await res.json()) as { status: string; layout_status: string; layout_approved: boolean; cells: ApiCell[] };
}

/** 取り込みでジョブを作り、1 ページ目を開く(fake バックエンド) */
async function importAndOpen(page: Page, request: APIRequestContext, stamp: string) {
  await page.goto(`/?layout=${Date.now()}#token=${TOKEN}`);
  await expect(page.getByTestId("connection")).toContainText("接続 OK");
  await page.getByLabel("ジョブ名").fill(stamp);
  await page.getByTestId("file-input").setInputFiles([{ name: "p.png", mimeType: "image/png", buffer: PNG }]);
  const row = page.locator("tr", { hasText: stamp });
  await expect(row.getByTestId(/job-status-/)).toContainText("確認待ち", { timeout: 20_000 });
  await row.getByRole("button").click();
  await expect(page.getByTestId("review")).toBeVisible();
  const jobs = (await (await request.get("/api/jobs", { headers: AUTH })).json()) as { id: number; name: string }[];
  const job = jobs.find((j) => j.name === stamp)!;
  const detail = (await (await request.get(`/api/jobs/${job.id}`, { headers: AUTH })).json()) as { pages: { id: number }[] };
  return detail.pages[0].id;
}

const input = (page: Page, row: number, col: number) => page.getByTestId(`cell-${row}-${col}`).locator("input");
const activeLabel = (page: Page) => page.evaluate(() => document.activeElement?.getAttribute("aria-label") ?? "");

test.afterEach(async ({ request }) => {
  await fake(request, {});
});

test("⑫ 列モード(Ctrl+L): 1 列ずつ表示して上から下へ進み、列が終わると次の列へ。切り替えてもフォーカスは保たれる", async ({ page, request }) => {
  await fake(request, { layout_ok: true });
  const pid = await importAndOpen(page, request, `列${Date.now() % 100000}`);
  const cells = (await apiPage(request, pid)).cells;
  const nav = (c: ApiCell) => c.kind !== "blank" || c.review;
  const col0 = cells.filter((c) => c.col === 0 && nav(c)).sort((a, b) => a.row - b.row);
  const col1 = cells.filter((c) => c.col === 1 && nav(c)).sort((a, b) => a.row - b.row);

  await input(page, col0[0].row, 0).click();
  const labelBefore = await activeLabel(page);
  await page.keyboard.press("Control+l");
  await expect(page.getByTestId("mode-toggle")).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByTestId("column-tabs")).toBeVisible();
  // 日付の列だけが出ている(科目の列のセルは DOM に無い)
  await expect(page.getByTestId(`cell-${col1[0].row}-1`)).toHaveCount(0);
  expect(await activeLabel(page)).toBe(labelBefore); // フォーカスは保たれる
  // Enter で日付の列を上から下へ。最後のセルの次は科目の列の先頭
  for (let i = 0; i < col0.length; i++) {
    await page.waitForTimeout(300); // 人が画像を見る時間(要確認セルは速すぎる Enter では通らない)
    await page.keyboard.press("Enter");
  }
  await expect(input(page, col1[0].row, 1)).toBeFocused();
  await expect(page.getByTestId(`coltab-1`)).toHaveAttribute("aria-selected", "true");
  await expect(page.getByTestId(`cell-${col0[0].row}-0`)).toHaveCount(0);
  // 行モードへ戻しても、そのセルにいる
  const label = await activeLabel(page);
  await page.keyboard.press("Control+l");
  await expect(page.getByTestId("mode-toggle")).toHaveAttribute("aria-pressed", "false");
  await expect(page.getByTestId(`cell-${col0[0].row}-0`)).toBeVisible();
  expect(await activeLabel(page)).toBe(label);
});

test("⑫b Ctrl+↓ で次の未通過の要確認セルへ飛ぶ", async ({ page, request }) => {
  const pid = await importAndOpen(page, request, `飛躍${Date.now() % 100000}`);
  const cells = (await apiPage(request, pid)).cells.filter((c) => c.review && !c.passed);
  expect(cells.length).toBeGreaterThan(3);
  const first = cells.sort((a, b) => a.row - b.row || a.col - b.col);
  await input(page, first[0].row, first[0].col).click();
  await page.keyboard.press("Control+ArrowDown");
  await expect(input(page, first[1].row, first[1].col)).toBeFocused();
  await page.keyboard.press("Control+ArrowUp");
  await expect(input(page, first[0].row, first[0].col)).toBeFocused();
});

test("⑬ レイアウト要確認のページ: 重ね表示で隅をドラッグして切り直すと、要確認が外れ認識をやり直す", async ({ page, request }) => {
  await fake(request, { layout_ok: false });
  const pid = await importAndOpen(page, request, `罫線${Date.now() % 100000}`);
  await expect(page.getByTestId("layout-banner")).toBeVisible();
  const before = await apiPage(request, pid);
  expect(before.layout_status).toBe("needs_review");

  await page.getByRole("button", { name: "罫線を確認・補正" }).click();
  const panel = page.getByTestId("layout-panel");
  await expect(panel).toBeVisible();
  await expect(page.getByTestId("layout-state")).toContainText("自動検出(自信なし)");
  // 動かす前は切り直せず、「境界は合っている」は押せる
  await expect(page.getByTestId("layout-recut")).toBeDisabled();
  await expect(page.getByTestId("layout-approve")).toBeEnabled();
  // ② 右上の隅をドラッグ
  const corner = page.getByTestId("corner-1");
  const box = (await corner.boundingBox())!;
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  const before1 = await corner.getAttribute("aria-valuetext");
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  await page.mouse.move(cx - 30, cy + 25, { steps: 6 });
  await page.mouse.up();
  expect(await corner.getAttribute("aria-valuetext")).not.toBe(before1);
  await expect(page.getByTestId("layout-recut")).toBeEnabled();
  await expect(page.getByTestId("layout-approve")).toBeDisabled(); // 動かしたら承認ではなく切り直しを選ぶ
  // 元の位置に戻せる
  await page.getByRole("button", { name: "元の位置に戻す" }).click();
  expect(await corner.getAttribute("aria-valuetext")).toBe(before1);
  // キーボードでも動かせる(矢印、Shift で 10 倍)
  await corner.focus();
  await page.keyboard.press("Shift+ArrowLeft");
  await page.keyboard.press("ArrowDown");
  expect(await corner.getAttribute("aria-valuetext")).not.toBe(before1);
  // 切り直す(確認のボタンを経る)
  await page.getByTestId("layout-recut").click();
  await page.getByTestId("layout-recut-confirm").click();
  await expect(panel).toHaveCount(0);
  await expect(page.getByTestId("review")).toBeVisible({ timeout: 20_000 });
  await expect(page.getByTestId("layout-banner")).toHaveCount(0);
  const after = await apiPage(request, pid);
  expect(after.layout_approved).toBeTruthy();
  expect(after.cells.some((c) => c.reasons.includes("page_needs_review"))).toBeFalsy();
  // 開き直すと「手動で指定済み」
  await page.getByTestId("layout-open").click();
  await expect(page.getByTestId("layout-state")).toContainText("手動で指定済み");
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("layout-panel")).toHaveCount(0);
});

test("⑭ 「境界は合っている」でレイアウト由来の要確認だけが外れる", async ({ page, request }) => {
  await fake(request, { layout_ok: false });
  const pid = await importAndOpen(page, request, `承認${Date.now() % 100000}`);
  await expect(page.getByTestId("layout-banner")).toBeVisible();
  await page.getByRole("button", { name: "罫線を確認・補正" }).click();
  await page.getByTestId("layout-approve").click();
  await expect(page.getByTestId("layout-panel")).toHaveCount(0);
  await expect(page.getByTestId("layout-banner")).toHaveCount(0);
  const after = await apiPage(request, pid);
  expect(after.layout_approved).toBeTruthy();
  expect(after.cells.some((c) => c.reasons.includes("page_needs_review"))).toBeFalsy();
  expect(after.cells.some((c) => c.review)).toBeTruthy(); // 認識自体の要確認(自信が低い等)は残る
});
