import AxeBuilder from "@axe-core/playwright";
import { type APIRequestContext, type Page } from "@playwright/test";
import { expect, test } from "./fixtures";

const TOKEN = "e2e-token-0123456789abcdef0123456789abcdef";
const AUTH = { Authorization: `Bearer ${TOKEN}` };
const ACCOUNTS = ["現金", "普通預金", "未払金", "売掛金"];
const STAMP = "切手代";
const DEFAULT = "課対仕入込10%";

const tab = (page: Page, name: string) => page.getByRole("navigation", { name: "画面" }).getByRole("button", { name });
const input = (page: Page, row: number, col: number) => page.getByTestId(`cell-${row}-${col}`).locator("input");

async function openApp(page: Page) {
  await page.goto(`/?tax=${Date.now()}#token=${TOKEN}`);
  await expect(page.getByTestId("connection")).toContainText("接続 OK");
}

async function setup(request: APIRequestContext) {
  const res = await request.put("/api/settings", {
    headers: AUTH,
    data: {
      wizard_done: true,
      payments: ACCOUNTS.map((a) => ({ name: a, credit_account: a, credit_sub: "" })),
      titles: [{ name: "旅費交通費", debit_account: "旅費交通費", debit_sub: "" }],
      default_tax: [{ account: "旅費交通費", tax_category: DEFAULT }],
    },
  });
  expect(res.ok()).toBeTruthy();
  await request.post("/api/tax-memory/clear", { headers: AUTH });
}

/** 1 ページのジョブを作り、row 0 の摘要を指定の値にする */
async function job(request: APIRequestContext, seed: number, desc: string) {
  const res = await request.post("/api/dev/jobs", {
    headers: AUTH,
    data: { pages: 1, seed, fill_rows: 5, error_rate: 0, review_rate: 0, ambiguous_rows: [] },
  });
  expect(res.ok()).toBeTruthy();
  const j = (await res.json()) as { job_id: number; pages: { id: number }[] };
  const pid = j.pages[0].id;
  const pg = (await (await request.get(`/api/pages/${pid}`, { headers: AUTH })).json()) as { cells: { id: number; row: number; col: number }[] };
  const cell = pg.cells.find((c) => c.row === 0 && c.col === 2)!;
  const r = await request.patch(`/api/pages/${pid}/cells/${cell.id}`, { headers: AUTH, data: { action: "edit", value: desc } });
  expect(r.ok()).toBeTruthy();
  await request.put(`/api/pages/${pid}/title`, { headers: AUTH, data: { title: "旅費交通費" } });
  return { jobId: j.job_id, pid };
}

async function openJob(page: Page, jobId: number) {
  await tab(page, "ジョブ").click();
  await page.getByTestId(`job-${jobId}`).getByRole("button").click();
  await expect(page.getByTestId("review")).toBeVisible();
}

test("㉖ F2 で行の税区分を選び、確定すると摘要を記憶して、次のページで「提案」と表示される。記憶は設定画面で編集・削除できる", async ({ page, request }) => {
  await setup(request);
  const a = await job(request, 921, STAMP);
  await openApp(page);
  await openJob(page, a.jobId);
  // 既定が出ている
  await expect(page.getByTestId("taxcell-0")).toHaveAttribute("data-source", "default");
  await expect(page.getByTestId("taxcell-0")).toContainText(DEFAULT);
  // 摘要のセルで F2 → 候補リスト(キーボードだけで操作)
  await input(page, 0, 2).click();
  await page.keyboard.press("F2");
  await expect(page.getByTestId("tax-picker")).toBeVisible();
  const list = page.getByRole("listbox", { name: "税区分" });
  await expect(list).toBeFocused();
  // 先頭は「既定に戻す」。↓ を押して「非課仕入」を探す
  const options = await list.getByRole("option").allTextContents();
  const idx = options.findIndex((t) => t.startsWith("非課仕入"));
  expect(idx).toBeGreaterThan(0);
  for (let i = 0; i < idx; i++) await page.keyboard.press("ArrowDown");
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("tax-picker")).toHaveCount(0);
  await expect(input(page, 0, 2)).toBeFocused(); // 元のセルに戻る
  await expect(page.getByTestId("taxcell-0")).toHaveAttribute("data-source", "user");
  await expect(page.getByTestId("taxcell-0")).toContainText("非課仕入");
  // Esc で閉じても値は変わらない
  await page.keyboard.press("F2");
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("taxcell-0")).toContainText("非課仕入");
  await page.getByTestId("confirm-page").click();
  await expect(page.getByTestId("review")).toHaveCount(0);

  // 同じ摘要の別ページ: 提案として出る(既定とは見た目で区別できる)
  const b = await job(request, 922, STAMP);
  await openApp(page); // ジョブ一覧を読み直す
  await openJob(page, b.jobId);
  await expect(page.getByTestId("taxcell-0")).toHaveAttribute("data-source", "suggested");
  await expect(page.getByTestId("taxcell-0")).toContainText("非課仕入");
  await expect(page.getByTestId("taxcell-0")).toContainText("提案");
  // 摘要を書き換えると提案は外れる
  await input(page, 0, 2).fill("文具代");
  await expect(page.getByTestId("taxcell-0")).toHaveAttribute("data-source", "default", { timeout: 5000 });
  await input(page, 0, 2).fill(STAMP);
  await expect(page.getByTestId("taxcell-0")).toHaveAttribute("data-source", "suggested", { timeout: 5000 });

  // 提案のままの行の数が、状態の表示に出る。確定して出力の検証へ → 提案の行の数とリンクが出る(出力は止まらない)
  await expect(page.getByTestId("suggested-count")).toContainText("提案」のまま");
  await page.getByTestId("confirm-page").click();
  await expect(page.getByTestId("review")).toHaveCount(0);
  await tab(page, "出力").click();
  await page.getByTestId(`candidate-${b.pid}`).getByRole("checkbox").check();
  await page.getByTestId("export-validate").click();
  await expect(page.getByTestId("export-suggested")).toContainText("提案の税区分で出力される行");
  await expect(page.getByTestId("export-suggested").getByRole("button", { name: "確認画面で開く" }).first()).toBeVisible();
  await expect(page.getByTestId("export-run")).toBeEnabled();

  // 設定画面: 記憶が見える。税区分を直す → 削除
  await tab(page, "設定").click();
  const mem = page.getByTestId("tax-memory");
  await expect(mem).toContainText(STAMP);
  await mem.getByLabel(`摘要「${STAMP}」の税区分`).selectOption("対外仕入");
  await expect(mem.getByLabel(`摘要「${STAMP}」の税区分`)).toHaveValue("対外仕入");
  await mem.getByRole("button", { name: `摘要「${STAMP}」の記憶を削除` }).click();
  await expect(page.getByTestId("tax-memory-empty")).toBeVisible();
});

test("㉗ 行の税区分の候補リストと記憶の画面に、アクセシビリティ上の重大な違反がない", async ({ page, request }) => {
  await setup(request);
  const a = await job(request, 923, STAMP);
  await openApp(page);
  await openJob(page, a.jobId);
  await input(page, 0, 2).click();
  await page.keyboard.press("F2");
  await expect(page.getByTestId("tax-picker")).toBeVisible();
  const scan = async (label: string) => {
    const res = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
    const bad = res.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
    expect(bad.map((v) => `${v.id}: ${v.nodes[0].target.join(" ")}`), `${label} のアクセシビリティ違反`).toEqual([]);
  };
  await scan("税区分の候補リスト");
  await page.keyboard.press("Escape");
  await scan("確認画面(税区分の列つき)");
  await request.put(`/api/pages/${a.pid}/rows/0/tax`, { headers: AUTH, data: { tax_category: "非課仕入" } });
  await tab(page, "設定").click();
  await expect(page.getByTestId("settings")).toBeVisible();
  await scan("設定(記憶の編集つき)");
});

test("㉘ 出力画面から、弥生の取り込み試験用のサンプル(架空の 2 行)をダウンロードできる。設定が足りないと理由が出る", async ({ page, request }) => {
  await request.put("/api/settings", { headers: AUTH, data: { wizard_done: true } }); // 設定が空
  const nExports = async () => ((await (await request.get("/api/exports", { headers: AUTH })).json()) as { exports: unknown[] }).exports.length;
  const before = await nExports(); // 試験は同じサーバを共有するので、ほかの試験の出力履歴が残っている
  await openApp(page);
  await tab(page, "出力").click();
  await page.getByTestId("sample-box").locator("summary").click();
  await page.getByTestId("download-sample").click();
  await expect(page.getByTestId("sample-message")).toContainText("設定が足りません");
  await setup(request);
  await openApp(page);
  await tab(page, "出力").click();
  await page.getByTestId("sample-box").locator("summary").click();
  await page.getByTestId("sample-date").fill("2026-04-01");
  const [dl] = await Promise.all([page.waitForEvent("download"), page.getByTestId("download-sample").click()]);
  expect(dl.suggestedFilename()).toBe("sample_yayoi_import.csv");
  const fs = await import("node:fs");
  const text = new TextDecoder("shift_jis").decode(fs.readFileSync((await dl.path())!));
  const lines = text.split("\r\n").filter((l) => l);
  expect(lines).toHaveLength(2);
  expect(lines[0].split(",")).toHaveLength(25);
  expect(lines[0]).toContain("2026/04/01");
  expect(lines[0]).toContain(DEFAULT);
  // 本番の出力ではないので、履歴には残らない
  expect(await nExports()).toBe(before);
});
