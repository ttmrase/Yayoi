import AxeBuilder from "@axe-core/playwright";
import { type APIRequestContext, type Page } from "@playwright/test";
import { expect, test } from "./fixtures";

// 主要な画面を axe-core で検査する(コントラスト・ラベル・ロール・フォーカスなど)。重大(critical)・深刻(serious)な違反があれば失敗。
const TOKEN = "e2e-token-0123456789abcdef0123456789abcdef";
const AUTH = { Authorization: `Bearer ${TOKEN}` };
const PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
  "base64",
);

async function scan(page: Page, label: string) {
  const res = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
  const bad = res.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
  const summary = bad.map((v) => `${v.id} (${v.impact}): ${v.nodes.length} 件 例: ${v.nodes[0].target.join(" ")}`);
  if (res.violations.length > bad.length) {
    console.log(`${label}: 軽微な指摘 ${res.violations.filter((v) => !bad.includes(v)).map((v) => v.id).join(", ")}`);
  }
  expect(summary, `${label} のアクセシビリティ違反`).toEqual([]);
}

async function setup(request: APIRequestContext) {
  await request.put("/api/settings", {
    headers: AUTH,
    data: {
      wizard_done: true,
      payments: [{ name: "現金", credit_account: "現金", credit_sub: "" }],
      titles: [{ name: "旅費交通費", debit_account: "旅費交通費", debit_sub: "" }],
      default_tax: [{ account: "旅費交通費", tax_category: "課対仕入込10%" }],
    },
  });
  const res = await request.post("/api/dev/jobs", { headers: AUTH, data: { pages: 2, seed: 777, fill_rows: 8 } });
  return (await res.json()) as { job_id: number; pages: { id: number }[] };
}

test("アクセシビリティ: ジョブ・確認(行/列)・罫線・設定・出力・学習", async ({ page, request }) => {
  test.setTimeout(120_000); // axe の走査を 7 画面で行う。遅いランナーでは 30 秒を超えうる
  await request.post("/api/dev/fake", { headers: AUTH, data: { layout_ok: false } });
  const job = await setup(request);
  await page.goto(`/?a11y=${Date.now()}#token=${TOKEN}`);
  await expect(page.getByTestId("connection")).toContainText("接続 OK");
  await expect(page.locator("html")).toHaveAttribute("lang", "ja");

  await scan(page, "ジョブ");
  await page.getByTestId(`job-${job.job_id}`).getByRole("button").click();
  await expect(page.getByTestId("review")).toBeVisible();
  await scan(page, "確認(行モード)");
  await page.getByTestId("mode-toggle").click();
  await expect(page.getByTestId("column-tabs")).toBeVisible();
  await scan(page, "確認(列モード)");

  // 取り込んだページ(レイアウト要確認)で罫線のパネルを開く
  await page.getByRole("navigation", { name: "画面" }).getByRole("button", { name: "ジョブ" }).click();
  const stamp = `a11y${Date.now() % 100000}`;
  await page.getByLabel("ジョブ名").fill(stamp);
  await page.getByTestId("file-input").setInputFiles([{ name: "p.png", mimeType: "image/png", buffer: PNG }]);
  const row = page.locator("tr", { hasText: stamp });
  await expect(row.getByTestId(/job-status-/)).toContainText("確認待ち", { timeout: 20_000 });
  await row.getByRole("button").click();
  await page.getByRole("button", { name: "罫線を確認・補正" }).click();
  await expect(page.getByTestId("layout-panel")).toBeVisible();
  await scan(page, "罫線の確認・補正");
  await page.keyboard.press("Escape");

  await page.getByRole("navigation", { name: "画面" }).getByRole("button", { name: "設定" }).click();
  await expect(page.getByTestId("settings")).toBeVisible();
  await scan(page, "設定");
  await page.getByRole("navigation", { name: "画面" }).getByRole("button", { name: "出力" }).click();
  await expect(page.getByTestId("export")).toBeVisible();
  await scan(page, "出力");
  await page.getByRole("navigation", { name: "画面" }).getByRole("button", { name: "学習" }).click();
  await expect(page.getByTestId("learning")).toBeVisible();
  await scan(page, "学習");
  await request.post("/api/dev/fake", { headers: AUTH, data: {} });
});
