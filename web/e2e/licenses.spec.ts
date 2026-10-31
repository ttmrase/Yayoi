import AxeBuilder from "@axe-core/playwright";
import { type Page } from "@playwright/test";
import { expect, test } from "./fixtures";

// ライセンス表示(Phase 6): 主な利用物の出典・ライセンス・変更点(帰属表示)が見える。依存ライブラリの一覧は配布物の作成時に生成するので、開発中の環境では空。
const TOKEN = "e2e-token-0123456789abcdef0123456789abcdef";
const tab = (page: Page, name: string) => page.getByRole("navigation", { name: "画面" }).getByRole("button", { name });

test("㊳ 「ライセンス」タブに、NDLOCR-Lite の CC BY 4.0 の帰属表示・PARSeq・Python が出て、アクセシビリティ上の重大な違反がない", async ({ page }) => {
  await page.goto(`/?lic=${Date.now()}#token=${TOKEN}`);
  await expect(page.getByTestId("connection")).toContainText("接続 OK");
  await tab(page, "ライセンス").click();
  const main = page.getByTestId("license-main");
  await expect(main.first()).toBeVisible();
  const ndl = main.filter({ hasText: "NDLOCR-Lite" });
  await expect(ndl).toContainText("CC BY 4.0");
  await expect(ndl).toContainText("国立国会図書館");
  await expect(ndl).toContainText("github.com/ndl-lab/ndlocr-lite");
  await expect(ndl).toContainText("変更点");
  await expect(main.filter({ hasText: "baudm/parseq" })).toContainText("Apache");
  await expect(main.filter({ hasText: "CPython" })).toContainText("PSF");
  await expect(page.getByTestId("licenses-not-generated")).toBeVisible(); // 開発中の環境では、依存の一覧はまだ生成されていない
  const res = await new AxeBuilder({ page }).include('[data-testid="licenses"]').analyze();
  const serious = res.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
  expect(serious, JSON.stringify(serious.map((v) => v.id))).toEqual([]);
});
