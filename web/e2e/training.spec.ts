import { type APIRequestContext, type Page } from "@playwright/test";
import { expect, test } from "./fixtures";

// 追加学習(Phase 4): 確定 → 学習 → ゲート → 採用/不採用 → ロールバック。学習器は開発用のモック(/api/dev/trainer)。
const TOKEN = "e2e-token-0123456789abcdef0123456789abcdef";
const AUTH = { Authorization: `Bearer ${TOKEN}` };

async function openApp(page: Page) {
  await page.goto(`/?tr=${Date.now()}#token=${TOKEN}`);
  await expect(page.getByTestId("connection")).toContainText("接続 OK");
}
const tab = (page: Page, name: string) => page.getByRole("navigation", { name: "画面" }).getByRole("button", { name });

async function post(request: APIRequestContext, path: string, data: object = {}) {
  const res = await request.post(path, { headers: AUTH, data });
  expect(res.ok(), `${path}: ${res.status()}`).toBeTruthy();
  return (await res.json()) as Record<string, unknown>;
}

/** 合成の確定ページを n 枚作る(116 セル。要確認なし)。 */
async function confirmPages(request: APIRequestContext, n: number, seed: number) {
  for (let i = 0; i < n; i++) {
    const job = (await post(request, "/api/dev/jobs", {
      pages: 1,
      seed: seed + i,
      fill_rows: 29,
      error_rate: 0,
      review_rate: 0,
      ambiguous_rows: [],
    })) as { pages: { id: number }[] };
    await post(request, `/api/pages/${job.pages[0].id}/confirm`);
  }
}

async function setTrainer(request: APIRequestContext, mode: string, delay = 0, auto?: boolean) {
  await post(request, "/api/dev/trainer", { mode, delay, ...(auto === undefined ? {} : { auto }) });
}

async function putTrainSettings(request: APIRequestContext, autoTrain: boolean, threshold = 200) {
  const cur = (await (await request.get("/api/settings", { headers: AUTH })).json()) as { settings: Record<string, unknown> };
  const res = await request.put("/api/settings", {
    headers: AUTH,
    data: { ...cur.settings, wizard_done: true, auto_train: autoTrain, train_threshold_cells: threshold },
  });
  expect(res.ok()).toBeTruthy();
}

test.beforeEach(async ({ request }) => {
  await post(request, "/api/generations/reset", { delete_learning_data: false }); // 出荷時の重みから始める(確定データは残る)
  await putTrainSettings(request, false);
  await setTrainer(request, "improve", 0, false);
});
test.afterEach(async ({ request }) => {
  await setTrainer(request, "improve", 0, false);
});

test("㉙ 今すぐ学習 → 採用 → 世代 1 が使用中 → 世代 0 に戻す → 世代 1 に戻す", async ({ page, request }) => {
  await confirmPages(request, 7, 1000);
  await setTrainer(request, "improve", 1.5);
  await openApp(page);
  await tab(page, "学習").click();
  await expect(page.getByTestId("training-counts")).toContainText("新しい確定セル");
  await expect(page.getByTestId("active-generation")).toHaveText("世代 0(出荷時)");
  await page.getByTestId("training-start").click();
  await expect(page.getByTestId("training-running")).toBeVisible();
  await expect(page.getByTestId("training-start")).toHaveCount(0); // 学習中は押せない
  await expect(page.getByTestId("training-last-status")).toHaveText("採用", { timeout: 20_000 });
  await expect(page.getByTestId("training-last")).toContainText("正解率が上がったため採用");
  // 世代の表が読み直され、世代 1 が使用中になる
  await expect(page.getByTestId("active-generation")).toHaveText(/世代 \d+$/);
  const active = await page.getByTestId("active-generation").innerText();
  expect(active).not.toContain("出荷時");
  const n = active.replace("世代 ", "");
  await expect(page.getByTestId(`generation-${n}`)).toContainText("使用中");
  await expect(page.getByTestId(`generation-${n}`)).toContainText("/"); // 学習 / 評価ページ
  // ロールバック: 世代 0 に戻す → 世代 n に戻す
  await page.getByTestId("rollback-0").click();
  await expect(page.getByTestId("active-generation")).toHaveText("世代 0(出荷時)");
  await expect(page.getByTestId("reset-message")).toContainText("使用中にしました");
  await page.getByTestId(`rollback-${n}`).click();
  await expect(page.getByTestId("active-generation")).toHaveText(`世代 ${n}`);
  // リセット(確定データを残す)で出荷時に戻る。世代 n は「リセットで破棄」
  await page.getByTestId("reset-learning").click();
  await page.getByTestId("reset-learning-confirm").click();
  await expect(page.getByTestId("active-generation")).toHaveText("世代 0(出荷時)");
  await expect(page.getByTestId(`generation-${n}`)).toContainText("リセットで破棄");
  await expect(page.getByTestId(`rollback-${n}`)).toHaveCount(0);
});

test("㉚ 悪化した世代は不採用(理由を表示)、使用中の世代は変わらない", async ({ page, request }) => {
  await confirmPages(request, 7, 2000);
  await setTrainer(request, "worsen");
  await openApp(page);
  await tab(page, "学習").click();
  await page.getByTestId("training-start").click();
  await expect(page.getByTestId("training-last-status")).toHaveText("不採用", { timeout: 20_000 });
  await expect(page.getByTestId("training-last")).toContainText("見逃し");
  await expect(page.getByTestId("active-generation")).toHaveText("世代 0(出荷時)");
  await expect(page.getByTestId("generations")).toContainText("不採用");
  await expect(page.getByTestId("generations")).toContainText("miss_rate_worse");
  // 失敗した学習は理由つきで表示する
  await setTrainer(request, "fail");
  await page.getByTestId("training-start").click();
  await expect(page.getByTestId("training-last-status")).toHaveText("失敗", { timeout: 20_000 });
});

test("㉛ 自動開始: 確定セルがしきい値たまると始まり、止めておけば始まらない。学習中も確認画面は使える", async ({ page, request }) => {
  await confirmPages(request, 6, 3000);
  await putTrainSettings(request, true, 50);
  await setTrainer(request, "improve", 6, true);
  await openApp(page);
  await tab(page, "学習").click();
  await expect(page.getByTestId("auto-train")).toBeChecked();
  await expect(page.getByTestId("train-threshold")).toHaveValue("50");
  await confirmPages(request, 1, 3100); // 確定の通知で自動開始
  await expect(page.getByTestId("training-running")).toBeVisible({ timeout: 8_000 });
  // 学習中も、確認画面(ジョブ)は開ける
  await tab(page, "ジョブ").click();
  await expect(page.getByRole("heading", { name: "ジョブ" })).toBeVisible();
  await tab(page, "学習").click();
  await expect(page.getByTestId("training-last-status")).toHaveText("採用", { timeout: 20_000 });
  await expect(page.getByTestId("training-last")).toContainText("自動");
  // 自動を止める(画面の設定)
  await page.getByTestId("auto-train").click(); // 保存してから画面が更新される(uncheck() は即時の状態変化を要求する)
  await expect.poll(async () => ((await (await request.get("/api/training", { headers: AUTH })).json()) as { auto_train: boolean }).auto_train).toBe(false);
});
