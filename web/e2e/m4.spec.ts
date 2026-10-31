import { type APIRequestContext, type Page } from "@playwright/test";
import { expect, test } from "./fixtures";

const TOKEN = "e2e-token-0123456789abcdef0123456789abcdef";
const AUTH = { Authorization: `Bearer ${TOKEN}` };
// fake.py の ACCOUNTS と同じ(合成ジョブの科目)
const ACCOUNTS = ["現金", "普通預金", "未払金", "売掛金"];

async function createJob(request: APIRequestContext, seed: number) {
  const res = await request.post("/api/dev/jobs", {
    headers: AUTH,
    data: { pages: 1, seed, fill_rows: 5, error_rate: 0, review_rate: 0, ambiguous_rows: [] },
  });
  expect(res.ok()).toBeTruthy();
  return (await res.json()) as { job_id: number; pages: { id: number }[] };
}

async function openApp(page: Page) {
  await page.goto(`/?m4=${Date.now()}#token=${TOKEN}`);
  await expect(page.getByTestId("connection")).toContainText("接続 OK");
}
const tab = (page: Page, name: string) => page.getByRole("navigation", { name: "画面" }).getByRole("button", { name });

async function putSettings(request: APIRequestContext, title = "旅費交通費") {
  const res = await request.put("/api/settings", {
    headers: AUTH,
    data: {
      wizard_done: true,
      payments: ACCOUNTS.map((a) => ({ name: a, credit_account: a, credit_sub: "" })),
      titles: [{ name: title, debit_account: title, debit_sub: "" }],
      default_tax: [{ account: title, tax_category: "課対仕入込10%" }],
    },
  });
  expect(res.ok()).toBeTruthy();
}

async function openJobFromList(page: Page, jobId: number) {
  await tab(page, "ジョブ").click();
  await page.getByTestId(`job-${jobId}`).getByRole("button").click();
  await expect(page.getByTestId("review")).toBeVisible();
}

test("⑮ 設定を画面で入力・保存でき、不正な入力は欄ごとに理由が出る", async ({ page, request }) => {
  const reset = await request.put("/api/settings", { headers: AUTH, data: { wizard_done: true } }); // 他の試験が入れた設定を空にする(ウィザードは終えた状態)
  expect(reset.ok()).toBeTruthy();
  await openApp(page);
  await tab(page, "設定").click();
  await expect(page.getByTestId("settings")).toBeVisible();
  await expect(page.getByTestId("settings-missing")).toContainText("支払手段が 1 つも登録されていません");
  // 支払手段を 2 行(同じ名前)→ 保存で重複が指摘される
  const payments = page.getByTestId("settings-payments");
  await page.getByTestId("add-payments").click();
  await page.getByTestId("add-payments").click();
  await payments.getByLabel("支払手段 → 貸方科目 1 行目の帳票に書かれる名前").fill("現金");
  await payments.getByLabel("支払手段 → 貸方科目 1 行目の貸方科目").fill("現金");
  await payments.getByLabel("支払手段 → 貸方科目 2 行目の帳票に書かれる名前").fill("現金");
  await page.getByTestId("settings-save").click();
  await expect(payments).toContainText("同じ名前が重複しています");
  await expect(page.getByTestId("settings-state")).toContainText("入力に誤りがあります");
  // 直して保存
  await payments.getByLabel("支払手段 → 貸方科目 2 行目の帳票に書かれる名前").fill("普通預金");
  await payments.getByLabel("支払手段 → 貸方科目 2 行目の貸方科目").fill("普通預金");
  await payments.getByLabel("支払手段 → 貸方科目 2 行目の貸方補助科目").fill("A 銀行");
  await page.getByTestId("add-titles").click();
  const titles = page.getByTestId("settings-titles");
  await titles.getByLabel("タイトル → 借方科目 1 行目の帳票に書かれるタイトル").fill("旅費交通費");
  await titles.getByLabel("タイトル → 借方科目 1 行目の借方科目").fill("旅費交通費");
  // 借方の税区分は必須: 未設定の間は「足りない設定」に出る。税率のない課税区分は保存できない
  await page.getByTestId("settings-save").click();
  await expect(page.getByTestId("settings-state")).toContainText("保存しました");
  await expect(page.getByTestId("settings-missing")).toContainText("既定の税区分が設定されていません");
  await page.getByTestId("add-default_tax").click();
  const tax = page.getByTestId("settings-default_tax");
  await tax.getByLabel("科目ごとの既定の税区分 1 行目の借方科目").fill("旅費交通費");
  await tax.getByLabel("科目ごとの既定の税区分 1 行目の税区分").fill("課対仕入");
  await page.getByTestId("settings-save").click();
  await expect(tax).toContainText("税率が書かれていません");
  await tax.getByLabel("科目ごとの既定の税区分 1 行目の税区分").fill("課対仕入込10%");
  await page.getByTestId("settings-save").click();
  await expect(page.getByTestId("settings-state")).toContainText("保存しました");
  await expect(page.getByTestId("settings-missing")).toHaveCount(0);
  // 読み込み直しても残っている
  await openApp(page);
  await tab(page, "設定").click();
  await expect(page.getByTestId("settings-payments").getByLabel("支払手段 → 貸方科目 2 行目の貸方補助科目")).toHaveValue("A 銀行");
});

test("⑯ タイトルを入れて確定 → 出力の検証・出力・履歴 → 同じページの再出力は承諾が要る", async ({ page, request }) => {
  await putSettings(request);
  const job = await createJob(request, 901);
  await openApp(page);
  await openJobFromList(page, job.job_id);
  // タイトル(Enter で確定すると最初のセルへ)
  const title = page.getByTestId("page-title");
  await title.fill("旅費交通費");
  await title.press("Enter");
  await expect(title).toHaveValue("旅費交通費");
  await page.getByTestId("confirm-page").click();
  await expect(page.getByTestId("review")).toHaveCount(0); // 1 ページのジョブなのでジョブ一覧へ戻る
  // 出力
  await tab(page, "出力").click();
  await expect(page.getByTestId("export-notes")).toContainText("Excel で開かず、弥生で直接取り込んでください");
  await expect(page.getByTestId("export-notes")).toContainText("元に戻せません");
  const cand = page.getByTestId(`candidate-${job.pages[0].id}`);
  await expect(cand).toContainText("旅費交通費");
  await expect(cand).toContainText("未出力");
  await cand.getByRole("checkbox").check();
  await page.getByTestId("export-validate").click();
  await expect(page.getByTestId("export-ok")).toContainText("5 行を出力できます");
  await page.getByTestId("export-run").click();
  await expect(page.getByTestId("export-message")).toContainText("出力しました(1 ページ・5 行・1 ファイル)");
  await expect(page.getByTestId("export-history").locator("tbody tr")).toHaveCount(1);
  // ダウンロードできる(Shift_JIS の CSV)
  const [dl] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: "ダウンロード" }).first().click()]);
  expect(dl.suggestedFilename()).toMatch(/^export_\d+_\d+_\d+_yayoi_import\.csv$/);
  // 同じページをもう一度 → 重複の警告、承諾するまで出力できない
  await expect(cand).toContainText("出力済み");
  await cand.getByRole("checkbox").check();
  await page.getByTestId("export-validate").click();
  await expect(page.getByTestId("export-duplicates")).toBeVisible();
  await expect(page.getByTestId("export-run")).toBeDisabled();
  await page.getByTestId("export-duplicates").getByRole("checkbox").check();
  await expect(page.getByTestId("export-run")).toBeEnabled();
});

test("⑰ 検証エラー(タイトル未入力)は確認画面へのリンクで直せる", async ({ page, request }) => {
  await putSettings(request);
  const job = await createJob(request, 902);
  await openApp(page);
  await openJobFromList(page, job.job_id);
  await page.getByTestId("confirm-page").click(); // タイトルを入れずに確定
  await expect(page.getByTestId("review")).toHaveCount(0);
  await tab(page, "出力").click();
  const cand = page.getByTestId(`candidate-${job.pages[0].id}`);
  await cand.getByRole("checkbox").check();
  await page.getByTestId("export-validate").click();
  await expect(page.getByTestId("export-issues")).toContainText("ページのタイトルが未入力です");
  await expect(page.getByTestId("export-run")).toBeDisabled();
  await page.getByTestId("export-issues").getByRole("button", { name: "確認画面で開く" }).first().click();
  await expect(page.getByTestId("review")).toBeVisible();
  // 確定済みページのタイトルは再オープンして直す
  await expect(page.getByTestId("page-title")).toBeDisabled();
  await page.getByTestId("confirm-page").click(); // = 再オープン
  await page.getByTestId("page-title").fill("旅費交通費");
  await page.getByTestId("page-title").press("Enter");
  await page.getByTestId("confirm-page").click();
  await expect(page.getByTestId("review")).toHaveCount(0);
  await tab(page, "出力").click();
  await page.getByTestId(`candidate-${job.pages[0].id}`).getByRole("checkbox").check();
  await page.getByTestId("export-validate").click();
  await expect(page.getByTestId("export-ok")).toBeVisible();
});

test("⑱ 学習画面: 確定したページの実績(値は出さない)と、世代 0 だけなら戻す操作は出ない、リセット(確定データを残す/削除する)は確認を経て実行", async ({ page, request }) => {
  await putSettings(request);
  const job = await createJob(request, 903);
  await openApp(page);
  await openJobFromList(page, job.job_id);
  await page.getByTestId("confirm-page").click();
  await expect(page.getByTestId("review")).toHaveCount(0);
  await tab(page, "学習").click();
  await expect(page.getByTestId("learning")).toBeVisible();
  await expect(page.getByTestId("learning-summary")).toContainText("確定したページ");
  await expect(page.getByTestId("trend")).toBeVisible();
  await expect(page.getByTestId("generations")).toContainText("世代 0(出荷時)");
  await expect(page.getByTestId("active-generation")).toHaveText("世代 0(出荷時)");
  await expect(page.getByRole("button", { name: "この世代に戻す" })).toHaveCount(0); // 戻せる過去の世代がまだない
  // リセット: 確定したページがあるので押せる。既定は「確定データを残す」(切り出し画像は残る)
  await expect(page.getByTestId("reset-keep-data")).toBeChecked();
  await expect(page.getByTestId("reset-learning")).toBeEnabled();
  await page.getByTestId("reset-learning").click();
  await page.getByTestId("reset-learning-confirm").click();
  await expect(page.getByTestId("reset-message")).toContainText("確定データは残してあります");
  // 「切り出し画像と学習状態も削除する」を選ぶと、確認を経て切り出し画像を削除する
  await page.getByTestId("reset-delete-data").check();
  await page.getByTestId("reset-learning").click();
  await expect(page.getByTestId("reset-learning-confirm")).toContainText("確定データも削除");
  await page.getByTestId("reset-learning-confirm").click();
  await expect(page.getByTestId("reset-message")).toContainText("切り出し画像");
  await expect(page.getByTestId("reset-message")).toContainText("削除しました");
});

test("⑲ タイトル欄は日本語入力の変換中の Enter で送らない(IME)", async ({ page, request }) => {
  await putSettings(request);
  const job = await createJob(request, 904);
  await openApp(page);
  await openJobFromList(page, job.job_id);
  const title = page.getByTestId("page-title");
  await title.click();
  await title.fill("りょひ");
  await title.evaluate((el) => {
    el.dispatchEvent(new CompositionEvent("compositionstart", { bubbles: true }));
    el.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", keyCode: 229, isComposing: true, bubbles: true, cancelable: true }));
  });
  await expect(title).toBeFocused(); // 変換の確定でセルへ移らない
  await expect(title).toHaveValue("りょひ");
});
