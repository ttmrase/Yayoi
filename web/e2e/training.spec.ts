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

/** 学習を始め、**この実行**が終わるまで待つ。前のテストの実行が `last` に残っているので、番号が新しくなったものだけを数える(古い結果で先へ進むと、新しい世代の採用の前にページを作ってしまう)。 */
async function startAndWait(request: APIRequestContext, status: string) {
  const lastRun = async () => ((await (await request.get("/api/training", { headers: AUTH })).json()) as { last: { id: number; status: string } | null }).last;
  const before = (await lastRun())?.id ?? 0;
  await post(request, "/api/training/start");
  await expect
    .poll(async () => {
      const last = await lastRun();
      return last && last.id > before ? last.status : null;
    })
    .toBe(status);
}

async function setTrainer(request: APIRequestContext, mode: string, delay = 0, auto?: boolean) {
  await post(request, "/api/dev/trainer", { mode, delay, ...(auto === undefined ? {} : { auto }) });
}

async function putTrainSettings(request: APIRequestContext, autoTrain: boolean, threshold = 200) {
  const cur = (await (await request.get("/api/settings", { headers: AUTH })).json()) as { settings: Record<string, unknown> };
  const res = await request.put("/api/settings", {
    headers: AUTH,
    data: { ...cur.settings, wizard_done: true, auto_train: autoTrain, suggest_mode: true, train_threshold_cells: threshold },
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
  await expect(page.getByTestId(`generation-${n}-eval`)).toContainText("正解率"); // 採否の評価(親 → この世代)
  // 新しい世代で認識して確定したページができると、推移のグラフに世代の切り替わりの印が出る
  await confirmPages(request, 1, 1100);
  await tab(page, "ジョブ").click();
  await tab(page, "学習").click();
  await expect(page.getByTestId("trend-switch").first()).toBeVisible();
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
  await expect(page.getByTestId("generations")).toContainText("見逃し(誤りなのに要確認がつかないセル)が増えるため"); // 理由の文言
  await expect(page.getByTestId("generations")).toContainText("正解率"); // 評価の値(親 → この世代)も残る
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

test("㉜ 採用した世代で認識したページの金額に「要確認でなかったのに直したセル」があると、自動で前の世代に戻り、画面の上部に通知が出る(確認で消える)", async ({ page, request }) => {
  await confirmPages(request, 7, 4000);
  await setTrainer(request, "improve", 0);
  await startAndWait(request, "adopted");
  // 新しい世代で認識したページ(合成): 要確認のないページの金額を利用者が直して確定する = 観測した見逃し
  const job = (await post(request, "/api/dev/jobs", {
    pages: 1,
    seed: 4100,
    fill_rows: 29,
    error_rate: 0,
    review_rate: 0,
    ambiguous_rows: [],
  })) as { pages: { id: number }[] };
  const pid = job.pages[0].id;
  const detail = (await (await request.get(`/api/pages/${pid}`, { headers: AUTH })).json()) as { cells: { id: number; column: string; kind: string }[] };
  const amount = detail.cells.find((c) => c.column === "amount" && c.kind === "text")!;
  const edit = await request.patch(`/api/pages/${pid}/cells/${amount.id}`, { headers: AUTH, data: { action: "edit", value: "999" } });
  expect(edit.ok()).toBeTruthy();
  const done = (await post(request, `/api/pages/${pid}/confirm`)) as { generation_halted?: { code: string } };
  expect(done.generation_halted?.code).toBe("halted_amount_miss");

  await openApp(page);
  await expect(page.getByTestId("notice-banner")).toContainText("安全のため停止しました");
  await expect(page.getByTestId("notice-banner")).toContainText("金額");
  await tab(page, "学習").click();
  await expect(page.getByTestId("active-generation")).toHaveText("世代 0(出荷時)"); // 親(出荷時)に戻っている
  await expect(page.getByTestId("generations")).toContainText("安全のため停止");
  await expect(page.getByRole("button", { name: "この世代に戻す" })).toHaveCount(0); // 停止した世代は戻せない
  await page.locator('[data-testid^="ack-notice-"]').first().click();
  await expect(page.getByTestId("notice-banner")).toHaveCount(0);
});

test("㉝ 提案モード: 安全基準には届かないが正解率が有意に高い世代は、入力済みの値の提案に使われ(要確認の判定は変えない)、新しい世代の採用で置き換わる", async ({ page, request }) => {
  await confirmPages(request, 7, 5000);
  await setTrainer(request, "suggest");
  await startAndWait(request, "suggested");
  await openApp(page);
  await tab(page, "学習").click();
  await expect(page.getByTestId("training-last-status")).toHaveText("提案モード");
  await expect(page.getByTestId("active-generation")).toHaveText("世代 0(出荷時)"); // 使用中の世代は変わらない
  await expect(page.getByTestId("suggest-banner")).toContainText("提案モードで使用中");
  await expect(page.getByTestId("suggest-banner")).toContainText("要確認の判定は変えません");
  const detail = page.locator('[data-testid^="suggest-detail-"]').first();
  await expect(detail).toContainText("符号検定");
  await expect(detail).toContainText("金額に「要確認でないのに誤り」");
  await expect(page.getByRole("button", { name: "この世代に戻す" })).toHaveCount(0); // 提案モードの世代は通常の世代としては戻せない
  // 設定の「提案モードを使う」(既定オン)をオフにすると、いまは使っていない表示になり、オンに戻すとバナーが戻る
  const sw = page.getByTestId("suggest-mode");
  await expect(sw).toBeChecked();
  await sw.click();
  await expect(sw).not.toBeChecked();
  await expect(page.getByTestId("suggest-off")).toContainText("オフ");
  await expect(page.getByTestId("suggest-banner")).toHaveCount(0);
  await sw.click();
  await expect(sw).toBeChecked();
  await expect(page.getByTestId("suggest-banner")).toBeVisible();
  // 新しい世代が採用されると、提案モードの世代は置き換わる(バナーが消える)
  await confirmPages(request, 1, 5100);
  await setTrainer(request, "improve");
  await startAndWait(request, "adopted");
  await tab(page, "ジョブ").click();
  await tab(page, "学習").click();
  await expect(page.getByTestId("suggest-banner")).toHaveCount(0);
  await expect(page.getByTestId("generations")).toContainText("新しい世代に置き換えられました");
});

const PNG = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
  "base64",
);
const cellInput = (page: Page, row: number, col: number) => page.getByTestId(`cell-${row}-${col}`).locator("input");

test("㉞ 提案モードで認識したページでは、金額が親の世代と違うセルは、候補を選ぶか入力するまで Enter で通せない", async ({ page, request }) => {
  await confirmPages(request, 7, 5200);
  await setTrainer(request, "suggest");
  await startAndWait(request, "suggested");
  await openApp(page);
  const stamp = `提案${Date.now() % 100000}`;
  await page.getByLabel("ジョブ名").fill(stamp);
  await page.getByTestId("file-input").setInputFiles([{ name: "p.png", mimeType: "image/png", buffer: PNG }]);
  const row = page.locator("tr", { hasText: stamp });
  await expect(row.getByTestId(/job-status-/)).toContainText("確認待ち", { timeout: 20_000 });
  const jobs = (await (await request.get("/api/jobs", { headers: AUTH })).json()) as { id: number; name: string }[];
  const job = jobs.find((j) => j.name === stamp)!;
  const detail = (await (await request.get(`/api/jobs/${job.id}`, { headers: AUTH })).json()) as { pages: { id: number }[] };
  const cells = ((await (await request.get(`/api/pages/${detail.pages[0].id}`, { headers: AUTH })).json()) as {
    cells: { row: number; col: number; reasons: string[]; value: string; candidates: { value: string }[] }[];
  }).cells;
  const amt = cells.find((c) => c.reasons.includes("suggest_amount_disagreement"))!;
  expect(amt.candidates.length).toBeGreaterThanOrEqual(2);
  await row.getByRole("button").click();
  await expect(page.getByTestId("review")).toBeVisible();
  await cellInput(page, amt.row, amt.col).click();
  const warn = page.getByTestId(`cell-${amt.row}-${amt.col}`).getByTestId("ambiguity");
  await expect(warn).toContainText("使用中の世代と提案モードで違います");
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("notice")).toContainText("Alt+1(提案の値)");
  await expect(cellInput(page, amt.row, amt.col)).toBeFocused(); // 進まない
  await page.keyboard.press("Alt+Digit1"); // 提案の値(入力済みと同じ)を明示的に選ぶ
  await expect(cellInput(page, amt.row, amt.col)).toHaveValue(amt.value);
  await expect(warn).toHaveCount(0);
  await page.keyboard.press("Enter");
  await expect(cellInput(page, amt.row, amt.col)).not.toBeFocused();
});
