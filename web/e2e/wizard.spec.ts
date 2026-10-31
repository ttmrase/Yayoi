import AxeBuilder from "@axe-core/playwright";
import { type APIRequestContext, type Page } from "@playwright/test";
import { expect, test } from "./fixtures";

const TOKEN = "e2e-token-0123456789abcdef0123456789abcdef";
const AUTH = { Authorization: `Bearer ${TOKEN}` };
const ACCOUNTS = ["現金", "普通預金", "未払金", "売掛金"];

const tab = (page: Page, name: string) => page.getByRole("navigation", { name: "画面" }).getByRole("button", { name });

async function openApp(page: Page) {
  await page.goto(`/?wiz=${Date.now()}#token=${TOKEN}`);
  await expect(page.getByTestId("connection")).toContainText("接続 OK");
}

/** 初回起動の状態(設定が空でウィザード未了)に戻す */
async function firstLaunch(request: APIRequestContext) {
  const res = await request.put("/api/settings", { headers: AUTH, data: {} });
  expect(res.ok()).toBeTruthy();
}

async function createJob(request: APIRequestContext, seed: number) {
  const res = await request.post("/api/dev/jobs", {
    headers: AUTH,
    data: { pages: 1, seed, fill_rows: 6, error_rate: 0, review_rate: 0, ambiguous_rows: [] },
  });
  expect(res.ok()).toBeTruthy();
  return (await res.json()) as { job_id: number; pages: { id: number }[] };
}

test("㉑ 初回起動の初期設定ウィザード: 不正な税区分は該当の欄で止まり、直して完了すると設定が保存される", async ({ page, request }) => {
  await firstLaunch(request);
  await openApp(page);
  await expect(page.getByTestId("wizard")).toBeVisible();
  await expect(tab(page, "ジョブ")).toBeDisabled(); // 終える(またはスキップ)までは他の画面へ行けない
  // axe(ウィザードの 4 つの段階)
  const scan = async (label: string) => {
    const res = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
    const bad = res.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
    expect(bad.map((v) => `${v.id}: ${v.nodes[0].target.join(" ")}`), `${label} のアクセシビリティ違反`).toEqual([]);
  };
  await scan("ウィザード 1");
  // 1) 事業者
  await page.getByTestId("wizard-next").click();
  // 2) 支払手段
  await expect(page.getByTestId("wizard-payments")).toBeVisible();
  await scan("ウィザード 2");
  await page.getByLabel("帳票に書かれる名前(1 行目)").fill("現金");
  await page.getByLabel("貸方科目(1 行目)").fill("現金");
  await page.getByTestId("wizard-add-payment").click();
  await page.getByLabel("帳票に書かれる名前(2 行目)").fill("普通預金");
  await page.getByLabel("貸方科目(2 行目)").fill("普通預金");
  await page.getByLabel("貸方補助科目(2 行目)").fill("A 銀行");
  await page.getByTestId("wizard-next").click();
  // 3) タイトル → 借方・税区分(税率なしの課税区分は保存できない)
  await expect(page.getByTestId("wizard-titles")).toBeVisible();
  await scan("ウィザード 3");
  await page.getByLabel("帳票に書かれるタイトル(1 行目)").fill("旅費交通費");
  await page.getByLabel("借方科目(1 行目)").fill("旅費交通費");
  await page.getByLabel("借方の税区分(1 行目)").fill("課対仕入");
  await page.getByTestId("wizard-next").click();
  // 4) 確認
  await expect(page.getByTestId("wizard-summary")).toContainText("支払手段 2 件");
  await scan("ウィザード 4");
  await page.getByTestId("wizard-finish").click();
  await expect(page.getByTestId("wizard-error")).toContainText("入力に誤りがあります");
  await expect(page.getByTestId("wizard-titles")).toContainText("税率が書かれていません");
  await expect(page.getByLabel("借方の税区分(1 行目)")).toHaveAttribute("aria-invalid", "true");
  // 直して完了
  await page.getByLabel("借方の税区分(1 行目)").fill("課対仕入込10%");
  await page.getByTestId("wizard-next").click();
  await page.getByTestId("wizard-finish").click();
  await expect(page.getByTestId("wizard")).toHaveCount(0);
  await expect(tab(page, "ジョブ")).toBeEnabled();
  // 設定画面に反映され、読み込み直してもウィザードは出ない
  await openApp(page);
  await expect(page.getByTestId("wizard")).toHaveCount(0);
  await tab(page, "設定").click();
  await expect(page.getByTestId("settings-payments").getByLabel("支払手段 → 貸方科目 2 行目の貸方補助科目")).toHaveValue("A 銀行");
  await expect(page.getByTestId("settings-default_tax").getByLabel("科目ごとの既定の税区分 1 行目の税区分")).toHaveValue("課対仕入込10%");
  await expect(page.getByTestId("settings-missing")).toHaveCount(0);
});

test("㉒ ウィザードをスキップすると二度と自動では出ず、確認画面に「科目の語彙が未設定」が出る。支払手段を入れれば消える", async ({ page, request }) => {
  await firstLaunch(request);
  const job = await createJob(request, 911);
  await openApp(page);
  await page.getByTestId("wizard-skip").click();
  await expect(page.getByTestId("wizard")).toHaveCount(0);
  await openApp(page);
  await expect(page.getByTestId("wizard")).toHaveCount(0); // スキップも「終えた」
  await page.getByTestId(`job-${job.job_id}`).getByRole("button").click();
  await expect(page.getByTestId("no-vocabulary")).toContainText("科目の語彙が未設定のため、科目はすべて要確認になります");
  // 支払手段を入れると出なくなる
  const put = await request.put("/api/settings", {
    headers: AUTH,
    data: { wizard_done: true, payments: ACCOUNTS.map((a) => ({ name: a, credit_account: a, credit_sub: "" })) },
  });
  expect(put.ok()).toBeTruthy();
  await openApp(page);
  await page.getByTestId(`job-${job.job_id}`).getByRole("button").click();
  await expect(page.getByTestId("review")).toBeVisible();
  await expect(page.getByTestId("no-vocabulary")).toHaveCount(0);
});

test("㉓ 出力: 形式を選べ、注意(警告)は出力を止めない。スマート取引取込は支払手段ごとのファイルに分かれ、タイトルが失われる注意が出る", async ({ page, request }) => {
  const put = await request.put("/api/settings", {
    headers: AUTH,
    data: {
      wizard_done: true,
      payments: ACCOUNTS.map((a) => ({ name: a, credit_account: a, credit_sub: "" })),
      titles: [{ name: "旅費交通費", debit_account: "旅費交通費", debit_sub: "" }],
      default_tax: [{ account: "旅費交通費", tax_category: "課対仕入込10%" }],
    },
  });
  expect(put.ok()).toBeTruthy();
  const job = await createJob(request, 912);
  await openApp(page);
  await page.getByTestId(`job-${job.job_id}`).getByRole("button").click();
  const title = page.getByTestId("page-title");
  await title.fill("旅費交通費");
  await title.press("Enter");
  await page.getByTestId("confirm-page").click();
  await expect(page.getByTestId("review")).toHaveCount(0);
  await tab(page, "出力").click();
  // 既定は弥生インポート形式。形式を切り替えると注意書きが変わる
  await expect(page.getByTestId("export-format")).toHaveValue("yayoi_import");
  await expect(page.getByTestId("export-notes")).toContainText("マッチングリスト");
  await page.getByTestId("export-format").selectOption("smart_torihiki");
  await expect(page.getByTestId("export-notes")).toContainText("取引手段");
  await page.getByTestId(`candidate-${job.pages[0].id}`).getByRole("checkbox").check();
  await page.getByTestId("export-validate").click();
  await expect(page.getByTestId("export-ok")).toBeVisible();
  await expect(page.getByTestId("export-warnings")).toContainText("ページのタイトル(借方の科目)は失われます");
  await expect(page.getByTestId("export-run")).toBeEnabled(); // 注意は出力を止めない
  await page.getByTestId("export-run").click();
  await expect(page.getByTestId("export-message")).toContainText("出力しました");
  // 履歴に、支払手段ごとのファイルが並ぶ(ファイル名に支払手段)
  const history = page.getByTestId("export-history");
  // 履歴の再読み込みを待つ(先頭の行が今の出力 = スマート取引取込になってから、その行のボタンを押す)
  await expect(history.locator("tbody tr").first()).toContainText("スマート取引取込");
  const downloads = history.locator("tbody tr").first().getByRole("button", { name: "ダウンロード" });
  expect(await downloads.count()).toBeGreaterThanOrEqual(1);
  const [dl] = await Promise.all([page.waitForEvent("download"), downloads.first().click()]);
  // ファイル名(日本語の支払手段名を含む)はサーバの記録で確認する。Linux のヘッドレス Chromium は非 ASCII の保存名を落とすため、保存名自体は検査しない
  const list = (await (await request.get("/api/exports", { headers: AUTH })).json()) as { exports: { files: { name: string; payment: string }[] }[] };
  const names = list.exports[0].files.map((f) => f.name);
  expect(names.length).toBeGreaterThanOrEqual(1);
  for (const f of list.exports[0].files) expect(f.name).toMatch(new RegExp(`^export_\\d+_\\d+_\\d+_smart_${f.payment}_1\\.csv$`));
  // 中身は Shift_JIS・CRLF・ヘッダーなし・全行 4 列
  const fs = await import("node:fs");
  const bytes = fs.readFileSync((await dl.path())!);
  const text = new TextDecoder("shift_jis").decode(bytes);
  const lines = text.split("\r\n");
  expect(lines.pop()).toBe(""); // 末尾の CRLF
  expect(lines.length).toBeGreaterThan(0);
  for (const l of lines) {
    expect(l.split(",")).toHaveLength(4);
    expect(l).toMatch(/^\d{4}\/\d{2}\/\d{2},,\d+,/); // 日付(西暦 4 桁)・入金(空)・出金
  }
});

test("㉔ 学習画面から精度レポート(統計のみ)を保存できる", async ({ page, request }) => {
  const put = await request.put("/api/settings", {
    headers: AUTH,
    data: { wizard_done: true, payments: ACCOUNTS.map((a) => ({ name: a, credit_account: a, credit_sub: "" })) },
  });
  expect(put.ok()).toBeTruthy();
  const job = await createJob(request, 913);
  await openApp(page);
  await page.getByTestId(`job-${job.job_id}`).getByRole("button").click();
  await page.getByTestId("confirm-page").click();
  await expect(page.getByTestId("review")).toHaveCount(0);
  await tab(page, "学習").click();
  const [dl] = await Promise.all([page.waitForEvent("download"), page.getByTestId("download-report").click()]);
  expect(dl.suggestedFilename()).toBe("accuracy_report.json");
  const path = await dl.path();
  const fs = await import("node:fs");
  const text = fs.readFileSync(path, "utf-8");
  const rep = JSON.parse(text) as Record<string, unknown>;
  expect(rep.schema).toBe(2);
  // 帳票の内容(科目名・タイトル)は含まれない
  for (const w of [...ACCOUNTS, "旅費交通費"]) expect(text).not.toContain(w);
});

test("㉕ 税区分の候補は課税方式・経理方式の許可リスト。税抜経理へ変えると古い税区分は選び直しになり、端数処理を選べる", async ({ page, request }) => {
  const put = await request.put("/api/settings", {
    headers: AUTH,
    data: {
      wizard_done: true,
      payments: ACCOUNTS.map((a) => ({ name: a, credit_account: a, credit_sub: "" })),
      titles: [{ name: "旅費交通費", debit_account: "旅費交通費", debit_sub: "" }],
      default_tax: [{ account: "旅費交通費", tax_category: "課対仕入込10%" }],
    },
  });
  expect(put.ok()).toBeTruthy();
  await openApp(page);
  await tab(page, "設定").click();
  const options = () => page.locator("#tax-category-options option").evaluateAll((els) => els.map((e) => (e as HTMLOptionElement).value));
  // 本則課税 × 税込経理(15 件)
  expect(await options()).toContain("共対仕入込軽減8%");
  expect(await options()).not.toContain("課対仕入");
  await expect(page.getByTestId("tax-rounding")).toHaveCount(0);
  // 税抜経理へ: 候補が内税・外税・別記に変わる。保存すると、込の税区分は選べないと指摘される
  await page.getByTestId("tax-method").selectOption("exclusive");
  expect((await options())[0]).toBe("課対仕入内10%");
  await expect(page.getByTestId("tax-rounding")).toBeVisible();
  await page.getByTestId("tax-rounding").selectOption("round");
  await page.getByTestId("settings-save").click();
  await expect(page.getByTestId("settings-default_tax")).toContainText("選べない税区分です");
  // 選び直して保存 → 読み込み直しても残る
  await page.getByLabel("科目ごとの既定の税区分 1 行目の税区分").fill("課対仕入内10%");
  await page.getByTestId("settings-save").click();
  await expect(page.getByTestId("settings-state")).toContainText("保存しました");
  await openApp(page);
  await tab(page, "設定").click();
  await expect(page.getByTestId("tax-method")).toHaveValue("exclusive");
  await expect(page.getByTestId("tax-rounding")).toHaveValue("round");
  // 免税事業者: 候補は「対象外」だけ
  await page.getByTestId("taxpayer").selectOption("exempt");
  expect(await options()).toEqual(["対象外"]);
  await request.put("/api/settings", { headers: AUTH, data: { wizard_done: true } });
});
