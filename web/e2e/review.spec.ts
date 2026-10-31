import { expect, test, type APIRequestContext, type Page } from "@playwright/test";

const TOKEN = "e2e-token-0123456789abcdef0123456789abcdef";
const AUTH = { Authorization: `Bearer ${TOKEN}` };

interface DevPage {
  id: number;
  truth: Record<string, string>;
}
interface ApiCell {
  id: number;
  row: number;
  col: number;
  column: string;
  kind: string;
  value: string;
  review: boolean;
  passed: boolean;
  candidates: { value: string; p: number }[];
  reasons: string[];
  expansion: { kind: string } | null;
}

async function createJob(request: APIRequestContext, body: Record<string, unknown> = {}) {
  const res = await request.post("/api/dev/jobs", { headers: AUTH, data: { pages: 2, fill_rows: 8, ...body } });
  expect(res.ok()).toBeTruthy();
  return (await res.json()) as { job_id: number; pages: DevPage[] };
}

async function apiPage(request: APIRequestContext, id: number) {
  const res = await request.get(`/api/pages/${id}`, { headers: AUTH });
  return (await res.json()) as { cells: ApiCell[]; summary: { unpassed: number[]; can_confirm: boolean } };
}

async function openJob(page: Page, jobId: number) {
  await page.goto(`/#token=${TOKEN}`);
  await expect(page.getByTestId("connection")).toContainText("接続 OK");
  await page.getByTestId(`job-${jobId}`).getByRole("button").click();
  await expect(page.getByTestId("review")).toBeVisible();
}

const LOOK_MS = 300; // 要確認セルの通過に要る滞在時間(250 ms)より少し長く

const navigable = (cells: ApiCell[]) => cells.filter((c) => c.kind !== "blank" || c.review);
const input = (page: Page, row: number, col: number) => page.getByTestId(`cell-${row}-${col}`).locator("input");
const activeLabel = (page: Page) => page.evaluate(() => document.activeElement?.getAttribute("aria-label") ?? "");

test("① Enter だけで 1 ページを通し、Ctrl+Enter で確定して次のページが開く(自動スクロールも)", async ({ page, request }) => {
  const job = await createJob(request, { seed: 11, error_rate: 0, review_rate: 1, ambiguous_rows: [] });
  const before = await apiPage(request, job.pages[0].id);
  const steps = navigable(before.cells).length;
  await openJob(page, job.job_id);
  await expect(page.getByTestId("page-no")).toContainText("ページ 1 / 2");
  await expect(input(page, 0, 0)).toBeFocused();
  const started = Date.now();
  for (let i = 0; i < steps; i++) {
    await page.waitForTimeout(LOOK_MS); // 人が画像を見る時間(これより速い Enter は要確認セルを通さない。⑨ で確かめる)
    await page.keyboard.press("Enter");
    if (i === Math.floor(steps / 2)) {
      // フォーカス行が画面の上から約 1/3 に保たれる
      const top = await page.evaluate(() => document.activeElement!.closest("[data-row]")!.getBoundingClientRect().top);
      const h = await page.evaluate(() => window.innerHeight);
      expect(top).toBeGreaterThan(0);
      expect(top).toBeLessThan(h * 0.6);
    }
  }
  await expect(page.getByTestId("prompt")).toBeVisible();
  await expect(page.getByTestId("confirm-page")).toBeEnabled();
  await page.keyboard.press("Control+Enter");
  await expect(page.getByTestId("page-no")).toContainText("ページ 2 / 2");
  const after = await apiPage(request, job.pages[0].id);
  expect(after.summary.can_confirm).toBeTruthy();
  const seconds = (Date.now() - started) / 1000;
  console.log(`Enter だけ(1 セル ${LOOK_MS} ms): ${steps} セルを ${seconds.toFixed(1)} 秒で通した`);
  expect(seconds).toBeLessThan(60); // 回帰の目安(実測は M5 で見積りと比べる)
});

test("② 上書きと候補の採用(Alt+数字)", async ({ page, request }) => {
  const job = await createJob(request, { seed: 21, error_rate: 0.7, review_rate: 1, ambiguous_rows: [] });
  const p = job.pages[0];
  const cells = (await apiPage(request, p.id)).cells;
  const wrong = cells.filter((c) => c.value !== p.truth[String(c.id)] && c.kind === "text" && c.column !== "description");
  const withCand = wrong.find((c) => c.candidates.some((x) => x.value === p.truth[String(c.id)]))!;
  const plain = wrong.find((c) => c.id !== withCand.id)!;
  await openJob(page, job.job_id);
  // 候補: Alt+(位置)でその値が入る
  await input(page, withCand.row, withCand.col).click();
  await expect(page.getByTestId("candidates")).toBeVisible();
  const idx = withCand.candidates.findIndex((x) => x.value === p.truth[String(withCand.id)]);
  await page.keyboard.press(`Alt+Digit${idx + 1}`);
  await expect(input(page, withCand.row, withCand.col)).toHaveValue(p.truth[String(withCand.id)]);
  // 上書き: 全選択されたところへ入力
  await input(page, plain.row, plain.col).click();
  await page.keyboard.press("Control+a");
  await page.keyboard.type(p.truth[String(plain.id)]);
  await page.keyboard.press("Enter");
  await expect
    .poll(async () => (await apiPage(request, p.id)).cells.find((c) => c.id === plain.id)!.value)
    .toBe(p.truth[String(plain.id)]);
});

test("③ 要確認セルを飛ばしたままでは確定できない(UI もサーバも)", async ({ page, request }) => {
  const job = await createJob(request, { seed: 31, error_rate: 0, review_rate: 1, ambiguous_rows: [] });
  await openJob(page, job.job_id);
  await expect(page.getByTestId("confirm-page")).toBeDisabled();
  await page.keyboard.press("Control+Enter");
  await expect(page.getByTestId("block")).toContainText("未通過の要確認");
  const res = await request.post(`/api/pages/${job.pages[0].id}/confirm`, { headers: AUTH, data: {} });
  expect(res.status()).toBe(409);
});

test("④ 末尾「-」の両仮説で 10 倍違う金額は、Alt+数字で選ぶまで進めない", async ({ page, request }) => {
  const job = await createJob(request, { seed: 41, error_rate: 0, review_rate: 1, ambiguous_rows: [3] });
  const cells = (await apiPage(request, job.pages[0].id)).cells;
  const amb = cells.find((c) => c.reasons.includes("dash_digit_ambiguity"))!;
  await openJob(page, job.job_id);
  await input(page, amb.row, amb.col).click();
  await expect(page.getByTestId("ambiguity")).toBeVisible();
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("notice")).toContainText("Alt+1 / Alt+2");
  await expect(input(page, amb.row, amb.col)).toBeFocused(); // 進まない
  await page.keyboard.press("Alt+Digit2");
  await expect(input(page, amb.row, amb.col)).toHaveValue("500");
  await expect(page.getByTestId("ambiguity")).toHaveCount(0);
  await page.keyboard.press("Enter");
  await expect(input(page, amb.row, amb.col)).not.toBeFocused();
});

test("⑤ 日本語入力の変換中の Enter ではセルが進まない(IME)", async ({ page, request }) => {
  const job = await createJob(request, { seed: 51, error_rate: 0, review_rate: 1, ambiguous_rows: [] });
  await openJob(page, job.job_id);
  const desc = input(page, 0, 2);
  await desc.click();
  await page.waitForTimeout(LOOK_MS);
  const label = await activeLabel(page);
  // 変換の開始 → 変換中の Enter(isComposing / keyCode 229)→ 変換の確定
  await desc.evaluate((el) => {
    el.dispatchEvent(new CompositionEvent("compositionstart", { bubbles: true }));
    el.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", code: "Enter", keyCode: 229, isComposing: true, bubbles: true, cancelable: true }),
    );
  });
  expect(await activeLabel(page)).toBe(label); // 動かない
  await desc.evaluate((el) => el.dispatchEvent(new CompositionEvent("compositionend", { bubbles: true, data: "漢字" })));
  await page.keyboard.press("Enter"); // 変換が終わったあとの Enter は送る
  expect(await activeLabel(page)).not.toBe(label);
  // keyCode 229 だけ(isComposing が付かない実装)でも動かない
  await page.waitForTimeout(LOOK_MS);
  const next = page.locator(`input[aria-label="${await activeLabel(page)}"]`);
  const label2 = await activeLabel(page);
  await next.evaluate((el) =>
    el.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", keyCode: 229, bubbles: true, cancelable: true })),
  );
  expect(await activeLabel(page)).toBe(label2);
});

test("⑥ 元に戻す(Ctrl+Z)", async ({ page, request }) => {
  const job = await createJob(request, { seed: 61, error_rate: 0, review_rate: 1, ambiguous_rows: [] });
  const cells = (await apiPage(request, job.pages[0].id)).cells;
  const amount = cells.find((c) => c.column === "amount" && c.kind === "text")!;
  await openJob(page, job.job_id);
  const field = input(page, amount.row, amount.col);
  await field.click();
  await page.keyboard.press("Control+a");
  await page.keyboard.type("777");
  await page.keyboard.press("Enter");
  await expect
    .poll(async () => (await apiPage(request, job.pages[0].id)).cells.find((c) => c.id === amount.id)!.value)
    .toBe("777");
  await page.keyboard.press("Control+z");
  await expect(field).toHaveValue(amount.value);
  await expect(field).toBeFocused();
  await page.keyboard.press("Control+y");
  await expect(field).toHaveValue("777");
});

test("⑦ 中断して開き直すと、同じページの同じセルから再開する", async ({ page, request }) => {
  const job = await createJob(request, { seed: 71, error_rate: 0, review_rate: 1, ambiguous_rows: [] });
  await openJob(page, job.job_id);
  const target = input(page, 4, 1);
  await target.click();
  const label = await activeLabel(page);
  await page.waitForResponse((r) => r.url().includes("/position") && r.request().method() === "PUT");
  await page.goto(`/?reopen=1#token=${TOKEN}`); // 別の URL にして、ページを読み直す(ハッシュだけの遷移では読み直さない)
  await expect(page.getByTestId("connection")).toContainText("接続 OK");
  await page.getByTestId(`job-${job.job_id}`).getByRole("button", { name: "続きから" }).click();
  await expect(page.getByTestId("review")).toBeVisible();
  await expect(input(page, 4, 1)).toBeFocused();
  expect(await activeLabel(page)).toBe(label);
});

test("⑧ 〃(展開済み)のバッジが付き、参照元を直すと〃は要確認に戻る", async ({ page, request }) => {
  const job = await createJob(request, { seed: 81, error_rate: 0, review_rate: 1, ambiguous_rows: [] });
  const cells = (await apiPage(request, job.pages[0].id)).cells;
  const ditto = cells.find((c) => c.expansion?.kind === "ditto")!;
  await openJob(page, job.job_id);
  await expect(page.getByTestId(`badge-${ditto.row}-${ditto.col}`)).toContainText("展開済み");
  // 参照元(同じ列の上のセル)を書き換える
  const src = cells.filter((c) => c.col === ditto.col && c.row < ditto.row && c.kind === "text").at(-1)!;
  await input(page, src.row, src.col).click();
  await page.keyboard.press("Control+a");
  await page.keyboard.type("書き換え後");
  await page.keyboard.press("Enter");
  await expect(input(page, ditto.row, ditto.col)).toHaveValue("書き換え後");
  await expect(page.getByTestId(`cell-${ditto.row}-${ditto.col}`)).toContainText("参照元が変わった");
});

test("⑳ Enter の連打(キーリピート)では要確認セルを通過できない。見てから(250 ms 以上)なら通る", async ({ page, request }) => {
  const job = await createJob(request, { seed: 91, error_rate: 0, review_rate: 1, ambiguous_rows: [] });
  const pid = job.pages[0].id;
  await openJob(page, job.job_id);
  const total = (await apiPage(request, pid)).cells.filter((c) => c.review).length;
  // 5 ms 間隔で 40 回(押しっぱなしのキーリピート相当)
  for (let i = 0; i < 40; i++) {
    await page.keyboard.press("Enter");
    await page.waitForTimeout(5);
  }
  await page.waitForTimeout(500);
  const passed = (await apiPage(request, pid)).cells.filter((c) => c.review && c.passed).length;
  expect(passed).toBeLessThan(total / 2); // ほとんどが未通過のまま
  await expect(page.getByTestId("confirm-page")).toBeDisabled();
  // サーバにも下限がある(クライアントが壊れていても、値を直さない連打は通過に数えない)
  const todo = (await apiPage(request, pid)).cells.filter((c) => c.review && !c.passed).slice(0, 3);
  const flags: boolean[] = [];
  for (const c of todo) {
    const r = await request.patch(`/api/pages/${pid}/cells/${c.id}`, { headers: AUTH, data: { action: "confirm" } });
    const cell = ((await r.json()) as { cells: ApiCell[] }).cells.find((x) => x.id === c.id)!;
    flags.push(cell.passed);
  }
  expect(flags.filter(Boolean).length).toBeLessThanOrEqual(1);
});
