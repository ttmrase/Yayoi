import { expect, test, type APIRequestContext } from "@playwright/test";

// 操作の遅延の機械計測(開発機での基準値。実際の PC での値は Windows 実機で測り直す)。値は標準出力に出し、緩い上限だけ検査する(回帰の検知用)。
const TOKEN = "e2e-token-0123456789abcdef0123456789abcdef";
const AUTH = { Authorization: `Bearer ${TOKEN}` };

const pct = (xs: number[], p: number) => [...xs].sort((a, b) => a - b)[Math.min(xs.length - 1, Math.floor((p / 100) * xs.length))];
const fmt = (xs: number[]) => `p50 ${pct(xs, 50).toFixed(0)} ms / p95 ${pct(xs, 95).toFixed(0)} ms / max ${Math.max(...xs).toFixed(0)} ms (n=${xs.length})`;

async function createJob(request: APIRequestContext) {
  const res = await request.post("/api/dev/jobs", {
    headers: AUTH,
    data: { pages: 2, seed: 4242, fill_rows: 29, error_rate: 0.2, review_rate: 0.8, ambiguous_rows: [] },
  });
  return (await res.json()) as { job_id: number; pages: { id: number }[] };
}

test("操作の遅延(Enter でフォーカスが動くまで・保存・ページを開く・確定して次を開く)", async ({ page, request }) => {
  const job = await createJob(request);
  const pid = job.pages[0].id;

  // サーバの往復(29 行 x 4 列 = 116 セルのページ)
  const get: number[] = [];
  const patch: number[] = [];
  const cells = ((await (await request.get(`/api/pages/${pid}`, { headers: AUTH })).json()) as { cells: { id: number; value: string; review: boolean }[] }).cells;
  for (let i = 0; i < 30; i++) {
    let t = performance.now();
    await request.get(`/api/pages/${pid}`, { headers: AUTH });
    get.push(performance.now() - t);
    const c = cells[i];
    t = performance.now();
    await request.patch(`/api/pages/${pid}/cells/${c.id}`, { headers: AUTH, data: { action: "edit", value: c.value } });
    patch.push(performance.now() - t);
  }

  // ページを開く(ジョブ一覧のボタン → 確認画面が表示され、最初のセルの切り出し画像が出るまで)
  await page.goto(`/?perf=${Date.now()}#token=${TOKEN}`);
  await expect(page.getByTestId("connection")).toContainText("接続 OK");
  const t0 = performance.now();
  await page.getByTestId(`job-${job.job_id}`).getByRole("button").click();
  await expect(page.getByTestId("review")).toBeVisible();
  await expect(page.locator(".cell img").first()).toBeVisible();
  // (アプリの CSP は eval を許さないので waitForFunction は使わず、evaluate を繰り返し確かめる)
  await expect
    .poll(() => page.evaluate(() => Array.from(document.images).slice(0, 8).every((i) => i.complete && i.naturalWidth > 0)), { intervals: [10] })
    .toBe(true);
  const open = performance.now() - t0;

  // Enter でフォーカスが次のセルへ動くまで(キー入力 → 画面の更新。保存は待たない)
  const enter: number[] = [];
  for (let i = 0; i < 40; i++) {
    await page.waitForTimeout(300); // 要確認セルは 250 ms 見ないと Enter で通らないので、人の速さで進める
    // ページの中で、Enter の keydown を送ってからフォーカスが動いた次のフレームまでを測る(Playwright の通信の遅れを含めない)
    enter.push(
      await page.evaluate(
        () =>
          new Promise<number>((resolve) => {
            const el = document.activeElement as HTMLElement;
            const before = el.getAttribute("aria-label");
            const t = performance.now();
            el.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true }));
            const tick = () => {
              if (document.activeElement?.getAttribute("aria-label") !== before) resolve(performance.now() - t);
              else if (performance.now() - t > 2000) resolve(Infinity); // 動かなかった(失敗として検出する)
              else requestAnimationFrame(tick);
            };
            requestAnimationFrame(tick);
          }),
      ),
    );
  }

  console.log(`計測(合成 29 行ページ・Chromium・開発機): 
  ページを開く(画像 8 枚が出るまで): ${open.toFixed(0)} ms
  Enter → 次のセルにフォーカス: ${fmt(enter)}
  サーバ GET /api/pages(116 セル): ${fmt(get)}
  サーバ PATCH(セルの保存): ${fmt(patch)}`);

  expect(open).toBeLessThan(3000);
  expect(pct(enter, 95)).toBeLessThan(250);
  expect(pct(patch, 95)).toBeLessThan(150);
  expect(pct(get, 95)).toBeLessThan(300);
});
