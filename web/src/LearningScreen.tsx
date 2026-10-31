import { useEffect, useState } from "react";
import TrainingPanel from "./TrainingPanel";
import { GEN_STATUS_TEXT, reasonText } from "./trainingText";
import { activateGeneration, fetchLearning, learningReport, resetGenerations } from "./pageApi";
import type { Generation, LearningData, PageMetric } from "./types";

const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(1)}%`);
const fmt = (t: number | null) => (t === null ? "—" : new Date(t * 1000).toLocaleString("ja-JP"));

/** 確定したページ順の折れ線(要確認率・直した割合・見逃しの観測)と、世代が切り替わった位置。外部ライブラリを使わない小さな SVG。 */
function Trend({ pages, genNumber }: { pages: PageMetric[]; genNumber: Record<number, number> }) {
  const W = 560;
  const H = 160;
  const pad = 28;
  const x = (i: number) => pad + (pages.length <= 1 ? 0.5 : i / (pages.length - 1)) * (W - 2 * pad);
  const y = (v: number) => H - pad - v * (H - 2 * pad);
  const rate = (p: PageMetric, key: "review_rate" | "correction_rate" | "miss") =>
    key === "miss" ? (p.n_text ? p.n_unflagged_corrected / p.n_text : null) : p[key];
  const line = (key: "review_rate" | "correction_rate" | "miss") =>
    pages
      .map((p, i) => {
        const v = rate(p, key);
        return v === null ? null : `${x(i).toFixed(1)},${y(v).toFixed(1)}`;
      })
      .filter((s): s is string => s !== null)
      .join(" ");
  const switches = pages.flatMap((p, i) => (i > 0 && p.generation !== pages[i - 1].generation ? [{ i, gen: p.generation }] : []));
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label="確定したページ順の、要確認率・直した割合・見逃しの観測の推移。世代が切り替わった位置に印があります"
      className="trend"
      data-testid="trend"
    >
      {[0, 0.5, 1].map((g) => (
        <g key={g}>
          <line x1={pad} x2={W - pad} y1={y(g)} y2={y(g)} className="grid-line" />
          <text x={pad - 4} y={y(g) + 4} textAnchor="end" fontSize="10">
            {g * 100}%
          </text>
        </g>
      ))}
      {switches.map(({ i, gen }) => {
        const xs = (x(i) + x(i - 1)) / 2;
        return (
          <g key={`g${i}`} data-testid="trend-switch">
            <line x1={xs} x2={xs} y1={pad - 8} y2={H - pad} className="switch-line" />
            <text x={xs + 3} y={pad - 1} fontSize="10">
              世代 {genNumber[gen] ?? "?"}
            </text>
          </g>
        );
      })}
      <polyline points={line("review_rate")} className="series review" fill="none" />
      <polyline points={line("correction_rate")} className="series corrected" fill="none" />
      <polyline points={line("miss")} className="series miss" fill="none" />
      {pages.map((p, i) => (
        <text key={p.page_id} x={x(i)} y={H - 8} textAnchor="middle" fontSize="10">
          {i + 1}
        </text>
      ))}
    </svg>
  );
}

export default function LearningScreen() {
  const [data, setData] = useState<LearningData | null>(null);
  const [error, setError] = useState(false);
  const [confirmReset, setConfirmReset] = useState(false);
  const [deleteData, setDeleteData] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const load = () => void fetchLearning().then(setData).catch(() => setError(true));
  useEffect(load, []);
  const doReset = async () => {
    try {
      const r = await resetGenerations(deleteData);
      setMessage(
        `出荷時の重み(世代 0)に戻しました(破棄した世代 ${r.discarded})。` +
          (deleteData ? `切り出し画像 ${r.deleted_images} 枚と学習状態も削除しました。` : "確定データは残してあります。"),
      );
    } catch {
      setMessage("戻せませんでした");
    }
    setConfirmReset(false);
    load();
  };
  const doRollback = async (g: Generation) => {
    try {
      await activateGeneration(g.id);
      setMessage(`世代 ${g.number} を使用中にしました。これから認識するページから使います(認識済みのページは変わりません)。`);
    } catch {
      setMessage("戻せませんでした(この世代の重みが見つかりません)");
    }
    load();
  };
  const downloadReport = async () => {
    try {
      const url = await learningReport();
      const a = document.createElement("a");
      a.href = url;
      a.download = "accuracy_report.json";
      a.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
    } catch {
      setMessage("レポートを作れませんでした");
    }
  };
  if (error) return <p role="alert">読み込めません</p>;
  if (!data) return <p>読み込み中…</p>;
  const s = data.summary;
  return (
    <section className="learning" data-testid="learning">
      <h1>学習</h1>
      <p className="hint">
        確認画面での実績です(値の内容は記録しません)。使用中の世代:{" "}
        <b data-testid="active-generation">
          {(() => {
            const g = data.generations.find((x) => x.id === data.active_id);
            return g ? (g.kind === "shipped" ? "世代 0(出荷時)" : `世代 ${g.number}`) : "—";
          })()}
        </b>
        。確定したページの摘要・科目・金額は、次のページの認識(辞書・語彙・金額の履歴)にすでに反映されています。
      </p>
      <div className="tiles" data-testid="learning-summary">
        <div>
          <b>{s.n_pages}</b>
          <span>確定したページ</span>
        </div>
        <div>
          <b>{pct(s.review_rate)}</b>
          <span>要確認だったセル(空欄を除く)</span>
        </div>
        <div>
          <b>{pct(s.correction_rate)}</b>
          <span>直したセル</span>
        </div>
        <div title="要確認でないのに利用者が直したセル。見逃しの観測された件数で、実際の見逃しはこれ以上あり得ます">
          <b>{s.unflagged_corrected}</b>
          <span>要確認でなかったのに直したセル</span>
        </div>
        <div>
          <b>{s.median_review_seconds === null ? "—" : `${Math.round(s.median_review_seconds)} 秒`}</b>
          <span>1 ページの確認時間(中央値)</span>
        </div>
      </div>
      <h2>推移(確定した順)</h2>
      {data.pages.length === 0 ? (
        <p>確定したページがまだありません。</p>
      ) : (
        <>
          <Trend pages={data.pages} genNumber={Object.fromEntries(data.generations.map((g) => [g.id, g.number]))} />
          <p className="legend">
            <span className="swatch review" /> 要確認率 <span className="swatch corrected" /> 直した割合 <span className="swatch miss" /> 見逃しの観測
            (要確認でないのに直したセル。実際の見逃しはこれ以上あり得ます)
          </p>
        </>
      )}
      <p>
        <button data-testid="download-report" onClick={() => void downloadReport()}>
          精度レポートを保存(統計のみ)
        </button>{" "}
        <span className="hint">ページ数・要確認率・直した割合・確認時間だけを含みます。帳票の内容(日付・科目・摘要・金額の値)は含まれません。</span>
      </p>
      <TrainingPanel onFinished={load} />
      <h2>世代</h2>
      <p className="hint">
        世代は、重みと確信度の組です。出荷時の重みが世代 0 で、あなたの確定データから追加学習した世代が 1 以降に並びます。「実績」は、その世代で認識して確定したページでの値です
        (要確認率・直した割合・見逃しの観測=要確認でないのに直したセルの割合)。
      </p>
      <table className="jobs" data-testid="generations">
        <thead>
          <tr>
            <th>世代</th>
            <th>状態</th>
            <th>作成</th>
            <th>学習 / 評価ページ</th>
            <th>採否の評価(評価用ページで 親 → この世代)</th>
            <th>認識したページ</th>
            <th>実績: ページ</th>
            <th>要確認率</th>
            <th>直した割合</th>
            <th>見逃し(観測)</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          {data.generations.map((g) => (
            <tr key={g.id} data-testid={`generation-${g.number}`} className={g.status === "active" ? "active-row" : undefined}>
              <td>{g.kind === "shipped" ? "世代 0(出荷時)" : `世代 ${g.number}`}</td>
              <td>
                {GEN_STATUS_TEXT[g.status] ?? g.status}
                {g.reject_reason && <div className="hint">{reasonText(g.reject_reason)}</div>}
              </td>
              <td>{g.kind === "shipped" ? "—" : fmt(g.created_at)}</td>
              <td>{g.kind === "shipped" ? "—" : `${g.n_train_pages ?? "—"} / ${g.n_eval_pages ?? "—"}`}</td>
              <td data-testid={`generation-${g.number}-eval`}>
                {g.eval ? (
                  <>
                    正解率 {pct(g.eval.parent.accuracy)} → {pct(g.eval.candidate.accuracy)}
                    <br />
                    要確認 {pct(g.eval.parent.review_rate)} → {pct(g.eval.candidate.review_rate)}
                    <br />
                    見逃し {g.eval.parent.n_miss} → {g.eval.candidate.n_miss} 件 / {g.eval.candidate.n_cells} セル
                    {g.eval.parent.calib && g.eval.candidate.calib && (
                      <>
                        <br />
                        較正(NLL) {g.eval.parent.calib.nll.toFixed(2)} → {g.eval.candidate.calib.nll.toFixed(2)} / 予測した見逃し{" "}
                        {g.eval.candidate.calib.expected_miss.toFixed(1)} 件に対し観測 {g.eval.candidate.calib.observed_miss} 件
                      </>
                    )}
                  </>
                ) : (
                  "—"
                )}
              </td>
              <td>{g.n_recognized}</td>
              <td>{g.observed?.n_pages ?? 0}</td>
              <td>{pct(g.observed?.review_rate)}</td>
              <td>{pct(g.observed?.correction_rate)}</td>
              <td>{pct(g.observed?.miss_rate)}</td>
              <td>
                {g.status !== "active" && g.restorable && (
                  <button data-testid={`rollback-${g.number}`} onClick={() => void doRollback(g)}>
                    この世代に戻す
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="hint">
        「この世代に戻す」は、重みが残っている世代(過去の世代・出荷時)にだけ出ます。何度でも切り替えられます。
      </p>
      <h2>出荷時の重みに戻す(リセット)</h2>
      <p className="hint">
        追加学習した世代をすべて捨てて、出荷時の重み(世代 0)に戻します。確定した値・出力の履歴・設定は残ります。確定データ(切り出し画像・辞書・金額の履歴)を残すかを選べます。
        元画像の保管の設定(出力後に削除)では、切り出し画像と追加学習用の入力は消えません(ここで削除します)。摘要ごとの税区分の記憶(摘要の文字列を保存しています)は、ここでは消えません。設定画面の「摘要ごとの税区分の記憶」から削除してください。
      </p>
      <fieldset className="reset-choice">
        <legend>確定データ</legend>
        <label>
          <input type="radio" name="reset-data" data-testid="reset-keep-data" checked={!deleteData} onChange={() => setDeleteData(false)} />
          残す(次の追加学習の元になります)
        </label>
        <br />
        <label>
          <input type="radio" name="reset-data" data-testid="reset-delete-data" checked={deleteData} onChange={() => setDeleteData(true)} />
          切り出し画像と学習状態も削除する(辞書・語彙・金額の履歴を最初から。確定済みのページでも切り出し画像は表示されません)
        </label>
      </fieldset>
      {!confirmReset ? (
        <button data-testid="reset-learning" disabled={!data.can_reset} onClick={() => setConfirmReset(true)}>
          出荷時の重みに戻す(リセット)
        </button>
      ) : (
        <span role="alert">
          <button data-testid="reset-learning-confirm" className="danger" onClick={() => void doReset()}>
            本当に戻す{deleteData ? "(確定データも削除)" : ""}
          </button>{" "}
          <button onClick={() => setConfirmReset(false)}>やめる</button>
        </span>
      )}
      {message && (
        <p role="status" data-testid="reset-message">
          {message}
        </p>
      )}
    </section>
  );
}
