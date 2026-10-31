import { useEffect, useState } from "react";
import { fetchLearning, resetLearning } from "./pageApi";
import type { LearningData, PageMetric } from "./types";

const pct = (v: number | null) => (v === null ? "—" : `${(v * 100).toFixed(1)}%`);

/** 確定したページ順の折れ線(要確認率・直した割合)。外部ライブラリを使わない小さな SVG。 */
function Trend({ pages }: { pages: PageMetric[] }) {
  const W = 560;
  const H = 160;
  const pad = 28;
  const x = (i: number) => pad + (pages.length <= 1 ? 0.5 : i / (pages.length - 1)) * (W - 2 * pad);
  const y = (v: number) => H - pad - v * (H - 2 * pad);
  const line = (key: "review_rate" | "correction_rate") =>
    pages
      .map((p, i) => (p[key] === null ? null : `${x(i).toFixed(1)},${y(p[key]!).toFixed(1)}`))
      .filter((s): s is string => s !== null)
      .join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="確定したページ順の、要確認率と直した割合の推移" className="trend" data-testid="trend">
      {[0, 0.5, 1].map((g) => (
        <g key={g}>
          <line x1={pad} x2={W - pad} y1={y(g)} y2={y(g)} className="grid-line" />
          <text x={pad - 4} y={y(g) + 4} textAnchor="end" fontSize="10">
            {g * 100}%
          </text>
        </g>
      ))}
      <polyline points={line("review_rate")} className="series review" fill="none" />
      <polyline points={line("correction_rate")} className="series corrected" fill="none" />
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
  const [message, setMessage] = useState<string | null>(null);
  const load = () => void fetchLearning().then(setData).catch(() => setError(true));
  useEffect(load, []);
  const doReset = async () => {
    try {
      const r = await resetLearning();
      setMessage(`削除しました(切り出し画像 ${r.deleted_images} 枚)。`);
    } catch {
      setMessage("削除できませんでした");
    }
    setConfirmReset(false);
    load();
  };
  if (error) return <p role="alert">読み込めません</p>;
  if (!data) return <p>読み込み中…</p>;
  const s = data.summary;
  return (
    <section className="learning" data-testid="learning">
      <h1>学習</h1>
      <p className="hint">
        確認画面での実績です(値の内容は記録しません)。いまは出荷時の重み(世代 0)だけを使っています。書き手ごとの追加学習(世代の追加・採否)は今後のバージョンで入ります。
        確定したページの摘要・科目・金額は、次のページの認識(辞書・語彙・金額の履歴)にすでに反映されています。
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
          <Trend pages={data.pages} />
          <p className="legend">
            <span className="swatch review" /> 要確認率 <span className="swatch corrected" /> 直した割合
          </p>
        </>
      )}
      <h2>世代</h2>
      <table className="jobs" data-testid="generations">
        <thead>
          <tr>
            <th>世代</th>
            <th>状態</th>
            <th>学習に使った確定ページ</th>
            <th>この世代で認識したページ</th>
          </tr>
        </thead>
        <tbody>
          {data.generations.map((g, i) => (
            <tr key={g.id}>
              <td>{i === 0 ? "世代 0(出荷時)" : `世代 ${i}`}</td>
              <td>{g.status === "active" ? "使用中" : g.status}</td>
              <td>{g.n_pages}</td>
              <td>{g.n_recognized}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p>
        <button disabled={!data.can_rollback} title="世代が増えると選べるようになります">
          この世代に戻す
        </button>{" "}
        <span className="hint">追加学習した世代ができるまでは、戻す対象がありません。</span>
      </p>
      <h2>学習データの削除(リセット)</h2>
      <p className="hint">
        確定したセルの切り出し画像(学習データ)を削除し、学習(摘要の辞書・科目の語彙・金額の履歴)を最初からやり直します。確定した値・出力の履歴・設定は残ります。
        削除後は、確定済みのページでも切り出し画像は表示されません。
      </p>
      {!confirmReset ? (
        <button data-testid="reset-learning" disabled={!data.can_reset} onClick={() => setConfirmReset(true)}>
          学習データを削除する(リセット)
        </button>
      ) : (
        <span role="alert">
          <button data-testid="reset-learning-confirm" className="danger" onClick={() => void doReset()}>
            本当に削除する
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
