import { useEffect, useState } from "react";
import { clearTaxMemory, deleteTaxMemory, fetchTaxMemory, putTaxMemory } from "./pageApi";
import type { TaxMemory } from "./types";

/**
 * 摘要ごとの税区分の記憶(確認画面で行の税区分を選んで確定すると増える)。編集・削除できる。
 * 摘要の文字列を保存しているので、精度レポートには含めない。税区分を選び直した記憶は、次のページから提案に使う。
 */
export default function TaxMemoryEditor() {
  const [mem, setMem] = useState<TaxMemory | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);
  useEffect(() => {
    void fetchTaxMemory()
      .then(setMem)
      .catch(() => setError("記憶を読み込めません"));
  }, []);

  const run = (p: Promise<TaxMemory>) => {
    setError(null);
    p.then(setMem).catch(() => setError("保存できませんでした"));
  };

  if (error && !mem) return <p role="alert">{error}</p>;
  if (!mem || mem.exempt) return null;
  return (
    <fieldset className="settings-table" data-testid="tax-memory">
      <legend>摘要ごとの税区分の記憶</legend>
      <p className="hint">
        確認画面で行の税区分を選んでページを確定すると、その摘要(空白や全角・半角の違いは無視して照合)と税区分の対応を覚え、次から同じ摘要の行に「提案」として出します。
        ここで税区分を直したり、記憶を消したりできます(摘要の文字列を保存しています)。
      </p>
      {error && (
        <p className="ng" role="alert">
          {error}
        </p>
      )}
      {mem.entries.length === 0 ? (
        <p data-testid="tax-memory-empty">まだ記憶はありません。</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>摘要</th>
              <th>税区分</th>
              <th>使った回数</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {mem.entries.map((e) => (
              <tr key={e.id} data-testid={`memory-${e.id}`}>
                <td>{e.description}</td>
                <td>
                  <select
                    aria-label={`摘要「${e.description}」の税区分`}
                    value={e.tax_category}
                    onChange={(ev) => run(putTaxMemory(e.id, ev.target.value))}
                  >
                    {!mem.options.includes(e.tax_category) && <option value={e.tax_category}>{e.tax_category}(今は選べません)</option>}
                    {mem.options.map((o) => (
                      <option key={o} value={o}>
                        {o}
                      </option>
                    ))}
                  </select>
                  {!e.allowed && <div className="ng small">今の課税方式・経理方式では選べないため、提案されません</div>}
                </td>
                <td>{e.n_used}</td>
                <td>
                  <button aria-label={`摘要「${e.description}」の記憶を削除`} onClick={() => run(deleteTaxMemory(e.id))}>
                    削除
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {mem.entries.length > 0 &&
        (confirmClear ? (
          <span role="alert">
            <button
              className="danger"
              data-testid="tax-memory-clear-confirm"
              onClick={() => {
                run(clearTaxMemory());
                setConfirmClear(false);
              }}
            >
              本当にすべて削除する
            </button>{" "}
            <button onClick={() => setConfirmClear(false)}>やめる</button>
          </span>
        ) : (
          <button data-testid="tax-memory-clear" onClick={() => setConfirmClear(true)}>
            記憶をすべて削除
          </button>
        ))}
    </fieldset>
  );
}
