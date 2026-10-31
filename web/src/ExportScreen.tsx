import { useCallback, useEffect, useState } from "react";
import { ApiError } from "./api";
import { exportFile, fetchExports, runExport, validateExport } from "./pageApi";
import { MISSING_LABEL, type ExportIssue, type ExportList, type ExportValidation } from "./types";

interface Props {
  /** 検証の指摘から確認画面の該当セルへ */
  onOpenCell: (pageId: number, cellId: number | null) => void;
}

const fmt = (t: number) => new Date(t * 1000).toLocaleString("ja-JP");

export default function ExportScreen({ onOpenCell }: Props) {
  const [data, setData] = useState<ExportList | null>(null);
  const [picked, setPicked] = useState<Set<number>>(new Set());
  const [format, setFormat] = useState("");
  const [result, setResult] = useState<ExportValidation | null>(null);
  const [acceptDup, setAcceptDup] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    void fetchExports().then((d) => {
      setData(d);
      setFormat((f) => f || d.formats[0]?.id || "");
    });
  }, []);
  useEffect(load, [load]);

  if (!data) return <p>読み込み中…</p>;
  const f = data.formats.find((x) => x.id === format);
  const ids = [...picked];
  const toggle = (id: number) => {
    const n = new Set(picked);
    if (n.has(id)) n.delete(id);
    else n.add(id);
    setPicked(n);
    setResult(null);
    setMessage(null);
  };

  const validate = async () => {
    setBusy(true);
    try {
      setResult(await validateExport(ids, format));
      setAcceptDup(false);
      setMessage(null);
    } catch (e) {
      setMessage(e instanceof ApiError ? `検証できません (${e.code || e.status})` : "検証できません");
    } finally {
      setBusy(false);
    }
  };

  const run = async () => {
    setBusy(true);
    try {
      const rec = await runExport(ids, format, acceptDup);
      setMessage(`出力しました(${rec.n_pages} ページ・${rec.n_rows} 行)。下の履歴からダウンロードできます。`);
      setResult(null);
      setPicked(new Set());
      load();
    } catch (e) {
      if (e instanceof ApiError && e.code === "duplicate_pages") setMessage("すでに出力したページが含まれています。承諾にチェックを入れてください。");
      else if (e instanceof ApiError && e.code === "validation_failed") setMessage("検証エラーがあるため出力しません。");
      else setMessage("出力できませんでした");
    } finally {
      setBusy(false);
    }
  };

  const download = async (id: number, name: string | null) => {
    const url = await exportFile(id);
    const a = document.createElement("a");
    a.href = url;
    a.download = name ?? `export_${id}.csv`;
    a.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
  };

  const blocked = !result || result.issues.length > 0 || result.missing_settings.length > 0 || result.n_rows === 0;
  const needsConsent = (result?.duplicates.length ?? 0) > 0 && !acceptDup;

  return (
    <section className="export" data-testid="export">
      <h1>出力</h1>
      <label>
        形式{" "}
        <select value={format} onChange={(e) => { setFormat(e.target.value); setResult(null); }}>
          {data.formats.map((x) => (
            <option key={x.id} value={x.id}>
              {x.label}
            </option>
          ))}
        </select>
      </label>
      {f?.provisional && (
        <div className="banner" data-testid="provisional-banner" role="note">
          この形式は仮のものです。弥生のインポート形式(列・必須項目・税区分の値)は確定後に差し替えます。
        </div>
      )}
      <h2>出力するページ(確定済み)</h2>
      {data.candidates.length === 0 ? (
        <p>確定したページがありません。確認画面でページを確定してください。</p>
      ) : (
        <table className="jobs" data-testid="export-candidates">
          <thead>
            <tr>
              <th />
              <th>ジョブ</th>
              <th>ページ</th>
              <th>タイトル</th>
              <th>出力</th>
            </tr>
          </thead>
          <tbody>
            {data.candidates.map((c) => (
              <tr key={c.id} data-testid={`candidate-${c.id}`}>
                <td>
                  <input type="checkbox" aria-label={`${c.job_name} ${c.idx + 1} ページを出力する`} checked={picked.has(c.id)} onChange={() => toggle(c.id)} />
                </td>
                <td>{c.job_name}</td>
                <td>{c.idx + 1}</td>
                <td>{c.title ?? <span className="ng">未入力</span>}</td>
                <td>{c.exported ? "出力済み" : "未出力"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p>
        <button
          onClick={() => setPicked(new Set(data.candidates.filter((c) => !c.exported).map((c) => c.id)))}
          disabled={data.candidates.every((c) => c.exported)}
        >
          未出力のページを全部選ぶ
        </button>{" "}
        <button data-testid="export-validate" disabled={ids.length === 0 || busy} onClick={() => void validate()}>
          検証
        </button>
      </p>
      {result && (
        <div data-testid="export-result">
          {result.missing_settings.length > 0 && (
            <div className="banner ng" role="alert">
              設定が足りません: {result.missing_settings.map((m) => MISSING_LABEL[m] ?? m).join("、")}(設定画面で入力してください)
            </div>
          )}
          {result.issues.length > 0 ? (
            <div className="banner ng" role="alert" data-testid="export-issues">
              検証エラー {result.issues.length} 件
              <ul>
                {result.issues.map((i: ExportIssue, k) => (
                  <li key={k}>
                    {i.label}(ページ ID {i.page_id}
                    {i.row !== null && ` ・ ${i.row + 1} 行目`})
                    <button onClick={() => onOpenCell(i.page_id, i.cell_id)}>確認画面で開く</button>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <p role="status" data-testid="export-ok">
              検証しました: {result.n_rows} 行を出力できます。
            </p>
          )}
          {result.duplicates.length > 0 && (
            <div className="banner" role="alert" data-testid="export-duplicates">
              すでに出力したページが {result.duplicates.length} 件含まれています(同じ画像・同じ確定値)。二重に取り込まないよう注意してください。
              <label>
                <input type="checkbox" checked={acceptDup} onChange={(e) => setAcceptDup(e.target.checked)} /> 重複を承知で出力する
              </label>
            </div>
          )}
          <button data-testid="export-run" disabled={blocked || needsConsent || busy} onClick={() => void run()}>
            出力する
          </button>
        </div>
      )}
      {message && (
        <p role="status" data-testid="export-message">
          {message}
        </p>
      )}
      <h2>出力の履歴</h2>
      {data.exports.length === 0 ? (
        <p>まだ出力していません。</p>
      ) : (
        <table className="jobs" data-testid="export-history">
          <thead>
            <tr>
              <th>日時</th>
              <th>形式</th>
              <th>ページ</th>
              <th>行数</th>
              <th>ファイルのハッシュ</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {data.exports.map((e) => (
              <tr key={e.id}>
                <td>{fmt(e.created_at)}</td>
                <td>{e.format_label}</td>
                <td>{e.n_pages}</td>
                <td>{e.n_rows}</td>
                <td title={e.file_sha256}>
                  <code>{e.file_sha256.slice(0, 12)}…</code>
                </td>
                <td>
                  <button onClick={() => void download(e.id, e.file_name)}>ダウンロード</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
