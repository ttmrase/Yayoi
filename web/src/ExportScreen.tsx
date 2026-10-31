import { useCallback, useEffect, useState } from "react";
import { ApiError } from "./api";
import { exportFile, exportSample, fetchExports, runExport, validateExport } from "./pageApi";
import { MISSING_LABEL, type ExportIssue, type ExportList, type ExportRecord, type ExportValidation } from "./types";

interface Props {
  /** 検証の指摘から確認画面の該当セルへ */
  onOpenCell: (pageId: number, cellId: number | null) => void;
}

const fmt = (t: number) => new Date(t * 1000).toLocaleString("ja-JP");

function IssueList({ issues, onOpenCell }: { issues: ExportIssue[]; onOpenCell: Props["onOpenCell"] }) {
  return (
    <ul>
      {issues.map((i, k) => (
        <li key={k}>
          {i.label}(ページ ID {i.page_id}
          {i.row !== null && ` ・ ${i.row + 1} 行目`})
          <button onClick={() => onOpenCell(i.page_id, i.cell_id)}>確認画面で開く</button>
        </li>
      ))}
    </ul>
  );
}

export default function ExportScreen({ onOpenCell }: Props) {
  const [data, setData] = useState<ExportList | null>(null);
  const [picked, setPicked] = useState<Set<number>>(new Set());
  const [format, setFormat] = useState("");
  const [result, setResult] = useState<ExportValidation | null>(null);
  const [acceptDup, setAcceptDup] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [sampleDate, setSampleDate] = useState(() => new Date().toLocaleDateString("sv-SE")); // yyyy-mm-dd
  const [sampleMessage, setSampleMessage] = useState<string | null>(null);

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
      setMessage(
        `出力しました(${rec.n_pages} ページ・${rec.n_rows} 行・${rec.files.length} ファイル)。下の履歴からダウンロードできます。`,
      );
      setResult(null);
      setPicked(new Set());
      load();
    } catch (e) {
      if (e instanceof ApiError && e.code === "duplicate_pages") setMessage("すでに出力した行が含まれています。承諾にチェックを入れてください。");
      else if (e instanceof ApiError && e.code === "validation_failed") setMessage("検証エラーがあるため出力しません。");
      else setMessage("出力できませんでした");
    } finally {
      setBusy(false);
    }
  };

  const download = async (id: number, idx: number, name: string | null) => {
    const url = await exportFile(id, idx);
    const a = document.createElement("a");
    a.href = url;
    a.download = name ?? `export_${id}.csv`;
    a.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
  };

  const downloadSample = async () => {
    setSampleMessage(null);
    try {
      const url = await exportSample(format, sampleDate.replace(/-/g, "/"));
      const a = document.createElement("a");
      a.href = url;
      a.download = `sample_${format}.csv`;
      a.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
    } catch (e) {
      setSampleMessage(
        e instanceof ApiError && e.code === "settings_incomplete"
          ? `設定が足りません: ${((e.body.missing_settings ?? []) as string[]).map((m) => MISSING_LABEL[m] ?? m).join("、")}`
          : "サンプルを作れませんでした",
      );
    }
  };

  const blocked = !result || result.errors.length > 0 || result.missing_settings.length > 0 || result.n_rows === 0;
  const needsConsent = (result?.duplicates.length ?? 0) > 0 && !acceptDup;

  return (
    <section className="export" data-testid="export">
      <h1>出力</h1>
      <label>
        形式{" "}
        <select
          data-testid="export-format"
          value={format}
          onChange={(e) => {
            setFormat(e.target.value);
            setResult(null);
          }}
        >
          {data.formats.map((x) => (
            <option key={x.id} value={x.id}>
              {x.label}
            </option>
          ))}
        </select>
      </label>
      {f && (
        <div className="banner" data-testid="export-notes" role="note">
          <b>取り込む前に</b>
          <ul>
            {f.notes.map((n, k) => (
              <li key={k}>{n}</li>
            ))}
          </ul>
        </div>
      )}
      <details className="sample" data-testid="sample-box">
        <summary>弥生の取り込み試験用のサンプル(架空の 2 行)</summary>
        <p className="hint">
          設定の科目・補助科目・税区分のとおりに作った、架空の 2 行の CSV です(実帳票の値は使いません。出力の履歴には残りません)。
          弥生の「テスト用の事業所データ」(または、弥生でバックアップを取ったあとのデータ)に取り込んで、マッチングリストが出ないことを確かめてください。
          日付は、取り込み先の会計期間内にしてください。
        </p>
        <label>
          日付{" "}
          <input type="date" data-testid="sample-date" value={sampleDate} onChange={(e) => setSampleDate(e.target.value)} />
        </label>{" "}
        <button data-testid="download-sample" onClick={() => void downloadSample()}>
          サンプルをダウンロード
        </button>
        {sampleMessage && (
          <p className="ng" role="alert" data-testid="sample-message">
            {sampleMessage}
          </p>
        )}
      </details>
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
              <th>行数</th>
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
                <td>{c.n_rows}</td>
                <td>{c.n_exported === 0 ? "未出力" : c.exported ? "出力済み" : `一部出力済み(${c.n_exported} 行)`}</td>
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
            <div className="banner ng" role="alert" data-testid="export-missing">
              設定が足りません: {result.missing_settings.map((m) => MISSING_LABEL[m] ?? m).join("、")}(設定画面で入力してください)
            </div>
          )}
          {result.errors.length > 0 ? (
            <div className="banner ng" role="alert" data-testid="export-issues">
              検証エラー {result.errors.length} 件(エラーがあるうちは出力できません)
              <IssueList issues={result.errors} onOpenCell={onOpenCell} />
            </div>
          ) : (
            <p role="status" data-testid="export-ok">
              検証しました: {result.n_rows} 行を出力できます。
            </p>
          )}
          {result.warnings.length > 0 && (
            <div className="banner" role="note" data-testid="export-warnings">
              注意 {result.warnings.length} 件(出力は止まりません)
              <IssueList issues={dedupe(result.warnings)} onOpenCell={onOpenCell} />
            </div>
          )}
          {result.duplicates.length > 0 && (
            <div className="banner" role="alert" data-testid="export-duplicates">
              すでに出力した行が含まれています(ページ {result.duplicates.length} 件・合計 {result.duplicates.reduce((a, d) => a + d.n_rows, 0)} 行。同じ画像・同じ確定値)。
              弥生は再インポートを警告せずに重複登録するので、二重に取り込まないよう注意してください。
              <label>
                <input type="checkbox" data-testid="accept-duplicates" checked={acceptDup} onChange={(e) => setAcceptDup(e.target.checked)} /> 重複を承知で出力する
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
              <th>ファイル(ハッシュ)</th>
            </tr>
          </thead>
          <tbody>
            {data.exports.map((e) => (
              <HistoryRow key={e.id} e={e} onDownload={download} />
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

/** 同じ理由の注意は 1 件にまとめる(行ごとに出すと長くなる)。先頭の 1 件へ移動できる。 */
function dedupe(list: ExportIssue[]): ExportIssue[] {
  const seen = new Set<string>();
  return list.filter((i) => {
    if (i.code !== "title_lost" && i.code !== "tax_amount_manual") return true;
    if (seen.has(i.code)) return false;
    seen.add(i.code);
    return true;
  });
}

function HistoryRow({ e, onDownload }: { e: ExportRecord; onDownload: (id: number, idx: number, name: string | null) => Promise<void> }) {
  return (
    <tr data-testid={`export-${e.id}`}>
      <td>{fmt(e.created_at)}</td>
      <td>{e.format_label}</td>
      <td>{e.n_pages}</td>
      <td>{e.n_rows}</td>
      <td>
        <ul className="plain">
          {e.files.map((file) => (
            <li key={file.idx} title={file.sha256}>
              <button data-testid={`download-${e.id}-${file.idx}`} onClick={() => void onDownload(e.id, file.idx, file.name)}>
                ダウンロード
              </button>{" "}
              {file.payment !== null && <b>{file.payment}</b>} {file.n_rows} 行 <code>{file.sha256.slice(0, 10)}…</code>
            </li>
          ))}
        </ul>
      </td>
    </tr>
  );
}
