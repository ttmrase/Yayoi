import { useEffect, useRef, useState } from "react";
import { ApiError } from "./api";
import {
  ackBackupResult,
  backupDownloadLink,
  cancelBackup,
  createBackup,
  deleteBackupUndo,
  discardBackup,
  fetchBackup,
  restoreBackup,
  uploadBackup,
} from "./pageApi";
import type { BackupStatus, BackupSummary } from "./types";

const POLL_BUSY_MS = 1000;
const POLL_IDLE_MS = 5000;

const BACKUP_ERROR: Record<string, string> = {
  not_a_zip: "バックアップのファイルではありません(zip として読めません)。",
  entry_name_invalid: "バックアップの中に、許可されていない名前のファイルがあります。安全のため読み込みません。",
  entry_type_invalid: "バックアップの中に、許可されていない種類のファイルがあります。安全のため読み込みません。",
  entry_duplicate: "バックアップの中に、同じ名前のファイルが重複しています。安全のため読み込みません。",
  hash_mismatch: "バックアップが壊れているか、書き換えられています(内容が記録と一致しません)。",
  hash_list_mismatch: "バックアップが壊れているか、書き換えられています(ファイルの一覧が一致しません)。",
  file_list_mismatch: "バックアップが壊れているか、書き換えられています(ファイルの一覧が一致しません)。",
  size_mismatch: "バックアップが壊れているか、書き換えられています(大きさが一致しません)。",
  manifest_missing: "このアプリのバックアップではありません(記録がありません)。",
  manifest_invalid: "このアプリのバックアップではありません(記録が読めません)。",
  format_unsupported: "このアプリのバックアップではありません。",
  version_unsupported: "バックアップの形式の版が違います。",
  db_too_new: "新しい版のアプリで作られたバックアップです。アプリを更新してから、もう一度お試しください。",
  db_missing: "バックアップの中にデータベースがありません。",
  db_invalid: "バックアップの中のデータベースが壊れています。",
  db_schema_mismatch: "バックアップの中のデータベースの構造が、このアプリのものと一致しません。安全のため読み込みません。",
  db_path_invalid: "バックアップの中のデータベースに、安全でない内容が含まれています。読み込みません。",
  too_many_entries: "バックアップが大きすぎるか、不自然な内容です(ファイルが多すぎます)。",
  entry_too_large: "バックアップが大きすぎるか、不自然な内容です。",
  total_too_large: "バックアップが大きすぎるか、不自然な内容です。",
  too_large: "バックアップが大きすぎます。",
  ratio_too_high: "バックアップが不自然に圧縮されています。安全のため読み込みません。",
  disk_full: "ディスクの空きが足りません。",
  busy: "認識または学習の最中は、復元の準備はできません。終わってからやり直してください。",
  backup_busy: "ほかのバックアップの操作の最中です。終わってからやり直してください。",
  empty_file: "空のファイルです。",
  timeout: "時間がかかりすぎたため止めました。",
  worker_crashed: "処理のプログラムが異常終了しました。",
};
const errorText = (code: string): string => BACKUP_ERROR[code] ?? `失敗しました(${code})`;

const STAGE_TEXT: Record<string, string> = {
  archive: "バックアップを作成中",
  verify: "バックアップの内容を検査中",
  extract: "復元の準備(取り出しと検査)中",
  check_db: "データベースを検査中",
};

const fmtBytes = (n: number): string => {
  if (n >= 1 << 30) return `${(n / (1 << 30)).toFixed(1)} GB`;
  if (n >= 1 << 20) return `${(n / (1 << 20)).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(n / 1024))} KB`;
};
const fmtTime = (t: number): string =>
  new Date(t * 1000).toLocaleString("ja-JP", { year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });

function Summary({ s, testid }: { s: BackupSummary; testid: string }) {
  return (
    <ul className="hint" data-testid={testid}>
      <li>作成: {fmtTime(s.created_at)}(アプリの版 {s.app_version || "—"})</li>
      <li>
        ページ {s.n_pages}(確定済み {s.n_confirmed_pages})・切り出し画像 {s.n_images}・追加学習の世代 {s.n_generations}
      </li>
      <li>
        大きさ: {fmtBytes(s.total_bytes)}(zip {fmtBytes(s.zip_bytes)})・{s.includes_originals ? "元画像を含む" : "元画像は含まない"}
      </li>
    </ul>
  );
}

/** バックアップの作成と、バックアップからの復元(設定画面)。保存先は「名前を付けて保存」で選ぶ。復元の入れ替えは、アプリを開き直したとき。 */
export default function BackupPanel() {
  const [st, setSt] = useState<BackupStatus | null>(null);
  const [originals, setOriginals] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [upload, setUpload] = useState<number | null>(null);
  const [confirming, setConfirming] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const load = () =>
    void fetchBackup()
      .then(setSt)
      .catch(() => setError("読み込めません"));
  useEffect(load, []);
  const busy = (st?.running ?? false) || upload !== null;
  useEffect(() => {
    const id = window.setInterval(load, busy ? POLL_BUSY_MS : POLL_IDLE_MS);
    return () => window.clearInterval(id);
  }, [busy]);

  const run = async (f: () => Promise<BackupStatus>) => {
    setError(null);
    try {
      setSt(await f());
    } catch (e) {
      setError(errorText(e instanceof ApiError ? e.code || String(e.status) : "network"));
    }
    load();
  };

  const save = async () => {
    setError(null);
    try {
      const { url } = await backupDownloadLink();
      const a = document.createElement("a");
      a.href = url;
      a.download = st?.created?.name ?? "yayoi-backup.zip";
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch {
      setError("保存を始められませんでした。もう一度お試しください。");
    }
  };

  const pick = async (file: File | undefined) => {
    if (!file) return;
    setError(null);
    setConfirming(false);
    setUpload(0);
    try {
      setSt(await uploadBackup(file, setUpload));
    } catch (e) {
      setError(errorText(e instanceof ApiError ? e.code || String(e.status) : "network"));
    } finally {
      setUpload(null);
      if (fileRef.current) fileRef.current.value = "";
    }
    load();
  };

  if (!st) return <section data-testid="backup">{error ? <p role="alert">{error}</p> : <p>読み込み中…</p>}</section>;
  const pct = st.total > 0 ? Math.min(100, Math.round((st.done / st.total) * 100)) : 0;
  const progressLabel = `${STAGE_TEXT[st.stage ?? ""] ?? "処理中"} ${pct}%`;
  const failure = st.error ? errorText(st.error) : error;

  return (
    <section data-testid="backup" aria-labelledby="backup-h">
      <h2 id="backup-h">バックアップと復元</h2>
      <p className="hint">
        データ(確定した値・切り出し画像・追加学習の世代・設定・税区分の記憶など)を 1 つの zip にまとめます。バックアップには、帳票の内容(確定した摘要・金額など)が含まれます。
        保管と持ち運びにご注意ください。
      </p>

      {st.pending && (
        <p role="status" className="notice-banner" data-testid="backup-pending">
          <b>復元の準備ができています。</b>アプリを一度閉じて開き直すと、バックアップ(ページ {st.pending.summary.n_pages})の内容に置き換わります。いまのデータは自動で退避されます。{" "}
          <button data-testid="backup-cancel-restore" onClick={() => void run(cancelBackup)}>
            取り消す
          </button>
        </p>
      )}
      {st.result && (
        <p role="status" className={st.result.ok ? "hint" : "ng"} data-testid="backup-result">
          {st.result.ok
            ? `バックアップから復元しました(${fmtTime(st.result.at)}${
                st.result.pairs ? `。追加学習の世代 ${st.result.pairs.checked} 件を照合${st.result.pairs.broken ? `、使えない世代 ${st.result.pairs.broken} 件` : ""}` : ""
              })。`
            : `復元の入れ替えに失敗しました(${st.result.code ?? ""})。データは元のままです。`}{" "}
          <button data-testid="backup-ack" onClick={() => void run(ackBackupResult)}>
            確認した
          </button>
        </p>
      )}
      {st.undo && (
        <p className="hint" data-testid="backup-undo">
          復元の前のデータを退避しています({fmtBytes(st.undo.bytes)}、{fmtTime(st.undo.at)})。復元の結果に問題がなければ、削除して容量を空けられます。{" "}
          <button data-testid="backup-delete-undo" onClick={() => void run(deleteBackupUndo)}>
            退避したデータを削除
          </button>
        </p>
      )}

      <h3>バックアップを作成</h3>
      <p>
        <label>
          <input type="checkbox" data-testid="backup-originals" checked={originals} onChange={(e) => setOriginals(e.target.checked)} /> 取り込んだ元画像(スキャンした帳票)も含める
        </label>
        <span className="hint"> (大きくなります。含めないと、確定前のページの再認識や罫線の再切り出しは、復元後にできません)</span>
      </p>
      <p>
        <button data-testid="backup-create" disabled={busy} onClick={() => void run(() => createBackup(originals))}>
          バックアップを作成
        </button>
      </p>
      {st.running && st.op === "create" && (
        <p role="status" data-testid="backup-progress">
          <progress max={100} value={pct} aria-label={progressLabel} /> {progressLabel}
        </p>
      )}
      {st.created && !st.running && (
        <div data-testid="backup-created">
          <p>
            <b>バックアップができました</b>({st.created.name}、{fmtBytes(st.created.bytes)})。
          </p>
          <Summary s={st.created.summary} testid="backup-created-summary" />
          <p>
            <button data-testid="backup-save" onClick={() => void save()}>
              保存する(保存先を選びます)
            </button>{" "}
            <button data-testid="backup-discard" onClick={() => void run(discardBackup)}>
              破棄
            </button>
          </p>
        </div>
      )}

      <h3>バックアップから復元</h3>
      <p className="hint">
        このアプリで作ったバックアップの zip を選びます。内容(名前・大きさ・ハッシュ・データベース)を検査してから、置き換えるかどうかを確認します。
        認識・追加学習の最中は始められません。
      </p>
      <p>
        <input
          ref={fileRef}
          type="file"
          accept=".zip,application/zip"
          data-testid="backup-file"
          aria-label="復元するバックアップの zip"
          disabled={busy}
          onChange={(e) => void pick(e.target.files?.[0])}
        />
      </p>
      {upload !== null && (
        <p role="status" data-testid="backup-upload-progress">
          <progress max={100} value={Math.round(upload * 100)} aria-label="バックアップを読み込み中" /> 読み込み中 {Math.round(upload * 100)}%
        </p>
      )}
      {st.running && (st.op === "inspect" || st.op === "extract") && (
        <p role="status" data-testid="backup-progress">
          <progress max={100} value={pct} aria-label={progressLabel} /> {progressLabel}
        </p>
      )}
      {st.incoming && !st.running && (
        <div data-testid="backup-incoming">
          <p>
            <b>このバックアップは使えます。</b>
          </p>
          <Summary s={st.incoming} testid="backup-incoming-summary" />
          {!confirming ? (
            <p>
              <button data-testid="backup-restore" onClick={() => setConfirming(true)}>
                このバックアップの内容に置き換える…
              </button>{" "}
              <button data-testid="backup-incoming-cancel" onClick={() => void run(cancelBackup)}>
                やめる
              </button>
            </p>
          ) : (
            <p role="alert" data-testid="backup-confirm">
              <b>いまのデータは、このバックアップの内容に置き換わります。</b>
              置き換えはアプリを開き直したときに行い、いまのデータは自動で退避します(あとで削除できます)。{" "}
              <button
                data-testid="backup-restore-go"
                onClick={() => {
                  setConfirming(false);
                  void run(restoreBackup);
                }}
              >
                置き換える
              </button>{" "}
              <button onClick={() => setConfirming(false)}>やめる</button>
            </p>
          )}
        </div>
      )}
      {failure && (
        <p role="alert" className="ng" data-testid="backup-error">
          {failure}
        </p>
      )}
    </section>
  );
}
