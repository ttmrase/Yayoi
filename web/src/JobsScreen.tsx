import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "./api";
import { ACCEPT, importErrorLabel, naturalSort, progressLabel, resumeTarget } from "./jobs";
import { createJob, deleteUpload, devCreateJob, fetchJob, fetchJobs, uploadFile } from "./pageApi";
import type { JobSummary, Upload } from "./types";

interface Props {
  dev: boolean;
  onOpenPage: (id: number) => void;
}

const POLL_MS = 1500;

export default function JobsScreen({ dev, onOpenPage }: Props) {
  const [jobs, setJobs] = useState<JobSummary[] | null>(null);
  const [busy, setBusy] = useState<string | null>(null); // 取り込み中の表示
  const [failures, setFailures] = useState<string[]>([]);
  const [name, setName] = useState("");
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const reload = useCallback(() => void fetchJobs().then(setJobs).catch(() => setJobs([])), []);
  useEffect(reload, [reload]);

  // 認識中のジョブがあるあいだは進捗を読み直す
  const processing = (jobs ?? []).some((j) => j.n_processing > 0);
  useEffect(() => {
    if (!processing) return;
    const t = window.setInterval(reload, POLL_MS);
    return () => window.clearInterval(t);
  }, [processing, reload]);

  const resume = async (j: JobSummary) => {
    const target = resumeTarget(j, await fetchJob(j.id));
    if (target !== null) onOpenPage(target);
  };

  const importFiles = async (list: FileList | File[]) => {
    const files = naturalSort([...list]);
    if (files.length === 0 || busy) return;
    const ups: Upload[] = [];
    const bad: string[] = [];
    for (const [i, f] of files.entries()) {
      setBusy(`アップロード中 ${i + 1} / ${files.length}`);
      try {
        ups.push(await uploadFile(f));
      } catch (e) {
        bad.push(`${f.name}: ${importErrorLabel(e instanceof ApiError ? e.code : "")}`);
      }
    }
    setFailures(bad);
    if (ups.length > 0) {
      setBusy("ジョブを作成中");
      try {
        await createJob(
          ups.map((u) => u.id),
          name.trim(),
        );
        setName("");
      } catch {
        setFailures((b) => [...b, "ジョブを作成できませんでした(認識の機能が無効です)"]);
        await Promise.all(ups.map((u) => deleteUpload(u.id).catch(() => undefined)));
      }
    }
    setBusy(null);
    if (fileInput.current) fileInput.current.value = "";
    reload();
  };

  return (
    <section>
      <h1>ジョブ</h1>
      <div
        className={`dropzone${dragging ? " over" : ""}`}
        data-testid="dropzone"
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          void importFiles(e.dataTransfer.files);
        }}
      >
        <p>
          ここに帳票のファイル(PDF・画像)をドロップするか、
          <label className="filebutton">
            ファイルを選ぶ
            <input
              ref={fileInput}
              data-testid="file-input"
              type="file"
              multiple
              accept={ACCEPT}
              disabled={busy !== null}
              onChange={(e) => e.target.files && void importFiles(e.target.files)}
            />
          </label>
        </p>
        <label>
          ジョブ名(省略可){" "}
          <input aria-label="ジョブ名" value={name} maxLength={200} onChange={(e) => setName(e.target.value)} />
        </label>
        {busy && (
          <p role="status" data-testid="import-busy">
            {busy}…
          </p>
        )}
        {failures.length > 0 && (
          <ul role="alert" data-testid="import-failures" className="ng">
            {failures.map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
        )}
      </div>
      {dev && (
        <p>
          <button
            data-testid="dev-create-job"
            onClick={() => void devCreateJob({ pages: 2, seed: Math.floor(Math.random() * 1e6) }).then(reload)}
          >
            合成ジョブを作る(開発用)
          </button>
        </p>
      )}
      {jobs === null ? (
        <p>読み込み中…</p>
      ) : jobs.length === 0 ? (
        <p>ジョブがありません。</p>
      ) : (
        <table className="jobs">
          <thead>
            <tr>
              <th>名前</th>
              <th>ページ</th>
              <th>状況</th>
              <th>確定</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {jobs.map((j) => (
              <tr key={j.id} data-testid={`job-${j.id}`}>
                <td>{j.name}</td>
                <td>{j.n_pages}</td>
                <td data-testid={`job-status-${j.id}`}>
                  {j.n_processing > 0 && <progress max={j.n_pages} value={j.n_pages - j.n_processing} aria-label="認識の進み" />}{" "}
                  {progressLabel(j)}
                </td>
                <td>
                  {j.n_confirmed} / {j.n_pages}
                </td>
                <td>
                  <button disabled={j.n_processing === j.n_pages} onClick={() => void resume(j)}>
                    {j.n_confirmed === j.n_pages ? "開く" : "続きから"}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
