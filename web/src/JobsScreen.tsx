import { useEffect, useState } from "react";
import { fetchJob, fetchJobs, devCreateJob } from "./pageApi";
import type { JobSummary } from "./types";

interface Props {
  dev: boolean;
  onOpenPage: (id: number) => void;
}

export default function JobsScreen({ dev, onOpenPage }: Props) {
  const [jobs, setJobs] = useState<JobSummary[] | null>(null);
  const reload = () => void fetchJobs().then(setJobs).catch(() => setJobs([]));
  useEffect(reload, []);

  const resume = async (j: JobSummary) => {
    const d = await fetchJob(j.id);
    const target =
      (j.last_page_id !== null && d.pages.some((p) => p.id === j.last_page_id) ? j.last_page_id : null) ??
      d.pages.find((p) => p.status !== "confirmed")?.id ??
      d.pages[0]?.id;
    if (target !== undefined) onOpenPage(target);
  };

  return (
    <section>
      <h1>ジョブ</h1>
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
              <th>確定</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {jobs.map((j) => (
              <tr key={j.id} data-testid={`job-${j.id}`}>
                <td>{j.name}</td>
                <td>{j.n_pages}</td>
                <td>
                  {j.n_confirmed} / {j.n_pages}
                </td>
                <td>
                  <button onClick={() => void resume(j)}>{j.n_confirmed === j.n_pages ? "開く" : "続きから"}</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
