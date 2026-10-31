import { useEffect, useRef, useState } from "react";
import { ApiError } from "./api";
import { cancelTraining, fetchSettings, fetchTraining, saveSettings, startTraining } from "./pageApi";
import { reasonText, START_ERROR, STATUS_TEXT } from "./trainingText";
import type { TrainingData } from "./types";

const fmt = (t: number | null) => (t === null ? "—" : new Date(t * 1000).toLocaleString("ja-JP"));
const POLL_RUNNING_MS = 2000;
const POLL_IDLE_MS = 3000; // 学習画面を開いたまま自動で始まった場合にも気づけるように

/** 追加学習の状態・開始・自動開始の設定。学習中は 2 秒ごと(それ以外は 3 秒ごと)に状態を読み、終わったら `onFinished` で世代の表を読み直させる。 */
export default function TrainingPanel({ onFinished }: { onFinished: () => void }) {
  const [t, setT] = useState<TrainingData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState("200");
  const wasRunning = useRef(false);
  const lastId = useRef<number | null | undefined>(undefined);

  const load = () =>
    void fetchTraining()
      .then((d) => {
        setT(d);
        setThreshold((cur) => (document.activeElement?.id === "train-threshold" ? cur : String(d.threshold)));
        // 学習が終わった(終わる瞬間を見逃しても、新しい実行の結果が現れたら)世代の表を読み直させる
        const newest = d.runs[0]?.id ?? null;
        if ((wasRunning.current && !d.running) || (lastId.current !== undefined && lastId.current !== newest && !d.running)) onFinished();
        wasRunning.current = d.running !== null;
        lastId.current = newest;
      })
      .catch(() => setError("読み込めません"));
  useEffect(load, []);
  useEffect(() => {
    const id = window.setInterval(load, t?.running ? POLL_RUNNING_MS : POLL_IDLE_MS);
    return () => window.clearInterval(id);
  }, [t?.running?.id]);

  const start = async () => {
    setError(null);
    try {
      await startTraining();
    } catch (e) {
      const code = e instanceof ApiError ? e.code : "";
      setError(START_ERROR[code] ?? "開始できませんでした");
    }
    load();
  };
  const cancel = async () => {
    await cancelTraining().catch(() => undefined);
    load();
  };
  const changeSettings = async (patch: { auto_train?: boolean; train_threshold_cells?: number }) => {
    setError(null);
    try {
      const cur = (await fetchSettings()).settings;
      await saveSettings({ ...cur, ...patch });
    } catch {
      setError("設定を保存できませんでした");
    }
    load();
  };

  if (!t) return error ? <p role="alert">{error}</p> : <p>読み込み中…</p>;
  const enough = t.eligible_pages >= t.min_pages;
  const pct = Math.round((t.running?.progress ?? 0) * 100);
  const last = t.last;
  return (
    <section data-testid="training" aria-labelledby="training-h">
      <h2 id="training-h">追加学習</h2>
      <p className="hint">
        確定したページから、あなたの書き方に合わせた重みをバックグラウンドで学習します。学習した世代は、確定済みのページの一部(新しい側)で試して、見逃しも正解率も悪くならないときだけ採用します。
        学習中も確認画面はいまの世代のまま使えます。学習はこのPCの中だけで行い、確定したセルの画像を外へ送ることはありません。
      </p>
      <p data-testid="training-counts">
        新しい確定セル <b>{t.new_cells}</b> 件 / 自動で始めるしきい値 {t.threshold} 件 · 学習に使える確定ページ <b>{t.eligible_pages}</b>
        (開始には {t.min_pages} ページ以上)
      </p>
      {t.running ? (
        <div role="status" data-testid="training-running">
          <progress value={pct} max={100} aria-label="追加学習の進み具合" /> {pct}%{" "}
          <span className="hint">({t.running.trigger === "auto" ? "自動" : "手動"}で開始 · {fmt(t.running.started_at)})</span>{" "}
          <button data-testid="training-cancel" onClick={() => void cancel()}>
            中止する
          </button>
        </div>
      ) : (
        <p>
          <button data-testid="training-start" disabled={!t.available || !enough} onClick={() => void start()}>
            今すぐ学習する
          </button>{" "}
          {!enough && <span className="hint">確定したページが {t.min_pages} ページに達すると使えます。</span>}
        </p>
      )}
      {last && (
        <p data-testid="training-last">
          前回({fmt(last.finished_at ?? last.started_at)}・{last.trigger === "auto" ? "自動" : "手動"}):{" "}
          <b data-testid="training-last-status">{STATUS_TEXT[last.status]}</b> — {reasonText(last.reason)}
          {last.n_train_pages !== null && (
            <span className="hint">
              {" "}
              (学習 {last.n_train_pages} ページ / 評価 {last.n_eval_pages} ページ)
            </span>
          )}
        </p>
      )}
      <p>
        <label>
          <input
            type="checkbox"
            data-testid="auto-train"
            checked={t.auto_train}
            onChange={(e) => void changeSettings({ auto_train: e.target.checked })}
          />{" "}
          新しい確定セルが
        </label>{" "}
        <input
          id="train-threshold"
          data-testid="train-threshold"
          type="number"
          min={50}
          max={5000}
          step={50}
          value={threshold}
          aria-label="自動で始める確定セル数"
          style={{ width: "5em" }}
          onChange={(e) => setThreshold(e.target.value)}
          onBlur={() => {
            const n = Number(threshold);
            if (Number.isInteger(n) && n >= 50 && n <= 5000 && n !== t.threshold) void changeSettings({ train_threshold_cells: n });
            else setThreshold(String(t.threshold));
          }}
        />{" "}
        件たまったら自動で学習を始める(50〜5000。認識の最中は始めません)
      </p>
      {error && (
        <p role="alert" data-testid="training-error">
          {error}
        </p>
      )}
    </section>
  );
}
