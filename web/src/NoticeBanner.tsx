import { useEffect, useState } from "react";
import { ackNotice, fetchNotices } from "./pageApi";
import { reasonText } from "./trainingText";
import type { Notice } from "./types";

const POLL_MS = 15000;

/** 使えなくなった世代(重みと確信度の組が壊れた / 採用後の監視で安全のため停止した)の通知。確認するまで全画面の上部に出す。 */
export default function NoticeBanner({ onOpenLearning }: { onOpenLearning: () => void }) {
  const [notices, setNotices] = useState<Notice[]>([]);
  const load = () => void fetchNotices().then((r) => setNotices(r.notices)).catch(() => undefined);
  useEffect(() => {
    load();
    const id = window.setInterval(load, POLL_MS);
    return () => window.clearInterval(id);
  }, []);
  if (notices.length === 0) return null;
  return (
    <div role="alert" className="notice-banner" data-testid="notice-banner">
      {notices.map((n) => (
        <p key={n.id} data-testid={`notice-${n.number}`}>
          <b>{n.status === "halted" ? `世代 ${n.number} を安全のため停止しました。` : `世代 ${n.number} が使えません。`}</b>{" "}
          {reasonText(n.code)}。{" "}
          <button onClick={onOpenLearning}>学習画面を開く</button>{" "}
          <button
            data-testid={`ack-notice-${n.number}`}
            onClick={() => void ackNotice(n.id).then(load)}
          >
            確認した
          </button>
        </p>
      ))}
    </div>
  );
}
