import type { FullConfig } from "@playwright/test";

// 共有サーバの試験が勝手に追加学習を始めないよう、自動開始を止めて始める(追加学習の試験は自分で有効にする)。
// 開発用の学習器(MockTrainer)の動作は /api/dev/trainer で選ぶ。
export default async function globalSetup(config: FullConfig) {
  const base = config.projects[0].use.baseURL as string;
  const token = "e2e-token-0123456789abcdef0123456789abcdef";
  const res = await fetch(`${base}/api/dev/trainer`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify({ mode: "improve", delay: 0, auto: false }),
  });
  if (!res.ok) throw new Error(`global setup failed: ${res.status}`);
}
