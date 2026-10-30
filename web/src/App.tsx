import { useEffect, useState } from "react";
import { ApiError, api, hasToken } from "./api";

type Health = { ok: boolean; version: string; schema: number };
type Profile = { id: number; name: string };

const TABS = ["ジョブ", "確認", "設定", "出力", "学習"] as const;

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [tab, setTab] = useState<(typeof TABS)[number]>("ジョブ");

  useEffect(() => {
    if (!hasToken()) {
      setError("トークンがありません(アプリから開いてください)");
      return;
    }
    api<Health>("/api/health")
      .then(setHealth)
      .catch((e: unknown) => setError(e instanceof ApiError ? `接続できません (${e.status})` : "接続できません"));
    api<Profile[]>("/api/profiles").then(setProfiles).catch(() => undefined);
  }, []);

  return (
    <div className="app">
      <header className="top">
        <strong>yayoi-ocr</strong>
        <nav aria-label="画面">
          {TABS.map((t) => (
            <button key={t} aria-current={t === tab ? "page" : undefined} onClick={() => setTab(t)}>
              {t}
            </button>
          ))}
        </nav>
        <label className="profile">
          書き手
          <select aria-label="書き手のプロファイル" disabled={profiles.length <= 1}>
            {profiles.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
        </label>
        <span role="status" data-testid="connection" className={health ? "ok" : "ng"}>
          {health ? `接続 OK (v${health.version})` : (error ?? "接続中…")}
        </span>
      </header>
      <main>
        <h1>{tab}</h1>
        <p>この画面は未実装です(M0: 骨組み)。</p>
      </main>
    </div>
  );
}
