import { useEffect, useState } from "react";
import { ApiError, api, hasToken } from "./api";
import ExportScreen from "./ExportScreen";
import JobsScreen from "./JobsScreen";
import LearningScreen from "./LearningScreen";
import NoticeBanner from "./NoticeBanner";
import ReviewScreen from "./ReviewScreen";
import { fetchSettings } from "./pageApi";
import SettingsScreen from "./SettingsScreen";
import SetupWizard from "./SetupWizard";
import type { AppSettings, TaxOptions } from "./types";

type Health = { ok: boolean; version: string; schema: number; dev: boolean };
type Profile = { id: number; name: string };

const TABS = ["ジョブ", "確認", "設定", "出力", "学習"] as const;

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [tab, setTab] = useState<(typeof TABS)[number]>("ジョブ");
  const [pageId, setPageId] = useState<number | null>(null); // 確認中のページ(確認タブ)
  const [startCell, setStartCell] = useState<number | null>(null); // 出力の検証の指摘から開くときの、最初のフォーカス
  const [wizard, setWizard] = useState<AppSettings | null>(null); // 初回起動の初期設定ウィザード(終えるまで他の画面は出さない)
  const [taxMap, setTaxMap] = useState<TaxOptions>({}); // 税区分の許可リスト(サーバ定義)

  useEffect(() => {
    if (!hasToken()) {
      setError("トークンがありません(アプリから開いてください)");
      return;
    }
    api<Health>("/api/health")
      .then(setHealth)
      .catch((e: unknown) => setError(e instanceof ApiError ? `接続できません (${e.status})` : "接続できません"));
    api<Profile[]>("/api/profiles").then(setProfiles).catch(() => undefined);
    fetchSettings()
      .then((r) => {
        setTaxMap(r.tax_options);
        setWizard(r.settings.wizard_done ? null : r.settings);
      })
      .catch(() => undefined);
  }, []);

  return (
    <div className="app">
      <header className="top">
        <strong>yayoi-ocr</strong>
        <nav aria-label="画面">
          {TABS.map((t) => (
            <button key={t} aria-current={t === tab ? "page" : undefined} disabled={wizard !== null} onClick={() => setTab(t)}>
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
      {wizard === null && <NoticeBanner onOpenLearning={() => setTab("学習")} />}
      <main>
        {wizard !== null && (
          <SetupWizard
            settings={wizard}
            taxMap={taxMap}
            onDone={() => {
              setWizard(null);
              setTab("ジョブ");
            }}
          />
        )}
        {wizard === null && tab === "ジョブ" && (
          <JobsScreen
            dev={health?.dev ?? false}
            onOpenPage={(id) => {
              setPageId(id);
              setTab("確認");
            }}
          />
        )}
        {wizard === null && tab === "確認" &&
          (pageId === null ? (
            <p>ジョブ一覧から開いてください。</p>
          ) : (
            <ReviewScreen
              pageId={pageId}
              startCell={startCell}
              onOpenPage={(id) => {
                setStartCell(null);
                setPageId(id);
              }}
              onExit={() => setTab("ジョブ")}
            />
          ))}
        {wizard === null && tab === "設定" && <SettingsScreen />}
        {wizard === null && tab === "出力" && (
          <ExportScreen
            onOpenCell={(id, cell) => {
              setStartCell(cell);
              setPageId(id);
              setTab("確認");
            }}
          />
        )}
        {wizard === null && tab === "学習" && <LearningScreen />}
      </main>
    </div>
  );
}
