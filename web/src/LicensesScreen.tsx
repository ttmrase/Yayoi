import { useEffect, useMemo, useState } from "react";
import { api } from "./api";

interface Main {
  name: string;
  license: string;
  owner: string;
  source: string;
  changes: string;
  note: string;
}
interface Component {
  name: string;
  version: string;
  license: string;
  kind: "python" | "npm";
  texts: string[];
}
interface LicenseData {
  main: Main[];
  components: Component[];
  generated: boolean;
}

const KIND_LABEL = { python: "Python", npm: "画面(npm)" } as const;

/** 利用しているものの出典・ライセンス・変更点(帰属表示)と、依存ライブラリの一覧。外部のページは開かない(全文は配布物の licenses フォルダ)。 */
export default function LicensesScreen() {
  const [data, setData] = useState<LicenseData | null>(null);
  const [error, setError] = useState(false);
  const [q, setQ] = useState("");
  useEffect(() => {
    api<LicenseData>("/api/licenses")
      .then(setData)
      .catch(() => setError(true));
  }, []);
  const shown = useMemo(() => {
    const k = q.trim().toLowerCase();
    return (data?.components ?? []).filter((c) => !k || c.name.toLowerCase().includes(k) || c.license.toLowerCase().includes(k));
  }, [data, q]);
  if (error) return <p role="alert">読み込めません</p>;
  if (!data) return <p>読み込み中…</p>;
  return (
    <section data-testid="licenses" aria-labelledby="licenses-h">
      <h2 id="licenses-h">ライセンス</h2>
      <p className="hint">
        このアプリは、次のものを利用しています。アプリは実行時にネットワークを使いません。ライセンスの全文は、配布物の <code>licenses</code> フォルダに入っています。
      </p>
      <h3>主な利用物(出典・ライセンス・変更点)</h3>
      <ul className="lic-main">
        {data.main.map((m) => (
          <li key={m.name} data-testid="license-main">
            <b>{m.name}</b>
            <br />
            ライセンス: {m.license}
            <br />
            提供: {m.owner}
            <br />
            出典: {m.source}
            <br />
            変更点: {m.changes}
            {m.note && (
              <>
                <br />
                {m.note}
              </>
            )}
          </li>
        ))}
      </ul>
      <h3>依存ライブラリ</h3>
      {!data.generated ? (
        <p className="hint" data-testid="licenses-not-generated">
          依存ライブラリの一覧は、配布物を作るときに生成されます(開発中の環境では空です)。
        </p>
      ) : (
        <>
          <p>
            <label>
              絞り込み{" "}
              <input type="search" data-testid="license-filter" value={q} onChange={(e) => setQ(e.target.value)} placeholder="名前またはライセンス" />
            </label>{" "}
            <span className="hint" data-testid="license-count">
              {shown.length} / {data.components.length} 件
            </span>
          </p>
          <table className="lic-table">
            <caption className="sr-only">依存ライブラリの一覧</caption>
            <thead>
              <tr>
                <th scope="col">名前</th>
                <th scope="col">版</th>
                <th scope="col">ライセンス</th>
                <th scope="col">種類</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((c) => (
                <tr key={`${c.kind}:${c.name}`} data-testid="license-row">
                  <td>{c.name}</td>
                  <td>{c.version}</td>
                  <td>{c.license}</td>
                  <td>{KIND_LABEL[c.kind]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}
