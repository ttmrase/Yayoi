import { useEffect, useState } from "react";
import { ApiError } from "./api";
import { fetchSettings, saveSettings } from "./pageApi";
import {
  MISSING_LABEL,
  SETTINGS_PROBLEM_LABEL,
  type AppSettings,
  type SettingsProblem,
} from "./types";

const BUSINESS = { individual: "個人事業", corporate: "法人" } as const;
const TAX = { inclusive: "税込", exclusive: "税抜", exempt: "免税" } as const;
const ACCOUNTING = { cash: "現金主義", accrual: "発生主義" } as const;

type Group = "payments" | "titles" | "default_tax";

const EMPTY_ROW = {
  payments: { name: "", credit_account: "", credit_sub: "" },
  titles: { name: "", debit_account: "", debit_sub: "" },
  default_tax: { account: "", tax_category: "" },
};

/** 設定の表。1 行 = 1 件。入力欄ごとに、保存時の検証エラー(path で指す)を出す。 */
function Table<G extends Group>({
  group,
  title,
  help,
  columns,
  rows,
  problems,
  onChange,
}: {
  group: G;
  title: string;
  help: string;
  columns: { key: string; label: string; wide?: boolean }[];
  rows: Record<string, string>[];
  problems: Record<string, string>;
  onChange: (rows: Record<string, string>[]) => void;
}) {
  return (
    <fieldset className="settings-table" data-testid={`settings-${group}`}>
      <legend>{title}</legend>
      <p className="hint">{help}</p>
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key}>{c.label}</th>
            ))}
            <th />
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {columns.map((c) => {
                const err = problems[`${group}[${i}].${c.key}`];
                return (
                  <td key={c.key}>
                    <input
                      aria-label={`${title} ${i + 1} 行目の${c.label}`}
                      aria-invalid={err ? true : undefined}
                      value={row[c.key]}
                      maxLength={c.key === "tax_category" ? 40 : 20}
                      onChange={(e) => onChange(rows.map((r, k) => (k === i ? { ...r, [c.key]: e.target.value } : r)))}
                    />
                    {err && <div className="ng small">{SETTINGS_PROBLEM_LABEL[err] ?? err}</div>}
                  </td>
                );
              })}
              <td>
                <button aria-label={`${title} ${i + 1} 行目を削除`} onClick={() => onChange(rows.filter((_, k) => k !== i))}>
                  削除
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <button data-testid={`add-${group}`} onClick={() => onChange([...rows, { ...EMPTY_ROW[group] }])}>
        行を追加
      </button>
    </fieldset>
  );
}

export default function SettingsScreen() {
  const [cfg, setCfg] = useState<AppSettings | null>(null);
  const [missing, setMissing] = useState<string[]>([]);
  const [problems, setProblems] = useState<Record<string, string>>({});
  const [state, setState] = useState<"clean" | "dirty" | "saved" | "error">("clean");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void fetchSettings()
      .then((r) => {
        setCfg(r.settings);
        setMissing(r.missing);
      })
      .catch(() => setError("設定を読み込めません"));
  }, []);

  if (error) return <p role="alert">{error}</p>;
  if (!cfg) return <p>読み込み中…</p>;

  const edit = (patch: Partial<AppSettings>) => {
    setCfg({ ...cfg, ...patch });
    setState("dirty");
  };

  const save = async () => {
    try {
      const r = await saveSettings(cfg);
      setCfg(r.settings);
      setMissing(r.missing);
      setProblems({});
      setState("saved");
    } catch (e) {
      if (e instanceof ApiError && e.code === "invalid_settings") {
        const list = (e.body.problems ?? []) as SettingsProblem[];
        setProblems(Object.fromEntries(list.map((p) => [p.path, p.code])));
        setState("error");
      } else if (e instanceof ApiError && e.status === 422) {
        setState("error");
        setProblems({});
        setError(null);
      } else {
        setError("保存できませんでした");
      }
    }
  };

  return (
    <section className="settings" data-testid="settings">
      <h1>設定</h1>
      <p className="hint">
        帳票の「科目」欄に書かれる支払手段と、ページ上部のタイトル(勘定科目名)を、仕訳の貸方・借方の科目に対応づけます。
        登録した支払手段は、認識が「科目」欄を読むときの語彙になります。変更は、すでに確定したページには遡りません(出力時の設定を使います)。
      </p>
      {missing.length > 0 && (
        <div className="banner" role="status" data-testid="settings-missing">
          出力にはまだ足りない設定があります: {missing.map((m) => MISSING_LABEL[m] ?? m).join("、")}
        </div>
      )}
      <fieldset>
        <legend>事業者(仮の選択肢。弥生のインポート形式で必要な値は確定後に差し替えます)</legend>
        <label>
          事業者区分{" "}
          <select value={cfg.business_type} onChange={(e) => edit({ business_type: e.target.value as AppSettings["business_type"] })}>
            {Object.entries(BUSINESS).map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
          </select>
        </label>{" "}
        <label>
          課税方式{" "}
          <select value={cfg.tax_method} onChange={(e) => edit({ tax_method: e.target.value as AppSettings["tax_method"] })}>
            {Object.entries(TAX).map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
          </select>
        </label>{" "}
        <label>
          経理方式{" "}
          <select value={cfg.accounting_method} onChange={(e) => edit({ accounting_method: e.target.value as AppSettings["accounting_method"] })}>
            {Object.entries(ACCOUNTING).map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
          </select>
        </label>
      </fieldset>
      <Table
        group="payments"
        title="支払手段 → 貸方科目"
        help="「科目」欄に書かれる名前(例: 現金)と、仕訳の貸方にする科目・補助科目。"
        columns={[
          { key: "name", label: "帳票に書かれる名前" },
          { key: "credit_account", label: "貸方科目" },
          { key: "credit_sub", label: "貸方補助科目" },
        ]}
        rows={cfg.payments as unknown as Record<string, string>[]}
        problems={problems}
        onChange={(rows) => edit({ payments: rows as unknown as AppSettings["payments"] })}
      />
      <Table
        group="titles"
        title="タイトル → 借方科目"
        help="ページ上部に書かれるタイトル(勘定科目名)と、仕訳の借方にする科目・補助科目。確認画面でページごとにタイトルを選びます。"
        columns={[
          { key: "name", label: "帳票に書かれるタイトル" },
          { key: "debit_account", label: "借方科目" },
          { key: "debit_sub", label: "借方補助科目" },
        ]}
        rows={cfg.titles as unknown as Record<string, string>[]}
        problems={problems}
        onChange={(rows) => edit({ titles: rows as unknown as AppSettings["titles"] })}
      />
      <Table
        group="default_tax"
        title="科目ごとの既定の税区分"
        help="借方科目ごとに出力する税区分(値は弥生のインポート形式に合わせて確定します)。"
        columns={[
          { key: "account", label: "借方科目" },
          { key: "tax_category", label: "税区分" },
        ]}
        rows={cfg.default_tax as unknown as Record<string, string>[]}
        problems={problems}
        onChange={(rows) => edit({ default_tax: rows as unknown as AppSettings["default_tax"] })}
      />
      <p>
        <button data-testid="settings-save" disabled={state === "clean" || state === "saved"} onClick={() => void save()}>
          保存
        </button>{" "}
        <span role="status" data-testid="settings-state">
          {state === "saved" && "保存しました"}
          {state === "dirty" && "未保存の変更があります"}
          {state === "error" && <span className="ng">入力に誤りがあります(赤い欄を直してください)</span>}
        </span>
      </p>
    </section>
  );
}
