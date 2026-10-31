import { useEffect, useState } from "react";
import { ApiError } from "./api";
import { fetchSettings, saveSettings } from "./pageApi";
import {
  MISSING_LABEL,
  ROUNDING_LABEL,
  SETTINGS_PROBLEM_LABEL,
  taxOptionsFor,
  type AppSettings,
  type SettingsProblem,
  type TaxOptions as TaxOptionMap,
} from "./types";

export const TAXPAYER = { taxable: "課税事業者", exempt: "免税事業者" } as const;
export const TAX_SCHEME = { general: "本則課税", simplified: "簡易課税" } as const;
export const TAX_METHOD = { inclusive: "税込経理", exclusive: "税抜経理" } as const;

type Group = "payments" | "titles" | "default_tax";

const EMPTY_ROW = {
  payments: { name: "", credit_account: "", credit_sub: "" },
  titles: { name: "", debit_account: "", debit_sub: "" },
  default_tax: { account: "", tax_category: "" },
};

/** 税区分の入力候補(サーバの許可リスト。今の課税方式・経理方式のもの)。弥生の登録名と完全に一致している。 */
export function TaxOptions({ options }: { options: string[] }) {
  return (
    <datalist id="tax-category-options">
      {options.map((t) => (
        <option key={t} value={t} />
      ))}
    </datalist>
  );
}

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
                      list={c.key === "tax_category" ? "tax-category-options" : undefined}
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
  const [taxMap, setTaxMap] = useState<TaxOptionMap>({});
  const [missing, setMissing] = useState<string[]>([]);
  const [problems, setProblems] = useState<Record<string, string>>({});
  const [state, setState] = useState<"clean" | "dirty" | "saved" | "error">("clean");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void fetchSettings()
      .then((r) => {
        setCfg(r.settings);
        setMissing(r.missing);
        setTaxMap(r.tax_options);
      })
      .catch(() => setError("設定を読み込めません"));
  }, []);

  if (error) return <p role="alert">{error}</p>;
  if (!cfg) return <p>読み込み中…</p>;

  const taxOptions = taxOptionsFor(taxMap, cfg);

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
      <fieldset data-testid="settings-business">
        <legend>事業者・経理の方式</legend>
        <label>
          事業者区分{" "}
          <select
            data-testid="taxpayer"
            value={cfg.taxpayer}
            onChange={(e) => edit({ taxpayer: e.target.value as AppSettings["taxpayer"] })}
          >
            {Object.entries(TAXPAYER).map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
          </select>
        </label>{" "}
        <label>
          課税方式{" "}
          <select
            data-testid="tax-scheme"
            value={cfg.tax_scheme}
            disabled={cfg.taxpayer === "exempt"}
            onChange={(e) => edit({ tax_scheme: e.target.value as AppSettings["tax_scheme"] })}
          >
            {Object.entries(TAX_SCHEME).map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
          </select>
        </label>{" "}
        <label>
          経理方式{" "}
          <select
            data-testid="tax-method"
            value={cfg.tax_method}
            onChange={(e) => edit({ tax_method: e.target.value as AppSettings["tax_method"] })}
          >
            {Object.entries(TAX_METHOD).map(([k, v]) => (
              <option key={k} value={k}>
                {v}
              </option>
            ))}
          </select>
        </label>{" "}
        {cfg.taxpayer === "taxable" && cfg.tax_method === "exclusive" && (
          <label>
            税額の端数処理{" "}
            <select
              data-testid="tax-rounding"
              value={cfg.tax_rounding}
              onChange={(e) => edit({ tax_rounding: e.target.value as AppSettings["tax_rounding"] })}
            >
              {Object.entries(ROUNDING_LABEL).map(([k, v]) => (
                <option key={k} value={k}>
                  {v}
                </option>
              ))}
            </select>
          </label>
        )}
        <p className="hint">
          免税事業者を選ぶと、借方・貸方の税区分はすべて「対象外」で出力します。税抜経理では、内税の行の借方税金額(金額 × 税率 ÷ (100 + 税率))を計算して出力します。
          外税・別記の行は借方税金額を空で出力します(弥生側で入力してください)。
        </p>
      </fieldset>
      <fieldset>
        <legend>データの保管</legend>
        <label>
          取り込んだ元画像(実帳票){" "}
          <select
            data-testid="original-retention"
            value={cfg.original_retention}
            onChange={(e) => edit({ original_retention: e.target.value as AppSettings["original_retention"] })}
          >
            <option value="keep">残す(既定)</option>
            <option value="delete_after_export">そのファイルの全ページを出力したら削除する</option>
          </select>
        </label>
        <p className="hint">
          削除すると、そのページの罫線の再切り出しはできなくなります(確定した値・出力の履歴は残ります)。どのジョブにも使われなかったアップロードは、
          起動時に 24 時間を過ぎていれば自動で削除されます。確定したセルの切り出し画像は学習データで、「学習」画面で削除できます。
        </p>
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
        help={
          cfg.taxpayer === "exempt"
            ? "免税事業者のため、すべて「対象外」で出力します(この表は使いません)。"
            : "借方科目ごとに出力する税区分(候補から選びます。弥生の登録名と完全に一致します)。税率のない課税区分(課対仕入 など)は弥生で 5% として取り込まれるため、使えません。"
        }
        columns={[
          { key: "account", label: "借方科目" },
          { key: "tax_category", label: "税区分" },
        ]}
        rows={cfg.default_tax as unknown as Record<string, string>[]}
        problems={problems}
        onChange={(rows) => edit({ default_tax: rows as unknown as AppSettings["default_tax"] })}
      />
      <TaxOptions options={taxOptions} />
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
