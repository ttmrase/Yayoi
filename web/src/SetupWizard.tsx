import { useEffect, useRef, useState } from "react";
import { ApiError } from "./api";
import { saveSettings } from "./pageApi";
import { TAX_METHOD, TAX_SCHEME, TAXPAYER, TaxOptions } from "./SettingsScreen";
import {
  ROUNDING_LABEL,
  SETTINGS_PROBLEM_LABEL,
  taxOptionsFor,
  type AppSettings,
  type SettingsProblem,
  type TaxOptions as TaxOptionMap,
} from "./types";

interface Props {
  settings: AppSettings;
  /** 税区分の許可リスト(サーバが返す。課税方式 × 経理方式ごと) */
  taxMap: TaxOptionMap;
  /** 完了またはスキップ。保存した設定を渡す */
  onDone: (s: AppSettings) => void;
}

interface TitleRow {
  name: string;
  debit_account: string;
  debit_sub: string;
  tax_category: string;
}

const STEPS = ["事業者・経理の方式", "支払手段(貸方)", "ページのタイトル(借方・税区分)", "確認"] as const;

const isBlank = (r: object) => Object.values(r).every((v) => v === "");

/**
 * 初期設定ウィザード(初回起動)。支払手段(名前 + 貸方科目・補助科目)、ページのタイトル(借方科目・補助科目・税区分)、
 * 事業者区分・課税方式・経理方式を順に入れる。どの段階でも「あとで設定する」で閉じられる(閉じた後は設定画面で入れる)。
 * 完了・スキップのどちらでも `wizard_done` を立てる(二度と自動では出ない)。
 */
export default function SetupWizard({ settings, taxMap, onDone }: Props) {
  const [step, setStep] = useState(0);
  const [taxpayer, setTaxpayer] = useState(settings.taxpayer);
  const [scheme, setScheme] = useState(settings.tax_scheme);
  const [method, setMethod] = useState(settings.tax_method);
  const [rounding, setRounding] = useState(settings.tax_rounding);
  const [payments, setPayments] = useState(() =>
    settings.payments.length > 0 ? settings.payments.map((p) => ({ ...p })) : [{ name: "", credit_account: "", credit_sub: "" }],
  );
  const [titles, setTitles] = useState<TitleRow[]>(() => {
    const tax = new Map(settings.default_tax.map((t) => [t.account, t.tax_category]));
    const rows = settings.titles.map((t) => ({ ...t, tax_category: tax.get(t.debit_account) ?? "" }));
    return rows.length > 0 ? rows : [{ name: "", debit_account: "", debit_sub: "", tax_category: "" }];
  });
  const [problems, setProblems] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => heading.current?.focus(), [step]);

  const exempt = taxpayer === "exempt";
  const taxOptions = taxOptionsFor(taxMap, { taxpayer, tax_scheme: scheme, tax_method: method });

  /** 入力から保存する設定を作る。表の行番号 → 画面の行番号の対応(検証エラーを指すため)も返す。 */
  const build = () => {
    const pays = payments.filter((r) => !isBlank(r));
    const tts = titles.filter((r) => !isBlank(r));
    const base = {
      ...settings,
      taxpayer,
      tax_scheme: scheme,
      tax_method: method,
      tax_rounding: rounding,
      wizard_done: true,
    };
    const taxRows: number[] = []; // default_tax[j] ← tts[taxRows[j]]
    let defaultTax = settings.default_tax;
    if (!exempt) {
      const mine = new Set(tts.map((t) => t.debit_account).filter((a) => a !== ""));
      defaultTax = settings.default_tax.filter((t) => !mine.has(t.account)); // 画面に出ていない科目の既定は残す
      taxRows.push(...defaultTax.map(() => -1));
      const seen = new Set<string>();
      tts.forEach((t, i) => {
        if (t.debit_account === "" || t.tax_category === "") return;
        const key = `${t.debit_account}\u0000${t.tax_category}`;
        if (seen.has(key)) return; // 同じ科目・同じ税区分の行は 1 件にまとめる
        seen.add(key);
        defaultTax = [...defaultTax, { account: t.debit_account, tax_category: t.tax_category }];
        taxRows.push(i);
      });
    }
    return {
      s: {
        ...base,
        payments: pays,
        titles: tts.map(({ name, debit_account, debit_sub }) => ({ name, debit_account, debit_sub })),
        default_tax: defaultTax,
      } satisfies AppSettings,
      pays,
      tts,
      taxRows,
    };
  };

  const finish = async () => {
    const { s, pays, tts, taxRows } = build();
    setBusy(true);
    setError(null);
    try {
      onDone((await saveSettings(s)).settings);
    } catch (e) {
      if (e instanceof ApiError && e.code === "invalid_settings") {
        const list = (e.body.problems ?? []) as SettingsProblem[];
        // 保存に使った行の並びから、画面の行(空行を除く前の並び)へ戻す
        const payIdx = payments.map((r, i) => (isBlank(r) ? -1 : i)).filter((i) => i >= 0);
        const titleIdx = titles.map((r, i) => (isBlank(r) ? -1 : i)).filter((i) => i >= 0);
        const out: Record<string, string> = {};
        for (const p of list) {
          const m = /^(payments|titles|default_tax)\[(\d+)\]\.(\w+)$/.exec(p.path);
          if (!m) continue;
          const j = Number(m[2]);
          if (m[1] === "payments" && j < pays.length) out[`payments[${payIdx[j]}].${m[3]}`] = p.code;
          else if (m[1] === "titles" && j < tts.length) out[`titles[${titleIdx[j]}].${m[3]}`] = p.code;
          else if (m[1] === "default_tax" && taxRows[j] >= 0) out[`titles[${titleIdx[taxRows[j]]}].tax_category`] = p.code;
        }
        setProblems(out);
        const keys = Object.keys(out);
        setStep(keys.some((k) => k.startsWith("payments")) ? 1 : 2);
        setError("入力に誤りがあります(赤い欄を直してください)");
      } else {
        setError("保存できませんでした");
      }
    } finally {
      setBusy(false);
    }
  };

  /** スキップ: 入力は捨てて、ウィザードを終えたことだけを保存する */
  const skip = async () => {
    setBusy(true);
    try {
      onDone((await saveSettings({ ...settings, wizard_done: true })).settings);
    } catch {
      setError("保存できませんでした");
    } finally {
      setBusy(false);
    }
  };

  const cell = (group: "payments" | "titles", i: number, key: string, label: string, value: string, set: (v: string) => void, list?: string) => {
    const err = problems[`${group}[${i}].${key}`];
    return (
      <td key={key}>
        <input
          aria-label={`${label}(${i + 1} 行目)`}
          aria-invalid={err ? true : undefined}
          list={list}
          maxLength={key === "tax_category" ? 40 : 20}
          value={value}
          onChange={(e) => set(e.target.value)}
        />
        {err && <div className="ng small">{SETTINGS_PROBLEM_LABEL[err] ?? err}</div>}
      </td>
    );
  };

  const filled = {
    payments: payments.filter((r) => !isBlank(r)).length,
    titles: titles.filter((r) => !isBlank(r)).length,
  };

  return (
    <section className="wizard" data-testid="wizard" aria-labelledby="wizard-heading">
      <p className="hint" data-testid="wizard-progress">
        初期設定 {step + 1} / {STEPS.length}
      </p>
      <h1 id="wizard-heading" tabIndex={-1} ref={heading}>
        {STEPS[step]}
      </h1>

      {step === 0 && (
        <fieldset>
          <legend className="sr-only">事業者・経理の方式</legend>
          <p className="hint">弥生の税区分の出力に使います。迷うときは、あとで設定画面から変えられます。</p>
          <p>
            <label>
              事業者区分{" "}
              <select data-testid="wizard-taxpayer" value={taxpayer} onChange={(e) => setTaxpayer(e.target.value as AppSettings["taxpayer"])}>
                {Object.entries(TAXPAYER).map(([k, v]) => (
                  <option key={k} value={k}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
          </p>
          <p>
            <label>
              課税方式{" "}
              <select value={scheme} disabled={exempt} onChange={(e) => setScheme(e.target.value as AppSettings["tax_scheme"])}>
                {Object.entries(TAX_SCHEME).map(([k, v]) => (
                  <option key={k} value={k}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
          </p>
          <p>
            <label>
              経理方式{" "}
              <select value={method} onChange={(e) => setMethod(e.target.value as AppSettings["tax_method"])}>
                {Object.entries(TAX_METHOD).map(([k, v]) => (
                  <option key={k} value={k}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
          </p>
          {!exempt && method === "exclusive" && (
            <p>
              <label>
                税額の端数処理{" "}
                <select value={rounding} onChange={(e) => setRounding(e.target.value as AppSettings["tax_rounding"])}>
                  {Object.entries(ROUNDING_LABEL).map(([k, v]) => (
                    <option key={k} value={k}>
                      {v}
                    </option>
                  ))}
                </select>
              </label>
            </p>
          )}
          {!exempt && method === "exclusive" && (
            <p className="hint">税抜経理では、内税の行の借方税金額(金額 × 税率 ÷ (100 + 税率))を計算して出力します。外税・別記の行は空で出力します。</p>
          )}
          {exempt && <p className="hint">免税事業者は、借方・貸方の税区分がすべて「対象外」で出力されます。</p>}
        </fieldset>
      )}

      {step === 1 && (
        <>
          <p className="hint">
            帳票の「科目」欄に書かれる支払手段(例: 現金、普通預金、カード名)と、仕訳の貸方にする科目・補助科目を登録します。
            登録した名前が、認識が「科目」欄を読むときの語彙になります。補助科目は、弥生の登録名と完全に一致させてください。
          </p>
          <table className="settings-table" data-testid="wizard-payments">
            <thead>
              <tr>
                <th>帳票に書かれる名前</th>
                <th>貸方科目</th>
                <th>貸方補助科目</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {payments.map((r, i) => (
                <tr key={i}>
                  {cell("payments", i, "name", "帳票に書かれる名前", r.name, (v) => setPayments(payments.map((x, k) => (k === i ? { ...x, name: v } : x))))}
                  {cell("payments", i, "credit_account", "貸方科目", r.credit_account, (v) =>
                    setPayments(payments.map((x, k) => (k === i ? { ...x, credit_account: v } : x))),
                  )}
                  {cell("payments", i, "credit_sub", "貸方補助科目", r.credit_sub, (v) =>
                    setPayments(payments.map((x, k) => (k === i ? { ...x, credit_sub: v } : x))),
                  )}
                  <td>
                    <button aria-label={`支払手段 ${i + 1} 行目を削除`} onClick={() => setPayments(payments.filter((_, k) => k !== i))}>
                      削除
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <button data-testid="wizard-add-payment" onClick={() => setPayments([...payments, { name: "", credit_account: "", credit_sub: "" }])}>
            行を追加
          </button>
        </>
      )}

      {step === 2 && (
        <>
          <p className="hint">
            ページ上部に書かれるタイトル(勘定科目名)と、仕訳の借方にする科目・補助科目を登録します。
            {exempt
              ? "免税事業者のため、税区分は入力不要です(すべて「対象外」で出力します)。"
              : "税区分は候補から選びます(弥生の登録名と完全に一致します)。税率のない課税区分は弥生で 5% として取り込まれるため、使えません。"}
          </p>
          <table className="settings-table" data-testid="wizard-titles">
            <thead>
              <tr>
                <th>帳票に書かれるタイトル</th>
                <th>借方科目</th>
                <th>借方補助科目</th>
                {!exempt && <th>借方の税区分</th>}
                <th />
              </tr>
            </thead>
            <tbody>
              {titles.map((r, i) => {
                const set = (patch: Partial<TitleRow>) => setTitles(titles.map((x, k) => (k === i ? { ...x, ...patch } : x)));
                return (
                  <tr key={i}>
                    {cell("titles", i, "name", "帳票に書かれるタイトル", r.name, (v) => set({ name: v }))}
                    {cell("titles", i, "debit_account", "借方科目", r.debit_account, (v) => set({ debit_account: v }))}
                    {cell("titles", i, "debit_sub", "借方補助科目", r.debit_sub, (v) => set({ debit_sub: v }))}
                    {!exempt && cell("titles", i, "tax_category", "借方の税区分", r.tax_category, (v) => set({ tax_category: v }), "tax-category-options")}
                    <td>
                      <button aria-label={`タイトル ${i + 1} 行目を削除`} onClick={() => setTitles(titles.filter((_, k) => k !== i))}>
                        削除
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <TaxOptions options={taxOptions} />
          <button data-testid="wizard-add-title" onClick={() => setTitles([...titles, { name: "", debit_account: "", debit_sub: "", tax_category: "" }])}>
            行を追加
          </button>
        </>
      )}

      {step === 3 && (
        <div data-testid="wizard-summary">
          <ul>
            <li>
              {TAXPAYER[taxpayer]}・{exempt ? "税区分はすべて「対象外」" : `${TAX_SCHEME[scheme]}・${TAX_METHOD[method]}`}
            </li>
            <li>
              支払手段 {filled.payments} 件
              {filled.payments === 0 && <span className="ng"> — 未登録のため、科目はすべて要確認になります</span>}
            </li>
            <li>
              ページのタイトル {filled.titles} 件
              {filled.titles === 0 && <span className="ng"> — 未登録のため、出力の前に設定が必要です</span>}
            </li>
          </ul>
          <p className="hint">この内容を保存します。あとから設定画面で変えられます。</p>
        </div>
      )}

      {error && (
        <p className="ng" role="alert" data-testid="wizard-error">
          {error}
        </p>
      )}
      <p className="wizard-nav">
        <button data-testid="wizard-back" disabled={step === 0 || busy} onClick={() => setStep(step - 1)}>
          戻る
        </button>{" "}
        {step < STEPS.length - 1 ? (
          <button data-testid="wizard-next" className="primary" disabled={busy} onClick={() => setStep(step + 1)}>
            次へ
          </button>
        ) : (
          <button data-testid="wizard-finish" className="primary" disabled={busy} onClick={() => void finish()}>
            保存して始める
          </button>
        )}{" "}
        <button data-testid="wizard-skip" disabled={busy} onClick={() => void skip()}>
          あとで設定する(スキップ)
        </button>
      </p>
    </section>
  );
}
