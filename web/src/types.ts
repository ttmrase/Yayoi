export type Column = "date" | "account" | "description" | "amount";
export type Kind = "blank" | "text" | "ditto" | "arrow" | "wave";
/** chosen = 候補から、入力済みの値と同じものを明示的に選んだ(値は変わらない) */
export type EditKind = "none" | "unchanged" | "corrected" | "candidate" | "chosen" | "expanded_edit";

/** 候補の選択か入力で明示するまで確定できない理由(Enter だけでは通せない) */
export const CHOICE_REASONS = ["dash_digit_ambiguity", "suggest_amount_disagreement"];
export const needsChoice = (c: { reasons: string[] }): boolean => c.reasons.some((r) => CHOICE_REASONS.includes(r));
/** 明示の選択が済んでいる編集の種類 */
export const isExplicitKind = (k: EditKind): boolean => k === "candidate" || k === "corrected" || k === "chosen";

export interface Expansion {
  kind: "ditto" | "year" | "arrow" | "wave";
  /** 参照元のセル(行, 列)。前のページから年を継承した行など、このページ内に無いときは null */
  source: [number, number] | null;
}

export interface Candidate {
  value: string;
  p: number;
}

export interface Cell {
  id: number;
  row: number;
  col: number;
  column: Column;
  kind: Kind;
  recognized_text: string;
  resolved_value: string;
  value: string;
  edit_kind: EditKind;
  expansion: Expansion | null;
  reasons: string[];
  candidates: Candidate[];
  review: boolean;
  passed: boolean;
  has_image: boolean;
}

export interface ValidationError {
  cell_id: number;
  row: number;
  col: number;
  code: string;
  label: string;
}

export interface Summary {
  review_total: number;
  review_passed: number;
  unpassed: number[];
  errors: ValidationError[];
  ambiguous: number[];
  can_confirm: boolean;
}

/** 行ごとの借方税区分。source: default = 科目ごとの既定 / suggested = 摘要の記憶からの提案 / user = 利用者が選んだ */
export interface RowTax {
  row: number;
  category: string;
  source: "default" | "suggested" | "user";
  default: string;
}

export interface PageTax {
  /** 免税事業者は false(税区分は全て「対象外」なので行ごとの指定はない) */
  enabled: boolean;
  /** 選べる税区分(許可リスト) */
  options: string[];
  rows: RowTax[];
}

export interface Page {
  id: number;
  job_id: number;
  idx: number;
  n_pages: number;
  prev_page_id: number | null;
  next_page_id: number | null;
  status: "processing" | "failed" | "review" | "reviewing" | "confirmed";
  /** 認識に失敗したときの原因(例外の型名。値は含まない) */
  error: string | null;
  /** ページのタイトル(帳票の上部の勘定科目名)。title = 確認した値、recognized = 認識した値、suggestion = 候補 */
  title: string | null;
  title_recognized: string | null;
  title_suggestion: string | null;
  layout_status: string;
  layout_reasons: string[];
  layout_approved: boolean;
  revision: number;
  cells: Cell[];
  summary: Summary;
  tax: PageTax;
  can_undo: boolean;
  can_redo: boolean;
}

export interface JobSummary {
  id: number;
  name: string;
  status: string;
  profile_id: number;
  n_pages: number;
  n_confirmed: number;
  n_processing: number;
  n_failed: number;
  last_page_id: number | null;
  last_cell: number | null;
}

export interface JobDetail extends JobSummary {
  pages: { id: number; idx: number; status: Page["status"]; layout_status: string; error: string | null }[];
}

export interface Upload {
  id: string;
  name: string;
  size: number;
  n_pages: number;
}

export const COLUMN_LABEL: Record<Column, string> = {
  date: "日付",
  account: "科目",
  description: "摘要",
  amount: "金額",
};

export const EXPANSION_LABEL: Record<Expansion["kind"], string> = {
  ditto: "〃",
  year: "年",
  arrow: "→",
  wave: "〜",
};

/** 理由コード → 短いラベル(実データの値は含まれない) */
export const REASON_LABEL: Record<string, string> = {
  low_confidence: "自信が低い",
  grammar_violation: "形式が不正",
  dash_digit_ambiguity: "末尾の「-」か数字か不明",
  row_inconsistent: "行の整合",
  ditto_source_uncertain: "〃の参照元が要確認",
  suggest_disagreement: "提案モード: 親の世代と値が違う",
  suggest_amount_disagreement: "金額が親の世代と違う(選んでください)",
  ditto_source_changed: "参照元が変わった",
  year_unknown: "年が不明",
  month_decreased: "月が戻った",
  year_jump: "年が飛んだ",
  oov_account: "知らない科目",
  cp932_unencodable: "インポートできない文字",
  too_long: "長すぎる",
  faint_ink: "薄い",
  class_uncertain: "種別が不確か",
  page_needs_review: "レイアウトに自信なし",
  layout_uncertain: "レイアウトに自信なし",
  ruled_lines_low: "罫線の検出率が低い",
  band_clipped: "見出し帯が画像の端で切れている",
  table_cut: "表が画像からはみ出している",
};

export type Pt = [number, number];

export interface LayoutInfo {
  /** 元画像の大きさ(px)。隅・罫線の座標はこの座標系 */
  width: number;
  height: number;
  /** 検出した表の 4 隅(左上・右上・右下・左下) */
  corners: Pt[];
  hlines: Pt[][];
  vlines: Pt[][];
  /** 罫線の位置(表の高さ・幅に対する 0〜1 の比) */
  hfrac: number[];
  vfrac: number[];
}

export interface LayoutData {
  info: LayoutInfo;
  view_scale: number;
  status: string;
  reasons: string[];
  approved: boolean;
  /** 利用者が隅を指定した結果か */
  manual: boolean;
  override: Pt[] | null;
}

// ---- 設定(M4)
export interface PaymentMap {
  name: string;
  credit_account: string;
  credit_sub: string;
}
export interface TitleMap {
  name: string;
  debit_account: string;
  debit_sub: string;
}
export interface AccountTax {
  account: string;
  tax_category: string;
}
export interface AppSettings {
  /** 事業者区分(免税事業者はすべて「対象外」) */
  taxpayer: "taxable" | "exempt";
  /** 課税方式 */
  tax_scheme: "general" | "simplified";
  /** 経理方式(税抜・内税の行は借方税金額を計算して出力。外税・別記は空で注意) */
  tax_method: "inclusive" | "exclusive";
  /** 税抜経理・内税の借方税金額の端数処理 */
  tax_rounding: "floor" | "round" | "ceil";
  /** 初期設定ウィザードを終えた(スキップを含む) */
  wizard_done: boolean;
  payments: PaymentMap[];
  titles: TitleMap[];
  default_tax: AccountTax[];
  /** 取り込んだ元画像の保管: keep = 残す(既定)/ delete_after_export = そのファイルの全ページを出力したら削除 */
  original_retention: "keep" | "delete_after_export";
  /** 追加学習: 確定した新しい非空欄セルがしきい値たまったら、バックグラウンドで自動で始める */
  auto_train: boolean;
  /** 提案モードを使う(提案モードの世代があるとき、親と両方で認識する。認識の時間が約 1.4 倍) */
  suggest_mode: boolean;
  train_threshold_cells: number;
}
export interface SettingsResponse {
  settings: AppSettings;
  tax_options: TaxOptions;
  /** 出力に足りない設定(コード) */
  missing: string[];
}
export interface SettingsProblem {
  path: string;
  code: string;
}

// ---- 出力(M4)
export interface ExportIssue {
  code: string;
  label: string;
  /** error = 出力できない / warning = 注意(出力は止めない) */
  severity: "error" | "warning";
  page_id: number;
  cell_id: number | null;
  row: number | null;
}
export interface ExportValidation {
  n_rows: number;
  issues: ExportIssue[];
  errors: ExportIssue[];
  warnings: ExportIssue[];
  duplicates: { page_id: number; n_rows: number; export_ids: number[] }[];
  missing_settings: string[];
  notes: string[];
  format: string;
}
export interface ExportCandidate {
  id: number;
  job_id: number;
  job_name: string;
  idx: number;
  title: string | null;
  n_rows: number;
  n_exported: number;
  exported: boolean;
}
export interface ExportFile {
  idx: number;
  name: string;
  sha256: string;
  n_rows: number;
  payment: string | null;
}
export interface ExportRecord {
  id: number;
  format: string;
  format_label: string;
  created_at: number;
  n_pages: number;
  n_rows: number;
  file_sha256: string;
  file_name: string | null;
  files: ExportFile[];
}
export interface ExportList {
  exports: ExportRecord[];
  formats: { id: string; label: string; notes: string[] }[];
  candidates: ExportCandidate[];
}

// ---- 学習(M4)
export interface PageMetric {
  page_id: number;
  generation: number;
  confirmed_at: number | null;
  review_seconds: number;
  n_text: number;
  n_review: number;
  n_corrected: number;
  n_unflagged_corrected: number;
  review_rate: number | null;
  correction_rate: number | null;
}
/** 世代で認識して確定したページの実績(値は含まない。見逃しの観測 = 要確認でないのに直したセル ÷ 非空欄セル) */
export interface GenerationObserved {
  n_pages: number;
  n_text: number;
  review_rate: number | null;
  correction_rate: number | null;
  miss_rate: number | null;
  n_unflagged_corrected: number;
}
/** 採否の評価の値(評価用ページを親・新世代の両方で認識し直した結果) */
/** 確信度の較正の質(評価セルのうち確認必須でないもの)。期待見逃し = 要確認でないセルの (1 - 確信度) の合計 */
export interface GenerationCalib {
  n: number;
  nll: number;
  brier: number;
  expected_miss: number;
  observed_miss: number;
}
export interface GenerationEvalSide {
  accuracy: number;
  review_rate: number;
  miss_rate: number;
  n_cells: number;
  n_text: number;
  n_miss: number;
  calib?: GenerationCalib | null;
  by_column?: Record<string, Omit<GenerationEvalSide, "by_column">>;
}
export interface Generation {
  id: number;
  number: number;
  kind: "shipped" | "trained";
  parent_id: number | null;
  created_at: number;
  activated_at: number | null;
  status: "active" | "retired" | "rejected" | "discarded" | "broken" | "halted" | "suggest";
  reject_reason: string | null;
  n_train_pages: number | null;
  n_eval_pages: number | null;
  eval: { parent: GenerationEvalSide; candidate: GenerationEvalSide } | null;
  recalibration: Record<string, unknown> | null;
  restorable: boolean;
  n_recognized: number | null;
  /** 安全のため停止したときの理由と件数(観測した見逃しの件数・全セル数) */
  /** 提案モード: 値が違うセルのうち、この世代が正しい件数 / 親が正しい件数 と、符号検定の片側 p */
  suggest?: { better: number; worse: number; p: number } | null;
  n_suggested?: number;
  suggest_in_effect?: boolean;
  halt?: { code: string; k_total?: number; k_amount?: number; n_cells?: number; n_pages?: number } | null;
  observed?: GenerationObserved | null;
}
/** 利用者に知らせること: 使えなくなった世代(重みと確信度の組が壊れた / 安全のため停止した)。確認するまで画面の上部に出す */
export interface Notice {
  id: number;
  number: number;
  status: "broken" | "halted";
  code: string;
  at: number | null;
}
/** 提案モードの効果(提案モードで認識した確定ページと、そうでないページの比較) */
export interface SuggestEffectSide {
  n_pages: number;
  n_text: number;
  n_review: number;
  n_review_unchanged: number;
  n_corrected: number;
  review_unchanged_share: number | null;
  review_per_page: number | null;
  corrected_per_page: number | null;
}
export interface SuggestEffect {
  suggest: SuggestEffectSide;
  normal: SuggestEffectSide;
}
export interface LearningData {
  profile_id: number;
  generations: Generation[];
  active_id: number | null;
  notices: Notice[];
  suggest_effect: SuggestEffect;
  /** 設定の「提案モードを使う」 */
  suggest_mode: boolean;
  pages: PageMetric[];
  summary: {
    n_pages: number;
    n_cells: number;
    review_rate: number | null;
    correction_rate: number | null;
    unflagged_corrected: number;
    median_review_seconds: number | null;
  };
  can_rollback: boolean;
  can_reset: boolean;
  phase: number;
}

// ---- 追加学習(Phase 4)
export type TrainingStatus = "running" | "adopted" | "suggested" | "rejected" | "deferred" | "failed" | "interrupted";
export interface TrainingRun {
  id: number;
  trigger: "auto" | "manual";
  status: TrainingStatus;
  started_at: number;
  finished_at: number | null;
  n_new_cells: number | null;
  n_train_pages: number | null;
  n_eval_pages: number | null;
  generation_id: number | null;
  /** 採否・見送り・失敗・中断の理由コード(値は含まない)。文言は trainingText.ts */
  reason: string | null;
  stage: string | null;
  progress: number | null;
}
export interface TrainingData {
  available: boolean;
  auto_train: boolean;
  suggest_mode: boolean;
  threshold: number;
  active_generation: { id: number; number: number };
  eligible_pages: number;
  min_pages: number;
  new_cells: number;
  running: TrainingRun | null;
  last: TrainingRun | null;
  runs: TrainingRun[];
}

/** 税区分の許可リストのキー(課税方式 × 経理方式)。リスト自体はサーバが返す(`/api/settings` の tax_options。app/tax.py が唯一の定義) */
export type TaxOptions = Record<string, string[]>;

/** 今の事業者区分・課税方式・経理方式で選べる税区分(免税事業者は「対象外」だけ) */
export function taxOptionsFor(
  opts: TaxOptions,
  s: Pick<AppSettings, "taxpayer" | "tax_scheme" | "tax_method">,
): string[] {
  if (s.taxpayer === "exempt") return ["対象外"];
  return opts[`${s.tax_scheme}_${s.tax_method}`] ?? [];
}

export const ROUNDING_LABEL = { floor: "切り捨て", round: "四捨五入", ceil: "切り上げ" } as const;

export const SETTINGS_PROBLEM_LABEL: Record<string, string> = {
  tax_rate_missing: "税率が書かれていません(税率なしの税区分は弥生で 5% として取り込まれます)。例: 課対仕入込10%",
  tax_percent_fullwidth: "% は半角で書いてください",
  tax_category_not_allowed: "選べない税区分です(今の課税方式・経理方式の候補から選んでください)",
  too_long_bytes: "長すぎます(全角 12 文字・24 バイトまで)",
  duplicate: "同じ名前が重複しています",
  empty: "空にできません",
  surrounding_space: "前後に空白があります",
  cp932_unencodable: "インポートできない文字が含まれています",
};
export const MISSING_LABEL: Record<string, string> = {
  no_payments: "支払手段が 1 つも登録されていません",
  payment_without_credit_account: "貸方科目が空の支払手段があります",
  no_titles: "タイトル → 借方科目が 1 つも登録されていません",
  title_without_debit_account: "借方科目が空のタイトルがあります",
  debit_account_without_tax_category: "借方科目に既定の税区分が設定されていません(借方税区分は必須です。免税事業者なら不要)",
  tax_category_not_allowed: "既定の税区分に、今の課税方式・経理方式で選べないものがあります(設定画面で選び直してください)",
};

export const TAX_SOURCE_LABEL: Record<RowTax["source"], string> = { default: "既定", suggested: "提案", user: "指定" };
export const TAX_SOURCE_HINT: Record<RowTax["source"], string> = {
  default: "科目ごとの既定の税区分",
  suggested: "同じ摘要で前に選んだ税区分を、記憶から提案しています(確定するとこの値で出力します。変えるには F2)",
  user: "この行について、選んだ税区分",
};

// ---- 摘要ごとの税区分の記憶(設定画面で編集)
export interface TaxMemoryEntry {
  id: number;
  description: string;
  tax_category: string;
  n_used: number;
  updated_at: number;
  /** 今の課税方式・経理方式で選べる税区分か(選べない記憶は提案されない) */
  allowed: boolean;
}
export interface TaxMemory {
  entries: TaxMemoryEntry[];
  options: string[];
  exempt: boolean;
}

/** バックアップの要約(件数・大きさ・日時。実データの値は含まない) */
export interface BackupSummary {
  created_at: number;
  app_version: string;
  schema: number;
  includes_originals: boolean;
  n_files: number;
  total_bytes: number;
  zip_bytes: number;
  n_pages: number;
  n_confirmed_pages: number;
  n_generations: number;
  n_images: number;
  n_pagedata: number;
}
export interface BackupStatus {
  op: "create" | "inspect" | "extract" | null;
  running: boolean;
  stage: string | null;
  done: number;
  total: number;
  error: string | null;
  /** 作成したバックアップ(ダウンロード待ち) */
  created: { name: string; bytes: number; summary: BackupSummary } | null;
  /** 検査済みの、復元しようとしている zip */
  incoming: BackupSummary | null;
  /** 復元の準備ができている(次の起動で入れ替わる) */
  pending: { at: number; summary: BackupSummary } | null;
  /** 直近の復元の結果 */
  result: { ok: boolean; code?: string; at: number; summary?: BackupSummary; pairs?: { checked: number; broken: number } } | null;
  /** 入れ替え前のデータの退避 */
  undo: { bytes: number; at: number } | null;
}
