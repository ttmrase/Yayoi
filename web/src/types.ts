export type Column = "date" | "account" | "description" | "amount";
export type Kind = "blank" | "text" | "ditto" | "arrow" | "wave";
export type EditKind = "none" | "unchanged" | "corrected" | "candidate" | "expanded_edit";

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
  business_type: "individual" | "corporate";
  tax_method: "inclusive" | "exclusive" | "exempt";
  accounting_method: "cash" | "accrual";
  payments: PaymentMap[];
  titles: TitleMap[];
  default_tax: AccountTax[];
  /** 取り込んだ元画像の保管: keep = 残す(既定)/ delete_after_export = そのファイルの全ページを出力したら削除 */
  original_retention: "keep" | "delete_after_export";
}
export interface SettingsResponse {
  settings: AppSettings;
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
  page_id: number;
  cell_id: number | null;
  row: number | null;
}
export interface ExportValidation {
  n_rows: number;
  issues: ExportIssue[];
  duplicates: { page_id: number; export_ids: number[] }[];
  missing_settings: string[];
  provisional: boolean;
}
export interface ExportCandidate {
  id: number;
  job_id: number;
  job_name: string;
  idx: number;
  title: string | null;
  exported: boolean;
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
}
export interface ExportList {
  exports: ExportRecord[];
  formats: { id: string; label: string; provisional: boolean }[];
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
export interface LearningData {
  profile_id: number;
  generations: { id: number; parent_id: number | null; created_at: number; n_pages: number; score: number | null; status: string; n_recognized: number }[];
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

export const SETTINGS_PROBLEM_LABEL: Record<string, string> = {
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
};
