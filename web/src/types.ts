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
};
