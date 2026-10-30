// ジョブ一覧まわりの純粋な関数(画面から切り離してテストする)

import type { JobDetail, JobSummary } from "./types";

/** 取り込めるファイルの拡張子(サーバの `ingest.SUPPORTED_SUFFIXES` と同じ) */
export const ACCEPT = ".pdf,.png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp";

/** ファイル名の数字を数として比べる(scan2 < scan10)。スキャナの連番の順に並べる。 */
export function naturalSort<T extends { name: string }>(files: T[]): T[] {
  return [...files].sort((a, b) => a.name.localeCompare(b.name, "ja", { numeric: true, sensitivity: "base" }));
}

export const IMPORT_ERROR: Record<string, string> = {
  unsupported_type: "対応していない形式です",
  unreadable_file: "読み込めないファイルです",
  empty_file: "空のファイルです",
  too_large: "大きすぎます",
  bad_page_count: "ページ数が多すぎます",
};

export function importErrorLabel(code: string): string {
  return IMPORT_ERROR[code] ?? "取り込めませんでした";
}

/** 認識が済んでいる(確認できる)ページ数 */
export function readyPages(j: JobSummary): number {
  return j.n_pages - j.n_processing - j.n_failed;
}

export function progressLabel(j: JobSummary): string {
  if (j.n_processing > 0) return `認識中 ${readyPages(j) + j.n_failed} / ${j.n_pages}`;
  if (j.n_failed > 0) return `失敗 ${j.n_failed}`;
  return j.n_confirmed === j.n_pages ? "確定済み" : "確認待ち";
}

/** 「続きから」で開くページ: 中断した位置のページ > 最初の未確定のページ > 最初のページ。認識待ちのページしか無ければ null。 */
export function resumeTarget(j: JobSummary, d: JobDetail): number | null {
  const open = d.pages.filter((p) => p.status !== "processing");
  if (open.length === 0) return null;
  const last = j.last_page_id !== null ? open.find((p) => p.id === j.last_page_id) : undefined;
  return (last ?? open.find((p) => p.status !== "confirmed" && p.status !== "failed") ?? open[0]).id;
}
