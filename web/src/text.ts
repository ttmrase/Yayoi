import type { Column } from "./types";

/** 入力の正規化: 日付・金額は全角を半角にする。金額はカンマ・空白・「-」を除いて数字だけにする(確定値は数字だけ)。 */
export function normalizeInput(column: Column, s: string): string {
  if (column === "amount") return s.normalize("NFKC").replace(/[,，\s\-ー−―]/g, "");
  if (column === "date") return s.normalize("NFKC").replace(/\s+/g, "");
  return s;
}
