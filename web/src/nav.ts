import type { Cell } from "./types";

/** Enter で順に進むセル: 空欄で要確認でないセルは飛ばす(要確認の空欄は「書いてあるのに空欄」の疑いなので止まる)。 */
export function isNavigable(c: Cell): boolean {
  return c.kind !== "blank" || c.review;
}

export function navigableIds(cells: Cell[]): number[] {
  return [...cells].sort((a, b) => a.row - b.row || a.col - b.col).filter(isNavigable).map((c) => c.id);
}

export function nextId(cells: Cell[], current: number): number | null {
  const ids = navigableIds(cells);
  const i = ids.indexOf(current);
  if (i < 0) return ids[0] ?? null;
  return ids[i + 1] ?? null;
}

export function prevId(cells: Cell[], current: number): number | null {
  const ids = navigableIds(cells);
  const i = ids.indexOf(current);
  return i > 0 ? ids[i - 1] : null;
}

/** 同じ列の上(-1)/下(+1)のセル(進める行まで) */
export function verticalId(cells: Cell[], current: number, dir: -1 | 1): number | null {
  const cur = cells.find((c) => c.id === current);
  if (!cur) return null;
  const col = cells
    .filter((c) => c.col === cur.col && isNavigable(c))
    .sort((a, b) => a.row - b.row);
  const i = col.findIndex((c) => c.id === current);
  return col[i + dir]?.id ?? null;
}

/** ページを開いたときの最初のフォーカス: 保存された位置 > 最初の未通過の要確認セル > 最初のセル */
export function initialFocus(cells: Cell[], saved: number | null): number | null {
  if (saved !== null && cells.some((c) => c.id === saved && isNavigable(c))) return saved;
  const sorted = [...cells].sort((a, b) => a.row - b.row || a.col - b.col);
  return sorted.find((c) => isNavigable(c) && c.review && !c.passed)?.id ?? navigableIds(cells)[0] ?? null;
}
