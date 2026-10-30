import type { Cell } from "./types";

/** 行モード = 紙の並び(行ごとに 4 列)、列モード = 1 列ずつ(上から下、列が終わったら次の列) */
export type Mode = "row" | "column";

const byMode = (mode: Mode) => (a: Cell, b: Cell) =>
  mode === "row" ? a.row - b.row || a.col - b.col : a.col - b.col || a.row - b.row;

/** Enter で順に進むセル: 空欄で要確認でないセルは飛ばす(要確認の空欄は「書いてあるのに空欄」の疑いなので止まる)。 */
export function isNavigable(c: Cell): boolean {
  return c.kind !== "blank" || c.review;
}

export function navigableIds(cells: Cell[], mode: Mode = "row"): number[] {
  return [...cells].sort(byMode(mode)).filter(isNavigable).map((c) => c.id);
}

export function nextId(cells: Cell[], current: number, mode: Mode = "row"): number | null {
  const ids = navigableIds(cells, mode);
  const i = ids.indexOf(current);
  if (i < 0) return ids[0] ?? null;
  return ids[i + 1] ?? null;
}

export function prevId(cells: Cell[], current: number, mode: Mode = "row"): number | null {
  const ids = navigableIds(cells, mode);
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
export function initialFocus(cells: Cell[], saved: number | null, mode: Mode = "row"): number | null {
  if (saved !== null && cells.some((c) => c.id === saved && isNavigable(c))) return saved;
  const sorted = [...cells].sort(byMode(mode));
  return sorted.find((c) => isNavigable(c) && c.review && !c.passed)?.id ?? navigableIds(cells, mode)[0] ?? null;
}

const needsPass = (c: Cell) => isNavigable(c) && (c.review || c.reasons.includes("dash_digit_ambiguity")) && !c.passed;

/** 今のセルより後(dir=1)/前(dir=-1)で、まだ通過していない要確認のセル。端まで行ったら反対側から探す(1 周)。 */
export function nextUnpassedId(cells: Cell[], current: number | null, dir: -1 | 1, mode: Mode = "row"): number | null {
  const sorted = [...cells].sort(byMode(mode));
  const todo = sorted.filter(needsPass);
  if (todo.length === 0) return null;
  const pos = sorted.findIndex((c) => c.id === current);
  const ordered = dir === 1 ? sorted : [...sorted].reverse();
  const start = pos < 0 ? -1 : dir === 1 ? pos : sorted.length - 1 - pos;
  for (let k = 1; k <= ordered.length; k++) {
    const c = ordered[(start + k) % ordered.length];
    if (needsPass(c) && c.id !== current) return c.id;
  }
  return null;
}

/** 列モードで表示する列(フォーカス中のセルの列。無ければ最初の列) */
export function activeColumn(cells: Cell[], focusId: number | null): number {
  return cells.find((c) => c.id === focusId)?.col ?? 0;
}
