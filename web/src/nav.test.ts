import { describe, expect, it } from "vitest";
import { activeColumn, initialFocus, navigableIds, nextId, nextUnpassedId, prevId, verticalId } from "./nav";
import type { Cell } from "./types";

function cell(id: number, row: number, col: number, over: Partial<Cell> = {}): Cell {
  return {
    id, row, col, column: "date", kind: "text", recognized_text: "", resolved_value: "", value: "",
    edit_kind: "none", expansion: null, reasons: [], candidates: [], review: true, passed: false,
    has_image: true, ...over,
  };
}

const cells: Cell[] = [
  cell(1, 0, 0), cell(2, 0, 1), cell(3, 1, 0, { kind: "blank", review: false }), // 要確認でない空欄は飛ばす
  cell(4, 1, 1, { kind: "blank", review: true }), // 要確認の空欄は止まる
  cell(5, 2, 0, { passed: true }), cell(6, 2, 1),
];

describe("navigation", () => {
  it("skips unflagged blank cells but stops on flagged blanks", () => {
    expect(nextId(cells, 2)).toBe(4);
    expect(nextId(cells, 4)).toBe(5);
    expect(prevId(cells, 4)).toBe(2);
    expect(nextId(cells, 6)).toBeNull();
    expect(prevId(cells, 1)).toBeNull();
  });
  it("moves vertically within a column", () => {
    expect(verticalId(cells, 1, 1)).toBe(5);
    expect(verticalId(cells, 5, -1)).toBe(1);
    expect(verticalId(cells, 1, -1)).toBeNull();
  });
  it("starts at the saved position, else the first unpassed review cell", () => {
    expect(initialFocus(cells, 6)).toBe(6);
    expect(initialFocus(cells, 3)).toBe(1); // 保存位置が飛ばすセルなら先頭の未通過
    expect(initialFocus(cells.map((c) => ({ ...c, passed: true })), null)).toBe(1);
  });
});

describe("column mode", () => {
  it("orders cells column by column (top to bottom, then the next column)", () => {
    expect(navigableIds(cells, "row")).toEqual([1, 2, 4, 5, 6]);
    expect(navigableIds(cells, "column")).toEqual([1, 5, 2, 4, 6]);
    expect(nextId(cells, 5, "column")).toBe(2); // 列の終わりで次の列の先頭へ
    expect(prevId(cells, 2, "column")).toBe(5);
    expect(nextId(cells, 6, "column")).toBeNull();
  });
  it("starts at the first unpassed review cell of the column order", () => {
    expect(initialFocus(cells.map((c) => (c.id === 1 ? { ...c, passed: true } : c)), null, "column")).toBe(2);
    expect(initialFocus(cells.map((c) => (c.id === 1 ? { ...c, passed: true } : c)), null, "row")).toBe(2);
  });
  it("shows the column of the focused cell", () => {
    expect(activeColumn(cells, 4)).toBe(1);
    expect(activeColumn(cells, null)).toBe(0);
  });
});

describe("jump to the next unpassed review cell", () => {
  it("goes forward/backward in the mode's order, skipping passed cells, and wraps around", () => {
    expect(nextUnpassedId(cells, 1, 1, "row")).toBe(2);
    expect(nextUnpassedId(cells, 4, 1, "row")).toBe(6); // 5 は通過済み
    expect(nextUnpassedId(cells, 6, 1, "row")).toBe(1); // 1 周して先頭へ
    expect(nextUnpassedId(cells, 1, -1, "row")).toBe(6);
    expect(nextUnpassedId(cells, 4, -1, "row")).toBe(2);
    expect(nextUnpassedId(cells, 1, 1, "column")).toBe(2); // 列順: 1, (5 は通過済み), 2
  });
  it("is null when nothing is left, and treats an unselected ambiguity as unpassed", () => {
    expect(nextUnpassedId(cells.map((c) => ({ ...c, passed: true })), 1, 1, "row")).toBeNull();
    const amb = cells.map((c) => ({ ...c, passed: true, review: false }));
    amb[5] = { ...amb[5], reasons: ["dash_digit_ambiguity"], passed: false };
    expect(nextUnpassedId(amb, 1, 1, "row")).toBe(6);
  });
});
