import { describe, expect, it } from "vitest";
import { initialFocus, nextId, prevId, verticalId } from "./nav";
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
