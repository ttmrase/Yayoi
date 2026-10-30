import { describe, expect, it } from "vitest";
import { classifyKey, isImeComposing } from "./keys";

describe("IME composition", () => {
  it("never treats Enter during composition as a cell advance", () => {
    expect(classifyKey({ key: "Enter", isComposing: true })).toEqual({ type: "none" });
    expect(classifyKey({ key: "Enter", keyCode: 229 })).toEqual({ type: "none" });
    expect(classifyKey({ key: "Process", keyCode: 229 })).toEqual({ type: "none" });
    expect(classifyKey({ key: "Enter", nativeEvent: { isComposing: true } })).toEqual({ type: "none" });
    expect(classifyKey({ key: "Enter", ctrlKey: true, isComposing: true })).toEqual({ type: "none" });
    expect(isImeComposing({ keyCode: 229 })).toBe(true);
    expect(isImeComposing({ keyCode: 13 })).toBe(false);
  });
  it("advances on a plain Enter", () => {
    expect(classifyKey({ key: "Enter", keyCode: 13 })).toEqual({ type: "confirm" });
  });
});

describe("key bindings", () => {
  it("maps Enter variants", () => {
    expect(classifyKey({ key: "Enter", shiftKey: true })).toEqual({ type: "prev" });
    expect(classifyKey({ key: "Enter", ctrlKey: true })).toEqual({ type: "confirmPage" });
  });
  it("maps Alt+digit to candidates by physical key", () => {
    expect(classifyKey({ key: "¡", code: "Digit1", altKey: true })).toEqual({ type: "candidate", index: 0 });
    expect(classifyKey({ key: "2", code: "Numpad2", altKey: true })).toEqual({ type: "candidate", index: 1 });
    expect(classifyKey({ key: "9", altKey: true })).toEqual({ type: "candidate", index: 8 });
    expect(classifyKey({ key: "0", code: "Digit0", altKey: true })).toEqual({ type: "none" });
  });
  it("maps undo/redo/escape/arrows", () => {
    expect(classifyKey({ key: "z", ctrlKey: true })).toEqual({ type: "undo" });
    expect(classifyKey({ key: "z", ctrlKey: true, shiftKey: true })).toEqual({ type: "redo" });
    expect(classifyKey({ key: "y", ctrlKey: true })).toEqual({ type: "redo" });
    expect(classifyKey({ key: "Escape" })).toEqual({ type: "revert" });
    expect(classifyKey({ key: "ArrowDown" })).toEqual({ type: "down" });
    expect(classifyKey({ key: "a" })).toEqual({ type: "none" });
  });
});
