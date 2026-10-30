// キー操作の判定(純粋関数。Vitest で試験する)。

/** 日本語入力(IME)の変換中か。変換中の Enter は「変換の確定」なので、セルの送りに使ってはいけない。 */
export function isImeComposing(e: {
  isComposing?: boolean;
  keyCode?: number;
  nativeEvent?: { isComposing?: boolean; keyCode?: number };
}): boolean {
  return Boolean(
    e.isComposing || e.nativeEvent?.isComposing || e.keyCode === 229 || e.nativeEvent?.keyCode === 229,
  );
}

export type KeyAction =
  | { type: "none" }
  | { type: "confirm" } // Enter
  | { type: "prev" } // Shift+Enter
  | { type: "up" }
  | { type: "down" }
  | { type: "candidate"; index: number } // Alt+1〜9
  | { type: "undo" }
  | { type: "redo" }
  | { type: "revert" } // Esc
  | { type: "confirmPage" } // Ctrl+Enter
  | { type: "toggleMode" } // Ctrl+L(行モード ↔ 列モード)
  | { type: "nextReview" } // Ctrl+↓(次の未通過の要確認セルへ)
  | { type: "prevReview" }; // Ctrl+↑

export interface KeyLike {
  key: string;
  code?: string;
  shiftKey?: boolean;
  altKey?: boolean;
  ctrlKey?: boolean;
  metaKey?: boolean;
  isComposing?: boolean;
  keyCode?: number;
  nativeEvent?: { isComposing?: boolean; keyCode?: number };
}

export function classifyKey(e: KeyLike): KeyAction {
  if (isImeComposing(e)) return { type: "none" }; // 変換中は何もしない(IME に任せる)
  const mod = Boolean(e.ctrlKey || e.metaKey);
  if (e.key === "Enter") {
    if (mod) return { type: "confirmPage" };
    return e.shiftKey ? { type: "prev" } : { type: "confirm" };
  }
  if (e.altKey && !mod) {
    // Alt+数字。キー配列によって key が変わるので code を優先する
    const m = /^(?:Digit|Numpad)([1-9])$/.exec(e.code ?? "") ?? /^([1-9])$/.exec(e.key);
    if (m) return { type: "candidate", index: Number(m[1]) - 1 };
    return { type: "none" };
  }
  if (mod && (e.key === "z" || e.key === "Z")) return e.shiftKey ? { type: "redo" } : { type: "undo" };
  if (mod && (e.key === "y" || e.key === "Y")) return { type: "redo" };
  if (mod && !e.altKey && (e.key === "l" || e.key === "L")) return { type: "toggleMode" };
  if (mod && e.key === "ArrowDown") return { type: "nextReview" };
  if (mod && e.key === "ArrowUp") return { type: "prevReview" };
  if (e.key === "Escape") return { type: "revert" };
  if (e.key === "ArrowUp" && !mod) return { type: "up" };
  if (e.key === "ArrowDown" && !mod) return { type: "down" };
  return { type: "none" };
}
