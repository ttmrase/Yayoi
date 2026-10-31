import { useEffect, useRef, useState } from "react";
import type { RowTax } from "./types";

interface Props {
  /** 何行目の税区分を選ぶか(表示用は 1 始まり) */
  row: number;
  current: RowTax;
  options: string[];
  /** null = 行の指定を外して、科目ごとの既定(または記憶からの提案)に戻す */
  onPick: (category: string | null) => void;
  onClose: () => void;
}

/**
 * 行ごとの税区分を選ぶ候補リスト(キーボードで操作できる)。↑↓ で動かし、Enter で決め、Esc で閉じる。先頭は「既定に戻す」。
 * 開いたときのフォーカスは今の値の上。閉じたら呼び出し側が元のセルへフォーカスを戻す。
 */
export default function TaxPicker({ row, current, options, onPick, onClose }: Props) {
  const items: (string | null)[] = [null, ...options];
  const start = Math.max(0, items.indexOf(current.source === "user" ? current.category : null));
  const [active, setActive] = useState(start);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => box.current?.focus(), []);
  useEffect(() => {
    box.current?.querySelector<HTMLElement>(`[data-i="${active}"]`)?.scrollIntoView({ block: "nearest" });
  }, [active]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") setActive((i) => Math.min(items.length - 1, i + 1));
    else if (e.key === "ArrowUp") setActive((i) => Math.max(0, i - 1));
    else if (e.key === "Home") setActive(0);
    else if (e.key === "End") setActive(items.length - 1);
    else if (e.key === "Enter") onPick(items[active]);
    else if (e.key === "Escape" || e.key === "F2") onClose();
    else return;
    e.preventDefault();
    e.stopPropagation();
  };

  return (
    <div className="taxpicker" role="dialog" aria-label={`${row + 1} 行目の税区分を選ぶ`} data-testid="tax-picker">
      <p className="hint">
        {row + 1} 行目の税区分(↑↓ で選び、Enter で決定、Esc で閉じる)
      </p>
      <div
        ref={box}
        role="listbox"
        tabIndex={0}
        aria-label="税区分"
        aria-activedescendant={`taxopt-${active}`}
        onKeyDown={onKeyDown}
        className="taxlist"
      >
        {items.map((it, i) => {
          const selected = it === null ? current.source !== "user" : current.source === "user" && current.category === it;
          return (
            <div
              key={it ?? "reset"}
              id={`taxopt-${i}`}
              data-i={i}
              role="option"
              aria-selected={i === active}
              className={`taxopt${i === active ? " active" : ""}${selected ? " current" : ""}`}
              onMouseDown={(e) => {
                e.preventDefault();
                onPick(it);
              }}
            >
              {it === null ? `既定に戻す(科目ごとの既定: ${current.default || "未設定"})` : it}
              {selected && <span className="sr-only">(今の値)</span>}
              {selected && <span aria-hidden> ✓</span>}
            </div>
          );
        })}
      </div>
      <button onClick={onClose}>閉じる</button>
    </div>
  );
}
