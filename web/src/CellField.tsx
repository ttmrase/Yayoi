import { useEffect, useRef, useState } from "react";
import { cellImage } from "./pageApi";
import { EXPANSION_LABEL, REASON_LABEL, type Cell } from "./types";

interface Props {
  pageId: number;
  cell: Cell;
  value: string;
  focused: boolean;
  locked: boolean;
  passMs: number;
  register: (el: HTMLInputElement | null) => void;
  onFocus: () => void;
  onBlur: () => void;
  onChange: (v: string) => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  onPass: () => void;
  onPickCandidate: (index: number) => void;
}

// 同じセルの画像を再取得しない(ページを行き来しても blob URL を使い回す)
const imageCache = new Map<string, string>();

function useCellImage(pageId: number, cell: Cell) {
  const key = `${pageId}/${cell.id}`;
  const [url, setUrl] = useState<string | null>(imageCache.get(key) ?? null);
  useEffect(() => {
    if (!cell.has_image) return;
    const hit = imageCache.get(key);
    if (hit) {
      setUrl(hit);
      return;
    }
    let alive = true;
    cellImage(pageId, cell.id)
      .then((u) => {
        imageCache.set(key, u);
        if (alive) setUrl(u);
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [key, cell.has_image, pageId, cell.id]);
  return url;
}

export function CellField(props: Props) {
  const { cell, value, focused, locked, passMs } = props;
  const url = useCellImage(props.pageId, cell);
  const [loaded, setLoaded] = useState(false);
  const ambiguity = cell.reasons.includes("dash_digit_ambiguity");
  const unresolved = ambiguity && cell.edit_kind !== "candidate" && cell.edit_kind !== "corrected";
  const onPass = useRef(props.onPass);
  onPass.current = props.onPass;

  // 通過: フォーカスされ、画像の読み込みが終わった状態が passMs 続いたとき
  useEffect(() => {
    if (!focused || !loaded || cell.passed) return;
    const t = window.setTimeout(() => onPass.current(), passMs);
    return () => window.clearTimeout(t);
  }, [focused, loaded, cell.passed, passMs]);

  const reasons = cell.reasons.filter((r) => REASON_LABEL[r]).slice(0, 2);
  const cls = [
    "cell",
    cell.column,
    cell.review ? "review" : "",
    cell.passed ? "passed" : "",
    focused ? "focused" : "",
    unresolved ? "ambig" : "",
    cell.edit_kind === "corrected" || cell.edit_kind === "candidate" || cell.edit_kind === "expanded_edit" ? "edited" : "",
  ].join(" ");

  return (
    <div className={cls} data-cell={cell.id} data-testid={`cell-${cell.row}-${cell.col}`}>
      <div className="crop">
        {url ? (
          <img src={url} alt={`${cell.row + 1} 行目の切り出し画像`} onLoad={() => setLoaded(true)} draggable={false} />
        ) : (
          <span className="noimg">画像なし</span>
        )}
      </div>
      <input
        ref={props.register}
        value={value}
        readOnly={locked}
        spellCheck={false}
        autoComplete="off"
        inputMode={cell.column === "amount" || cell.column === "date" ? "numeric" : "text"}
        aria-label={`${cell.row + 1} 行目 ${cell.col + 1} 列`}
        onFocus={props.onFocus}
        onBlur={props.onBlur}
        onChange={(e) => props.onChange(e.target.value)}
        onKeyDown={props.onKeyDown}
      />
      <div className="meta">
        {cell.expansion && (
          <span className="badge" title="展開済み" data-testid={`badge-${cell.row}-${cell.col}`}>
            展開済み {EXPANSION_LABEL[cell.expansion.kind]}
          </span>
        )}
        {reasons.map((r) => (
          <span key={r} className="chip">
            {REASON_LABEL[r]}
          </span>
        ))}
      </div>
      {unresolved && (
        <div className="warn" role="alert" data-testid="ambiguity">
          値が 10 倍違うかもしれません。
          {cell.candidates.map((c, i) => (
            <button key={c.value} type="button" tabIndex={-1} onMouseDown={(e) => e.preventDefault()} onClick={() => props.onPickCandidate(i)}>
              Alt+{i + 1} {c.value}
            </button>
          ))}
        </div>
      )}
      {focused && !unresolved && cell.candidates.length > 0 && (
        <div className="cands" data-testid="candidates">
          {cell.candidates.slice(0, 9).map((c, i) => (
            <button key={c.value} type="button" tabIndex={-1} onMouseDown={(e) => e.preventDefault()} onClick={() => props.onPickCandidate(i)}>
              Alt+{i + 1} {c.value}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
