import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError } from "./api";
import { CellField } from "./CellField";
import { classifyKey } from "./keys";
import { initialFocus, nextId, prevId, verticalId } from "./nav";
import {
  approveLayout,
  confirmPage,
  fetchJob,
  fetchPage,
  patchCell,
  putPosition,
  redoPage,
  reopenPage,
  undoPage,
  type CellAction,
} from "./pageApi";
import { normalizeInput } from "./text";
import { COLUMN_LABEL, REASON_LABEL, type Cell, type Column, type Page, type Summary } from "./types";

const PASS_MS = 250; // 要確認セルを「通過した」と数える最小の滞在時間(連打で飛ばせない)
const DRAFT_MS = 300; // 入力中の自動保存のデバウンス

interface Props {
  pageId: number;
  onOpenPage: (id: number) => void;
  onExit: () => void;
}

export default function ReviewScreen({ pageId, onOpenPage, onExit }: Props) {
  const [page, setPage] = useState<Page | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<number, string>>({});
  const [focusId, setFocusId] = useState<number | null>(null);
  const [prompt, setPrompt] = useState(false); // 最後のセルで Enter → 「ページを確定」の確認
  const [block, setBlock] = useState<Summary | null>(null); // 確定を拒否された理由
  const inputs = useRef(new Map<number, HTMLInputElement>());
  const queue = useRef<Promise<unknown>>(Promise.resolve());
  const timers = useRef(new Map<number, number>());
  const pageRef = useRef<Page | null>(null);
  const draftsRef = useRef<Record<number, string>>({});
  const active = useRef({ last: performance.now(), seconds: 0 });
  const saved = useRef<number | null>(null);
  pageRef.current = page;
  draftsRef.current = drafts;

  // ---- 経過時間(指標。キー・ポインタの操作の間隔が 30 秒を超えたら離席とみなして数えない)
  useEffect(() => {
    const tick = () => {
      const now = performance.now();
      active.current.seconds += Math.min(30, (now - active.current.last) / 1000);
      active.current.last = now;
    };
    window.addEventListener("keydown", tick);
    window.addEventListener("pointerdown", tick);
    return () => {
      window.removeEventListener("keydown", tick);
      window.removeEventListener("pointerdown", tick);
    };
  }, []);

  // ---- 読み込み
  useEffect(() => {
    let alive = true;
    setPage(null);
    setDrafts({});
    setPrompt(false);
    setBlock(null);
    setNotice(null);
    active.current = { last: performance.now(), seconds: 0 };
    (async () => {
      try {
        const p = await fetchPage(pageId);
        const job = await fetchJob(p.job_id);
        if (!alive) return;
        saved.current = job.last_page_id === pageId ? job.last_cell : null;
        setPage(p);
        setFocusId(initialFocus(p.cells, saved.current));
      } catch (e) {
        if (alive) setError(e instanceof ApiError ? `ページを読み込めません (${e.status})` : "ページを読み込めません");
      }
    })();
    return () => {
      alive = false;
    };
  }, [pageId]);

  // ---- 位置の保存(中断・再開)
  useEffect(() => {
    if (!page || focusId === null) return;
    const t = window.setTimeout(() => void putPosition(page.job_id, page.id, focusId).catch(() => undefined), 400);
    return () => window.clearTimeout(t);
  }, [page?.id, page?.job_id, focusId]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- フォーカスと、フォーカス行を画面の上から 1/3 に保つスクロール
  useEffect(() => {
    if (focusId === null) return;
    const el = inputs.current.get(focusId);
    if (!el) return;
    if (document.activeElement !== el) el.focus({ preventScroll: true });
    const row = el.closest<HTMLElement>("[data-row]");
    if (row) {
      const top = row.getBoundingClientRect().top + window.scrollY - window.innerHeight / 3;
      window.scrollTo({ top: Math.max(0, top) });
    }
  }, [focusId, page?.id]);

  // ---- サーバへの書き込みは 1 本の列にして、順序を守る
  const enqueue = useCallback(<T,>(fn: () => Promise<T>): Promise<T | undefined> => {
    const next = queue.current.then(fn);
    queue.current = next.catch(() => undefined);
    return next.catch((e: unknown) => {
      if (e instanceof ApiError && e.status === 409 && e.code === "ambiguity_unresolved") {
        setNotice("末尾が「-」か数字かを、Alt+1 / Alt+2 で選んでください");
      } else if (e instanceof ApiError && e.code === "page_confirmed") {
        setNotice("確定済みのページです(再オープンしてください)");
      } else {
        setNotice("保存できませんでした。もう一度試してください");
      }
      return undefined;
    });
  }, []);

  const patch = useCallback(
    (cellId: number, action: CellAction, extra: { value?: string; candidate_index?: number } = {}) =>
      enqueue(async () => {
        const p = await patchCell(pageId, cellId, action, extra);
        setPage(p);
        return p;
      }),
    [enqueue, pageId],
  );

  const flushDraft = useCallback(
    (cellId: number) => {
      const t = timers.current.get(cellId);
      if (t !== undefined) {
        window.clearTimeout(t);
        timers.current.delete(cellId);
      }
      const v = draftsRef.current[cellId];
      if (v === undefined) return Promise.resolve(undefined);
      const cell = pageRef.current?.cells.find((c) => c.id === cellId);
      setDrafts((d) => {
        const { [cellId]: _removed, ...rest } = d;
        void _removed;
        return rest;
      });
      if (cell && cell.value === v) return Promise.resolve(undefined);
      return patch(cellId, "edit", { value: v });
    },
    [patch],
  );

  const onChange = (cell: Cell, raw: string) => {
    const v = normalizeInput(cell.column, raw);
    setDrafts((d) => ({ ...d, [cell.id]: v }));
    setPrompt(false);
    const old = timers.current.get(cell.id);
    if (old !== undefined) window.clearTimeout(old);
    timers.current.set(
      cell.id,
      window.setTimeout(() => {
        timers.current.delete(cell.id);
        void flushDraft(cell.id);
      }, DRAFT_MS),
    );
  };

  const markPassed = useCallback(
    (cellId: number) => {
      const c = pageRef.current?.cells.find((x) => x.id === cellId);
      if (c && (c.review || c.reasons.includes("dash_digit_ambiguity")) && !c.passed) void patch(cellId, "pass");
    },
    [patch],
  );

  const shown = (c: Cell) => drafts[c.id] ?? c.value;

  const tryConfirmPage = useCallback(async () => {
    await queue.current;
    const p = pageRef.current;
    if (!p) return;
    try {
      const done = await confirmPage(p.id, Math.round(active.current.seconds));
      setPage(done);
      setBlock(null);
      setPrompt(false);
      if (done.next_page_id !== null) onOpenPage(done.next_page_id);
      else onExit();
    } catch (e) {
      if (e instanceof ApiError && e.status === 409 && e.body.summary) {
        const s = e.body.summary as Summary;
        setBlock(s);
        const first = s.unpassed[0] ?? s.ambiguous[0] ?? s.errors[0]?.cell_id;
        if (first !== undefined) setFocusId(first);
      } else {
        setNotice("確定できませんでした");
      }
    }
  }, [onExit, onOpenPage]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>, cell: Cell) => {
    const p = pageRef.current;
    if (!p) return;
    const action = classifyKey(e);
    if (action.type === "none") return; // 変換中の Enter などはここで何もしない(IME に任せる)
    e.preventDefault();
    switch (action.type) {
      case "confirm": {
        setNotice(null);
        const value = shown(cell);
        // 末尾「-」の両仮説で 10 倍違う金額は、候補の選択か入力で明示するまで進まない(サーバも同じ検査をする)
        const explicit =
          cell.edit_kind === "candidate" || cell.edit_kind === "corrected" || value !== cell.resolved_value;
        if (cell.reasons.includes("dash_digit_ambiguity") && !explicit) {
          setNotice("末尾が「-」か数字かを、Alt+1 / Alt+2 で選んでください");
          break;
        }
        const nxt = nextId(p.cells, cell.id);
        const timer = timers.current.get(cell.id);
        if (timer !== undefined) window.clearTimeout(timer);
        timers.current.delete(cell.id);
        setDrafts((d) => {
          const { [cell.id]: _removed, ...rest } = d;
          void _removed;
          return rest;
        });
        // 待たずに次のセルへ(連続で Enter を押しても遅れない)。保存は順序どおりに送る
        if (nxt !== null) setFocusId(nxt);
        else setPrompt(true);
        void patch(cell.id, "confirm", { value });
        break;
      }
      case "prev": {
        void flushDraft(cell.id);
        const t = prevId(p.cells, cell.id);
        if (t !== null) setFocusId(t);
        break;
      }
      case "up":
      case "down": {
        void flushDraft(cell.id);
        const t = verticalId(p.cells, cell.id, action.type === "up" ? -1 : 1);
        if (t !== null) setFocusId(t);
        break;
      }
      case "candidate": {
        if (action.index < cell.candidates.length) {
          setDrafts((d) => {
            const { [cell.id]: _removed, ...rest } = d;
            void _removed;
            return rest;
          });
          void patch(cell.id, "candidate", { candidate_index: action.index });
        }
        break;
      }
      case "revert":
        setDrafts((d) => {
          const { [cell.id]: _removed, ...rest } = d;
          void _removed;
          return rest;
        });
        void patch(cell.id, "revert");
        break;
      case "undo":
      case "redo":
        setDrafts({});
        void enqueue(async () => {
          const r = await (action.type === "undo" ? undoPage(p.id) : redoPage(p.id));
          setPage(r.page);
          if (r.cell_id !== null) setFocusId(r.cell_id);
        });
        break;
      case "confirmPage":
        void flushDraft(cell.id).then(() => tryConfirmPage());
        break;
    }
  };

  const rows = useMemo(() => {
    const m = new Map<number, Cell[]>();
    for (const c of page?.cells ?? []) m.set(c.row, [...(m.get(c.row) ?? []), c]);
    return [...m.entries()].sort((a, b) => a[0] - b[0]).map(([r, cs]) => [r, cs.sort((a, b) => a.col - b.col)] as const);
  }, [page?.cells]);

  if (error) return <p role="alert">{error}</p>;
  if (!page) return <p>読み込み中…</p>;
  const s = page.summary;
  const confirmed = page.status === "confirmed";

  return (
    <div className="review" data-testid="review">
      <div className="statusbar" role="status">
        <span data-testid="page-no">
          ページ {page.idx + 1} / {page.n_pages}
        </span>
        <progress max={Math.max(1, s.review_total)} value={s.review_passed} aria-label="通過済みの要確認セル" />
        <span data-testid="progress">
          通過 {s.review_passed} / {s.review_total}
        </span>
        <span data-testid="errors" className={s.errors.length ? "ng" : ""}>
          検証エラー {s.errors.length}
        </span>
        {s.ambiguous.length > 0 && <span className="ng">未選択の金額 {s.ambiguous.length}</span>}
        <button disabled={!page.can_undo} onClick={() => void enqueue(async () => { const r = await undoPage(page.id); setPage(r.page); if (r.cell_id !== null) setFocusId(r.cell_id); })}>
          元に戻す
        </button>
        <button
          data-testid="confirm-page"
          disabled={confirmed ? false : !s.can_confirm}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.nativeEvent.isComposing) {
              e.preventDefault();
              if (!confirmed) void tryConfirmPage();
            }
          }}
          onClick={() => (confirmed ? void reopenPage(page.id).then(setPage) : void tryConfirmPage())}
        >
          {confirmed ? "再オープン" : "ページを確定 (Ctrl+Enter)"}
        </button>
        <button onClick={onExit}>ジョブ一覧へ</button>
      </div>

      {page.layout_status !== "ok" && (
        <div className="banner" data-testid="layout-banner" role="alert">
          表の罫線の検出に自信がありません({page.layout_reasons.map((r) => REASON_LABEL[r] ?? r).join("、") || "要確認"})。
          <button onClick={() => void approveLayout(page.id).then(setPage)}>境界は合っている</button>
        </div>
      )}
      {notice && (
        <div className="banner ng" role="alert" data-testid="notice">
          {notice}
        </div>
      )}
      {prompt && !block && (
        <div className="banner" data-testid="prompt">
          最後のセルです。{s.can_confirm ? "Ctrl+Enter でこのページを確定します。" : "確定できない項目が残っています。"}
        </div>
      )}
      {block && (
        <div className="banner ng" role="alert" data-testid="block">
          確定できません:
          {block.unpassed.length > 0 && <> 未通過の要確認 {block.unpassed.length} 件。</>}
          {block.errors.length > 0 && <> 検証エラー {block.errors.length} 件({[...new Set(block.errors.map((x) => x.label))].join("、")})。</>}
          {block.ambiguous.length > 0 && <> 金額の末尾が未選択 {block.ambiguous.length} 件。</>}
        </div>
      )}

      <div className="grid" role="grid" aria-label="確認するセル">
        <div className="row head">
          <span className="rowno" />
          {(["date", "account", "description", "amount"] as Column[]).map((c) => (
            <span key={c} className={`colhead ${c}`}>
              {COLUMN_LABEL[c]}
            </span>
          ))}
        </div>
        {rows.map(([r, cs]) => (
          <div className="row" data-row={r} key={r}>
            <span className="rowno">{r + 1}</span>
            {cs.map((c) => (
              <CellField
                key={c.id}
                pageId={page.id}
                cell={c}
                value={shown(c)}
                focused={focusId === c.id}
                locked={confirmed}
                register={(el) => (el ? inputs.current.set(c.id, el) : inputs.current.delete(c.id))}
                onFocus={() => {
                  setFocusId(c.id);
                  setPrompt(false);
                }}
                onBlur={() => void flushDraft(c.id)}
                onChange={(v) => onChange(c, v)}
                onKeyDown={(e) => onKeyDown(e, c)}
                onPass={() => markPassed(c.id)}
                onPickCandidate={(i) => void patch(c.id, "candidate", { candidate_index: i })}
                passMs={PASS_MS}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
