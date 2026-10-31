import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError } from "./api";
import { CellField } from "./CellField";
import { classifyKey, isImeComposing } from "./keys";
import LayoutPanel from "./LayoutPanel";
import { activeColumn, initialFocus, isNavigable, nextId, nextUnpassedId, prevId, verticalId, type Mode } from "./nav";
import {
  approveLayout,
  confirmPage,
  fetchJob,
  fetchPage,
  fetchSettings,
  patchCell,
  putPosition,
  putTitle,
  redoPage,
  reopenPage,
  retryPage,
  undoPage,
  type CellAction,
} from "./pageApi";
import { normalizeInput } from "./text";
import { COLUMN_LABEL, REASON_LABEL, type Cell, type Column, type Page, type Summary } from "./types";

const PASS_MS = 250; // 要確認セルを「通過した」と数える最小の滞在時間(連打で飛ばせない)
const DRAFT_MS = 300; // 入力中の自動保存のデバウンス

interface Props {
  pageId: number;
  /** 最初にフォーカスするセル(出力の検証の指摘から開いたとき) */
  startCell?: number | null;
  onOpenPage: (id: number) => void;
  onExit: () => void;
}

export default function ReviewScreen({ pageId, startCell = null, onOpenPage, onExit }: Props) {
  const [page, setPage] = useState<Page | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<number, string>>({});
  const [focusId, setFocusId] = useState<number | null>(null);
  const [prompt, setPrompt] = useState(false); // 最後のセルで Enter → 「ページを確定」の確認
  const [block, setBlock] = useState<Summary | null>(null); // 確定を拒否された理由
  const [reloadKey, setReloadKey] = useState(0); // 認識が終わった・再試行したときに読み直す
  const [titles, setTitles] = useState<string[]>([]); // 設定のタイトル(入力の候補)
  const [noVocabulary, setNoVocabulary] = useState(false); // 支払手段が未登録 = 科目の語彙がない
  const [titleDraft, setTitleDraft] = useState<string | null>(null);
  const [mode, setMode] = useState<Mode>("row"); // 行モード(紙の並び)/ 列モード(1 列ずつ)
  const [layoutOpen, setLayoutOpen] = useState(false); // 罫線の確認・補正のパネル
  const modeRef = useRef<Mode>("row");
  const looked = useRef(new Set<number>()); // 画像を PASS_MS 以上見たセル(サーバへの通過の記録より先に、手元で判定する)
  modeRef.current = mode;
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
    setTitleDraft(null);
    looked.current = new Set();
    active.current = { last: performance.now(), seconds: 0 };
    (async () => {
      try {
        const p = await fetchPage(pageId);
        const job = await fetchJob(p.job_id);
        if (!alive) return;
        saved.current = startCell ?? (job.last_page_id === pageId ? job.last_cell : null);
        setPage(p);
        setFocusId(initialFocus(p.cells, saved.current, modeRef.current));
      } catch (e) {
        if (alive) setError(e instanceof ApiError ? `ページを読み込めません (${e.status})` : "ページを読み込めません");
      }
    })();
    return () => {
      alive = false;
    };
  }, [pageId, reloadKey]);

  // ---- 認識待ちのページ: できるまで待つ(先読みで認識されている最中に開いたとき)
  useEffect(() => {
    if (page?.status !== "processing") return;
    const t = window.setInterval(() => {
      void fetchPage(pageId)
        .then((p) => p.status !== "processing" && setReloadKey((k) => k + 1))
        .catch(() => undefined);
    }, 1000);
    return () => window.clearInterval(t);
  }, [page?.status, pageId]);

  // ---- タイトルの候補(設定から)
  useEffect(() => {
    void fetchSettings()
      .then((r) => {
        setTitles(r.settings.titles.map((t) => t.name));
        setNoVocabulary(r.settings.payments.length === 0);
      })
      .catch(() => undefined);
  }, []);

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
  }, [focusId, page?.id, mode]);

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
      looked.current.add(cellId);
      const c = pageRef.current?.cells.find((x) => x.id === cellId);
      if (c && (c.review || c.reasons.includes("dash_digit_ambiguity")) && !c.passed) void patch(cellId, "pass");
    },
    [patch],
  );

  const shown = (c: Cell) => drafts[c.id] ?? c.value;

  const saveTitle = (value: string | null) => {
    const p = pageRef.current;
    if (!p || value === null || value.trim() === (p.title ?? "")) {
      setTitleDraft(null);
      return;
    }
    setTitleDraft(null);
    void enqueue(async () => {
      setPage(await putTitle(p.id, value));
    });
  };

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
        // 要確認のセルは、画像を一定時間見てからでないと Enter で通さない(連打・キーリピートで飛ばさない)。
        // 値を直した・候補を選んだ場合は、見て判断した証拠なので待たない(サーバも同じ下限を持つ)
        const needsLook = (cell.review || cell.reasons.includes("dash_digit_ambiguity")) && !cell.passed;
        const decided = drafts[cell.id] !== undefined || value !== cell.resolved_value || cell.edit_kind === "candidate";
        if (needsLook && !decided && !looked.current.has(cell.id)) break;
        const nxt = nextId(p.cells, cell.id, modeRef.current);
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
        const t = prevId(p.cells, cell.id, modeRef.current);
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
      case "toggleMode":
        void flushDraft(cell.id);
        setMode((m) => (m === "row" ? "column" : "row")); // フォーカス中のセルはそのまま
        break;
      case "nextReview":
      case "prevReview": {
        void flushDraft(cell.id);
        const t = nextUnpassedId(p.cells, cell.id, action.type === "nextReview" ? 1 : -1, modeRef.current);
        if (t !== null) setFocusId(t);
        else setNotice("通過していない要確認のセルはありません");
        break;
      }
    }
  };

  const column = activeColumn(page?.cells ?? [], focusId);
  const columnCells = useMemo(
    () => (page?.cells ?? []).filter((c) => c.col === column && isNavigable(c)).sort((a, b) => a.row - b.row),
    [page?.cells, column],
  );
  const rows = useMemo(() => {
    const m = new Map<number, Cell[]>();
    for (const c of page?.cells ?? []) m.set(c.row, [...(m.get(c.row) ?? []), c]);
    return [...m.entries()].sort((a, b) => a[0] - b[0]).map(([r, cs]) => [r, cs.sort((a, b) => a.col - b.col)] as const);
  }, [page?.cells]);

  if (error) return <p role="alert">{error}</p>;
  if (!page) return <p>読み込み中…</p>;
  if (page.status === "processing" || page.status === "failed") {
    return (
      <div className="review pending" data-testid="review-pending">
        <p data-testid="page-no">
          ページ {page.idx + 1} / {page.n_pages}
        </p>
        {page.status === "processing" ? (
          <p role="status">このページを認識しています…(終わると自動で開きます)</p>
        ) : (
          <div role="alert" className="banner ng">
            このページの認識に失敗しました({page.error ?? "原因不明"})。
            <button
              data-testid="retry-page"
              onClick={() => void retryPage(page.id).then(() => setReloadKey((k) => k + 1))}
            >
              もう一度認識する
            </button>
          </div>
        )}
        <p>
          {page.prev_page_id !== null && <button onClick={() => onOpenPage(page.prev_page_id!)}>前のページ</button>}{" "}
          {page.next_page_id !== null && <button onClick={() => onOpenPage(page.next_page_id!)}>次のページ</button>}{" "}
          <button onClick={onExit}>ジョブ一覧へ</button>
        </p>
      </div>
    );
  }
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
          data-testid="mode-toggle"
          aria-pressed={mode === "column"}
          title="行モード(紙の並び)と列モード(1 列ずつ)を切り替えます"
          onClick={() => setMode((m) => (m === "row" ? "column" : "row"))}
        >
          {mode === "row" ? "列モードへ (Ctrl+L)" : "行モードへ (Ctrl+L)"}
        </button>
        <button data-testid="layout-open" onClick={() => setLayoutOpen(true)}>
          罫線を確認
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

      <div className="titlebar" data-testid="titlebar">
        <label>
          ページのタイトル(勘定科目名){" "}
          <input
            data-testid="page-title"
            list="title-options"
            maxLength={40}
            disabled={confirmed}
            value={titleDraft ?? page.title ?? ""}
            placeholder={page.title_suggestion ?? ""}
            onChange={(e) => setTitleDraft(e.target.value)}
            onBlur={() => saveTitle(titleDraft)}
            onKeyDown={(e) => {
              if (e.key !== "Enter" || isImeComposing(e)) return; // 変換中の Enter は変換の確定(送らない)
              e.preventDefault();
              saveTitle(titleDraft);
              const first = initialFocus(page.cells, null, modeRef.current);
              if (first !== null) setFocusId(first);
            }}
          />
        </label>
        <datalist id="title-options">
          {titles.map((t) => (
            <option key={t} value={t} />
          ))}
        </datalist>
        {!confirmed && !page.title && page.title_suggestion && titleDraft === null && (
          <button data-testid="title-suggest" onClick={() => saveTitle(page.title_suggestion)}>
            「{page.title_suggestion}」にする
          </button>
        )}
        {!page.title && titleDraft === null && <span className="ng small">未入力(出力に必要です)</span>}
      </div>

      {noVocabulary && (
        <div className="banner" data-testid="no-vocabulary" role="note">
          科目の語彙が未設定のため、科目はすべて要確認になります。設定画面で支払手段を登録してください(確定した科目は、次のページから語彙に加わります)。
        </div>
      )}
      {page.layout_status !== "ok" && (
        <div className="banner" data-testid="layout-banner" role="alert">
          表の罫線の検出に自信がありません({page.layout_reasons.map((r) => REASON_LABEL[r] ?? r).join("、") || "要確認"})。
          <button onClick={() => setLayoutOpen(true)}>罫線を確認・補正</button>{" "}
          <button onClick={() => void approveLayout(page.id).then(setPage)}>境界は合っている</button>
        </div>
      )}
      {layoutOpen && (
        <LayoutPanel
          pageId={page.id}
          onClose={() => setLayoutOpen(false)}
          onApprove={() => void approveLayout(page.id).then((p) => { setPage(p); setLayoutOpen(false); })}
          onRecut={() => {
            setLayoutOpen(false);
            setReloadKey((k) => k + 1);
          }}
        />
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

      {mode === "column" && (
        <div className="coltabs" role="tablist" aria-label="列" data-testid="column-tabs">
          {(["date", "account", "description", "amount"] as Column[]).map((c, i) => {
            const todo = page.cells.filter((x) => x.col === i && isNavigable(x) && x.review && !x.passed).length;
            return (
              <button
                key={c}
                role="tab"
                aria-selected={i === column}
                data-testid={`coltab-${i}`}
                onClick={() => {
                  const first = page.cells
                    .filter((x) => x.col === i && isNavigable(x))
                    .sort((a, b) => a.row - b.row);
                  const t = first.find((x) => x.review && !x.passed) ?? first[0];
                  if (t) setFocusId(t.id);
                }}
              >
                {COLUMN_LABEL[c]}
                {todo > 0 && <span className="todo"> ({todo})</span>}
              </button>
            );
          })}
        </div>
      )}
      <div className={`grid${mode === "column" ? " colmode" : ""}`} role="group" aria-label="確認するセル">
        {mode === "row" && (
          <div className="row head">
            <span className="rowno" />
            {(["date", "account", "description", "amount"] as Column[]).map((c) => (
              <span key={c} className={`colhead ${c}`}>
                {COLUMN_LABEL[c]}
              </span>
            ))}
          </div>
        )}
        {(mode === "row" ? rows : columnCells.map((c) => [c.row, [c]] as const)).map(([r, cs]) => (
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
