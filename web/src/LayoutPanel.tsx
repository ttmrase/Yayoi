import { useEffect, useRef, useState } from "react";
import { ApiError } from "./api";
import { clampTo, isValidQuad, polyline, previewGrid, viewBoxFor } from "./layoutGeometry";
import { fetchLayout, layoutImage, recutPage } from "./pageApi";
import { REASON_LABEL, type LayoutData, type Pt } from "./types";

interface Props {
  pageId: number;
  onClose: () => void;
  /** 「境界は合っている」 */
  onApprove: () => void;
  /** 切り直しを依頼した(ページは認識中に戻る) */
  onRecut: () => void;
}

const CORNER_LABEL = ["① 左上", "② 右上", "③ 右下", "④ 左下"];
const KEY_STEP = 1;

/** 元画像に検出した罫線を重ねて見せ、表の 4 隅をドラッグして切り直す。 */
export default function LayoutPanel({ pageId, onClose, onApprove, onRecut }: Props) {
  const [data, setData] = useState<LayoutData | null>(null);
  const [src, setSrc] = useState<string | null>(null);
  const [corners, setCorners] = useState<Pt[]>([]);
  const [dragging, setDragging] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirmRecut, setConfirmRecut] = useState(false);
  const svg = useRef<SVGSVGElement>(null);

  useEffect(() => {
    let alive = true;
    let url: string | null = null;
    (async () => {
      try {
        const [d, u] = await Promise.all([fetchLayout(pageId), layoutImage(pageId)]);
        if (!alive) {
          URL.revokeObjectURL(u);
          return;
        }
        url = u;
        setData(d);
        setSrc(u);
        setCorners(d.override ?? d.info.corners);
      } catch (e) {
        if (alive) setError(e instanceof ApiError && e.status === 404 ? "このページには元画像の情報がありません" : "読み込めません");
      }
    })();
    return () => {
      alive = false;
      if (url) URL.revokeObjectURL(url);
    };
  }, [pageId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !e.isComposing) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (error) {
    return (
      <div className="layout-panel" role="dialog" aria-label="罫線の確認・補正">
        <p role="alert">{error}</p>
        <button onClick={onClose}>閉じる</button>
      </div>
    );
  }
  if (!data || !src) return <div className="layout-panel">読み込み中…</div>;

  const { info } = data;
  const detected = data.override ?? info.corners;
  const moved = corners.some((p, i) => Math.hypot(p[0] - detected[i][0], p[1] - detected[i][1]) > 0.5);
  const valid = isValidQuad(corners);
  const grid = moved ? previewGrid(corners, info.hfrac, info.vfrac) : { h: info.hlines, v: info.vlines };
  const box = viewBoxFor(info.width, info.height, detected);
  const clampPt = (p: Pt): Pt => clampTo(p, box, info.width, info.height);
  const handleR = info.width * 0.012;

  const toSvg = (e: React.PointerEvent): Pt => {
    const el = svg.current!;
    const pt = el.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const m = el.getScreenCTM()!.inverse();
    const q = pt.matrixTransform(m);
    return clampPt([q.x, q.y]);
  };
  const move = (i: number, p: Pt) => setCorners((c) => c.map((x, k) => (k === i ? p : x)));

  const doRecut = async () => {
    try {
      await recutPage(pageId, corners.map(([x, y]) => [Math.round(x * 10) / 10, Math.round(y * 10) / 10] as Pt));
      onRecut();
    } catch (e) {
      setConfirmRecut(false);
      setError(e instanceof ApiError && e.code === "bad_corners" ? "隅の指定が不正です(左上・右上・右下・左下の順で、表を囲むように)" : "切り直せませんでした");
    }
  };

  return (
    <div className="layout-panel" role="dialog" aria-modal="true" aria-label="罫線の確認・補正" data-testid="layout-panel">
      <div className="layout-head">
        <strong>罫線の確認・補正</strong>
        <span data-testid="layout-state">
          {data.manual ? "手動で指定済み" : data.status === "ok" ? "自動検出(問題なし)" : "自動検出(自信なし)"}
          {data.reasons.length > 0 && `: ${data.reasons.map((r) => REASON_LABEL[r] ?? r).join("、")}`}
        </span>
        <button onClick={onClose}>閉じる (Esc)</button>
      </div>
      <p className="hint">
        青い線が検出した罫線です。ずれているときは、表を囲む 4 つの丸(① 左上 → ② 右上 → ③ 右下 → ④ 左下。左上は見出し帯の下の左端)を
        ドラッグするか、丸にフォーカスして矢印キー(Shift で 10 倍)で動かし、「この位置で切り直す」を押します。
      </p>
      <svg
        ref={svg}
        data-testid="layout-svg"
        viewBox={`${box.x} ${box.y} ${box.w} ${box.h}`}
        className="layout-svg"
        role="group"
        aria-label="元画像に罫線を重ねた表示。表の 4 隅の丸を動かせます"
        onPointerMove={(e) => dragging !== null && move(dragging, toSvg(e))}
        onPointerUp={() => setDragging(null)}
        onPointerCancel={() => setDragging(null)}
      >
        <image href={src} x={0} y={0} width={info.width} height={info.height} />
        <g className={moved ? "grid-preview" : "grid-detected"} fill="none">
          {grid.h.map((l, i) => (
            <polyline key={`h${i}`} points={polyline(l)} />
          ))}
          {grid.v.map((l, i) => (
            <polyline key={`v${i}`} points={polyline(l)} />
          ))}
        </g>
        <polygon className={valid ? "quad" : "quad bad"} points={polyline(corners)} fill="none" />
        {corners.map((p, i) => (
          <g key={i}>
            <circle
              cx={p[0]}
              cy={p[1]}
              r={handleR}
              className="handle"
              tabIndex={0}
              role="slider"
              aria-label={`表の隅 ${CORNER_LABEL[i]}`}
              aria-valuetext={`x ${Math.round(p[0])}, y ${Math.round(p[1])}`}
              data-testid={`corner-${i}`}
              onPointerDown={(e) => {
                (e.target as Element).setPointerCapture(e.pointerId);
                setDragging(i);
                setConfirmRecut(false);
              }}
              onKeyDown={(e) => {
                const step = (e.shiftKey ? 10 : 1) * KEY_STEP;
                const d: Record<string, Pt> = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
                const v = d[e.key];
                if (!v) return;
                e.preventDefault();
                e.stopPropagation();
                setConfirmRecut(false);
                move(i, clampPt([p[0] + v[0], p[1] + v[1]]));
              }}
            />
            <text x={p[0] + handleR * 1.2} y={p[1] - handleR * 1.2} className="handle-label" fontSize={handleR * 1.6}>
              {CORNER_LABEL[i]}
            </text>
          </g>
        ))}
      </svg>
      <div className="layout-actions">
        <button data-testid="layout-approve" onClick={onApprove} disabled={moved}>
          境界は合っている
        </button>
        {!confirmRecut ? (
          <button data-testid="layout-recut" disabled={!moved || !valid} onClick={() => setConfirmRecut(true)}>
            この位置で切り直す
          </button>
        ) : (
          <button data-testid="layout-recut-confirm" className="danger" onClick={() => void doRecut()}>
            切り直す(このページの修正は破棄し、認識をやり直します)
          </button>
        )}
        <button disabled={!moved} onClick={() => setCorners(detected)}>
          元の位置に戻す
        </button>
        {!valid && <span className="ng">隅の順序か形が不正です</span>}
      </div>
    </div>
  );
}
