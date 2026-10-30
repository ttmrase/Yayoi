// 表の 4 隅から罫線の見込みを描くための幾何(純粋関数。Vitest で試験する)。

import type { Pt } from "./types";

/** 単位正方形 (0,0)(1,0)(1,1)(0,1) → 4 隅(左上・右上・右下・左下)の射影変換。(u, v) → 画像上の点。 */
export function quadMap(c: Pt[]): (u: number, v: number) => Pt {
  const [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = c;
  const dx1 = x1 - x2;
  const dx2 = x3 - x2;
  const dx3 = x0 - x1 + x2 - x3;
  const dy1 = y1 - y2;
  const dy2 = y3 - y2;
  const dy3 = y0 - y1 + y2 - y3;
  let g = 0;
  let h = 0;
  if (Math.abs(dx3) > 1e-9 || Math.abs(dy3) > 1e-9) {
    const den = dx1 * dy2 - dx2 * dy1;
    g = (dx3 * dy2 - dx2 * dy3) / den;
    h = (dx1 * dy3 - dx3 * dy1) / den;
  }
  const a = x1 - x0 + g * x1;
  const b = x3 - x0 + h * x3;
  const d = y1 - y0 + g * y1;
  const e = y3 - y0 + h * y3;
  return (u, v) => {
    const w = g * u + h * v + 1;
    return [(a * u + b * v + x0) / w, (d * u + e * v + y0) / w];
  };
}

/** 隅から見込みの罫線(横罫線 = hfrac の数、縦罫線 = vfrac の数)の折れ線を作る。 */
export function previewGrid(
  corners: Pt[],
  hfrac: number[],
  vfrac: number[],
  steps = 8,
): { h: Pt[][]; v: Pt[][] } {
  const m = quadMap(corners);
  const ts = Array.from({ length: steps + 1 }, (_, i) => i / steps);
  return {
    h: hfrac.map((y) => ts.map((x) => m(x, y))),
    v: vfrac.map((x) => ts.map((y) => m(x, y))),
  };
}

/** 隅が「左上・右上・右下・左下の順の、時計回りの凸四角形」か(サーバの `validate_corners` と同じ向きの検査)。 */
export function isValidQuad(c: Pt[]): boolean {
  if (c.length !== 4) return false;
  for (let i = 0; i < 4; i++) {
    const a = c[i];
    const b = c[(i + 1) % 4];
    const d = c[(i + 2) % 4];
    if ((b[0] - a[0]) * (d[1] - b[1]) - (b[1] - a[1]) * (d[0] - b[0]) <= 0) return false;
  }
  return true;
}

export const polyline = (pts: Pt[]): string => pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * 重ね表示の表示範囲: 元画像 + 検出した隅(画像の外に出ていることがある)に余白を足した範囲。
 * 画像の外(紙が写っていない側)に隅があっても、丸が見えて動かせるようにする。
 */
export function viewBoxFor(width: number, height: number, corners: Pt[]): Box {
  const m = 0.06 * Math.max(width, height);
  const xs = [0, width, ...corners.map((p) => p[0])];
  const ys = [0, height, ...corners.map((p) => p[1])];
  const x0 = Math.min(...xs) - m;
  const y0 = Math.min(...ys) - m;
  return { x: x0, y: y0, w: Math.max(...xs) + m - x0, h: Math.max(...ys) + m - y0 };
}

/** 表示範囲の中に収める(サーバが受け付ける範囲 = 画像の幅・高さの −0.5〜1.5 倍も超えない) */
export function clampTo(p: Pt, box: Box, width: number, height: number): Pt {
  const lo: Pt = [Math.max(box.x, -0.5 * width), Math.max(box.y, -0.5 * height)];
  const hi: Pt = [Math.min(box.x + box.w, 1.5 * width), Math.min(box.y + box.h, 1.5 * height)];
  return [Math.min(hi[0], Math.max(lo[0], p[0])), Math.min(hi[1], Math.max(lo[1], p[1]))];
}
