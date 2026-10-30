import { describe, expect, it } from "vitest";
import { clampTo, isValidQuad, previewGrid, quadMap, viewBoxFor } from "./layoutGeometry";
import type { Pt } from "./types";

const RECT: Pt[] = [
  [100, 200],
  [500, 200],
  [500, 800],
  [100, 800],
];
const SKEW: Pt[] = [
  [120, 180],
  [520, 220],
  [480, 820],
  [90, 790],
];

describe("quadMap", () => {
  it("単位正方形の 4 隅は指定した隅に写る", () => {
    for (const q of [RECT, SKEW]) {
      const m = quadMap(q);
      [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
      ].forEach(([u, v], i) => {
        const [x, y] = m(u, v);
        expect(x).toBeCloseTo(q[i][0], 6);
        expect(y).toBeCloseTo(q[i][1], 6);
      });
    }
  });
  it("長方形なら中央は中央、辺は直線(アフィン)", () => {
    const [x, y] = quadMap(RECT)(0.5, 0.5);
    expect([x, y]).toEqual([300, 500]);
  });
  it("台形(遠近)では OpenCV の射影変換と一致する(遠い側=狭い辺が縮むので、紙の中央は画像の中央より狭い辺の側に寄る)", () => {
    const m = quadMap([
      [150, 200],
      [450, 200],
      [500, 800],
      [100, 800],
    ]);
    const [x, y] = m(0.5, 0.5);
    expect(x).toBeCloseTo(300, 6);
    expect(y).toBeCloseTo(457.142857, 4); // cv2.getPerspectiveTransform で同じ 4 点から求めた値
  });
});

describe("previewGrid", () => {
  it("罫線の数と折れ線の点数", () => {
    const g = previewGrid(SKEW, [0, 0.5, 1], [0, 0.25, 0.5, 0.75, 1], 4);
    expect(g.h).toHaveLength(3);
    expect(g.v).toHaveLength(5);
    expect(g.h[0]).toHaveLength(5);
    expect(g.h[0][0]).toEqual(SKEW[0]);
    expect(g.h[2][4][0]).toBeCloseTo(SKEW[2][0], 6);
  });
});

describe("isValidQuad", () => {
  it("時計回りの凸四角形は可、ねじれ・逆回りは不可", () => {
    expect(isValidQuad(RECT)).toBe(true);
    expect(isValidQuad(SKEW)).toBe(true);
    expect(isValidQuad([RECT[0], RECT[1], RECT[3], RECT[2]])).toBe(false);
    expect(isValidQuad([...RECT].reverse())).toBe(false);
    expect(isValidQuad(RECT.slice(0, 3))).toBe(false);
  });
});

describe("viewBoxFor / clampTo", () => {
  it("画像より外に出た隅も含む範囲になり、その中に収められる", () => {
    const corners: Pt[] = [
      [-100, 226],
      [870, 100],
      [1068, 1404],
      [92, 1565],
    ];
    const box = viewBoxFor(988, 1602, corners);
    expect(box.x).toBeLessThan(-100);
    expect(box.x + box.w).toBeGreaterThan(1068);
    expect(box.y).toBeLessThan(0);
    expect(box.y + box.h).toBeGreaterThan(1602);
    expect(clampTo([-9999, 5000], box, 988, 1602)).toEqual([Math.max(box.x, -494), Math.min(box.y + box.h, 2403)]);
    expect(clampTo([300, 400], box, 988, 1602)).toEqual([300, 400]);
  });
  it("サーバが受け付ける範囲(−0.5〜1.5 倍)を超えない", () => {
    const box = viewBoxFor(100, 100, [
      [-90, 0],
      [200, 0],
      [200, 200],
      [-90, 200],
    ]);
    const [x, y] = clampTo([-9999, 9999], box, 100, 100);
    expect(x).toBeGreaterThanOrEqual(-50);
    expect(y).toBeLessThanOrEqual(150);
  });
});
