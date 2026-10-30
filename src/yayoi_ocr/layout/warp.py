"""正準ページへのワープ。

射影変換 H で大まかに正した上に、罫線 1 本ずつのずれ(紙のたわみ)を変位場として足し込む。
結果の正準ページ画像では、すべての罫線がテンプレートの位置に真っ直ぐ並ぶ。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..template import Box, FormTemplate
from .grid import GridFit, LineTrack


@dataclass
class PageGeometry:
    """正準単位 ↔ 元画像 px の写像と、正準ページ画像の座標系。"""

    tpl: FormTemplate
    H: np.ndarray  # 正準単位(ずれ補正後)→ 作業画像 px
    work_scale: float  # 元画像 → 作業画像の倍率
    hlines: list[LineTrack]
    vlines: list[LineTrack]
    width: int  # 正準ページ画像の幅 px
    height: int

    @property
    def px_per_unit(self) -> float:
        return float(self.tpl.canon_width)

    def unit_to_px(
        self, u: np.ndarray | float, v: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        b = self.tpl.page_box
        return (np.asarray(u) - b.x0) * self.px_per_unit, (np.asarray(v) - b.y0) * self.px_per_unit

    def px_to_unit(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        b = self.tpl.page_box
        return np.asarray(x) / self.px_per_unit + b.x0, np.asarray(y) / self.px_per_unit + b.y0

    # --- 変位場 -------------------------------------------------------------

    def _displace(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """正準単位の格子点 (u: (Wp,), v: (Hp,)) に対する (dX: Hp x Wp, dY: Hp x Wp)。"""
        ys = np.array(self.tpl.hline_y)
        xs = np.array(self.tpl.vline_x)
        # 横罫線 j の x 方向の関数としてのずれ → E[j, u]
        e = np.stack([hl.interp(u) for hl in self.hlines])  # J x Wp
        j = np.clip(np.searchsorted(ys, v, side="right") - 1, 0, len(ys) - 2)
        t = np.clip((v - ys[j]) / (ys[j + 1] - ys[j]), 0.0, 1.0)
        dy = (1 - t)[:, None] * e[j] + t[:, None] * e[j + 1]  # Hp x Wp
        # 縦罫線 i の y 方向の関数としてのずれ → D[i, v]
        d = np.stack([vl.interp(v) for vl in self.vlines])  # I x Hp
        i = np.clip(np.searchsorted(xs, u, side="right") - 1, 0, len(xs) - 2)
        s = np.clip((u - xs[i]) / (xs[i + 1] - xs[i]), 0.0, 1.0)
        dx = ((1 - s)[None, :] * d[i].T) + (s[None, :] * d[i + 1].T)  # Hp x Wp
        return dx, dy

    def source_maps(self, rows: slice | None = None) -> tuple[np.ndarray, np.ndarray]:
        """正準ページ画像の各画素 → 元画像の座標 (map_x, map_y)(cv2.remap 用)。"""
        b = self.tpl.page_box
        u = b.x0 + (np.arange(self.width) + 0.5) / self.px_per_unit
        v = b.y0 + (np.arange(self.height) + 0.5) / self.px_per_unit
        dx, dy = self._displace(u, v)
        uu = u[None, :] + dx
        vv = v[:, None] + dy
        h = self.H
        den = h[2, 0] * uu + h[2, 1] * vv + h[2, 2]
        xw = (h[0, 0] * uu + h[0, 1] * vv + h[0, 2]) / den
        yw = (h[1, 0] * uu + h[1, 1] * vv + h[1, 2]) / den
        # 作業画像 px → 元画像 px(画素中心の規約: cv2.resize は中心を 0.5 ずらす)
        s = self.work_scale
        return ((xw + 0.5) / s - 0.5).astype(np.float32), ((yw + 0.5) / s - 0.5).astype(np.float32)

    def unit_points_to_source(self, pts: np.ndarray) -> np.ndarray:
        """正準単位の点列 (N,2) → 元画像 px (N,2)(変位場を考慮)。"""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        out = np.zeros_like(pts)
        for k, (u, v) in enumerate(pts):
            dx, dy = self._displace(np.array([u]), np.array([v]))
            uu, vv = u + dx[0, 0], v + dy[0, 0]
            h = self.H
            den = h[2, 0] * uu + h[2, 1] * vv + h[2, 2]
            xw = (h[0, 0] * uu + h[0, 1] * vv + h[0, 2]) / den
            yw = (h[1, 0] * uu + h[1, 1] * vv + h[1, 2]) / den
            out[k] = ((xw + 0.5) / self.work_scale - 0.5, (yw + 0.5) / self.work_scale - 0.5)
        return out

    def box_polygon_source(self, box: Box, per_edge: int = 2) -> list[tuple[float, float]]:
        """矩形(正準単位)の周を元画像座標の多角形にする(辺ごとに中間点を入れて曲がりに追従)。"""
        t = np.linspace(0.0, 1.0, per_edge + 1)[:-1]
        top = [(box.x0 + (box.x1 - box.x0) * a, box.y0) for a in t]
        right = [(box.x1, box.y0 + (box.y1 - box.y0) * a) for a in t]
        bottom = [(box.x1 - (box.x1 - box.x0) * a, box.y1) for a in t]
        left = [(box.x0, box.y1 - (box.y1 - box.y0) * a) for a in t]
        pts = np.array(top + right + bottom + left)
        src = self.unit_points_to_source(pts)
        return [(float(x), float(y)) for x, y in src]

    def box_to_px(self, box: Box) -> tuple[int, int, int, int]:
        x0, y0 = self.unit_to_px(box.x0, box.y0)
        x1, y1 = self.unit_to_px(box.x1, box.y1)
        return (
            round(float(x0)),
            round(float(y0)),
            round(float(x1)),
            round(float(y1)),
        )


def make_geometry(fit: GridFit, tpl: FormTemplate, work_scale: float) -> PageGeometry:
    b = tpl.page_box
    w = round((b.x1 - b.x0) * tpl.canon_width)
    h = round((b.y1 - b.y0) * tpl.canon_width)
    return PageGeometry(
        tpl=tpl,
        H=fit.H,
        work_scale=work_scale,
        hlines=fit.hlines,
        vlines=fit.vlines,
        width=w,
        height=h,
    )


def rectify(img: np.ndarray, geom: PageGeometry) -> np.ndarray:
    """元画像(BGR/グレー)を正準ページ画像へ。元画像が大きいときは先に面積平均で縮める。"""
    # 元画像 1px あたりの正準 px 数(おおよそ)。1 より小さい = 縮小方向。
    src_h, src_w = img.shape[:2]
    table_w_src = float(np.hypot(geom.H[0, 0], geom.H[1, 0])) / geom.work_scale
    ratio = geom.px_per_unit / max(float(table_w_src), 1.0)  # 正準 px / 元 px
    if ratio < 0.67:  # 1.5 倍より強く縮小するなら事前に縮める
        f = ratio * 1.2
        img = cv2.resize(
            img, (max(1, round(src_w * f)), max(1, round(src_h * f))), interpolation=cv2.INTER_AREA
        )
        pre = f
    else:
        pre = 1.0
    map_x, map_y = geom.source_maps()
    if pre != 1.0:
        map_x = ((map_x + 0.5) * pre - 0.5).astype(np.float32)
        map_y = ((map_y + 0.5) * pre - 0.5).astype(np.float32)
    return cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if img.ndim == 3 else 255,
    )
