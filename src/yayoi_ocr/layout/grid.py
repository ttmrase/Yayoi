"""罫線格子の位置合わせ。

手順(粗 → 細):
 0. 見出し帯の下辺の両端から相似変換(粗い正準座標への変換)を作る。
 1. 横罫線の周期パターンを全体で照合し、縦方向の倍率・オフセットを決める(周期の取り違え防止)。
 2. 期待位置の近くで横罫線(30 本)・縦罫線(5 本)を 1 本ずつ、区間に分けて探す(DP で滑らかに)。
 3. 罫線の交点から射影変換を再推定し、2 を繰り返す(探索幅を狭めながら)。
 4. 最後に残る罫線ごとのずれ(紙のたわみ)を「変位場」として保持する。

座標は「正準単位」(表の左上 = 原点、表の幅 = 1.0、y 下向き)。検出時のビュー画像は
`scale`(px / 単位)で正準単位から作る。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import cv2
import numpy as np

from ..template import FormTemplate
from .band import BandResult


@dataclass
class View:
    """正準単位の矩形 [x0,x1]x[y0,y1] を `scale` px/単位で切り出したビュー。"""

    x0: float
    y0: float
    x1: float
    y1: float
    scale: float

    @property
    def size(self) -> tuple[int, int]:  # (width, height) px
        return round((self.x1 - self.x0) * self.scale), round((self.y1 - self.y0) * self.scale)

    def px_to_unit(self) -> np.ndarray:
        s = self.scale
        return np.array([[1 / s, 0, self.x0], [0, 1 / s, self.y0], [0, 0, 1]], dtype=np.float64)

    def unit_to_px(self) -> np.ndarray:
        s = self.scale
        return np.array([[s, 0, -self.x0 * s], [0, s, -self.y0 * s], [0, 0, 1]], dtype=np.float64)


@dataclass
class LineTrack:
    """1 本の罫線の追跡結果。`offset` は期待位置からのずれ(正準単位)、`conf` は 0〜1。"""

    expected: float  # 期待位置(単位)。横罫線なら y、縦罫線なら x
    centers: np.ndarray  # 区間の中心(単位)。横罫線なら x、縦罫線なら y
    offset: np.ndarray
    conf: np.ndarray
    good: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def interp(self, at: np.ndarray | float) -> np.ndarray:
        """任意位置でのずれ(良好な区間のみから線形補間。両端は端の値を延長)。"""
        g = self.good
        at = np.asarray(at, dtype=np.float64)
        if g.sum() == 0:
            return np.zeros_like(at)
        return np.interp(at, self.centers[g], self.offset[g])


@dataclass
class GridFit:
    H: np.ndarray  # 正準単位 → 作業画像 px の射影変換(3x3)
    hlines: list[LineTrack]
    vlines: list[LineTrack]
    good_h: float  # 良好に検出できた横罫線の割合(行内の区間の 6 割以上が良好な線)
    good_v: float
    outer_good: bool  # 外枠 4 辺(上・下・左・右)のうち 3 辺以上が良好
    rms_resid_px: float  # 交点の射影変換あてはめ残差(ビュー px)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- 粗い変換


def coarse_homography(band: BandResult) -> np.ndarray:
    """帯の下辺の両端 → 正準単位 (0,0), (1,0) の相似変換の逆(正準単位 → 作業画像 px)。"""
    bl, br = band.bottom_left.astype(np.float64), band.bottom_right.astype(np.float64)
    vec = br - bl
    length = float(np.hypot(*vec))
    right = vec / length
    down = np.array([-right[1], right[0]])
    return np.array(
        [
            [right[0] * length, down[0] * length, bl[0]],
            [right[1] * length, down[1] * length, bl[1]],
            [0.0, 0.0, 1.0],
        ]
    )


def warp_to_view(img: np.ndarray, H: np.ndarray, view: View, border: int = 255) -> np.ndarray:
    """作業画像を正準ビューへ射影変換する。H: 正準単位 → 作業 px。"""
    M = H @ view.px_to_unit()  # ビュー px → 作業 px
    w, h = view.size
    return cv2.warpPerspective(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


# ---------------------------------------------------------------- 応答(尾根)画像


def _ridge_maps(view_img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(横線の尾根, 縦線の尾根, 上端の段差) 画像。インク(暗い)側を正にする。"""
    d = 255.0 - view_img.astype(np.float32)
    rh = cv2.GaussianBlur(d, (0, 0), sigmaX=2.0, sigmaY=1.0) - cv2.GaussianBlur(
        d, (0, 0), sigmaX=2.0, sigmaY=4.0
    )
    rv = cv2.GaussianBlur(d, (0, 0), sigmaX=1.0, sigmaY=2.0) - cv2.GaussianBlur(
        d, (0, 0), sigmaX=4.0, sigmaY=2.0
    )
    g = cv2.GaussianBlur(d, (0, 0), sigmaX=2.0, sigmaY=1.5)
    step = np.zeros_like(g)
    k = 5
    step[k:-k] = g[: -2 * k] - g[2 * k :]  # 上が暗く下が明るい境界で正
    return np.clip(rh, 0, None), np.clip(rv, 0, None), np.clip(step, 0, None)


# ---------------------------------------------------------------- DP による 1 本の線の追跡


def _track_line(
    profiles: np.ndarray,
    expected_px: float,
    half_window: int,
    ref: float,
    smooth: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    """区間ごとの断面プロファイル(K x L)から、滑らかに連なる線の位置(px)と信頼度を求める。

    `ref` は「この線の典型的な応答の強さ」。戻り値: (位置 px, 信頼度 0〜1)。
    """
    k_seg, length = profiles.shape
    offs = np.arange(-half_window, half_window + 1)
    n_states = len(offs)
    idx = np.clip(np.round(expected_px).astype(int) + offs, 0, length - 1)
    in_range = ((np.round(expected_px).astype(int) + offs) >= 0) & (
        (np.round(expected_px).astype(int) + offs) < length
    )
    score = np.minimum(profiles[:, idx] / max(ref, 1e-6), 1.5)  # K x S
    score = np.where(in_range[None, :], score, 0.0)
    # Viterbi(L1 平滑ペナルティ)
    cost = -score
    dp = np.zeros((k_seg, n_states))
    bp = np.zeros((k_seg, n_states), dtype=np.int32)
    dp[0] = cost[0]
    pen = smooth * np.abs(offs[:, None] - offs[None, :])  # [to, from]
    for k in range(1, k_seg):
        tot = dp[k - 1][None, :] + pen
        bp[k] = np.argmin(tot, axis=1)
        dp[k] = tot[np.arange(n_states), bp[k]] + cost[k]
    path = np.zeros(k_seg, dtype=np.int32)
    path[-1] = int(np.argmin(dp[-1]))
    for k in range(k_seg - 1, 0, -1):
        path[k - 1] = bp[k, path[k]]
    pos = np.zeros(k_seg)
    conf = np.zeros(k_seg)
    for k in range(k_seg):
        s = path[k]
        p0 = score[k, s]
        # 放物線補間でサブピクセル化
        if 0 < s < n_states - 1:
            a, b, c = score[k, s - 1], score[k, s], score[k, s + 1]
            den = a - 2 * b + c
            sub = 0.5 * (a - c) / den if den < -1e-9 else 0.0
            sub = float(np.clip(sub, -0.5, 0.5))
        else:
            sub = 0.0
        pos[k] = expected_px + offs[s] + sub
        conf[k] = float(np.clip(p0 / 0.6, 0.0, 1.0))
    return pos, conf


def _segment_profiles(resp: np.ndarray, edges_px: np.ndarray, axis: int) -> np.ndarray:
    """応答画像を区間に分け、区間内を線に沿って中央値でまとめた断面プロファイル(K x L)。

    axis=1: 横罫線用(列方向 = 線に沿う方向を区間化し、行ごとの中央値)。
    axis=0: 縦罫線用(行方向を区間化し、列ごとの中央値)。
    """
    h, w = resp.shape
    out = []
    for a, b in pairwise(edges_px):
        a_i, b_i = int(max(0, round(a))), int(min((w if axis == 1 else h), round(b)))
        if b_i - a_i < 3:
            out.append(np.zeros(h if axis == 1 else w, dtype=np.float32))
            continue
        seg = resp[:, a_i:b_i] if axis == 1 else resp[a_i:b_i, :]
        out.append(np.median(seg, axis=axis).astype(np.float32))
    return np.stack(out)


# ---------------------------------------------------------------- 検出(1 回分)


@dataclass
class _Params:
    h_window_units: float  # 横罫線の探索半幅(単位)
    v_window_units: float
    n_h_seg: int = 10
    n_v_seg_per_row: int = 1


def detect_lines(
    view_img: np.ndarray,
    view: View,
    tpl: FormTemplate,
    params: _Params,
) -> tuple[list[LineTrack], list[LineTrack]]:
    rh, rv, step = _ridge_maps(view_img)
    s = view.scale
    # --- 横罫線
    x_edges_u = np.linspace(0.02, 0.98, params.n_h_seg + 1)
    x_edges_px = (x_edges_u - view.x0) * s
    prof_h = _segment_profiles(rh, x_edges_px, axis=1)  # K x Hpx
    prof_step = _segment_profiles(step, x_edges_px, axis=1)
    centers_x = 0.5 * (x_edges_u[:-1] + x_edges_u[1:])
    hw_h = max(2, round(params.h_window_units * s))
    # 典型的な線の強さ: 期待位置周辺の最大値の中央値(線が大半なので頑健)
    peaks = []
    for y in tpl.hline_y[1:]:
        c = round((y - view.y0) * s)
        lo, hi = max(0, c - hw_h), min(prof_h.shape[1], c + hw_h + 1)
        if hi > lo:
            peaks.append(prof_h[:, lo:hi].max(axis=1))
    ref_h = float(np.median(np.concatenate(peaks))) if peaks else 1.0
    hlines: list[LineTrack] = []
    for j, y in enumerate(tpl.hline_y):
        exp_px = (y - view.y0) * s
        if j == 0:  # 帯の下辺: 段差として探す
            ref0 = max(float(np.median(prof_step[:, int(exp_px) - 2 : int(exp_px) + 3])), 1e-3)
            pos, conf = _track_line(prof_step, exp_px, hw_h, max(ref0, 0.5 * ref_h), smooth=0.12)
        else:
            pos, conf = _track_line(prof_h, exp_px, hw_h, ref_h)
        hlines.append(
            LineTrack(
                expected=y,
                centers=centers_x,
                offset=(pos - exp_px) / s,
                conf=conf,
                good=conf >= 0.45,
            )
        )
    # --- 縦罫線
    ys = np.array(tpl.hline_y)
    y_edges_u = np.concatenate(
        [np.linspace(ys[i], ys[i + 1], params.n_v_seg_per_row + 1)[:-1] for i in range(tpl.n_rows)]
        + [[ys[-1]]]
    )
    y_edges_px = (y_edges_u - view.y0) * s
    prof_v = _segment_profiles(rv, y_edges_px, axis=0)  # K x Wpx
    centers_y = 0.5 * (y_edges_u[:-1] + y_edges_u[1:])
    hw_v = max(2, round(params.v_window_units * s))
    peaks = []
    for x in tpl.vline_x:
        c = round((x - view.x0) * s)
        lo, hi = max(0, c - hw_v), min(prof_v.shape[1], c + hw_v + 1)
        if hi > lo:
            peaks.append(prof_v[:, lo:hi].max(axis=1))
    ref_v = float(np.median(np.concatenate(peaks))) if peaks else 1.0
    vlines: list[LineTrack] = []
    for x in tpl.vline_x:
        exp_px = (x - view.x0) * s
        pos, conf = _track_line(prof_v, exp_px, hw_v, ref_v)
        vlines.append(
            LineTrack(
                expected=x,
                centers=centers_y,
                offset=(pos - exp_px) / s,
                conf=conf,
                good=conf >= 0.45,
            )
        )
    return hlines, vlines


# ------------------------------------------------ 全体の縦倍率の照合(周期の取り違え防止)


def _global_vertical_fit(
    view_img: np.ndarray, view: View, tpl: FormTemplate
) -> tuple[float, float, float, float]:
    """左右それぞれで、横罫線 30 本の周期パターンを照合して y = oy + sy*y_j を求める。

    戻り値: (左の (oy, sy) …は使わず) → (sy_left, oy_left, sy_right, oy_right)
    """
    rh, _, _ = _ridge_maps(view_img)
    s = view.scale
    pitch = float(np.median(np.diff(tpl.hline_y)))
    res = []
    for lo_u, hi_u in ((0.03, 0.5), (0.5, 0.97)):
        a, b = int((lo_u - view.x0) * s), int((hi_u - view.x0) * s)
        prof = np.median(rh[:, a:b], axis=1)
        prof = np.convolve(prof, np.ones(3) / 3, mode="same")
        ys = np.array(tpl.hline_y[1:])  # 帯の下辺(j=0)は別扱い
        best = (-1.0, 1.0, 0.0)
        for sy in np.arange(0.84, 1.19, 0.002):
            for oy_u in np.arange(-0.35 * pitch, 0.35 * pitch + 1e-9, 1.0 / s):
                pos = (oy_u + sy * ys - view.y0) * s
                ok = (pos >= 0) & (pos < prof.shape[0] - 1)
                if ok.sum() < 0.7 * len(ys):
                    continue
                v = np.interp(pos[ok], np.arange(prof.shape[0]), prof).sum() / len(ys)
                if v > best[0]:
                    best = (v, sy, oy_u)
        res.append(best)
    return res[0][1], res[0][2], res[1][1], res[1][2]


# ---------------------------------------------------------------- 射影変換の推定


def _intersections(
    hlines: list[LineTrack], vlines: list[LineTrack]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(テンプレート点 Nx2, 観測点 Nx2, 重み N) 単位座標。良好な線どうしの交点のみ。"""
    src, dst, wts = [], [], []
    for hl in hlines:
        for vl in vlines:
            x0, y0 = vl.expected, hl.expected
            # 交点付近の区間が両方良好かを確認
            ih = int(np.argmin(np.abs(hl.centers - x0)))
            iv = int(np.argmin(np.abs(vl.centers - y0)))
            if not (hl.good[ih] and vl.good[iv]):
                continue
            dx = float(vl.interp(y0))
            dy = float(hl.interp(x0 + dx))
            src.append((x0, y0))
            dst.append((x0 + dx, y0 + dy))
            wts.append(min(hl.conf[ih], vl.conf[iv]))
    if not src:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    return np.array(src), np.array(dst), np.array(wts)


def _fit_homography(
    src: np.ndarray, dst: np.ndarray, thr_units: float
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(src) < 8:
        return None
    k = 1000.0  # 数値条件のため px 相当にスケール
    H, inl = cv2.findHomography(
        (src * k).astype(np.float64),
        (dst * k).astype(np.float64),
        method=cv2.RANSAC,
        ransacReprojThreshold=thr_units * k,
        maxIters=3000,
        confidence=0.999,
    )
    if H is None or inl is None or inl.sum() < 8:
        return None
    S = np.diag([k, k, 1.0])
    Hu = np.linalg.inv(S) @ H @ S
    return Hu, inl.ravel().astype(bool)


def _count_good(tracks: list[LineTrack]) -> float:
    if not tracks:
        return 0.0
    return float(np.mean([t.good.mean() >= 0.6 for t in tracks]))


def _outer_ok(hl: list[LineTrack], vl: list[LineTrack]) -> bool:
    ok = [
        hl[0].good.mean() >= 0.6,
        hl[-1].good.mean() >= 0.6,
        vl[0].good.mean() >= 0.5,
        vl[-1].good.mean() >= 0.5,
    ]
    return sum(ok) >= 3


def fit_grid(
    flat: np.ndarray, band: BandResult, tpl: FormTemplate, det_width: int = 1200
) -> GridFit:
    """作業画像 `flat` と帯の検出結果から、罫線格子への位置合わせを行う。"""
    notes: list[str] = []
    H = coarse_homography(band)
    margin = 0.06
    view = View(-margin, -0.07, 1 + margin, tpl.table_height + 0.06, float(det_width))

    # 0〜1. 縦方向の倍率・オフセットの全体照合 → 4 点(上下 × 左右)で射影変換を補正
    img = warp_to_view(flat, H, view)
    sy_l, oy_l, sy_r, oy_r = _global_vertical_fit(img, view, tpl)
    ht = tpl.table_height
    xl, xr = 0.265, 0.735
    src = np.array([[xl, 0.0], [xr, 0.0], [xl, ht], [xr, ht]])
    dst = np.array(
        [
            [xl, 0.0],  # 上端(帯の下辺)は帯から得た位置を信じる
            [xr, 0.0],
            [xl, oy_l + sy_l * ht],
            [xr, oy_r + sy_r * ht],
        ]
    )
    Hc = cv2.getPerspectiveTransform(src.astype(np.float32) * 1000, dst.astype(np.float32) * 1000)
    S = np.diag([1000.0, 1000.0, 1.0])
    Hc = np.linalg.inv(S) @ Hc @ S
    H = H @ Hc

    # 2〜3. 1 本ずつの検出と射影変換の再推定(探索幅を狭めながら)
    pitch = float(np.median(np.diff(tpl.hline_y)))
    schedule = [
        (_Params(h_window_units=0.30 * pitch, v_window_units=0.030), 0.006),
        (_Params(h_window_units=0.18 * pitch, v_window_units=0.014), 0.003),
    ]
    rms = float("nan")
    for params, thr in schedule:
        img = warp_to_view(flat, H, view)
        hl, vl = detect_lines(img, view, tpl, params)
        src_p, dst_p, _ = _intersections(hl, vl)
        fit = _fit_homography(src_p, dst_p, thr)
        if fit is None:
            notes.append("交点が少なく射影変換を更新できなかった")
            continue
        Hu, _ = fit
        H = H @ Hu
    # 最終検出(変位場用)
    img = warp_to_view(flat, H, view)
    hl, vl = detect_lines(
        img, view, tpl, _Params(h_window_units=0.14 * pitch, v_window_units=0.010)
    )
    src_p, dst_p, _ = _intersections(hl, vl)
    if len(src_p):
        # 最終の射影変換に対する交点の残差(これは変位場で吸収される量)
        rms = float(np.sqrt(np.mean(np.sum((dst_p - src_p) ** 2, axis=1))) * view.scale)
    return GridFit(
        H=H,
        hlines=hl,
        vlines=vl,
        good_h=_count_good(hl),
        good_v=_count_good(vl),
        outer_good=_outer_ok(hl, vl),
        rms_resid_px=rms,
        notes=notes,
    )
