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
    expected_px: np.ndarray,
    half_window: int,
    ref: float,
    smooth: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    """区間ごとの断面プロファイル(K x L)から、滑らかに連なる線の位置(px)と信頼度を求める。

    `expected_px` は区間ごとの期待位置(K,)。`ref` は「この線の典型的な応答の強さ」。
    戻り値: (位置 px (K,), 信頼度 0〜1 (K,))。
    """
    k_seg, length = profiles.shape
    expected_px = np.broadcast_to(np.asarray(expected_px, dtype=np.float64), (k_seg,))
    centers = np.round(expected_px).astype(int)
    offs = np.arange(-half_window, half_window + 1)
    n_states = len(offs)
    raw_idx = centers[:, None] + offs[None, :]  # K x S
    in_range = (raw_idx >= 0) & (raw_idx < length)
    idx = np.clip(raw_idx, 0, length - 1)
    score = np.minimum(profiles[np.arange(k_seg)[:, None], idx] / max(ref, 1e-6), 1.5)
    score = np.where(in_range, score, 0.0)
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
        st = path[k]
        p0 = score[k, st]
        sub = 0.0
        if 0 < st < n_states - 1:  # 放物線補間でサブピクセル化
            a, b, c = score[k, st - 1], score[k, st], score[k, st + 1]
            den = a - 2 * b + c
            if den < -1e-9:
                sub = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
        pos[k] = centers[k] + offs[st] + sub
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


def _smooth_pitch(pitch: np.ndarray) -> np.ndarray:
    """区間ごとの行間隔を、x に対して直線あてはめで滑らかにする(遠近による一次変化を許す)。"""
    k = np.arange(len(pitch))
    if len(pitch) >= 3:
        coef = np.polyfit(k, pitch, 1)
        return np.polyval(coef, k)
    return pitch


def detect_lines(
    view_img: np.ndarray,
    view: View,
    tpl: FormTemplate,
    params: _Params,
) -> tuple[list[LineTrack], list[LineTrack]]:
    """横罫線は上から 1 本ずつ(直前の行間隔から次を予測)、縦罫線は期待位置の近くで追跡する。"""
    rh, rv, step = _ridge_maps(view_img)
    s = view.scale
    pitch_u = float(np.median(np.diff(tpl.hline_y)))
    # --- 横罫線
    x_edges_u = np.linspace(0.02, 0.98, params.n_h_seg + 1)
    x_edges_px = (x_edges_u - view.x0) * s
    prof_h = _segment_profiles(rh, x_edges_px, axis=1)  # K x Hpx
    prof_step = _segment_profiles(step, x_edges_px, axis=1)
    centers_x = 0.5 * (x_edges_u[:-1] + x_edges_u[1:])
    k_h = len(centers_x)
    hw_h = max(2, round(params.h_window_units * s))
    # 典型的な線の強さ: 期待位置周辺の最大値の中央値(線が大半なので頑健)
    peaks = []
    for y in tpl.hline_y[1:]:
        c = round((y - view.y0) * s)
        lo, hi = max(0, c - hw_h), min(prof_h.shape[1], c + hw_h + 1)
        if hi > lo:
            peaks.append(prof_h[:, lo:hi].max(axis=1))
    ref_h = float(np.median(np.concatenate(peaks))) if peaks else 1.0
    n_h = len(tpl.hline_y)
    pos = np.zeros((n_h, k_h))
    conf = np.zeros((n_h, k_h))
    # j = 0: 帯の下辺(段差)。位置は帯の検出から ±数 px で分かっているので窓を狭める
    # (帯の中の白抜き文字の上端も同じ向きの段差になるため、広い窓では吸われる)
    exp0 = np.full(k_h, (tpl.hline_y[0] - view.y0) * s)
    c0 = round(exp0[0])
    ref0 = max(float(np.median(prof_step[:, max(0, c0 - 2) : c0 + 3])), 1e-3)
    hw0 = max(2, round(min(params.h_window_units, 0.16 * pitch_u) * s))
    pos[0], conf[0] = _track_line(prof_step, exp0, hw0, max(ref0, 0.5 * ref_h))
    pos[0] = np.where(conf[0] >= 0.3, pos[0], exp0)
    for j in range(1, n_h):
        if j == 1:
            pitch = np.full(k_h, (tpl.hline_y[1] - tpl.hline_y[0]) * s)
        else:
            lo = max(0, j - 4)
            pitch = _smooth_pitch((pos[j - 1] - pos[lo]) / (j - 1 - lo))
        pred = pos[j - 1] + pitch
        p, c = _track_line(prof_h, pred, hw_h, ref_h)
        pos[j] = np.where(c >= 0.3, p, pred)  # 見つからない区間は予測で繋ぐ(次の予測のため)
        conf[j] = c
    hlines = [
        LineTrack(
            expected=y,
            centers=centers_x,
            offset=(pos[j] - (y - view.y0) * s) / s,
            conf=conf[j],
            good=conf[j] >= 0.45,
        )
        for j, y in enumerate(tpl.hline_y)
    ]
    # --- 縦罫線(各行 1 区間)
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
        exp_px = np.full(len(centers_y), (x - view.x0) * s)
        p, c = _track_line(prof_v, exp_px, hw_v, ref_v)
        vlines.append(
            LineTrack(
                expected=x,
                centers=centers_y,
                offset=(p - exp_px) / s,
                conf=c,
                good=c >= 0.45,
            )
        )
    return hlines, vlines


def _vertical_pattern_fit(
    view_img: np.ndarray, view: View, tpl: FormTemplate
) -> tuple[float, float, float] | None:
    """縦罫線 5 本の配置パターン(間隔が不等)を、表の上半分で全体照合する。

    見出し帯の端が画像の外に切れていると、帯から決めた左端・倍率がずれる。縦罫線の並びは
    不等間隔で特徴的なので、スケール・オフセット・傾きをまとめて探す(x = a + b*u + sl*y)。
    戻り値: (a, b, sl) 正準単位。見つからなければ None。
    """
    _, rv, _ = _ridge_maps(view_img)
    f = 0.5
    small = cv2.resize(rv, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    s = view.scale * f
    h, w = small.shape
    y0 = (0.03 * tpl.table_height - view.y0) * s
    y1 = (0.55 * tpl.table_height - view.y0) * s
    r0, r1 = round(y0), round(y1)
    if r1 - r0 < 20:
        return None
    ymid = 0.5 * (y0 + y1)
    xs = np.array(tpl.vline_x)
    a_grid = np.arange(-0.22, 0.22, 1.0 / s)  # 単位
    idx = np.arange(w, dtype=np.float64)
    best = (-1.0, 0.0, 1.0, 0.0)  # 全体の最良 (score, a, b, sl)
    near = (-1.0, 0.0, 1.0, 0.0)  # 恒等に近い解の最良(帯から決めた倍率・左端がほぼ正しい場合)
    for sl in np.arange(-0.16, 0.1601, 0.008):
        m = np.array([[1.0, -sl, sl * ymid], [0.0, 1.0, 0.0]])
        sh = cv2.warpAffine(small, m, (w, h), flags=cv2.INTER_LINEAR)
        prof = np.mean(np.minimum(sh[r0:r1], 40.0), axis=0)
        for b in np.arange(0.82, 1.2001, 0.01):
            # 行の中央(ymid)での線位置(単位)= a + b*x_i。ビュー px へ。
            pos = ((a_grid[:, None] + b * xs[None, :]) - view.x0) * s  # A x 5
            ok = (pos >= 0) & (pos <= w - 1)
            score = np.where(ok, np.interp(pos, idx, prof), 0.0).sum(axis=1)
            k = int(np.argmax(score))
            if score[k] > best[0]:
                best = (float(score[k]), float(a_grid[k]), float(b), float(sl))
            if abs(b - 1.0) <= 0.03:
                near_ok = np.abs(a_grid) <= 0.03
                if near_ok.any():
                    kk = int(np.argmax(np.where(near_ok, score, -1.0)))
                    if score[kk] > near[0]:
                        near = (float(score[kk]), float(a_grid[kk]), float(b), float(sl))
    if best[0] <= 0:
        return None
    # 恒等に近い解が全体の最良の 85% 以上なら、そちらを採る(誤った周期への飛びつきを防ぐ)
    _, a, b, sl = near if near[0] >= 0.85 * best[0] else best
    y_mid_units = (ymid / s) + view.y0
    return a - (sl * y_mid_units), b, sl


def _global_vertical_lines_fit(
    view_img: np.ndarray, view: View, tpl: FormTemplate
) -> list[tuple[float, float, float]]:
    """縦罫線 5 本それぞれについて、傾きとオフセットを全体照合(Hough 的)で求める。

    戻り値: 各線の (スコア, x(y=0), x(y=table_height)) 正準単位。強い遠近・回転が残っていても、
    1 本ずつの追跡に入る前に大きなずれを取り除くための処理。
    """
    _, rv, _ = _ridge_maps(view_img)
    f = 0.5  # 計算量削減のため半分に縮小して探索
    small = cv2.resize(rv, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    s = view.scale * f
    h, w = small.shape
    y_top = (0.0 - view.y0) * s
    y_bot = (tpl.table_height - view.y0) * s
    r0, r1 = round(y_top), round(y_bot)
    xs_exp = [(x - view.x0) * s for x in tpl.vline_x]
    win = round(0.05 * s)
    slopes = np.arange(-0.12, 0.1201, 0.004)
    results = [[-1.0, 0.0, 0.0] for _ in tpl.vline_x]  # (score, x_at_top, slope)
    for sl in slopes:
        # 傾き sl の線 x = x0 + sl*(y - y_top) を垂直にするシア: x' = x - sl*(y - y_top)
        m = np.array([[1.0, -sl, sl * y_top], [0.0, 1.0, 0.0]])
        sh = cv2.warpAffine(small, m, (w, h), flags=cv2.INTER_LINEAR)
        prof = np.mean(np.minimum(sh[r0:r1], 40.0), axis=0)  # 手書きの強い応答を抑える
        for i, xe in enumerate(xs_exp):
            lo, hi = max(0, round(xe) - win), min(w, round(xe) + win + 1)
            if hi - lo < 5:
                continue
            seg = prof[lo:hi]
            k = int(np.argmax(seg))
            if seg[k] > results[i][0]:
                results[i] = [float(seg[k]), float(lo + k), float(sl)]
    out = []
    for score, x_top, sl in results:
        x_bot = x_top + sl * (y_bot - y_top)
        out.append((score, x_top / s + view.x0, x_bot / s + view.x0))
    return out


def _robust_line(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, thr: float
) -> tuple[float, float] | None:
    """重み付き直線 y = a + b x。残差が thr を超える点を順に除く(最低 3 点)。"""
    keep = np.ones(len(x), bool)
    for _ in range(len(x)):
        if keep.sum() < 3:
            return None
        coef = np.polyfit(x[keep], y[keep], 1, w=np.sqrt(w[keep]))
        res = np.abs(np.polyval(coef, x) - y)
        worst = int(np.argmax(np.where(keep, res, -1)))
        if res[worst] <= thr:
            return float(coef[1]), float(coef[0])
        keep[worst] = False
    return None


def _vertical_corner_correction(vfit, tpl: FormTemplate) -> np.ndarray | None:
    """縦罫線の全体照合の結果を 5 本の整合性(直線性)で確かめ、4 隅の x を求めて補正射影を返す。"""
    xs = np.array(tpl.vline_x)
    sc = np.array([v[0] for v in vfit])
    xt = np.array([v[1] for v in vfit])
    xb = np.array([v[2] for v in vfit])
    good = sc >= 0.5 * sc.max()
    if good.sum() < 3 or sc.max() <= 0:
        return None
    wts = np.where(good, sc, 1e-3)
    # 同じ線が上下の両方で整合する必要がある: 上と下を同時に除外
    keep = good.copy()
    ft = fb = None
    for _ in range(3):
        if keep.sum() < 3:
            return None
        ft = _robust_line(xs[keep], xt[keep], wts[keep], 0.02)
        fb = _robust_line(xs[keep], xb[keep], wts[keep], 0.02)
        if ft is None or fb is None:
            return None
        res = np.maximum(np.abs(ft[0] + ft[1] * xs - xt), np.abs(fb[0] + fb[1] * xs - xb))
        new_keep = keep & (res < 0.02)
        if (new_keep == keep).all():
            break
        keep = new_keep
    ht = tpl.table_height
    src = np.array([[xs[0], 0.0], [xs[-1], 0.0], [xs[0], ht], [xs[-1], ht]])
    dst = np.array(
        [
            [ft[0] + ft[1] * xs[0], 0.0],
            [ft[0] + ft[1] * xs[-1], 0.0],
            [fb[0] + fb[1] * xs[0], ht],
            [fb[0] + fb[1] * xs[-1], ht],
        ]
    )
    if np.any(np.abs(dst - src) > 0.2):
        return None
    return _homography_units(src, dst)


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


def _homography_units(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """正準単位の 4 点対応 → 射影変換(単位座標系のまま)。"""
    k = 1000.0
    h = cv2.getPerspectiveTransform((src * k).astype(np.float32), (dst * k).astype(np.float32))
    sc = np.diag([k, k, 1.0])
    return np.linalg.inv(sc) @ h.astype(np.float64) @ sc


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
    flat: np.ndarray,
    band: BandResult,
    tpl: FormTemplate,
    det_width: int = 1200,
    trace: list | None = None,
) -> GridFit:
    """作業画像 `flat` と帯の検出結果から、罫線格子への位置合わせを行う。"""
    notes: list[str] = []
    H = coarse_homography(band)
    margin = 0.06
    view = View(-margin, -0.07, 1 + margin, tpl.table_height + 0.06, float(det_width))

    # 0a. 縦罫線の配置パターンの全体照合(帯の端が画像の外に切れていてもスケール・左端を合わせる)
    img = warp_to_view(flat, H, view)
    # 帯の端が画像の端で切れているときだけ使う(切れていなければ帯の倍率・左端を信頼する。
    # 強い遠近で縦罫線が扇状に開いていると、全体照合が誤った周期に飛びつくため)
    pat = _vertical_pattern_fit(img, view, tpl) if band.clipped else None
    if pat is not None:
        a, b, sl = pat
        # 単位座標のアフィン: x' = a + b*x + sl*y, y' = y(スケール b は x 方向のみ)
        A = np.array([[b, sl, a], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        if abs(b - 1.0) > 0.015 or abs(a) > 0.01 or abs(sl) > 0.01:
            H = H @ A
            notes.append("縦罫線の配置から倍率・左端を補正")
        if trace is not None:
            trace.append(("vpattern", H.copy()))
    # 0b. 縦罫線の傾き・位置の全体照合(強い遠近・回転が残っていても外れないように)
    img = warp_to_view(flat, H, view)
    corr = _vertical_corner_correction(_global_vertical_lines_fit(img, view, tpl), tpl)
    if corr is not None:
        H = H @ corr
        notes.append("縦罫線の傾きを補正")
    if trace is not None:
        trace.append(("vlines-global", H.copy()))

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
        if trace is not None:
            trace.append((f"iter-{params.v_window_units}", H.copy()))
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
