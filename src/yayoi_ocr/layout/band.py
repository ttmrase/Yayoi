"""黒い見出し帯(「日付/科目/摘要/金額」)の検出。

見出し帯は様式の中で最も目立つ手がかり(大きく黒い横長の長方形)で、
- 向き(0/90/180/270°)の判定
- 傾き・拡大率
- 表の上端の 2 隅(帯の下辺の両端)
を与える。用紙の端が写っていなくても使える。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BandResult:
    """検出した帯。座標はすべて作業画像(検出用に正規化した解像度)上の (x, y)。"""

    top_left: np.ndarray  # 帯の上辺の左端(正しい向きにしたときの左上)
    top_right: np.ndarray
    bottom_left: np.ndarray  # 帯の下辺の左端 = 表の左上の隅
    bottom_right: np.ndarray  # 帯の下辺の右端 = 表の右上の隅
    right_vec: np.ndarray  # 正しい向きでの「右」方向の単位ベクトル
    down_vec: np.ndarray  # 正しい向きでの「下」方向の単位ベクトル(帯 → 表)
    rotation_cw: int  # 画像を時計回りに何度回すと正立するか(0/90/180/270)
    confidence: float  # 0〜1
    clipped: bool  # 帯が画像の端に接している(表の隅が写っていない可能性)
    length: float  # 帯の長さ(px)
    thickness: float  # 帯の太さ(px)
    far_known: bool = False  # 帯の反対側(紙の余白)が画像内で確認できた


DARK_THRESHOLDS = (105, 130, 155)  # 帯の暗さのしきい値(低コントラストの撮影では帯が灰色になる)


def _dark_mask(flat: np.ndarray, threshold: int = 105) -> np.ndarray:
    """帯(黒ベタ+白抜き文字)を 1 つの塊にした二値画像。"""
    blur = cv2.GaussianBlur(flat, (0, 0), 3.0)
    mask = (blur < threshold).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def _thin_structure(flat: np.ndarray) -> np.ndarray:
    """細い暗い線(罫線・文字の線)らしさ。ゆるやかな明暗(影・紙の縁)には反応しにくい。"""
    f = flat.astype(np.float32)
    fine = cv2.GaussianBlur(f, (0, 0), 1.0)
    coarse = cv2.GaussianBlur(f, (0, 0), 5.0)
    return np.clip(coarse - fine, 0, 255)


def _candidates(mask: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """(スコア, 画素座標 Nx2) の候補。横長・面積大・矩形に近い塊ほど高スコア。"""
    h, w = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out: list[tuple[float, np.ndarray]] = []
    min_area = 0.004 * h * w
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        ys, xs = np.nonzero(labels == i)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        _, (rw, rh), _ = cv2.minAreaRect(pts)
        long_, short_ = max(rw, rh), min(rw, rh)
        if short_ < 4:
            continue
        aspect = long_ / short_
        fill = area / max(1.0, rw * rh)
        if aspect < 6 or fill < 0.55:
            continue
        score = float(area) * min(fill, 1.0) * min(aspect / 12.0, 1.5)
        out.append((score, pts))
    out.sort(key=lambda t: -t[0])
    return out


def _edge_lines(pts: np.ndarray, u: np.ndarray, n: np.ndarray, origin: np.ndarray):
    """u(長軸)・n(法線)座標で見た帯の上辺・下辺を直線フィットする。

    戻り値: (s_min, s_max, (a_lo, b_lo), (a_hi, b_hi)) ただし辺は n = a + b*s の形。
    lo は n が小さい側の辺、hi は大きい側の辺。
    """
    rel = pts - origin
    s = rel @ u
    t = rel @ n
    s_min, s_max = np.percentile(s, [0.5, 99.5])
    nb = 24
    edges = np.linspace(s_min, s_max, nb + 1)
    lo_pts, hi_pts = [], []
    for k in range(nb):
        sel = (s >= edges[k]) & (s < edges[k + 1])
        if sel.sum() < 10:
            continue
        sc = 0.5 * (edges[k] + edges[k + 1])
        lo_pts.append((sc, np.percentile(t[sel], 1.0)))
        hi_pts.append((sc, np.percentile(t[sel], 99.0)))

    def fit(p):
        p = np.asarray(p, dtype=np.float64)
        if len(p) < 4:
            return None
        # 外れ値に強い直線フィット(2 回の反復で残差の大きい点を除く)
        sel = np.ones(len(p), bool)
        coef = np.zeros(2)
        for _ in range(3):
            coef = np.polyfit(p[sel, 0], p[sel, 1], 1)
            res = np.abs(np.polyval(coef, p[:, 0]) - p[:, 1])
            thr = max(1.5, 2.5 * np.median(res[sel]))
            new = res < thr
            if new.sum() < 4 or (new == sel).all():
                break
            sel = new
        return coef[1], coef[0]  # (a, b): t = a + b*s

    lo, hi = fit(lo_pts), fit(hi_pts)
    if lo is None or hi is None:
        return None
    return s_min, s_max, lo, hi


MIN_TABLE_PEAKS = 4  # 帯の表側に必要な罫線らしいピークの最少本数(これ未満は帯と認めない)
MAX_CANDIDATES = 6
OPEN_KERNEL_SIZES = (9, 15, 21)  # 帯に接した筆跡を切り離す開演算のカーネル径(px)
MIN_FAR_BRIGHTNESS = 170  # 帯の反対側(紙の余白)の明るさの下限
MAX_FIRST_LINE_RATIO = 2.5  # 最初の罫線は帯の下辺から帯の太さのこの倍数以内に始まる


def find_header_band(flat: np.ndarray) -> BandResult | None:
    """作業画像(照明補正済みグレー)から見出し帯を探す。見つからなければ None。

    候補(黒く横長の塊)を、しきい値・開演算のカーネルを変えて集め、次の検証を通ったものから最長を採る:
    表側に罫線らしい周期(MIN_TABLE_PEAKS 本以上)が帯のすぐ下から始まり、反対側(紙の余白)が明るい。
    紙の外の暗い縁・影は横長でも、この検証を通らない。
    検証を通る候補がなければ、最初の候補を自信の低い結果として返す。
    """
    h_img, w_img = flat.shape
    margin = 3  # 帯の画素がこの距離以内まで画像の端に達していたら「切れている」

    def touches(pts: np.ndarray) -> bool:
        return bool(
            pts[:, 0].min() <= margin
            or pts[:, 1].min() <= margin
            or pts[:, 0].max() >= w_img - 1 - margin
            or pts[:, 1].max() >= h_img - 1 - margin
        )

    fallback: BandResult | None = None
    valid: list[BandResult] = []
    # 等方的なカーネルで開演算(向きに依存しない)。帯(太さ約 35px)は残り、細い筆跡は切り離される。
    # 筆跡・照明で最適なカーネルが違うので、複数試して検証を通った最長の候補を採る。
    openers = [None] + [
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)) for k in OPEN_KERNEL_SIZES
    ]
    for thr in DARK_THRESHOLDS:
        base = _dark_mask(flat, thr)
        for opener in openers:
            mask = base if opener is None else cv2.morphologyEx(base, cv2.MORPH_OPEN, opener)
            cands = _candidates(mask)[:MAX_CANDIDATES]
            for _, pts in sorted(cands, key=lambda c: (touches(c[1]), -c[0])):
                res = _band_from_points(flat, pts, touches(pts))
                if res is None:
                    continue
                band, peaks, first, far = res
                far_dark = np.isfinite(far) and far < MIN_FAR_BRIGHTNESS
                if peaks >= MIN_TABLE_PEAKS and first <= MAX_FIRST_LINE_RATIO and not far_dark:
                    band.far_known = bool(np.isfinite(far))
                    valid.append(band)
                elif fallback is None:
                    fallback = band
                    fallback.confidence = min(fallback.confidence, 0.3)
    if valid:
        # 照明むらで帯の一部だけが暗く判定された断片より、全体が取れた最長の候補を採る
        return max(valid, key=lambda b: (b.far_known, not b.clipped, b.length))
    return fallback


def _band_from_points(
    flat: np.ndarray, pts: np.ndarray, clipped: bool
) -> tuple[BandResult, float, float, float] | None:
    """候補の点群から帯の幾何を求める。

    戻り値: (帯, 罫線ピーク数, 最初の罫線までの距離/帯の太さ, 反対側の平均の明るさ)。
    """
    # 主軸(長軸)
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    _, evecs = np.linalg.eigh(cov)
    u = evecs[:, 1]
    u = u / np.linalg.norm(u)
    n = np.array([-u[1], u[0]])
    fitted = _edge_lines(pts, u, n, mean)
    if fitted is None:
        return None
    s_min, s_max, lo, hi = fitted

    def corner(s: float, coef) -> np.ndarray:
        return mean + u * s + n * (coef[0] + coef[1] * s)

    p_lo_l, p_lo_r = corner(s_min, lo), corner(s_max, lo)
    p_hi_l, p_hi_r = corner(s_min, hi), corner(s_max, hi)
    thickness = float(np.hypot(*(0.5 * (p_hi_l + p_hi_r) - 0.5 * (p_lo_l + p_lo_r))))
    length = float(np.hypot(*(p_lo_r - p_lo_l)))

    # 表のある側(= 帯から見て「下」)を判定: 帯の両側で「罫線らしい細い構造」の密度を比べる。
    # (平均の暗さは、紙の外の暗い領域や影に引きずられるので使わない)
    h, w = flat.shape
    thin = _thin_structure(flat)
    ys, xs = np.mgrid[0:h:3, 0:w:3]
    grid = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    rel = grid - mean
    s = rel @ u
    t = rel @ n
    inside_s = (s > s_min + 0.05 * length) & (s < s_max - 0.05 * length)
    t_mid_lo = lo[0] + lo[1] * s
    t_mid_hi = hi[0] + hi[1] * s
    depth = 12.0 * thickness
    side_a = inside_s & (t < t_mid_lo - 0.5 * thickness) & (t > t_mid_lo - depth)
    side_b = inside_s & (t > t_mid_hi + 0.5 * thickness) & (t < t_mid_hi + depth)

    def density(sel: np.ndarray, sign: float, edge_t: np.ndarray) -> tuple[float, float]:
        """帯から離れる向きの断面にある「罫線らしいピーク」の (本数, 最初のピークまでの距離)。

        表側には等間隔の罫線が十数本あり、最初の罫線は帯のすぐ下に始まる。
        紙の外の暗い縁や題名の文字は数本しか出ない/表までの距離が大きい。
        """
        if sel.sum() < 50:
            return -1.0, float("inf")
        dist = (sign * (t[sel] - edge_t[sel])).astype(int)
        vals = thin[grid[sel, 1].astype(int), grid[sel, 0].astype(int)]
        nb = int(dist.max()) + 1
        tot = np.bincount(dist, weights=vals, minlength=nb)
        cnt = np.bincount(dist, minlength=nb)
        prof = np.where(cnt >= 20, tot / np.maximum(cnt, 1), 0.0)
        peaks = (prof[1:-1] > 6.0) & (prof[1:-1] >= prof[:-2]) & (prof[1:-1] > prof[2:])
        idx = np.nonzero(peaks)[0] + 1
        return float(len(idx)), float(idx[0]) if len(idx) else float("inf")

    (dens_a, first_a), (dens_b, first_b) = (
        density(side_a, -1.0, t_mid_lo),
        density(side_b, 1.0, t_mid_hi),
    )
    if dens_b >= dens_a:
        down = n.copy()
        near_edge = (p_hi_l, p_hi_r)
        far_edge = (p_lo_l, p_lo_r)
    else:
        down = -n
        near_edge = (p_lo_l, p_lo_r)
        far_edge = (p_hi_l, p_hi_r)
    sep = abs(dens_b - dens_a) / max(dens_a, dens_b, 1e-6) if max(dens_a, dens_b) > 0 else 0.0
    # 右方向: (x右, y下) が右手系になるよう right = (down_y, -down_x)
    right = np.array([down[1], -down[0]])

    # 帯の 4 隅を「正しい向き」の左右で並べ替える
    def by_right(a, b):
        return (a, b) if (a - b) @ right <= 0 else (b, a)

    bl, br = by_right(*near_edge)
    tl, tr = by_right(*far_edge)

    rotation_cw = _rotation_to_upright(down)

    aspect = length / max(thickness, 1.0)
    conf = float(
        np.clip(0.35 + 0.15 * min(aspect / 12.0, 2.0) + 0.5 * min(sep / 6.0, 1.0), 0.0, 1.0)
    )
    band = BandResult(
        top_left=tl,
        top_right=tr,
        bottom_left=bl,
        bottom_right=br,
        right_vec=right,
        down_vec=down,
        rotation_cw=rotation_cw,
        confidence=conf,
        clipped=clipped,
        length=length,
        thickness=thickness,
    )
    first = first_b if dens_b >= dens_a else first_a
    # 帯の「表と反対側」(紙の上の余白)は明るいはず。紙の外の暗い縁は反対側が暗い/画像の外。
    far_sel = (
        inside_s & (t < t_mid_lo - 0.5 * thickness) & (t > t_mid_lo - 2.5 * thickness)
        if dens_b >= dens_a
        else inside_s & (t > t_mid_hi + 0.5 * thickness) & (t < t_mid_hi + 2.5 * thickness)
    )
    far_mean = (
        float(flat[grid[far_sel, 1].astype(int), grid[far_sel, 0].astype(int)].mean())
        if far_sel.sum() >= 50
        else float("nan")
    )
    return band, max(dens_a, dens_b), first / max(thickness, 1.0), far_mean


def _rotation_to_upright(down: np.ndarray) -> int:
    """画像を時計回りに 0/90/180/270° 回して down が +y(画面の下)になる角度。"""
    # 画像座標(y 下向き)で、時計回りに 90° 回すと (x, y) → (-y, x)。
    best, best_val = 0, -2.0
    v = down.astype(np.float64)
    for k in range(4):
        if v[1] > best_val:
            best, best_val = k * 90, v[1]
        v = np.array([-v[1], v[0]])
    return best
