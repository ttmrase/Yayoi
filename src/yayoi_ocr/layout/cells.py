"""セルの切り出し: 罫線除去・二値化・連結成分の帰属・空欄判定。

入力は「正準ページ画像」(罫線がテンプレート位置に真っ直ぐ並んでいる)。
罫線は位置が分かっているので、強度ではなく位置で消す(薄い鉛筆書きの文字を消さないため)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..models import CellFlag
from ..preprocess import flatten_illumination, normalize_contrast, to_gray
from .warp import PageGeometry

# セル切り出しの余白(行の高さに対する比率)。はみ出した文字・上に小さく書かれた注記を拾うため。
PAD_Y_FRAC = 0.32
PAD_X_PX_FRAC = 0.012  # 表の幅に対する比率
BLANK_INK_PX = (
    14  # 空欄判定: セル本体のインク画素数がこれ未満(正準 W=2000 の場合)。〃は 60 画素以上
)
# 薄い記入(鉛筆の小さな〃など)の拾い上げ。通常のインク抽出より感度を上げた別マスク。
# 空欄判定にだけ使う(見逃し=書いてあるのに空欄、を避けるため)。
FAINT_WEAK = 22.0
FAINT_STRONG = 40.0
FAINT_MIN_AREA = 40  # 連結成分の最小面積(W=2000 の場合)
BLANK_FAINT_PX = 40  # セル本体にこれ以上の薄いインクがあれば空欄としない
MIN_SPECK = 6  # これ未満の面積の連結成分はノイズとして除去(正準 W=2000 の場合)
STROKE_SE = 19  # ブラックハット変換の構造要素(筆の太さより大きく。W=2000 の場合)
INK_WEAK = 26.0  # ヒステリシス閾値(ブラックハット応答)。ブランク写真の雑音は最大でも約 38
INK_STRONG = 50.0  # 薄い鉛筆書きの応答は 70 以上


@dataclass
class PageInk:
    """ページ全体のインク解析の結果(正準ページ座標)。"""

    gray: np.ndarray  # 照明補正・コントラスト正規化済みグレー(紙 = 255)
    ink: np.ndarray  # bool: 罫線除去後のインク
    line_mask: np.ndarray  # bool: 罫線として除去した領域
    labels: np.ndarray  # int32: インクの連結成分(膨張して束ねたもの)
    owner: np.ndarray  # int32: 成分ごとの帰属セル id(-1 = 位置で決める)
    cell_id: np.ndarray  # int16: 画素ごとの位置上のセル id(-1 = 表の外)
    faint: np.ndarray | None = None  # bool: 薄い記入を含む高感度のインク(空欄判定用)
    threshold: float = 0.0
    notes: list[str] = field(default_factory=list)


def _line_halfwidth(gray: np.ndarray, centers: list[int], axis: int, hw_min: int = 2) -> list[int]:
    """各罫線の半幅(px)を断面プロファイルの半値幅から推定する(解像度に追従)。"""
    out = []
    h, w = gray.shape
    for c in centers:
        lo, hi = max(0, c - 10), min((h if axis == 0 else w), c + 11)
        if hi - lo < 5:
            out.append(hw_min + 1)
            continue
        prof = np.median(255.0 - (gray[lo:hi, :] if axis == 0 else gray[:, lo:hi]), axis=1 - axis)
        peak = float(prof.max())
        if peak < 8:
            out.append(hw_min + 1)
            continue
        above = np.nonzero(prof >= 0.18 * peak)[0]
        width = (above.max() - above.min() + 1) if len(above) else 2 * hw_min
        out.append(int(np.clip(round(width / 2 + 2.0), hw_min, 10)))
    return out


def ink_response(gray: np.ndarray, canon_width: int) -> np.ndarray:
    """細く暗い構造(筆跡・罫線)の応答。広い暗部(紙の外・影・見出し帯)や緩やかなむらには反応しない。"""
    k = max(5, round(STROKE_SE * canon_width / 2000.0) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se).astype(np.float32)


def analyze_page_ink(rect_color: np.ndarray, geom: PageGeometry) -> PageInk:
    """正準ページ画像から、罫線を除いたインク画素と連結成分の帰属を求める。"""
    tpl = geom.tpl
    gray0 = to_gray(rect_color)
    gray = normalize_contrast(flatten_illumination(gray0, kernel_frac=0.06))
    h, w = gray.shape
    resp = ink_response(gray, tpl.canon_width)
    # ヒステリシス閾値: 強い画素につながる弱い画素も採る(細い・薄い筆跡の途切れ防止)
    strong = resp >= INK_STRONG
    weak = resp >= INK_WEAK
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[np.unique(lab[strong])] = True
    keep[0] = False
    ink = keep[lab]
    thr = INK_WEAK

    # --- 罫線を位置で除去(横・縦)
    ys = [round(float(geom.unit_to_px(0, y)[1])) for y in tpl.hline_y]
    xs = [round(float(geom.unit_to_px(x, 0)[0])) for x in tpl.vline_x]
    hw_h = _line_halfwidth(gray, ys, axis=0)
    hw_v = _line_halfwidth(gray, xs, axis=1)
    # 表の範囲(罫線の延長部分だけを消す)
    tx0, tx1 = xs[0] - max(hw_v[0], 3), xs[-1] + max(hw_v[-1], 3)
    ty0, ty1 = ys[0] - max(hw_h[0], 3), ys[-1] + max(hw_h[-1], 3)
    line_mask = np.zeros((h, w), dtype=bool)
    for y, hw in zip(ys, hw_h, strict=True):
        line_mask[max(0, y - hw) : y + hw + 1, max(0, tx0) : tx1 + 1] = True
    for x, hw in zip(xs, hw_v, strict=True):
        line_mask[max(0, ty0) : ty1 + 1, max(0, x - hw) : x + hw + 1] = True
    # 罫線の位置が不確実な区間(紙の端で切れている等)は、除去する幅を広げる
    wide = max(8, round(0.012 * tpl.canon_width))
    u_cols = geom.px_to_unit(np.arange(w), 0)[0]
    v_rows = geom.px_to_unit(0, np.arange(h))[1]
    for hl, y in zip(geom.hlines, ys, strict=True):
        if not hl.good.all():
            bad = ~hl.good[np.abs(u_cols[:, None] - hl.centers[None, :]).argmin(axis=1)]
            cols = np.nonzero(bad)[0]
            cols = cols[(cols >= tx0) & (cols <= tx1)]
            line_mask[max(0, y - wide) : y + wide + 1, cols] = True
    for vl, x in zip(geom.vlines, xs, strict=True):
        if not vl.good.all():
            bad = ~vl.good[np.abs(v_rows[:, None] - vl.centers[None, :]).argmin(axis=1)]
            rows = np.nonzero(bad)[0]
            rows = rows[(rows >= ty0) & (rows <= ty1)]
            line_mask[rows, max(0, x - wide) : x + wide + 1] = True
    # 見出し帯(黒ベタ)は丸ごとインクではない: 帯の領域は除外
    bx0, by0, bx1, by1 = geom.box_to_px(tpl.band)
    band_zone = np.zeros((h, w), dtype=bool)
    band_zone[max(0, by0 - 6) : max(0, by1 + 4), max(0, bx0 - 6) : bx1 + 7] = True

    cleaned = ink & ~line_mask
    # --- 罫線をまたぐ筆跡の橋渡し: 罫線帯の上下(帯のすぐ外)にともにインクがあれば帯の中も埋める
    for y, hw in zip(ys, hw_h, strict=True):
        a, b = y - hw - 1, y + hw + 1
        if a < 1 or b >= h - 1:
            continue
        above = cleaned[a - 1 : a + 1, :].any(axis=0)
        below = cleaned[b : b + 2, :].any(axis=0)
        bridge = above & below
        # 孤立した 1 列は橋渡ししない(ノイズ防止): 近傍列にも橋渡し候補があるものだけ
        bridge = bridge & (np.convolve(bridge.astype(np.int8), np.ones(3, np.int8), "same") >= 2)
        if bridge.any():
            sl = slice(max(0, y - hw), y + hw + 1)
            seg = ink[sl, :] & bridge[None, :]
            cleaned[sl, :] |= seg
    for x, hw in zip(xs, hw_v, strict=True):
        a, b = x - hw - 1, x + hw + 1
        if a < 1 or b >= w - 1:
            continue
        left = cleaned[:, a - 1 : a + 1].any(axis=1)
        right = cleaned[:, b : b + 2].any(axis=1)
        bridge = left & right
        bridge = bridge & (np.convolve(bridge.astype(np.int8), np.ones(3, np.int8), "same") >= 2)
        if bridge.any():
            sl = slice(max(0, x - hw), x + hw + 1)
            seg = ink[:, sl] & bridge[:, None]
            cleaned[:, sl] |= seg
    cleaned &= ~band_zone

    # --- 微小ノイズ除去(解像度に比例した面積)
    speck = max(2, round(MIN_SPECK * (tpl.canon_width / 2000.0) ** 2))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cleaned.astype(np.uint8), connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < speck
    small[0] = False
    cleaned = cleaned & ~small[lab]

    # --- 連結成分(近接する筆画を束ねる)とセル位置の対応
    k = max(3, round(0.0025 * tpl.canon_width) | 1)
    bundled = cv2.dilate(
        cleaned.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    n_lab, labels = cv2.connectedComponents(bundled, connectivity=8)
    labels = np.where(cleaned, labels, 0).astype(np.int32)

    cell_id = np.full((h, w), -1, dtype=np.int16)
    n_cols = len(tpl.columns)
    # 罫線の中心で分割(罫線の位置 = セル境界)
    for r in range(tpl.n_rows):
        for c in range(n_cols):
            cell_id[ys[r] : ys[r + 1], xs[c] : xs[c + 1]] = r * n_cols + c
    owner = _assign_owners(labels, cell_id, n_lab, n_cols * tpl.n_rows)
    # 高感度マスク(薄い記入用)。罫線領域・見出し帯は除く。
    kw = (tpl.canon_width / 2000.0) ** 2
    nf, labf = cv2.connectedComponents((resp >= FAINT_WEAK).astype(np.uint8), connectivity=8)
    keepf = np.zeros(nf, dtype=bool)
    keepf[np.unique(labf[resp >= FAINT_STRONG])] = True
    keepf[0] = False
    faint = keepf[labf] & ~line_mask & ~band_zone
    _, labfc, stf, _ = cv2.connectedComponentsWithStats(faint.astype(np.uint8), connectivity=8)
    smallf = stf[:, cv2.CC_STAT_AREA] < FAINT_MIN_AREA * kw
    smallf[0] = False
    faint &= ~smallf[labfc]
    return PageInk(gray, cleaned, line_mask, labels, owner, cell_id, faint=faint, threshold=thr)


def _assign_owners(labels: np.ndarray, cell_id: np.ndarray, n_lab: int, n_cells: int) -> np.ndarray:
    """成分ごとに、インク画素の 75% 以上が属するセルを帰属先とする。なければ -1(位置で決める)。"""
    owner = np.full(n_lab, -1, dtype=np.int32)
    sel = (labels > 0) & (cell_id >= 0)
    lab = labels[sel].astype(np.int64)
    cid = cell_id[sel].astype(np.int64)
    counts = np.bincount(lab * n_cells + cid, minlength=n_lab * n_cells).reshape(n_lab, n_cells)
    total = counts.sum(axis=1)
    best = counts.argmax(axis=1)
    frac = counts[np.arange(n_lab), best] / np.maximum(total, 1)
    good = (total > 0) & (frac >= 0.75)
    owner[good] = best[good]
    return owner


@dataclass
class CellCrop:
    color: np.ndarray  # 余白付きの元画像(ワープ済みカラー)
    binary: np.ndarray  # 罫線除去・二値化(インク = 0, 背景 = 255)。他セルのインクは除去済み
    window: tuple[int, int, int, int]  # 正準ページ座標での (x0, y0, x1, y1)(余白込み)
    cell_rect: tuple[int, int, int, int]  # 正準ページ座標でのセル本体
    ink_px: int  # セル本体(罫線の内側)のインク画素数
    ink_ratio: float
    blank: bool
    overflow: bool  # 帰属するインクがセル本体の外へ出ている
    flags: set[CellFlag] = field(default_factory=set)


def _crop(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, fill) -> np.ndarray:
    h, w = img.shape[:2]
    out_shape = (y1 - y0, x1 - x0, *img.shape[2:])
    out = np.full(out_shape, fill, dtype=img.dtype)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def extract_region(
    rect_color: np.ndarray,
    pink: PageInk,
    rect: tuple[int, int, int, int],
    cell_index: int | None,
    pad: tuple[int, int],
    inner: tuple[int, int, int, int],
    blank_px: float,
) -> CellCrop:
    """正準ページ上の矩形 `rect` を切り出す。`cell_index` は帰属判定用のセル id(なければ None)。"""
    x0, y0, x1, y1 = rect
    px, py = pad
    win = (x0 - px, y0 - py, x1 + px, y1 + py)
    color = _crop(rect_color, *win, 255)
    ink_win = _crop(pink.ink, *win, False)
    lab_win = _crop(pink.labels, *win, 0)
    cid_win = _crop(pink.cell_id, *win, -1)
    if cell_index is None:
        mine = ink_win
    else:
        own = pink.owner[lab_win]
        mine = ink_win & ((own == cell_index) | ((own == -1) & (cid_win == cell_index)))
    binary = np.where(mine, 0, 255).astype(np.uint8)
    # セル本体のインク量(罫線の内側 inner = (左, 上, 右, 下) px を除く)
    ix0, iy0 = px + inner[0], py + inner[1]
    ix1, iy1 = px + (x1 - x0) - inner[2], py + (y1 - y0) - inner[3]
    body = mine[iy0:iy1, ix0:ix1]
    ink_px = int(body.sum())
    area = max(1, body.size)
    thr_px = blank_px
    primary_blank = ink_px < thr_px
    faint_px = 0
    if pink.faint is not None:
        faint_win = _crop(pink.faint, *win, False)
        faint_px = int(faint_win[iy0:iy1, ix0:ix1].sum())
    blank = primary_blank and faint_px < BLANK_FAINT_PX * blank_px / BLANK_INK_PX
    faint = primary_blank and not blank
    # はみ出し: 帰属するインクがセル本体(罫線を含む外形)の外側にある
    outside = mine.copy()
    outside[py : py + (y1 - y0), px : px + (x1 - x0)] = False
    overflow = bool(outside.sum() > 0.5 * thr_px) and not blank
    flags: set[CellFlag] = set()
    if blank:
        flags.add(CellFlag.BLANK)
    if overflow:
        flags.add(CellFlag.OVERFLOW)
    if faint:
        flags.add(CellFlag.NEEDS_REVIEW)  # 薄い記入の可能性(空欄と断定しない)
    return CellCrop(
        color=color,
        binary=binary,
        window=win,
        cell_rect=rect,
        ink_px=ink_px,
        ink_ratio=ink_px / area,
        blank=blank,
        overflow=overflow,
        flags=flags,
    )
