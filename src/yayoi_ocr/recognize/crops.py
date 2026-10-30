"""セル切り出し(layout の `CellCrop`)から、認識器への入力画像を作る。

- `recognition_image`: PARSeq(R1)用。罫線・他セルのインクを除いたグレー画像を、
  インクの外接矩形 + 余白で切る。縦横比は変えない(エンジン側で NDLOCR-Lite と同じ手順で
  (幅, 24) にリサイズする)。
- `classifier_input`: セル種別分類器用。罫線除去済みのセル全体(余白込み)を高さ 64 px にして、
  固定幅に左詰めで置く。
- 確認画面(UI)に出す切り出しは別: layout の `CellCrop.color` を正準解像度のまま使う。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..layout.cells import CellCrop

CLF_HEIGHT = 64
CLF_WIDTH = 448  # 摘要欄(最も広い)でも高さ 64 px で収まる幅(縦横比を保って左詰め)
# 低い記入(「-」「.」だけ等)でも、行高のこの割合までは縦に広げて文字の大きさをそろえる
MIN_TEXT_FRAC = 0.45
PAD_Y_FRAC = 0.22  # 外接矩形(行高で下限を掛けた高さ)に対する上下の余白
PAD_X_FRAC = 0.30  # 同・左右の余白(高さに対する比)


@dataclass
class RecognitionImage:
    image: np.ndarray  # BGR uint8(グレーを 3 チャネルにしたもの)
    bbox: tuple[int, int, int, int]  # CellCrop のウィンドウ内での切り出し範囲 (x0, y0, x1, y1)
    ink_height_frac: float  # インクの高さ / ウィンドウの高さ(はみ出し・注記の検出に使う)
    ink_width_frac: float  # インクの幅 / ウィンドウの幅


def ink_bbox(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    """二値(インク = 0)の外接矩形 (x0, y0, x1, y1)。インクが無ければ None。"""
    ys, xs = np.nonzero(binary == 0)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def recognition_image(crop: CellCrop, row_h_px: int) -> RecognitionImage | None:
    """PARSeq 用の画像。インクが無い(空欄・薄い記入だけ)セルは None。"""
    if crop.gray is None:
        raise ValueError("CellCrop.gray がありません(layout.cells.extract_region が作る)")
    bb = ink_bbox(crop.binary)
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    h, w = crop.gray.shape
    ink_h = y1 - y0
    ref_h = max(ink_h, round(MIN_TEXT_FRAC * row_h_px))
    cy = 0.5 * (y0 + y1)
    py = max(2, round(PAD_Y_FRAC * ref_h))
    px = max(2, round(PAD_X_FRAC * ref_h))
    ty0 = round(cy - ref_h / 2) - py
    ty1 = round(cy + ref_h / 2) + py
    tx0, tx1 = x0 - px, x1 + px
    # ウィンドウの外は紙(白)で埋める(隣のセルの内容を持ち込まない)
    out = np.full((ty1 - ty0, tx1 - tx0), 255, np.uint8)
    sx0, sy0, sx1, sy1 = max(0, tx0), max(0, ty0), min(w, tx1), min(h, ty1)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - ty0 : sy1 - ty0, sx0 - tx0 : sx1 - tx0] = crop.gray[sy0:sy1, sx0:sx1]
    img = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return RecognitionImage(img, (tx0, ty0, tx1, ty1), ink_h / h, (x1 - x0) / w)


def classifier_input(crop: CellCrop) -> np.ndarray:
    """分類器用: セル全体(罫線除去済み・余白込み)を高さ 64 px・幅 448 px(左詰め)のグレーにする。"""
    if crop.gray is None:
        raise ValueError("CellCrop.gray がありません")
    h, w = crop.gray.shape
    scale = CLF_HEIGHT / h
    nw = max(1, round(w * scale))
    small = cv2.resize(crop.gray, (nw, CLF_HEIGHT), interpolation=cv2.INTER_AREA)
    out = np.full((CLF_HEIGHT, CLF_WIDTH), 255, np.uint8)
    n = min(nw, CLF_WIDTH)
    out[:, :n] = small[:, :n]
    return out
