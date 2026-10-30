"""レイアウト検出の評価指標(セル多角形の IoU)。"""

from __future__ import annotations

import cv2
import numpy as np


def polygon_iou(a: np.ndarray, b: np.ndarray, max_side: int = 600) -> float:
    """2 つの多角形(N,2)の IoU。両者を含む範囲を縮小してラスタライズして求める。"""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    pts = np.vstack([a, b])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = float(max((hi - lo).max(), 1.0))
    f = min(8.0, max_side / span)  # セルは細長いので、拡大してから描く(薄い差も拾う)
    size = np.ceil((hi - lo) * f).astype(int) + 3
    ma = np.zeros((size[1], size[0]), np.uint8)
    mb = np.zeros_like(ma)
    cv2.fillPoly(ma, [np.round((a - lo) * f + 1).astype(np.int32)], 1)
    cv2.fillPoly(mb, [np.round((b - lo) * f + 1).astype(np.int32)], 1)
    inter = int(np.logical_and(ma, mb).sum())
    union = int(np.logical_or(ma, mb).sum())
    return inter / union if union else 0.0
