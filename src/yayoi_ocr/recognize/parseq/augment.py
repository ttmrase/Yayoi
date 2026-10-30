"""認識画像(グレー、紙 = 255)の拡張。ファインチューニングで強めにかけ、書き手・紙質・撮影の違いに強くする。

回転・せん断・拡大縮小・位置のずれ(外接矩形の余白のばらつき)・線の太さ・濃さ・ぼけ・低解像度・雑音。
"""

from __future__ import annotations

import cv2
import numpy as np


def augment_gray(img: np.ndarray, rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    """img: (H, W) uint8(紙 = 255, インク = 暗い)。返り値も同じ向き・おおよそ同じ大きさの uint8。"""
    h, w = img.shape
    s = strength
    out = img
    # 幾何: 回転・せん断・拡大縮小(中心まわり)。はみ出しは紙の色で埋める
    ang = np.deg2rad(rng.normal(0, 2.0 * s))
    shear = rng.normal(0, 0.10 * s)
    sx = float(np.exp(rng.normal(0, 0.10 * s)))
    sy = float(np.exp(rng.normal(0, 0.10 * s)))
    c, sn = np.cos(ang), np.sin(ang)
    a = np.array([[sx * c, -sy * sn + shear, 0.0], [sx * sn, sy * c, 0.0]], dtype=np.float64)
    cx, cy = w / 2.0, h / 2.0
    a[:, 2] = np.array([cx, cy]) - a[:, :2] @ np.array([cx, cy])
    out = cv2.warpAffine(
        out, a, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )
    # 外接矩形の余白のばらつき(左右・上下の切り詰めと追加)
    mx = round(abs(rng.normal(0, 0.04 * s)) * w)
    my = round(abs(rng.normal(0, 0.06 * s)) * h)
    if rng.random() < 0.5 and mx > 0:
        out = out[:, min(mx, w // 4) :]
    if rng.random() < 0.5 and my > 0:
        out = out[min(my, h // 6) :, :]
    pad_l, pad_r = (
        int(rng.integers(0, 2 + int(0.06 * s * w))),
        int(rng.integers(0, 2 + int(0.06 * s * w))),
    )
    out = cv2.copyMakeBorder(out, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT, value=255)
    # 線の太さ(細く・太く)
    r = rng.random()
    if r < 0.2 * s:
        out = cv2.erode(out, np.ones((2, 2), np.uint8))  # 暗い部分を広げる = 太く
    elif r < 0.35 * s:
        out = cv2.dilate(out, np.ones((2, 2), np.uint8))  # 暗い部分を縮める = 細く
    # 濃さ(筆圧)とコントラスト
    gain = float(rng.uniform(1.0 - 0.4 * min(s, 1.5), 1.1))
    f = 255.0 - (255.0 - out.astype(np.float32)) * gain
    # 低解像度(縮小 → 拡大)とぼけ
    if rng.random() < 0.35 * s:
        k = float(rng.uniform(0.45, 0.9))
        sm = cv2.resize(f, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        f = cv2.resize(sm, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.4 * s:
        f = cv2.GaussianBlur(f, (0, 0), float(rng.uniform(0.3, 1.1)))
    f = f + rng.normal(0, rng.uniform(0, 5.0 * s), f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)
