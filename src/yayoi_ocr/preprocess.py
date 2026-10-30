"""前処理: 照明補正(背景を推定して割る)・コントラスト正規化・ノイズ除去。

画像の解像度に依存しないよう、内部処理では一定の作業解像度へ正規化する。
"""

from __future__ import annotations

import cv2
import numpy as np

WORK_LONG_SIDE = 1400  # 検出処理を行う作業解像度(長辺 px)


def to_gray(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def resize_to_long_side(
    img: np.ndarray, long_side: int = WORK_LONG_SIDE
) -> tuple[np.ndarray, float]:
    """長辺を `long_side` に揃える。戻り値は (画像, 倍率)。"""
    h, w = img.shape[:2]
    s = long_side / float(max(h, w))
    if abs(s - 1.0) < 1e-6:
        return img, 1.0
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=interp), s


def estimate_background(gray: np.ndarray, kernel_frac: float = 0.06) -> np.ndarray:
    """紙面の明るさ(照明むら・影)を推定する。ペン・罫線・見出し帯を除いた「紙」の明るさ。

    縮小 → 大きなカーネルで膨張(暗いものを消す)→ 収縮 → ぼかし → 拡大。
    `kernel_frac` は長辺に対する比率で、見出し帯の太さより大きくなるよう選ぶ。
    """
    h, w = gray.shape[:2]
    small_scale = 256.0 / max(h, w)
    small = cv2.resize(gray, (max(8, round(w * small_scale)), max(8, round(h * small_scale))),
                       interpolation=cv2.INTER_AREA)  # fmt: skip
    k = max(3, round(max(small.shape) * kernel_frac) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, se)
    bg = cv2.medianBlur(bg, 5)
    bg = cv2.GaussianBlur(bg, (0, 0), max(1.0, k / 3.0))
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(bg, 1, 255).astype(np.uint8)


def flatten_illumination(gray: np.ndarray, kernel_frac: float = 0.06) -> np.ndarray:
    """背景を推定して割る。紙は 255 付近・インクは暗いまま、照明むらが消える。"""
    bg = estimate_background(gray, kernel_frac).astype(np.float32)
    out = gray.astype(np.float32) / bg * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def normalize_contrast(flat: np.ndarray, low_pct: float = 0.5) -> np.ndarray:
    """紙=255 を保ったまま、最も暗い部分(濃いインク)が 0 付近になるよう線形に伸ばす。"""
    lo = float(np.percentile(flat, low_pct))
    hi = 255.0
    if hi - lo < 20:  # ほぼ白紙・ほぼ一様: 伸ばさない(ノイズ増幅防止)
        return flat
    out = (flat.astype(np.float32) - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def denoise(gray: np.ndarray, strength: float = 4.0) -> np.ndarray:
    """JPEG ノイズ等の除去(細い線を潰さない非局所平均)。"""
    if strength <= 0:
        return gray
    return cv2.fastNlMeansDenoising(
        gray, None, h=strength, templateWindowSize=5, searchWindowSize=15
    )


def preprocess_for_detection(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """検出用の作業画像を作る。戻り値は (照明補正・正規化済みグレー, 元画像→作業画像の倍率)。"""
    gray = to_gray(bgr)
    work, scale = resize_to_long_side(gray)
    flat = flatten_illumination(work)
    flat = normalize_contrast(flat)
    return flat, scale
