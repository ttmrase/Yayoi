"""NDLOCR-Lite の PARSeq 推論(src/parseq.py の `preprocess`)と同じ前処理。

出典: ndl-lab/ndlocr-lite(国立国会図書館, CC BY 4.0)の src/parseq.py。
- 縦書き行(高さ > 幅 x 0.8)は反時計回りに 90° 回す(`rotate_tall=True`)。
- 縦横比を無視して (幅, 高さ) へ cv2.INTER_LINEAR でリサイズ。
- BGR → RGB、/127.5 - 1.0、CHW、バッチ次元を追加。

帳票のセルは縦書きではないので、エンジンは `rotate_tall=False` で使う(短い 1 文字のセルを回転させない)。
NDLOCR-Lite と完全に同じ入力にしたいテスト(ONNX との一致)では `rotate_tall=True` を使う。
"""

from __future__ import annotations

import cv2
import numpy as np


def ndl_preprocess(
    img_bgr: np.ndarray, width: int, height: int = 24, rotate_tall: bool = False
) -> np.ndarray:
    """BGR(または 2 次元グレー)画像 → float32 (1, 3, height, width)。"""
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    h, w = img_bgr.shape[:2]
    if rotate_tall and h > w * 0.8:
        img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    resized = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    x = np.ascontiguousarray(resized[:, :, ::-1]).astype(np.float32)
    x /= 127.5
    x -= 1.0
    return x.transpose(2, 0, 1)[np.newaxis]
