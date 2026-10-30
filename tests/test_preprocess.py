from __future__ import annotations

import numpy as np

from yayoi_ocr.preprocess import (
    flatten_illumination,
    normalize_contrast,
    preprocess_for_detection,
    resize_to_long_side,
)


def test_flatten_removes_illumination_gradient():
    h, w = 400, 300
    paper = np.full((h, w), 230.0)
    paper[100:104, 50:250] = 60  # 細い暗い線
    grad = np.linspace(0.45, 1.0, w)[None, :]
    shaded = (paper * grad).astype(np.uint8)
    flat = flatten_illumination(shaded)
    left, right = flat[200:300, 5:40].mean(), flat[200:300, 260:295].mean()
    assert abs(left - right) < 12  # 照明むらが消える
    assert flat[100:104, 100:200].mean() < 0.5 * left  # 線は残る


def test_normalize_contrast_keeps_paper_white():
    img = np.full((50, 50), 255, np.uint8)
    img[10:12, 10:40] = 120
    out = normalize_contrast(img)
    assert out[0, 0] == 255 and out[10, 20] < 120


def test_resize_to_long_side():
    img = np.zeros((100, 50), np.uint8)
    out, s = resize_to_long_side(img, 200)
    assert out.shape == (200, 100) and abs(s - 2.0) < 1e-9


def test_preprocess_resolution_independent():
    rng = np.random.default_rng(0)
    small = (rng.random((300, 200, 3)) * 30 + 200).astype(np.uint8)
    flat, scale = preprocess_for_detection(small)
    assert max(flat.shape) == 1400 and scale > 1
