"""表の隅を手動で指定する再切り出し(確認画面の罫線補正)のテスト。フォント不要の部分と、合成帳票を使う部分。"""

from __future__ import annotations

import numpy as np
import pytest

from yayoi_ocr.layout.pipeline import analyze_image, corners_to_homography, overlay_info, view_jpeg
from yayoi_ocr.template import load_template

SHAPE = (1000, 700)  # (高さ, 幅)
GOOD = [(60, 120), (640, 120), (640, 900), (60, 900)]


def test_corners_make_a_homography_that_maps_the_unit_square_onto_them():
    tpl = load_template()
    H = corners_to_homography(GOOD, tpl, 1.0, SHAPE)
    th = tpl.table_height
    for (u, v), (x, y) in zip([(0, 0), (1, 0), (1, th), (0, th)], GOOD, strict=True):
        p = H @ np.array([u, v, 1.0])
        assert p[0] / p[2] == pytest.approx(x + 0.5 - 0.5, abs=1e-3)
        assert p[1] / p[2] == pytest.approx(y, abs=1e-3)


def test_work_scale_is_applied():
    tpl = load_template()
    H1 = corners_to_homography(GOOD, tpl, 1.0, SHAPE)
    H2 = corners_to_homography(GOOD, tpl, 0.5, SHAPE)
    p1 = H1 @ np.array([1.0, 0.0, 1.0])
    p2 = H2 @ np.array([1.0, 0.0, 1.0])
    assert (p2[0] / p2[2] + 0.5) == pytest.approx(0.5 * (p1[0] / p1[2] + 0.5), abs=1e-3)


@pytest.mark.parametrize(
    "bad",
    [
        [(60, 120), (640, 120), (640, 900)],  # 3 点
        [(60, 120), (640, 120), (60, 900), (640, 900)],  # 順序が不正(ねじれ)
        [(60, 120), (60, 900), (640, 900), (640, 120)],  # 逆回り
        [(300, 300), (330, 300), (330, 330), (300, 330)],  # 小さすぎる
        [(60, 120), (9000, 120), (640, 900), (60, 900)],  # 画像から離れすぎ
        [(60, 120), (640, float("nan")), (640, 900), (60, 900)],
    ],
)
def test_bad_corners_are_rejected(bad):
    with pytest.raises(ValueError):
        corners_to_homography(bad, load_template(), 1.0, SHAPE)


def test_flat_page_without_lines_still_returns_a_result_for_manual_corners():
    """罫線が全く写っていない画像でも、隅の指定があれば例外にならず、要確認で返る(隅の位置は尊重する)。"""
    img = np.full((1000, 700, 3), 235, np.uint8)
    res = analyze_image(img, corners=GOOD)
    assert len(res.page.rows) == 29 and all(len(r.cells) == 4 for r in res.page.rows)
    assert res.page.status.value == "needs_review"
    info = overlay_info(res)
    assert info["width"] == 700 and info["height"] == 1000
    for got, want in zip(info["corners"], GOOD, strict=True):
        assert got == pytest.approx(list(want), abs=2.0)
    assert len(info["hlines"]) == 30 and len(info["vlines"]) == 5


def test_view_jpeg_scales_to_the_long_side():
    img = np.full((1000, 700, 3), 200, np.uint8)
    res = analyze_image(img, corners=GOOD)
    data, f = view_jpeg(res, long_side=500)
    assert f == pytest.approx(0.5) and data[:2] == b"\xff\xd8"
    data2, f2 = view_jpeg(res, long_side=5000)
    assert f2 == 1.0 and data2[:2] == b"\xff\xd8"
