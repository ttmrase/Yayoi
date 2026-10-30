"""合成帳票によるレイアウト検出のテスト(画像はテスト中に合成。コミットしない)。"""

from __future__ import annotations

import numpy as np
import pytest

from yayoi_ocr.layout.metrics import polygon_iou
from yayoi_ocr.layout.pipeline import analyze_image
from yayoi_ocr.models import CellFlag, ColumnKind, PageStatus
from yayoi_ocr.synth.generator import SynthConfig, generate_sample

pytestmark = pytest.mark.needs_font


def _mean_iou(res, s) -> float:
    return float(
        np.mean(
            [
                polygon_iou(np.array(res.page.rows[r].cells[c].polygon), s.cell_polys[(r, c)])
                for r in range(29)
                for c in range(4)
            ]
        )
    )


def test_clean_page_is_accurate():
    s = generate_sample(SynthConfig(seed=11, dpi=150, degrade=False))
    res = analyze_image(s.image)
    assert res.page.status == PageStatus.OK
    assert len(res.page.rows) == 29 and all(len(r.cells) == 4 for r in res.page.rows)
    assert _mean_iou(res, s) > 0.93


@pytest.mark.parametrize("rot", [90, 180, 270])
def test_orientation_is_detected(rot):
    s = generate_sample(SynthConfig(seed=12, dpi=130, rotate90=rot))
    res = analyze_image(s.image)
    assert res.page.image.rotation_applied == (360 - rot) % 360
    assert _mean_iou(res, s) > 0.85


@pytest.mark.slow
def test_degraded_pages_meet_targets():
    ious, silent = [], 0
    n = 24
    for seed in range(100, 100 + n):
        s = generate_sample(SynthConfig(seed=seed))
        res = analyze_image(s.image)
        m = _mean_iou(res, s)
        ious.append(m)
        if res.page.status == PageStatus.OK and m < 0.8:
            silent += 1
    assert silent == 0
    assert float(np.mean(ious)) >= 0.9, ious


def test_blank_cells_follow_writing():
    s = generate_sample(SynthConfig(seed=21, dpi=200, degrade=False, fill_rows=17))
    res = analyze_image(s.image)
    wrong_blank = wrong_ink = total = 0
    for (r, c), text in s.texts.items():
        cell = res.page.rows[r].cells[c]
        is_blank = CellFlag.BLANK in cell.flags
        total += 1
        if text and is_blank:
            wrong_blank += 1  # 書いてあるのに空欄 = 見逃し(重大)
        if not text and not is_blank:
            wrong_ink += 1
    assert wrong_blank == 0
    assert wrong_ink <= 0.03 * total


def test_blank_image_returns_needs_review_without_error():
    res = analyze_image(np.full((600, 400, 3), 255, np.uint8))
    assert res.page.status == PageStatus.NEEDS_REVIEW
    assert len(res.page.rows) == 29
    assert all(CellFlag.LAYOUT_UNCERTAIN in c.flags for r in res.page.rows for c in r.cells)


def test_noise_image_returns_needs_review_without_error():
    rng = np.random.default_rng(0)
    res = analyze_image(rng.integers(0, 255, (500, 400, 3), dtype=np.uint8))
    assert res.page.status == PageStatus.NEEDS_REVIEW


def test_cell_crops_have_two_variants():
    s = generate_sample(SynthConfig(seed=31, dpi=150))
    res = analyze_image(s.image)
    crop = res.cell_crop(3, 2)
    assert crop.color.ndim == 3 and crop.binary.ndim == 2
    assert crop.color.shape[:2] == crop.binary.shape
    assert set(np.unique(crop.binary)) <= {0, 255}
    assert res.page.rows[3].cells[2].column == ColumnKind.DESCRIPTION
