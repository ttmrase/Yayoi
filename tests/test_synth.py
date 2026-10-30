from __future__ import annotations

import numpy as np
import pytest

from yayoi_ocr.groundtruth import GTPage
from yayoi_ocr.synth.content import ARROW, DITTO, ELIDE, plan_page
from yayoi_ocr.synth.generator import SynthConfig, generate_sample
from yayoi_ocr.synth.writer import make_writer

pytestmark = pytest.mark.needs_font


def test_generator_is_deterministic():
    a = generate_sample(SynthConfig(seed=5, dpi=100))
    b = generate_sample(SynthConfig(seed=5, dpi=100))
    assert np.array_equal(a.image, b.image)
    assert a.texts == b.texts


def test_cell_polygons_and_texts():
    s = generate_sample(SynthConfig(seed=1, dpi=100, degrade=False))
    assert len(s.cell_polys) == 29 * 4
    assert s.visible_fraction > 0.99
    # 行は上から下へ、列は左から右へ並ぶ
    c00, c01, c10 = s.cell_polys[(0, 0)], s.cell_polys[(0, 1)], s.cell_polys[(1, 0)]
    assert c01[0, 0] > c00[0, 0] and c10[0, 1] > c00[0, 1]
    assert any(t for t in s.texts.values())


def test_rotate90_changes_shape_and_keeps_polygons_inside():
    s = generate_sample(SynthConfig(seed=2, dpi=100, degrade=False, rotate90=90))
    h, w = s.image.shape[:2]
    p = s.cell_polys[(0, 0)]
    assert (p[:, 0] >= -1).all() and (p[:, 0] <= w + 1).all() and (p[:, 1] <= h + 1).all()


def test_writer_is_consistent_and_distinct():
    a1, a2, b = make_writer(7), make_writer(7), make_writer(8)
    assert a1 == a2  # 同じ ID は同じ癖
    assert a1 != b
    # 同じ書き手なら、別ページでも同じパラメータ(同じ癖)で書く
    s1 = generate_sample(SynthConfig(seed=1, writer_id=7, dpi=100, degrade=False))
    s2 = generate_sample(SynthConfig(seed=2, writer_id=7, dpi=100, degrade=False))
    assert s1.writer == s2.writer and s1.plan.title != "" and s1.texts != s2.texts


def test_ground_truth_is_consistent_with_marks():
    seen = {"ditto": 0, "arrow": 0, "elide": 0}
    for seed in range(40):
        plan = plan_page(np.random.default_rng(seed))
        gt = plan.to_ground_truth("x.png")
        assert isinstance(gt, GTPage) and len(gt.rows) == 29
        prev = {"date": "", "account": "", "description": ""}
        for row in gt.rows:
            for col in ("date", "account", "description"):
                f = getattr(row, col)
                if f.raw == DITTO:
                    seen["ditto"] += 1
                    assert f.resolved == prev[col] != ""  # 〃は直前の resolved に展開される
                if f.raw == ARROW:
                    seen["arrow"] += 1
                    assert f.resolved not in ("", ARROW)
                if ELIDE in f.raw:
                    seen["elide"] += 1
                    head, tail = f.raw.split(ELIDE)
                    assert f.resolved.startswith(head) and f.resolved.endswith(tail)
                if f.resolved:
                    prev[col] = f.resolved
            a = row.amount
            assert a.resolved == "" or a.resolved.isdigit()
            if a.raw:
                assert a.raw.replace(",", "").rstrip("-") == a.resolved
            if row.date.resolved:
                assert row.date.resolved.startswith("R") and row.date.resolved.count(".") == 2
    assert all(v > 0 for v in seen.values()), seen


def test_amounts_are_random_not_a_fixed_list():
    amounts = {
        c.resolved
        for seed in range(30)
        for row in plan_page(np.random.default_rng(seed)).rows
        for c in [row["amount"]]
        if c.resolved
    }
    assert len(amounts) > 100  # 固定の候補リストではなく乱数で作っている
