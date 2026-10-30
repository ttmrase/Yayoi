from __future__ import annotations

import pytest

from yayoi_ocr.models import ColumnKind
from yayoi_ocr.template import FormTemplate, load_template


def test_form_v1_structure():
    t = load_template()
    assert t.n_rows == 29
    assert [c.kind for c in t.columns] == [
        ColumnKind.DATE,
        ColumnKind.ACCOUNT,
        ColumnKind.DESCRIPTION,
        ColumnKind.AMOUNT,
    ]
    assert len(t.row_edges) == 30 and t.row_edges[0] == 0.0
    assert t.vline_x[0] == 0.0 and t.vline_x[-1] == 1.0
    assert t.band.y1 == 0.0 and t.band.y0 < 0
    assert t.title_box.y1 <= 0 and t.number_box.y1 <= 0
    # 表の縦横比は A4 の様式として妥当な範囲
    assert 1.25 < t.table_height < 1.5


def test_cell_box_matches_edges():
    t = load_template()
    b = t.cell_box(3, 2)
    assert (b.x0, b.x1) == (t.columns[2].x0, t.columns[2].x1)
    assert (b.y0, b.y1) == (t.row_edges[3], t.row_edges[4])


def test_invalid_templates_rejected():
    base = load_template().model_dump()
    bad = dict(base, row_edges=base["row_edges"][:-1])
    with pytest.raises(ValueError):
        FormTemplate.model_validate(bad)
    cols = [dict(c) for c in base["columns"]]
    cols[1]["x0"] += 0.05
    with pytest.raises(ValueError):
        FormTemplate.model_validate(dict(base, columns=cols))
