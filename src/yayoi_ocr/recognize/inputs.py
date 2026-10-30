"""認識への入力の作成: 合成データセット(PageData)・layout の結果(LayoutResult)→ PageInput。"""

from __future__ import annotations

import numpy as np

from ..layout.pipeline import LayoutResult
from ..models import CellFlag
from ..synth.dataset import COLUMNS as DS_COLUMNS
from ..synth.dataset import PageData
from .celltype import stats_vector
from .crops import classifier_input, recognition_image
from .page import CellInput, PageInput


def from_page_data(page: PageData) -> PageInput:
    cells = []
    for i, c in enumerate(page.cells):
        amount_blank = page.cell(c.row, 3).blank
        cells.append(
            CellInput(
                row=c.row,
                col=c.col,
                column=c.column,
                rec=page.rec.get(i),
                clf=page.clf[i],
                stats=stats_vector(
                    c.column,
                    c.ink_ratio,
                    c.flags,
                    c.faint_only,
                    c.ink_h_frac,
                    c.ink_w_frac,
                    amount_blank,
                ),
                flags=list(c.flags),
                blank=c.blank,
                faint_only=c.faint_only,
                ink_h_frac=c.ink_h_frac,
            )
        )
    return PageInput(
        cells,
        page.fields.get("title"),
        page.fields.get("number"),
        page.meta.get("status", "ok"),
        list(page.meta.get("notes", [])),
    )


def from_layout(res: LayoutResult) -> PageInput:
    """layout の結果から(実運用)。切り出し画像は layout が作ったものを使う。"""
    row_h = round(float(np.median(np.diff(res.tpl.row_edges))) * res.tpl.canon_width)
    cells = []
    for r in range(len(res.page.rows)):
        amount_blank = res.crops[(r, len(DS_COLUMNS) - 1)].blank
        for c, col in enumerate(DS_COLUMNS):
            crop = res.crops[(r, c)]
            lc = res.page.rows[r].cells[c]
            ri = recognition_image(crop, row_h)
            flags = sorted(f.value for f in lc.flags if isinstance(f, CellFlag))
            cells.append(
                CellInput(
                    row=r,
                    col=c,
                    column=col,
                    rec=None if ri is None else ri.image[:, :, 0],
                    clf=classifier_input(crop),
                    stats=stats_vector(
                        col,
                        crop.ink_ratio,
                        flags,
                        crop.faint_only,
                        0.0 if ri is None else ri.ink_height_frac,
                        0.0 if ri is None else ri.ink_width_frac,
                        amount_blank,
                    ),
                    flags=flags,
                    blank=crop.blank,
                    faint_only=crop.faint_only,
                    ink_h_frac=0.0 if ri is None else ri.ink_height_frac,
                )
            )
    title = number = None
    for name, fc in (("title", res.title_crop), ("number", res.number_crop)):
        if fc is not None and fc.gray is not None:
            fri = recognition_image(fc, round(fc.gray.shape[0] / 1.5))
            if fri is not None:
                if name == "title":
                    title = fri.image[:, :, 0]
                else:
                    number = fri.image[:, :, 0]
    return PageInput(cells, title, number, res.page.status.value, list(res.page.notes))
