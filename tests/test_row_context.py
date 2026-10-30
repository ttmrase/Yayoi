"""行の文脈による空欄判定(薄いインクだけのセルの扱い)。"""

from __future__ import annotations

import numpy as np

from yayoi_ocr.layout.cells import CellCrop
from yayoi_ocr.layout.pipeline import _apply_row_context
from yayoi_ocr.models import CellFlag
from yayoi_ocr.template import load_template


def _crop(*, blank: bool, faint_only: bool = False) -> CellCrop:
    flags = {CellFlag.BLANK} if blank else set()
    if faint_only:
        flags.add(CellFlag.NEEDS_REVIEW)
    z = np.zeros((4, 4), np.uint8)
    return CellCrop(
        color=z, binary=z, window=(0, 0, 4, 4), cell_rect=(0, 0, 4, 4), ink_px=0, ink_ratio=0.0,
        blank=blank, overflow=False, faint_only=faint_only, flags=flags,
    )  # fmt: skip


def _prep(crops):
    tpl = load_template()
    return [(col, None, None, c, True) for col, c in zip(tpl.columns, crops, strict=True)], tpl


def test_amount_blank_row_suppresses_faint_flags_in_other_columns():
    # 薄いインクだけの 3 セル + 空欄の金額欄 → 行全体が空欄(要確認なし)
    prep, tpl = _prep([_crop(blank=False, faint_only=True) for _ in range(3)] + [_crop(blank=True)])
    _apply_row_context(prep, tpl)
    for _, _, _, crop, _ in prep[:3]:
        assert crop.blank and CellFlag.BLANK in crop.flags
        assert CellFlag.NEEDS_REVIEW not in crop.flags and CellFlag.DITTO_LIKELY not in crop.flags


def test_filled_amount_row_picks_up_faint_ink_as_possible_ditto():
    prep, tpl = _prep(
        [_crop(blank=False, faint_only=True)] * 3 + [_crop(blank=False)]  # 金額欄に記入あり
    )
    _apply_row_context(prep, tpl)
    for _, _, _, crop, _ in prep[:3]:
        assert not crop.blank
        assert CellFlag.DITTO_LIKELY in crop.flags and CellFlag.NEEDS_REVIEW not in crop.flags


def test_amount_cell_keeps_high_sensitivity():
    # 金額欄が薄いインクだけ → 空欄とせず要確認のまま(行の文脈では変えない)
    prep, tpl = _prep([_crop(blank=True)] * 3 + [_crop(blank=False, faint_only=True)])
    _apply_row_context(prep, tpl)
    amount = prep[3][3]
    assert not amount.blank and CellFlag.NEEDS_REVIEW in amount.flags
    # 金額欄は記入あり扱いなので、他の列は(薄いインクが無ければ)変わらない
    assert all(p[3].blank for p in prep[:3])
