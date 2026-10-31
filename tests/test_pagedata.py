"""認識の入力(PageInput)の保存・読み込み。許可リスト形式(pickle なし)で、形が違うファイルは拒否する。"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")  # PageInput が属するモジュールが torch を読み込む

from yayoi_ocr.recognize.page import COLUMNS, CellInput, PageInput
from yayoi_ocr.recognize.pagedata import load_page_input, save_page_input


def make_page(seed=0) -> PageInput:
    rng = np.random.default_rng(seed)
    cells = []
    for i in range(116):
        r, c = divmod(i, 4)
        has = i % 3 != 0
        cells.append(
            CellInput(
                row=r,
                col=c,
                column=COLUMNS[c],
                rec=rng.integers(0, 255, (24 + i % 5, 40 + i), dtype=np.uint8) if has else None,
                clf=rng.integers(0, 255, (64, 448), dtype=np.uint8),
                stats=rng.random(9).astype(np.float32),
                flags=["ditto_likely"] if i % 7 == 0 else [],
                blank=not has,
                faint_only=i % 11 == 0,
                ink_h_frac=0.25,
            )
        )
    return PageInput(
        cells,
        rng.integers(0, 255, (20, 80), dtype=np.uint8),
        None,
        "needs_review",
        ["横罫線が少ない"],
    )


def test_roundtrip_preserves_every_cell_and_the_previous_year(tmp_path):
    page = make_page()
    f = tmp_path / "sub" / "7.npz"
    save_page_input(f, page, 8)
    assert not list(tmp_path.rglob("*.tmp"))  # 一時ファイルは残さない
    back, prev = load_page_input(f)
    assert (
        prev == 8
        and back.layout_status == "needs_review"
        and back.layout_notes == ["横罫線が少ない"]
    )
    assert back.number_rec is None and np.array_equal(back.title_rec, page.title_rec)
    for a, b in zip(page.cells, back.cells, strict=True):
        assert (a.row, a.col, a.column, a.flags, a.blank, a.faint_only) == (
            b.row,
            b.col,
            b.column,
            b.flags,
            b.blank,
            b.faint_only,
        )
        assert (a.rec is None) == (b.rec is None)
        if a.rec is not None:
            assert np.array_equal(a.rec, b.rec)
        assert np.array_equal(a.clf, b.clf) and np.allclose(a.stats, b.stats)
    save_page_input(f, make_page(1), None)  # 上書きできる(再認識)
    assert load_page_input(f)[1] is None


def test_wrong_shape_or_format_is_rejected(tmp_path):
    f = tmp_path / "a.npz"
    np.savez(
        f,
        meta=np.array('{"format": 99, "cells": []}'),
        clf=np.zeros((0, 64, 448), np.uint8),
        stats=np.zeros((0, 9)),
    )
    with pytest.raises(ValueError):
        load_page_input(f)
    g = tmp_path / "b.npz"
    np.savez(
        g,
        meta=np.array('{"format": 1, "cells": [], "layout_status": "ok", "layout_notes": []}'),
        clf=np.zeros((0, 64, 448), np.uint8),
        stats=np.zeros((0, 9)),
    )
    with pytest.raises(ValueError):
        load_page_input(g)
    h = tmp_path / "pickle.npz"
    np.savez(h, meta=np.array([object()], dtype=object))
    with pytest.raises((ValueError, KeyError)):
        load_page_input(h)  # allow_pickle=False なので pickle は読まない
