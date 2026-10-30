from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from yayoi_ocr.recognize.learning import ConfirmedCell, confirm_page
from yayoi_ocr.recognize.memory import RecognitionMemory
from yayoi_ocr.recognize.page import CellResult, JobContext, PageResult
from yayoi_ocr.recognize.parseq.engine import ParseqResult, RecognizeOutput


def _cell(row: int, col: int, column: str, emb: np.ndarray | None) -> CellResult:
    r = CellResult(row, col, column, "text")
    if emb is not None:
        r.out = RecognizeOutput([ParseqResult("x", -1.0, [], 0.9, "greedy", ())], 256, emb)
    return r


def test_confirm_page_updates_memory_dictionary_vocab_and_year():
    emb = np.ones(8, np.float32)
    results = [
        _cell(0, 0, "date", emb),
        _cell(0, 1, "account", emb),
        _cell(0, 2, "description", emb),
        _cell(0, 3, "amount", emb),
        _cell(1, 2, "description", None),  # 〃(R1 を走らせていない)
    ]
    truth = [
        ConfirmedCell(0, 0, "date", "text", "R8 8.21", "R8.8.21"),
        ConfirmedCell(0, 1, "account", "text", "現金", "現金"),
        ConfirmedCell(0, 2, "description", "text", "切手代", "切手代"),
        ConfirmedCell(0, 3, "amount", "text", "500-", "500"),
        ConfirmedCell(1, 2, "description", "ditto", "〃", "切手代"),
    ]
    ctx = JobContext(accounts=["普通預金"], memory=RecognitionMemory())
    n = confirm_page(ctx, PageResult(results), truth)
    assert n == 4 and len(ctx.memory) == 4  # 〃は埋め込みが無く、text でもないので入らない
    assert ctx.dictionary == {"切手代": 2}  # 〃も展開後の値で数える
    assert ctx.accounts == ["普通預金", "現金"] and ctx.prev_year == 8
    assert ctx.memory.query("amount", emb)[0].text == "500-"


def test_confirm_page_without_memory_still_updates_dictionary():
    ctx = JobContext()
    results = [_cell(0, 2, "description", None)]
    confirm_page(
        ctx, PageResult(results), [ConfirmedCell(0, 2, "description", "text", "会議費", "会議費")]
    )
    assert ctx.dictionary == {"会議費": 1}


def test_confirm_page_updates_amount_history_by_description_and_account():
    ctx = JobContext()
    cells = [
        ConfirmedCell(0, 0, "date", "text", "8.21", "R8.8.21"),
        ConfirmedCell(0, 1, "account", "text", "現金", "現金"),
        ConfirmedCell(0, 2, "description", "text", "電話代 1月分", "電話代 1月分"),
        ConfirmedCell(0, 3, "amount", "text", "5,800-", "5800"),
    ]
    results = [_cell(0, c.col, c.column, None) for c in cells]
    confirm_page(
        ctx,
        PageResult(results),
        cells + [ConfirmedCell(1, i, "x", "blank", "", "") for i in range(4)][:0],
    )
    confirm_page(ctx, PageResult(results), cells)
    assert ctx.amount_history["d:電話代1月分"] == {5800: 2}  # 摘要は NFKC・空白除去で正規化
    assert ctx.amount_history["a:現金"] == {5800: 2}
