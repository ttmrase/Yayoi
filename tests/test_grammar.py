from __future__ import annotations

import pytest

from yayoi_ocr.recognize.grammar import (
    DateValue,
    amount_surfaces,
    date_surfaces,
    is_valid_date,
    norm_text,
    parse_amount,
    parse_date,
    repair_amount,
    repair_date,
)


def test_parse_date_grammar():
    assert parse_date("R7 11.19") == DateValue(7, 11, 19)
    assert parse_date("R711.19") == DateValue(7, 11, 19)  # スペースは省略可
    assert parse_date("R7.11.19") == DateValue(7, 11, 19)
    assert parse_date("3.5") == DateValue(None, 3, 5)
    assert parse_date("12.31") == DateValue(None, 12, 31)
    for bad in ("13.1", "2.30", "0.5", "5.", ".5", "R7", "5.17.", "R99 1.1", "abc"):
        assert parse_date(bad) is None, bad


def test_calendar_validity():
    assert is_valid_date(6, 2, 29)  # 令和 6 年 = 2024 年は閏年
    assert not is_valid_date(7, 2, 29)  # 2025 年は平年
    assert is_valid_date(None, 2, 29)  # 年不明は閏年として許す
    assert not is_valid_date(7, 4, 31)


def test_date_format_and_surfaces():
    d = DateValue(7, 1, 5)
    assert d.format() == "R7.1.5" and d.western_year() == 2025
    assert DateValue(None, 11, 19).format() == "11.19"
    assert date_surfaces(d) == ["R7 1.5", "R71.5", "R7.1.5"]
    assert date_surfaces(DateValue(None, 3, 4)) == ["3.4"]
    assert d.with_year(8) == DateValue(8, 1, 5)


def test_repair_date_missing_dot_and_ambiguity():
    assert DateValue(None, 5, 17) in repair_date("517")  # ピリオド欠落
    assert DateValue(None, 5, 17) in repair_date("5.17")
    c = repair_date("111")  # 1.11 と 11.1 のどちらもありうる
    assert DateValue(None, 1, 11) in c and DateValue(None, 11, 1) in c
    c = repair_date("R6 55")
    assert DateValue(6, 5, 5) in c
    c = repair_date("R711.19")
    assert DateValue(7, 11, 19) in c
    assert repair_date("５．１７") == [DateValue(None, 5, 17)]  # 全角
    assert repair_date("R") == [] and repair_date("..") == [] and repair_date("") == []
    assert all(is_valid_date(v.era_year, v.month, v.day) for v in repair_date("R1234"))


def test_parse_amount_grammar():
    assert parse_amount("3,456-") == 3456
    assert parse_amount("3456") == 3456
    assert parse_amount("1,000,000-") == 1_000_000
    assert parse_amount("500-") == 500
    for bad in ("12,34-", "0", "0-", "-", "", ",500", "1,2345", "5,0", "abc", "1234567890"):
        assert parse_amount(bad) is None, bad


def test_amount_surfaces():
    assert amount_surfaces(3456) == ["3,456-", "3456-", "3,456", "3456"]
    assert amount_surfaces(500) == ["500-", "500"]  # 重複は除く


def test_repair_amount_strips_stray_marks():
    assert repair_amount("-1,280-") == [1280]  # 先頭の余計な線
    assert repair_amount(",500-") == [500]
    assert repair_amount("1,280-,") == [1280]
    assert repair_amount("12,34-") == [1234]  # カンマ位置が崩れていても数字は読めている
    assert repair_amount("-") == [] and repair_amount("") == []
    assert repair_amount("2100-") == [2100]


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("電話代 8月分", "電話代8月分"),
        ("（株）テスト", "(株)テスト"),
        ("ＡＢＣ）ＳＡＭＰＬＥ", "ABC)SAMPLE"),
    ],
)
def test_norm_text(a, b):
    assert norm_text(a) == norm_text(b)
