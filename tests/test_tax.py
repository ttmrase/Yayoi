"""税区分の許可リストと税額の計算(app/tax.py)。仕様は docs/yayoi_export.md。"""

from __future__ import annotations

import pytest

from yayoi_ocr.app import tax

RATES = ["10%", "軽減8%", "8%", "5%"]
TAIL = ["非課仕入", "対外仕入", "対象外"]


def test_general_inclusive_is_list_a():
    got = tax.categories("taxable", "general", "inclusive")
    assert got == [f"{k}仕入込{r}" for k in ("課対", "非対", "共対") for r in RATES] + TAIL
    assert len(got) == 15


def test_general_exclusive_is_list_b():
    got = tax.categories("taxable", "general", "exclusive")
    want = [f"{k}仕入{c}{r}" for k in ("課対", "非対", "共対") for c in "内外別" for r in RATES]
    assert got == want + TAIL
    assert got[0] == "課対仕入内10%"  # 税抜経理の既定は内税の 10%


def test_simplified_lists_are_c_and_d():
    assert (
        tax.categories("taxable", "simplified", "inclusive")
        == [f"課対仕入込{r}" for r in RATES] + TAIL
    )
    assert (
        tax.categories("taxable", "simplified", "exclusive")
        == [f"課対仕入{c}{r}" for c in "内外別" for r in RATES] + TAIL
    )


def test_exempt_has_only_the_exempt_category():
    for scheme in ("general", "simplified"):
        for method in ("inclusive", "exclusive"):
            assert tax.categories("exempt", scheme, method) == ["対象外"]


def test_default_category_by_method():
    assert tax.default_category("taxable", "general", "inclusive") == "課対仕入込10%"
    assert tax.default_category("taxable", "general", "exclusive") == "課対仕入内10%"
    assert tax.default_category("exempt", "general", "exclusive") == "対象外"


def test_all_options_cover_the_four_combinations_and_use_half_width_percent():
    opts = tax.all_options()
    assert set(opts) == {
        "general_inclusive",
        "general_exclusive",
        "simplified_inclusive",
        "simplified_exclusive",
    }
    assert all("％" not in c for cs in opts.values() for c in cs)
    # 税率のない課税区分(「課対仕入」など)は選択肢にない。非課仕入・対外仕入・対象外だけが税率なしで正しい
    for cs in opts.values():
        for c in cs:
            assert tax.parse_category(c) is not None
            assert "%" in c or c in tax.NO_RATE


@pytest.mark.parametrize(
    ("category", "want"),
    [
        ("課対仕入込10%", ("課対", "込", 10)),
        ("共対仕入込軽減8%", ("共対", "込", 8)),
        ("非対仕入内5%", ("非対", "内", 5)),
        ("課対仕入外8%", ("課対", "外", 8)),
        ("課対仕入別軽減8%", ("課対", "別", 8)),
        ("対象外", ("", "", None)),
        ("非課仕入", ("", "", None)),
    ],
)
def test_parse_category(category, want):
    info = tax.parse_category(category)
    assert info is not None and (info.kind, info.calc, info.rate) == want


@pytest.mark.parametrize(
    "bad", ["課対仕入", "課対仕入込", "課対仕入込10％", "課対仕入込 10%", "売上", ""]
)
def test_parse_category_rejects_unknown_names(bad):
    assert tax.parse_category(bad) is None


def test_rate_less_taxable_names_are_recognized_as_the_5_percent_trap():
    assert tax.has_rate_but_wrong("課対仕入") and tax.has_rate_but_wrong("課対仕入込")
    assert not tax.has_rate_but_wrong("課対仕入込10%") and not tax.has_rate_but_wrong("対象外")


def test_tax_amount_floor_round_ceil():
    # 3,456 円・10%: 3456*10/110 = 314.18…
    assert tax.tax_amount(3456, "課対仕入内10%", "floor") == 314
    assert tax.tax_amount(3456, "課対仕入内10%", "round") == 314
    assert tax.tax_amount(3456, "課対仕入内10%", "ceil") == 315
    # 軽減 8%: 1080*8/108 = 80 ちょうど
    assert tax.tax_amount(1080, "課対仕入内軽減8%", "ceil") == 80
    # 四捨五入は 0.5 を切り上げ: 55*10/110 = 5.0、165*10/110 = 15.0、 0.5 になる例: 11*5/105=0.52…
    assert tax.tax_amount(11, "課対仕入内5%", "round") == 1
    assert tax.tax_amount(10, "課対仕入内5%", "round") == 0  # 0.476…


def test_tax_amount_is_only_for_inner_tax():
    for cat in (
        "課対仕入込10%",
        "課対仕入外10%",
        "課対仕入別10%",
        "非課仕入",
        "対外仕入",
        "対象外",
        "xx",
    ):
        assert tax.tax_amount(1000, cat) is None
    assert tax.is_manual_tax("課対仕入外10%") and tax.is_manual_tax("共対仕入別軽減8%")
    assert not tax.is_manual_tax("課対仕入内10%") and not tax.is_manual_tax("対象外")


def test_normalize_description_is_for_matching_only():
    assert tax.normalize_description("切手代") == tax.normalize_description(" 切手 代 ")
    assert tax.normalize_description("ＡＢＣ　ｓｈｏｐ") == "abcshop"  # 全角英数・全角空白
    assert tax.normalize_description("") == ""
