"""提案モード: 親と提案モードの世代の合わせ方(要確認 = 親 OR 提案 OR 値が違う)と、安全性(見逃しは親の見逃しの部分集合)、対象の判定(符号検定)。torch なし。"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from yayoi_ocr.app import suggest


def cell(
    value="100",
    review=False,
    kind="text",
    reasons=None,
    candidates=None,
    conf=0.9,
    column="description",
):
    return SimpleNamespace(
        column=column,
        kind=kind,
        value=value,
        review=review,
        reasons=list(reasons or []),
        candidates=list(candidates or []),
        confidence=conf,
    )


def test_the_merged_review_flag_is_parent_or_suggestion_or_a_different_value():
    parent = [cell("1"), cell("2"), cell("3", review=True), cell("4"), cell("5", kind="text")]
    sugg = [
        cell("1"),  # 同じ値、どちらも要確認でない → 要確認でない
        cell("2", review=True),  # 提案モードだけが要確認 → 要確認
        cell("3"),  # 親だけが要確認 → 要確認
        cell("9"),  # 値が違う → 要確認(親の値を候補に入れる)
        cell("", kind="blank"),  # 種別が違う → 要確認
    ]
    n = suggest.merge(parent, sugg)
    assert [c.review for c in sugg] == [False, True, True, True, True]
    assert n == 2  # 値(または種別)が違うセルの数
    assert suggest.R_DISAGREE in sugg[3].reasons and suggest.R_DISAGREE not in sugg[0].reasons
    assert sugg[3].candidates[0] == ("4", 0.9)  # 親の値を先頭の候補に(利用者が戻せる)
    assert [c.value for c in sugg] == ["1", "2", "3", "9", ""]  # 入力済みの値は提案モードのもの
    # 同じ親の値がすでに候補にあれば重複させない
    p2, s2 = [cell("4")], [cell("9", candidates=[("4", 0.2)])]
    suggest.merge(p2, s2)
    assert s2[0].candidates == [("4", 0.2)]


def test_unflagged_cells_after_merging_are_a_subset_of_the_parents_and_have_the_parents_value():
    """合成の乱数で、性質を確かめる: 提案モードで要確認でないセル ⊆ 親の要確認でないセル、かつ値が一致。したがって、見逃しの集合は親の見逃しの部分集合。"""
    rng = random.Random(0)
    for _ in range(200):
        n = 116
        truth = [str(rng.randrange(1000)) for _ in range(n)]

        def make(acc, rev, rng=rng, truth=truth, n=n):
            return [
                cell(
                    truth[i] if rng.random() < acc else str(rng.randrange(1000)),
                    review=rng.random() < rev,
                )
                for i in range(n)
            ]

        parent, sugg = make(0.6, 0.5), make(0.8, 0.4)
        snap_parent = [(c.value, c.review) for c in parent]
        suggest.merge(parent, sugg)
        assert [(c.value, c.review) for c in parent] == snap_parent  # 親は書き換えない
        unflagged = [i for i, c in enumerate(sugg) if not c.review]
        assert all(not parent[i].review and sugg[i].value == parent[i].value for i in unflagged)
        miss_parent = {i for i, c in enumerate(parent) if c.value != truth[i] and not c.review}
        miss_sugg = {i for i, c in enumerate(sugg) if c.value != truth[i] and not c.review}
        assert miss_sugg <= miss_parent  # 安全性は親以上
        # 要確認のセルの数は増えるが(OR)、入力済みの値の正しさは提案モードのもの
        assert sum(c.review for c in sugg) >= sum(c.review for c in parent)


def test_sign_test_one_sided_p_values():
    assert suggest.sign_test_p(10, 2) == pytest.approx((66 + 12 + 1) / 4096)
    assert suggest.sign_test_p(5, 5) == 1.0 and suggest.sign_test_p(2, 8) == 1.0
    assert suggest.sign_test_p(0, 0) == 1.0
    assert suggest.sign_test_p(8, 0) == pytest.approx(1 / 256)
    assert 0 < suggest.sign_test_p(6, 2) < 1


def test_only_safety_side_rejections_with_a_significantly_better_accuracy_become_suggestions():
    for reason in ("amount_miss", "miss_rate_worse", "overconfident", "calibration_worse"):
        assert suggest.eligible(reason, 20, 2)
        assert not suggest.eligible(reason, 6, 2)  # 有意でない(p ≈ 0.145)
        assert not suggest.eligible(reason, 2, 20)
    for reason in ("no_gain", "accuracy_dropped", "review_rate_increased", "eval_too_small"):
        assert not suggest.eligible(reason, 50, 0)  # 安全側以外の理由は対象外


def test_a_minimum_number_of_differing_cells_is_required_even_if_the_p_value_is_small():
    # 5 対 0 は p = 0.031 < 0.05 だが、標本が少なすぎる(偶然で提案モードの世代が出る)
    assert suggest.sign_test_p(5, 0) < suggest.ALPHA
    assert not suggest.eligible("miss_rate_worse", 5, 0)
    assert not suggest.eligible("miss_rate_worse", suggest.MIN_DIFFER - 1, 0)
    assert suggest.eligible("miss_rate_worse", suggest.MIN_DIFFER, 0)
    assert suggest.eligible("amount_miss", 15, 5)  # 20 件ちょうど、p = 0.0207


def test_amount_cells_that_differ_need_an_explicit_choice_with_both_values_as_candidates():
    parent = [
        cell("1000", column="amount"),
        cell("2000", column="amount"),
        cell("abc", column="description"),
        cell("", kind="blank", column="amount"),
    ]
    sugg = [
        cell("1500", column="amount", candidates=[("1700", 0.1)]),  # 違う
        cell("2000", column="amount"),  # 同じ
        cell("abd", column="description"),  # 金額以外は通常の不一致(明示の選択は要らない)
        cell("300", column="amount", conf=0.7),  # 親は空欄
    ]
    suggest.merge(parent, sugg)
    assert suggest.R_AMOUNT_CHOICE in sugg[0].reasons and suggest.R_DISAGREE in sugg[0].reasons
    assert [v for v, _ in sugg[0].candidates] == ["1500", "1000", "1700"]  # [提案, 親, 他]
    assert suggest.R_AMOUNT_CHOICE not in sugg[1].reasons and not sugg[1].review
    assert suggest.R_AMOUNT_CHOICE not in sugg[2].reasons and sugg[2].review
    assert [v for v, _ in sugg[2].candidates] == ["abc"]
    assert suggest.R_AMOUNT_CHOICE in sugg[3].reasons  # 親が空欄で提案が値(種別も違う)
    assert [v for v, _ in sugg[3].candidates] == ["300"]


def test_combine_takes_the_screen_result_from_the_suggestion_but_last_year_from_the_parent():
    parent = SimpleNamespace(cells=[cell("1")], last_year=7)
    sugg = SimpleNamespace(cells=[cell("2")], last_year=8)
    result, last_year, parent_cells = suggest.combine(parent, sugg)
    assert result is sugg and last_year == 7 and parent_cells is parent.cells
    assert sugg.cells[0].review and suggest.R_DISAGREE in sugg.cells[0].reasons
