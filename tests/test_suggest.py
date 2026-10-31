"""提案モード: 親と提案モードの世代の合わせ方(要確認 = 親 OR 提案 OR 値が違う)と、安全性(見逃しは親の見逃しの部分集合)、対象の判定(符号検定)。torch なし。"""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from yayoi_ocr.app import suggest


def cell(value="100", review=False, kind="text", reasons=None, candidates=None, conf=0.9):
    return SimpleNamespace(
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
