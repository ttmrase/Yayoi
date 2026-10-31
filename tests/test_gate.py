"""Phase 4 の採用ゲート・ページの分割(純粋関数。torch なし)。値(摘要・金額)は一切使わない。"""

from __future__ import annotations

import pytest

from yayoi_ocr.app import gate
from yayoi_ocr.app.gate import Decision, EvalCounts, decide, split_pages


def counts(n_cells=1000, n_text=600, correct=540, review=180, miss=5) -> EvalCounts:
    return EvalCounts(n_cells, n_text, correct, review, miss)


def test_adopts_when_accuracy_goes_up_without_worse_miss_or_review():
    d = decide(counts(), counts(correct=560, review=170, miss=4))
    assert d == Decision("adopt", gate.R_ADOPTED_ACCURACY) and d.adopted


def test_adopts_when_only_the_review_rate_goes_down():
    d = decide(counts(), counts(review=150))
    assert d == Decision("adopt", gate.R_ADOPTED_REVIEW)


@pytest.mark.parametrize(
    ("cand", "reason"),
    [
        (
            counts(correct=600, miss=6),
            gate.R_MISS_WORSE,
        ),  # 正解率が上がっても見逃しが増えたら不採用
        (
            counts(correct=530, miss=5),
            gate.R_ACCURACY_DROPPED,
        ),  # 見逃しが同じでも正解率が下がったら不採用
        (counts(), gate.R_NO_GAIN),  # まったく同じ = 得がない
        (counts(correct=541, review=180 + 12 + 1, miss=5), gate.R_REVIEW_INCREASED),  # +2pt 超
    ],
)
def test_rejects_with_a_reason_code(cand, reason):
    d = decide(counts(), cand)
    assert d.outcome == "reject" and d.reason == reason and not d.adopted


def test_review_increase_up_to_two_points_is_allowed():
    # 600 件の 2pt = 12 件までは増えてよい
    assert decide(counts(), counts(correct=541, review=192)).outcome == "adopt"


def test_miss_rule_is_checked_before_accuracy():
    d = decide(counts(), counts(correct=500, miss=9))
    assert d.reason == gate.R_MISS_WORSE


def test_defers_when_the_evaluation_is_too_small():
    small = counts(n_cells=290, n_text=149, correct=130, review=30, miss=1)
    better = counts(n_cells=290, n_text=149, correct=140, review=20, miss=0)
    d = decide(small, better)
    assert d == Decision("defer", gate.R_EVAL_TOO_SMALL)
    assert (
        decide(counts(n_text=150, correct=135), counts(n_text=150, correct=145)).outcome == "adopt"
    )


def test_rates_and_dict_roundtrip():
    c = counts()
    assert c.accuracy == pytest.approx(0.9) and c.review_rate == pytest.approx(0.3)
    assert c.miss_rate == pytest.approx(0.005)
    d = c.to_dict()
    assert EvalCounts.from_dict(d) == c
    assert all(not isinstance(v, str) for v in d.values())  # 件数と割合だけ
    assert EvalCounts(0, 0, 0, 0, 0).accuracy == 0.0


@pytest.mark.parametrize(
    ("n", "n_eval"),
    [(6, 2), (7, 2), (10, 3), (12, 3), (20, 5), (39, 10), (40, 10), (100, 10)],
)
def test_split_takes_the_newest_quarter_clamped_to_2_to_10_pages(n, n_eval):
    ids = list(range(100, 100 + n))
    train, ev = split_pages(ids)
    assert len(ev) == n_eval and train + ev == ids  # 時系列で、評価用が新しい側
    assert max(train) < min(ev)


def test_split_needs_at_least_six_pages():
    assert split_pages(list(range(5))) is None
    assert split_pages([]) is None
