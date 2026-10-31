"""Phase 4 の採用ゲート・ページの分割(純粋関数。torch なし)。値(摘要・金額)は一切使わない。"""

from __future__ import annotations

import math

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


def test_same_value_normalizes_only_text_columns():
    assert gate.same_value("description", "ＡＢＣ　旅費", "ABC旅費")
    assert not gate.same_value("amount", "1,000", "1000")
    assert not gate.same_value("date", "R7.1.1", "R07.1.1")


# ---- 感度の高い指標(金額 1 件で不採用・列ごと・期待見逃しとの比較・較正の質)


def calib(n=500, nll=0.40, expected=3.0, observed=2):
    return gate.Calib(n, nll, nll * 0.8, expected, observed)


def side(total=None, calib_=None, **by_column):
    """by_column: 列 -> (n_miss, observed_miss, expected_miss)。"""
    t = total or counts()
    by = {
        c: EvalCounts(n_cells=250, n_text=150, n_correct=135, n_review=40, n_miss=m)
        for c, (m, _o, _e) in by_column.items()
    }
    cb = {c: calib(150, 0.4, e, o) for c, (_m, o, e) in by_column.items()}
    return gate.EvalSide(t, calib_ or calib(), by, cb)


def test_poisson_upper_tail_matches_closed_forms():
    assert gate.poisson_sf(0, 0.3) == 1.0
    assert gate.poisson_sf(1, 0.3) == pytest.approx(1 - math.exp(-0.3))
    assert gate.poisson_sf(2, 0.5) == pytest.approx(1 - math.exp(-0.5) * 1.5)
    assert gate.poisson_sf(3, 0.0) == 0.0


def test_one_unflagged_error_in_the_amount_column_rejects_even_a_better_generation():
    parent = side(counts(correct=500, miss=5), amount=(0, 0, 0.3))
    cand = side(counts(correct=580, review=150, miss=2), amount=(1, 1, 0.5))
    assert decide(parent, cand) == Decision("reject", gate.R_AMOUNT_MISS)
    ok = side(counts(correct=580, review=150, miss=2), amount=(0, 0, 0.5))
    assert decide(parent, ok).adopted


def test_a_column_whose_misses_grow_rejects_even_if_the_total_does_not_get_worse():
    parent = side(counts(miss=5), account=(0, 0, 0.2), date=(5, 5, 4.0))
    cand = side(counts(correct=560, miss=5), account=(1, 1, 1.0), date=(4, 4, 4.0))
    assert decide(parent, cand) == Decision("reject", gate.R_MISS_WORSE)


def test_confident_errors_more_than_predicted_reject_by_the_poisson_test():
    parent = side(counts(miss=3), calib_=calib(expected=3.0, observed=3))
    # 予測 0.5 件に対して 4 件観測(P ≈ 0.0017)。見逃しの件数は親と同じでも、自信のある誤りが予測より有意に多い
    over = side(counts(correct=560, review=170, miss=3), calib_=calib(expected=0.5, observed=4))
    assert decide(parent, over) == Decision("reject", gate.R_OVERCONFIDENT)
    # 予測 3 件に対して 4 件は偶然の範囲
    fine = side(counts(correct=560, review=170, miss=3), calib_=calib(expected=3.0, observed=4))
    assert decide(parent, fine).adopted
    # 列ごとは 1% の水準: 予測 0.3 件に対して 2 件(P ≈ 0.037)は全体の 5% では引っかかるが列では通る
    col_ok = side(counts(correct=560, review=170, miss=3), date=(2, 2, 0.3))
    assert decide(parent, col_ok).adopted
    col_ng = side(counts(correct=560, review=170, miss=3), date=(3, 3, 0.3))  # P ≈ 0.0036
    assert decide(parent, col_ng).reason in (gate.R_OVERCONFIDENT, gate.R_MISS_WORSE)


def test_calibration_quality_must_not_get_worse():
    parent = side(counts(miss=2), calib_=calib(nll=0.40, expected=3, observed=2))
    worse = side(
        counts(correct=570, review=170, miss=2), calib_=calib(nll=0.55, expected=3, observed=2)
    )
    assert decide(parent, worse) == Decision("reject", gate.R_CALIBRATION_WORSE)
    better = side(
        counts(correct=570, review=170, miss=2), calib_=calib(nll=0.30, expected=3, observed=2)
    )
    assert decide(parent, better).adopted


def test_gate_still_accepts_plain_counts_and_missing_calibration():
    assert decide(counts(), counts(correct=560, review=170, miss=4)).adopted
    assert decide(side(counts()), counts(correct=560, review=170, miss=4)).adopted


def test_eval_side_roundtrips_through_a_dict():
    s = side(counts(), amount=(0, 0, 0.3), date=(2, 2, 1.0))
    back = gate.EvalSide.from_dict(s.to_dict())
    assert back.total == s.total and back.calib == s.calib
    assert back.by_column == s.by_column and back.calib_by_column == s.calib_by_column
    flat = gate.EvalSide.from_dict(EvalCounts(10, 5, 4, 1, 0).to_dict())
    assert flat.calib is None and flat.by_column == {}
