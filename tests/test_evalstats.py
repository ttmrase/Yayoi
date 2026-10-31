"""評価用ページの観測からの要確認の判定の再現・件数・較正の質、クロスフィッティング、親より緩めない床、採用直後のウォームアップ(純 numpy)。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from yayoi_ocr.app import evalstats, recalibrate
from yayoi_ocr.app.evalstats import CellObs
from yayoi_ocr.recognize.confidence import (
    FEATURE_NAMES,
    ConfidenceModel,
    GroupModel,
    LogisticRegression,
)


def lr() -> LogisticRegression:
    m = LogisticRegression()
    m.w = np.zeros(len(FEATURE_NAMES))
    m.b = 0.0
    return m


def model(tau=0.9, amount_tau=None) -> ConfidenceModel:
    at = tau if amount_tau is None else amount_tau
    return ConfidenceModel(
        {
            "text:amount": GroupModel(lr(), 0.5, at, 1000),
            "ditto:amount": GroupModel(lr(), 0.5, at, 100),
            "text:description": GroupModel(lr(), 0.5, tau, 1000),
            "blank:amount": GroupModel(None, 0.99, 0.5, 100),
        }
    )


def cell(column="amount", kind="text", p=0.95, correct=True, hard=False, **kw):
    return CellObs(column, kind, p, correct, hard, **kw)


def page(n_ok=40, amount_wrong=(), page_no=0):
    """金額 n_ok 個(確信度 0.90〜0.99)+ 指定した確信度の金額の見逃し候補。"""
    obs = [cell(p=0.90 + 0.09 * i / max(n_ok - 1, 1), page=page_no, idx=i) for i in range(n_ok)]
    obs += [
        cell(p=p, correct=False, page=page_no, idx=n_ok + j) for j, p in enumerate(amount_wrong)
    ]
    return obs


# ---- 要確認の判定の再現(PageRecognizer._finalize と同じ規則)


def test_review_flags_follow_the_threshold_the_hard_reasons_and_the_ditto_source():
    m = model(0.9)
    obs = [
        cell(p=0.95, idx=0),  # しきい値以上 → 要確認でない
        cell(p=0.85, idx=1),  # 未満 → 要確認
        cell(p=0.99, hard=True, idx=2),  # 確認必須の理由
        cell(kind="ditto", p=0.99, idx=3, src=1),  # 参照元(idx 1)が要確認 → 同上も要確認
        cell(kind="ditto", p=0.99, idx=4, src=0),  # 参照元が要確認でない → 要確認でない
        cell(kind="arrow", p=0.0, idx=5),  # 矢印・波線は常に要確認
        cell(column="description", p=0.89, idx=6),  # 列ごとのしきい値(摘要も 0.9)
        cell(kind="blank", p=0.6, idx=7),  # 定数の群: しきい値 0.5 → 要確認でない
        cell(column="date", kind="text", p=0.1, idx=8),  # 群が無い列: 既定のしきい値(0.9)
    ]
    assert evalstats.review_flags(obs, m) == [
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        False,
        True,
    ]
    # しきい値を変えれば、認識をやり直さずに判定が変わる
    assert evalstats.review_flags(obs, model(0.97))[0] is True


def test_counts_and_calibration_quality():
    m = model(0.9)
    obs = [
        cell(p=0.95, correct=True, idx=0),
        cell(p=0.95, correct=False, idx=1),  # 要確認でない誤り = 見逃し
        cell(p=0.80, correct=False, idx=2),  # 要確認 → 見逃しではない
        cell(p=0.99, correct=True, hard=True, idx=3),  # 確認必須: 較正の対象外
        cell(kind="blank", p=0.99, correct=True, nonblank=False, idx=4),
    ]
    s = evalstats.side_from([obs], [evalstats.review_flags(obs, m)])
    t = s.total
    assert (t.n_cells, t.n_text, t.n_correct, t.n_review, t.n_miss) == (5, 4, 2, 2, 1)
    assert s.by_column["amount"].n_miss == 1
    c = s.calib
    assert c.n == 4  # 確認必須のセルを除く
    # 要確認でないセル(idx 0, 1, 4): 期待される見逃し = 0.05 + 0.05 + 0.01、観測 = 1
    assert c.expected_miss == pytest.approx(0.11) and c.observed_miss == 1
    expect_nll = -(math.log(0.95) + math.log(0.05) + math.log(0.2) + math.log(0.99)) / 4
    assert c.nll == pytest.approx(expect_nll)


# ---- クロスフィッティング: 自分に合わせたしきい値では見逃しを数えない


def test_cross_fitting_counts_a_confident_error_that_only_its_own_page_would_hide():
    """1 ページに自信のある誤り(p = 0.97)が 1 件ある。金額の許容は 0 件なので、全ページで再較正するとしきい値がそれを要確認にする(自分に合わせた = 楽観的)。
    クロスフィッティングでは、そのページの判定は他のページ(誤りなし)から決めたしきい値なので、見逃しとして数える。"""
    base = model(0.9)
    pages = [page(page_no=0), page(page_no=1), page(amount_wrong=[0.97], page_no=2)]
    final, _calibs, cf = recalibrate.cross_fit(base, pages)
    # 全評価ページで決めたしきい値は、誤りを要確認にする(採用した世代が実際に使う値)
    assert final.groups["text:amount"].threshold > 0.97
    optimistic = evalstats.side_from(pages, [evalstats.review_flags(pg, final) for pg in pages])
    honest = evalstats.side_from(pages, cf)
    assert optimistic.by_column["amount"].n_miss == 0  # 循環: 必ず通る
    assert honest.by_column["amount"].n_miss == 1  # クロスフィッティング: 見逃しを数える
    # ほかのページ(誤りのないページ 0)の判定は、誤りのあるページ 2 から決めたしきい値になる(厳しくなる)
    assert sum(cf[0]) > sum(evalstats.review_flags(pages[0], base))


def test_cross_fitting_tightens_pages_by_the_errors_of_the_other_pages():
    base = model(0.9)
    pages = [page(amount_wrong=[0.97], page_no=0), page(page_no=1), page(page_no=2)]
    _final, _calibs, cf = recalibrate.cross_fit(base, pages)
    honest = evalstats.side_from(pages, cf)
    # ページ 1・2 の判定は、ページ 0 の誤り(p = 0.97)から決めたしきい値 → 確信度の高いセルまで要確認になる(誤りのあるページ自身は数える)
    assert sum(cf[1]) > sum(evalstats.review_flags(pages[1], base))
    assert honest.by_column["amount"].n_miss == 1


def test_a_single_evaluation_page_uses_the_base_thresholds():
    base = model(0.9)
    pages = [page(amount_wrong=[0.97])]
    _final, _calibs, cf = recalibrate.cross_fit(base, pages)
    assert cf[0] == evalstats.review_flags(pages[0], base)


# ---- 確定ページが少ない間は親より緩めない


def test_floor_keeps_thresholds_at_least_as_strict_as_the_parent():
    base = model(0.90)
    parent = model(0.95)
    obs = page(n_ok=300)  # 見逃しなし。ページ数が少なく緩められない → 通常は据え置き(0.90)
    m, calibs = recalibrate.recalibrate(base, obs, {"amount": 29 * 10}, floor=parent)
    c = next(x for x in calibs if x.column == "amount")
    assert c.action == "floor" and c.tau_after == 0.95
    assert m.groups["text:amount"].threshold == 0.95
    # 親が緩ければ床は効かない
    m2, calibs2 = recalibrate.recalibrate(base, obs, {"amount": 290}, floor=model(0.80))
    assert next(x for x in calibs2 if x.column == "amount").action == "keep"
    assert m2.groups["text:amount"].threshold == 0.90
    # 観測した見逃しで厳しくしたあとも、親より緩くならない
    obs2 = page(amount_wrong=[0.91])
    m3, _ = recalibrate.recalibrate(base, obs2, {"amount": 290}, floor=parent)
    assert m3.groups["text:amount"].threshold >= 0.95


def test_apply_floor_does_not_loosen_and_does_not_touch_constant_groups():
    out = evalstats.apply_floor(model(0.90), model(0.95))
    assert out.groups["text:amount"].threshold == 0.95
    assert out.groups["blank:amount"].threshold == 0.5
    assert evalstats.apply_floor(model(0.90), None).groups["text:amount"].threshold == 0.90


# ---- 採用直後のウォームアップ(期待される見逃し率が目標 × 0.8 以下になるしきい値)


def test_warm_threshold_is_stricter_only_when_the_predicted_miss_rate_exceeds_the_budget():
    base = model(0.90)
    # 金額 0.1% × 3,000 セル × 0.8 = 予測 2.4 件まで。確信度 0.97 のセルが 200 個 → 予測 (1 - 0.97) × 200 = 6 件で超える
    obs = [cell(p=0.97, idx=i) for i in range(200)] + [
        cell(p=0.9995, idx=200 + i) for i in range(100)
    ]
    warm = recalibrate.warm_thresholds(base, obs, {"amount": 3000})
    w = warm.groups["text:amount"].warm_threshold
    assert w is not None and w > 0.97  # 0.97 のセルを要確認に回すしきい値
    kept = sum(1.0 - o.p for o in obs if o.p >= w)
    assert kept <= 0.8 * 0.001 * 3000
    assert warm.groups["text:amount"].threshold == 0.90  # 通常のしきい値は変わらない
    # 予測が予算内なら、通常のしきい値と同じ
    easy = [cell(p=0.9999, idx=i) for i in range(50)]
    w2 = (
        recalibrate.warm_thresholds(base, easy, {"amount": 3000})
        .groups["text:amount"]
        .warm_threshold
    )
    assert w2 == 0.90
    assert base.groups["text:amount"].warm_threshold is None  # 元のモデルは変えない


def test_warm_threshold_is_saved_and_applied_only_when_asked(tmp_path):
    base = model(0.90)
    obs = [cell(p=0.97, idx=i) for i in range(200)]
    warm = recalibrate.warm_thresholds(base, obs, {"amount": 3000})
    path = tmp_path / "confidence.json"
    warm.save(path)
    normal = ConfidenceModel.load(path)
    strict = ConfidenceModel.load(path, warm=True)
    g_n, g_w = normal.groups["text:amount"], strict.groups["text:amount"]
    assert (
        g_n.threshold == 0.90 and g_w.threshold > 0.97 and g_n.warm_threshold == g_w.warm_threshold
    )
    # 古い形式(warm_threshold なし)も読める
    base.save(tmp_path / "old.json")
    assert (
        ConfidenceModel.load(tmp_path / "old.json", warm=True).groups["text:amount"].threshold
        == 0.90
    )
