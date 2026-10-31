"""追加学習した世代の再較正(純 numpy)。厳しくするのは観測した見逃しが許容数を超えたときだけ、緩めるのは CP 上側限界が目標以下と示せたときだけ。"""

from __future__ import annotations

import numpy as np

from yayoi_ocr.app.recalibrate import CellObs, recalibrate, summarize
from yayoi_ocr.recognize.confidence import (
    FEATURE_NAMES,
    ConfidenceModel,
    GroupModel,
    LogisticRegression,
    cp_upper,
)


def lr() -> LogisticRegression:
    m = LogisticRegression()
    m.w = np.zeros(len(FEATURE_NAMES))
    m.b = 0.0
    return m


def base(tau=0.9) -> ConfidenceModel:
    return ConfidenceModel(
        {
            "text:amount": GroupModel(lr(), 0.5, tau, 1000),
            "ditto:amount": GroupModel(lr(), 0.5, tau, 100),
            "blank:amount": GroupModel(None, 0.99, 0.5, 100),  # 定数の群(しきい値を調整しない)
            "text:description": GroupModel(lr(), 0.5, tau, 1000),
        }
    )


def obs(column, ps, correct=True, kind="text", hard=False):
    return [CellObs(column, kind, p, correct, hard) for p in ps]


def test_tightens_only_when_observed_misses_exceed_the_allowed_count():
    # 金額 0.1% × 300 セル = 許容 0 件。p の高い見逃しが 3 件
    o = obs("amount", np.linspace(0.9, 1.0, 200)) + obs("amount", [0.95, 0.97, 0.99], correct=False)
    model, calibs = recalibrate(base(), o, {"amount": 300})
    c = next(x for x in calibs if x.column == "amount")
    assert c.action == "tighten" and c.k_before == 3 and c.k_after == 0
    assert c.tau_after > 0.99 and c.tau_after > c.tau_before
    assert (
        model.groups["text:amount"].threshold
        == c.tau_after
        == model.groups["ditto:amount"].threshold
    )
    assert model.groups["blank:amount"].threshold == 0.5  # 定数の群は触らない


def test_tightening_raises_tau_just_enough_to_reach_the_allowed_count():
    # 摘要 0.5% × 600 セル = 許容 3 件。見逃しが 5 件 → 2 件分(p の低い見逃しから)だけ要確認に回す
    wrong = [0.91, 0.92, 0.93, 0.96, 0.99]
    o = obs("description", np.linspace(0.9, 1.0, 100)) + obs("description", wrong, correct=False)
    _, calibs = recalibrate(base(), o, {"description": 600})
    c = next(x for x in calibs if x.column == "description")
    assert c.allowed == 3 and c.action == "tighten" and c.k_after == 3
    assert (
        0.92 < c.tau_after <= 0.9200011
    )  # 0.91 と 0.92 の見逃しだけが要確認になる(要確認の増加が最小)


def test_keeps_the_threshold_when_misses_are_within_the_allowance_and_cannot_be_shown_to_loosen():
    o = obs("amount", np.linspace(0.5, 1.0, 200)) + obs("amount", [0.99], correct=False)
    # 金額 0.1% × 3000 セル = 許容 3 件。見逃し 1 件は許容内。CP 上側限界(約 0.16%)は目標 0.1% を超えるので緩めない
    model, calibs = recalibrate(base(), o, {"amount": 3000})
    c = next(x for x in calibs if x.column == "amount")
    assert c.action == "keep" and c.tau_after == c.tau_before == 0.9
    assert model.groups["text:amount"].threshold == 0.9


def test_loosens_only_when_the_cp_upper_bound_proves_the_target_is_met():
    # 見逃し 0 件・3,000 セル: 上側限界 ≈ 0.0998% ≤ 0.1%。最も低い p まで緩められる
    assert cp_upper(0, 3000) <= 0.001
    o = obs("amount", np.linspace(0.5, 1.0, 300))
    model, calibs = recalibrate(base(), o, {"amount": 3000})
    c = next(x for x in calibs if x.column == "amount")
    assert c.action == "loosen" and c.tau_after == 0.5 and c.cp_upper is not None
    assert model.groups["text:amount"].threshold == 0.5
    # 同じ観測でも、セル数が足りなければ緩めない(評価 10 ページ以内 = 約 290 セルでは示せない)
    _, calibs = recalibrate(base(), o, {"amount": 290})
    assert next(x for x in calibs if x.column == "amount").action == "keep"


def test_does_not_change_the_base_model_and_ignores_hard_review_cells_and_correct_cells():
    b = base()
    o = obs("amount", [0.99, 0.98], correct=False, hard=True) + obs("amount", [0.95, 0.99])
    model, calibs = recalibrate(b, o, {"amount": 300})
    assert (
        next(x for x in calibs if x.column == "amount").k_before == 0
    )  # 要確認が必須のセルは見逃しではない
    assert b.groups["text:amount"].threshold == 0.9
    assert model is not b and model.groups["text:amount"] is not b.groups["text:amount"]


def test_unreachable_when_fixed_groups_alone_exceed_the_allowance():
    # 定数の群(blank:amount、p=0.99 ≥ しきい値 0.5)の見逃しが多い → しきい値では減らせない。要確認だらけにしない
    o = obs("amount", [0.95], kind="blank", correct=False) * 4 + obs("amount", [0.95, 0.96])
    model, calibs = recalibrate(base(), o, {"amount": 300})
    c = next(x for x in calibs if x.column == "amount")
    assert c.action == "unreachable" and c.tau_after == c.tau_before
    assert model.groups["text:amount"].threshold == 0.9


def test_learned_feature_weights_stay_zero_and_summary_has_no_values():
    from yayoi_ocr.recognize.confidence import LEARNED_FEATURES

    o = obs("amount", np.linspace(0.9, 1.0, 50)) + obs("amount", [0.99], correct=False)
    model, calibs = recalibrate(base(), o, {"amount": 300})
    for g in model.groups.values():
        if g.lr is not None:
            for name in LEARNED_FEATURES:
                assert g.lr.w[FEATURE_NAMES.index(name)] == 0.0
    s = summarize(calibs)
    assert set(s["columns"]) == {"amount", "description"}
    assert "amount" in s["changed"]
    assert all(
        not isinstance(v, str)
        for c in s["columns"].values()
        for k, v in c.items()
        if k != "column" and k != "action"
    )
