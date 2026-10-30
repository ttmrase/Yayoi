from __future__ import annotations

import numpy as np

from yayoi_ocr.recognize.confidence import (
    FEATURE_NAMES,
    ConfidenceModel,
    LogisticRegression,
    Sample,
    choose_threshold,
    featurize,
    fit_confidence,
    samples_from_dump,
)


def test_logistic_regression_learns_separable_signal():
    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(size=(n, len(FEATURE_NAMES)))
    y = (x[:, 0] + 0.5 * x[:, 3] + 0.3 * rng.normal(size=n) > 0).astype(float)
    lr = LogisticRegression(l2=1.0).fit(x, y)
    p = lr.predict_proba(x)
    assert ((p > 0.5) == (y > 0.5)).mean() > 0.9
    assert lr.w[0] > 0 and lr.w[3] > 0


def test_featurize_defaults_and_scaling():
    v = featurize({"r1_logp": -20.0, "n_cands": 4.0}, 0.8)
    f = dict(zip(FEATURE_NAMES, v, strict=True))
    assert f["cls_p"] == 0.8 and f["r1_logp"] == -2.0 and abs(f["log_n_cands"] - np.log(4)) < 1e-9
    assert f["mass"] == 0.0  # 欠けている特徴は 0


def test_choose_threshold_respects_miss_budget():
    p = np.array([0.99, 0.98, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    correct = np.array([1, 1, 1, 0, 1, 0, 0, 0], dtype=bool)
    # 列の全セルが 1000 のとき、許される見逃し件数 = 目標 x 1000
    tau0 = choose_threshold(p, correct, 1000, 0.0005)  # 0.5 件 → 0 件: 最初の誤り(0.8)の手前まで
    assert (p >= tau0).sum() == 3
    tau1 = choose_threshold(p, correct, 1000, 0.001)  # 1 件: 誤り 1 件を含む 5 件目(0.7)まで
    assert (p >= tau1).sum() == 5
    tau2 = choose_threshold(p, correct, 1000, 0.002)  # 2 件: 2 件目の誤り(0.6)まで
    assert (p >= tau2).sum() == 6
    assert choose_threshold(p, ~correct, 1000, 0.0) == 1.1  # 先頭がいきなり誤りなら、全部確認


def test_fit_confidence_and_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    samples = []
    for _ in range(800):
        top_p = float(rng.uniform(0, 1))
        correct = bool(rng.uniform() < top_p**2)
        feats = {"top_p": top_p, "post_top": top_p, "mass": top_p, "n_cands": 2.0}
        samples.append(Sample("amount", "text", featurize(feats, 1.0), correct, False))
    samples.append(Sample("amount", "arrow", featurize({}, 1.0), False, True))
    model = fit_confidence(samples, {"amount": 800})
    g = model.groups["text:amount"]
    assert g.lr is not None and g.lr.w[FEATURE_NAMES.index("top_p")] != 0
    assert model.groups["arrow:amount"].threshold == 1.1
    # 較正データ上の見逃しは、列の全セルの 0.1% 以下
    p = np.array([g.prob(s.x) for s in samples if s.kind == "text"])
    c = np.array([s.correct for s in samples if s.kind == "text"])
    assert ((p >= g.threshold) & ~c).sum() <= 1
    path = tmp_path / "conf.json"
    model.save(path)
    m2 = ConfidenceModel.load(path)
    x = samples[0].x
    assert abs(m2.groups["text:amount"].prob(x) - g.prob(x)) < 1e-12
    assert m2.groups["text:amount"].threshold == g.threshold


def test_samples_from_dump_marks_hard_cases():
    rows = [
        {
            "column": "date",
            "pred_kind": "text",
            "correct": True,
            "page_review": False,
            "reasons": [],
            "cls_probs": {"text": 0.9},
            "features": {"top_p": 0.9},
        },
        {
            "column": "date",
            "pred_kind": "text",
            "correct": False,
            "page_review": False,
            "reasons": ["year_unknown"],
            "cls_probs": {"text": 0.9},
            "features": {},
        },
        {
            "column": "date",
            "pred_kind": "arrow",
            "correct": False,
            "page_review": False,
            "reasons": ["arrow_candidate"],
            "cls_probs": {"arrow": 0.9},
            "features": {},
        },
        {
            "column": "date",
            "pred_kind": "text",
            "correct": True,
            "page_review": True,
            "reasons": [],
            "cls_probs": {"text": 0.9},
            "features": {},
        },
    ]
    ss = samples_from_dump(rows)
    assert [s.hard for s in ss] == [False, True, True, True]
