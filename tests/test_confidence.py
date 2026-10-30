from __future__ import annotations

import numpy as np

from yayoi_ocr.recognize.confidence import (
    FEATURE_NAMES,
    NONNEG_FEATURES,
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


def test_cp_upper_matches_known_values():
    from yayoi_ocr.recognize.confidence import cp_upper

    assert abs(cp_upper(0, 1000) - (1 - 0.05 ** (1 / 1000))) < 1e-6  # k = 0 の閉じた式
    assert (
        abs(cp_upper(3, 1000) - 0.007735) < 2e-6
    )  # Clopper-Pearson 片側 95%(Poisson 近似は 0.007754)
    assert (
        abs(cp_upper(1, 100_000) - 4.744e-5) < 1e-8 and abs(cp_upper(2, 100_000) - 6.296e-5) < 1e-8
    )
    assert cp_upper(0, 0) == 1.0 and cp_upper(5, 5) == 1.0
    assert cp_upper(2, 5000) > cp_upper(2, 50000)  # データが増えるほど限界は小さくなる


def test_choose_threshold_uses_upper_confidence_limit():
    p = np.array([0.99, 0.98, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    correct = np.array([1, 1, 1, 0, 1, 0, 0, 0], dtype=bool)
    # 列の全セルが少ないと、見逃し 0 件でも上側限界(約 3 / n)が目標を超えるので、全部確認
    assert choose_threshold(p, correct, 1000, 0.001) == 1.1
    # 十分に多ければ、見逃しが目標以下の範囲で確認なしにできる(最初の誤り 0.8 の直前まで)
    # 目標 5e-5: 見逃し 1 件の上側限界 4.74e-5 は目標以内、2 件の 6.30e-5 は超える → 見逃し ≦ 1 件の範囲(上位 5 件)
    tau = choose_threshold(p, correct, 100_000, 5e-5)
    assert (p >= tau).sum() == 5
    assert choose_threshold(p, ~correct, 100_000, 0.0) == 1.1


def test_fit_confidence_and_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    samples = []
    for _ in range(800):
        top_p = float(rng.uniform(0, 1))
        correct = bool(rng.uniform() < top_p**2)
        feats = {"top_p": top_p, "post_top": top_p, "mass": top_p, "n_cands": 2.0}
        samples.append(Sample("amount", "text", featurize(feats, 1.0), correct, False))
    samples.append(Sample("amount", "arrow", featurize({}, 1.0), False, True))
    model = fit_confidence(samples, {"amount": 200_000})
    g = model.groups["text:amount"]
    assert g.lr is not None and g.lr.w[FEATURE_NAMES.index("top_p")] != 0
    assert model.groups["arrow:amount"].threshold == 1.1
    # 較正データ上の見逃しは、列の全セルの 0.1% 以下
    p = np.array([g.prob(s.x) for s in samples if s.kind == "text"])
    c = np.array([s.correct for s in samples if s.kind == "text"])
    from yayoi_ocr.recognize.confidence import cp_upper

    k = int(((p >= g.threshold) & ~c).sum())
    assert (
        cp_upper(k, 200_000) <= 0.001 or g.threshold == 1.1
    )  # 見逃し率の上側限界が目標(金額 0.1%)以下
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


def test_history_feature_weights_are_never_negative():
    rng = np.random.default_rng(3)
    n = 500
    x = rng.normal(size=(n, len(FEATURE_NAMES)))
    i = FEATURE_NAMES.index("hist_desc")
    y = (x[:, 0] - 2.0 * x[:, i] + 0.2 * rng.normal(size=n) > 0).astype(
        float
    )  # 履歴の一致が負に効く合成データ
    samples = [Sample("amount", "text", x[k], bool(y[k]), False) for k in range(n)]
    model = fit_confidence(samples, {"amount": 1_000_000})
    lr = model.groups["text:amount"].lr
    assert lr is not None and all(lr.w[FEATURE_NAMES.index(f)] >= 0.0 for f in NONNEG_FEATURES)


def test_samples_from_confirmed_counts_cells_and_judges_by_confirmed_value():
    from types import SimpleNamespace

    from yayoi_ocr.recognize.confidence import samples_from_confirmed

    def cell(col, value, kind="text", reasons=()):
        return SimpleNamespace(
            column=col,
            kind=kind,
            value=value,
            reasons=list(reasons),
            features={"r1_logp": -1.0},
            cls_probs={kind: 0.9},
        )

    result = SimpleNamespace(
        layout_status="ok",
        cells=[
            cell("amount", "500"),
            cell("amount", "800"),
            cell("description", "", kind="blank"),
            cell("amount", "100", kind="arrow"),
        ],
    )
    confirmed = [
        SimpleNamespace(resolved="500"),
        SimpleNamespace(resolved="900"),
        SimpleNamespace(resolved=""),
        SimpleNamespace(resolved="100"),
    ]
    samples, n_cells = samples_from_confirmed([(result, confirmed)])
    assert n_cells == {"amount": 3, "description": 1}
    assert [s.correct for s in samples] == [True, False, True, True]
    assert [s.hard for s in samples] == [False, False, False, True]  # 矢印は常に確認
    flagged = SimpleNamespace(layout_status="needs_review", cells=result.cells[:1])
    assert samples_from_confirmed([(flagged, confirmed[:1])])[0][0].hard  # 要確認ページは対象外


def test_replace_columns_swaps_only_the_given_columns():
    from yayoi_ocr.recognize.confidence import GroupModel, replace_columns

    a = ConfidenceModel(
        {
            "text:date": GroupModel(None, 0.9, 0.5, 10),
            "text:description": GroupModel(None, 0.9, 0.5, 10),
            "ditto:description": GroupModel(None, 0.9, 0.5, 10),
        },
        default_threshold=0.9,
    )
    b = ConfidenceModel(
        {
            "text:date": GroupModel(None, 0.1, 1.1, 99),
            "text:description": GroupModel(None, 0.1, 1.1, 99),
        }
    )
    m = replace_columns(a, b, ["description"])
    assert m.groups["text:date"].n_train == 10  # 差し替えない列はそのまま
    assert m.groups["text:description"].n_train == 99
    assert "ditto:description" not in m.groups  # 差し替え元に無い群は消える(古い群を残さない)
    assert m.default_threshold == 0.9 and a.groups["text:description"].n_train == 10  # 元は変えない
