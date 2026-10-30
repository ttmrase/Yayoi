from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from yayoi_ocr.recognize.celltype import (
    CLASSES,
    N_STATS,
    CellTypeClassifier,
    CellTypeNet,
    fit_temperature,
    kind_to_class,
    stats_vector,
    to_tensor,
)


def test_kind_to_class_maps_elided_to_wave():
    assert [CLASSES[kind_to_class(k)] for k in ("blank", "ditto", "arrow", "elided", "text")] == [
        "blank",
        "ditto",
        "arrow",
        "wave",
        "text",
    ]


def test_stats_vector_layout():
    v = stats_vector("amount", 0.05, ["ditto_likely", "overflow"], True, 0.3, 0.2, True)
    assert v.shape == (N_STATS,) and v.dtype == np.float32
    assert v[1] == 1.0 and v[2] == 1.0 and v[3] == 0.0 and v[4] == 1.0 and v[7] == 1.0
    assert v[8:12].tolist() == [0.0, 0.0, 0.0, 1.0]  # 列のワンホット(金額)


def test_to_tensor_inverts_ink():
    x = np.full((2, 64, 448), 255, np.uint8)
    x[0, 10, 10] = 0
    t = to_tensor(x)
    assert t.shape == (2, 1, 64, 448) and t[0, 0, 10, 10] == 1.0 and t[1].max() == 0.0


def test_forward_save_load_roundtrip(tmp_path):
    net = CellTypeNet()
    imgs = np.random.default_rng(0).integers(0, 255, (3, 64, 448)).astype(np.uint8)
    st = np.stack([stats_vector("date", 0.02, [], False, 0.3, 0.2, False)] * 3)
    clf = CellTypeClassifier(net, temperature=1.7)
    p = clf.predict_proba(imgs, st)
    assert p.shape == (3, len(CLASSES)) and np.allclose(p.sum(1), 1.0, atol=1e-5)
    path = tmp_path / "c.pt"
    clf.save(path, {"note": "test"})
    clf2 = CellTypeClassifier.load(path)
    assert clf2.temperature == 1.7
    assert np.allclose(clf2.predict_proba(imgs, st), p, atol=1e-6)
    pred = clf2.predict(imgs, st)[0]
    assert pred.cls in CLASSES and abs(sum(pred.probs.values()) - 1.0) < 1e-5


def test_temperature_scaling_softens_overconfident_logits():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, 2000)
    logits = np.zeros((2000, 5), np.float32)
    correct = rng.random(2000) < 0.6  # 実際の正解率は 6 割なのに、ロジットは過信している
    pred = np.where(correct, y, (y + 1) % 5)
    logits[np.arange(2000), pred] = 8.0
    assert fit_temperature(logits, y) > 2.0
