"""追加学習のワーカーの純粋な部分(エポック数・リハーサルの混ぜ方・ラベルの候補・リハーサルの束・ページの入力の保存)。重い学習は slow の結合試験。"""

from __future__ import annotations

import numpy as np

from yayoi_ocr.app import train_worker
from yayoi_ocr.recognize.labels import choose_label, label_candidates
from yayoi_ocr.recognize.rehearsal import load_rehearsal, save_rehearsal


def img(n=1):
    return np.full((8, 12), n, np.uint8)


def test_epochs_are_clamped_and_cap_the_total_passes_at_4000():
    assert train_worker.epochs_for(100) == 40
    assert train_worker.epochs_for(800) == 5
    assert train_worker.epochs_for(4000) == 1
    assert train_worker.epochs_for(9000) == 1
    assert train_worker.epochs_for(0) == 40


def test_rehearsal_is_half_of_the_user_cells_and_total_passes_are_capped(monkeypatch):
    rng = np.random.default_rng(0)
    user = [(img(i % 250), f"u{i}") for i in range(100)]
    pool = [(img(200), f"r{i}") for i in range(500)]
    samples, k = train_worker.mix_rehearsal(user, pool, rng)
    assert k == 50 and len(samples) == 150
    assert sum(1 for _, t in samples if t.startswith("u")) == 100  # 利用者のセルはすべて入る
    samples, k = train_worker.mix_rehearsal(user, pool[:10], rng)  # 束が小さければ、あるだけ
    assert k == 10 and len(samples) == 110
    samples, k = train_worker.mix_rehearsal(user, [], rng)
    assert k == 0 and len(samples) == 100
    monkeypatch.setattr(train_worker, "MAX_PASSES", 60)
    samples, _ = train_worker.mix_rehearsal(user, pool, rng)
    assert len(samples) == 60


def test_label_candidates_cover_the_ways_it_may_have_been_written():
    d = label_candidates("date", "R7.11.19")
    assert "R7.11.19" in d and "11.19" in d  # 年が書かれていない書き方も
    a = label_candidates("amount", "3456")
    assert {"3456", "3,456", "3,456-", "3456-"} <= set(a)
    assert label_candidates("description", "旅費 交通費") == ["旅費交通費"]
    assert label_candidates("date", "壊れた値") == ["壊れた値"]


def test_choose_label_picks_the_most_likely_surface_with_the_engine():
    class Res:
        def __init__(self, lp):
            self.logprob = lp

    class Eng:
        def score_many(self, crop, texts, column=None, *, width=None):
            assert crop.shape[-1] == 3
            return [Res(-1.0 if t == "3,456-" else -9.0) for t in texts]

    assert choose_label(Eng(), img(), "amount", "3456") == "3,456-"
    assert choose_label(None, img(), "amount", "3456") == label_candidates("amount", "3456")[0]
    assert choose_label(Eng(), img(), "description", "") is None


def test_rehearsal_pack_roundtrip(tmp_path):
    items = [(256, "date", img(3), "R7.1.1"), (384, "description", img(4), "雑費")]
    save_rehearsal(tmp_path / "r.npz", items)
    out = load_rehearsal(tmp_path / "r.npz")
    assert set(out) == {256, 384} and out[256][0][1] == "R7.1.1"
    assert out[384][0][0].dtype == np.uint8 and out[384][0][0].shape == (8, 12)


def test_pick_device_prefers_gpu_only_when_available():
    class Cuda:
        def __init__(self, ok):
            self.cuda = type("C", (), {"is_available": staticmethod(lambda: ok)})

    assert train_worker.pick_device({"device": "auto"}, Cuda(True)) == "cuda"
    assert train_worker.pick_device({"device": "auto"}, Cuda(False)) == "cpu"
    assert train_worker.pick_device({"device": "cpu"}, Cuda(True)) == "cpu"  # 明示すれば CPU
    assert train_worker.pick_device({"device": "cuda"}, Cuda(False)) == "cpu"  # GPU が無ければ CPU


# ---- リハーサルの束の作成: 入力が合成データの署名を持たなければ失敗する(I-10)


def _write_page(path, *, with_signature: bool):
    import json

    page = {"seed": 1, "writer_id": 2, "font": "FakeFont-Regular"}
    if not with_signature:
        page = {"title": "署名のないページ"}
    meta = {"page": page, "cells": []}
    np.savez(path, meta=np.array(json.dumps(meta)), clf=np.zeros((0, 64, 448), np.uint8))


def _build(tmp_path, monkeypatch, with_signature: bool, fonts: bool):
    import importlib.util
    import sys
    from pathlib import Path

    private = tmp_path / "private"
    (private / "fonts").mkdir(parents=True)
    if fonts:
        (private / "fonts" / "FakeFont-Regular.ttf").write_bytes(b"x")
    data = tmp_path / "ds"
    data.mkdir()
    _write_page(data / "page_000.npz", with_signature=with_signature)
    monkeypatch.setenv("YAYOI_PRIVATE_DATA", str(private))
    spec = importlib.util.spec_from_file_location(
        "build_rehearsal", Path(__file__).resolve().parent.parent / "scripts" / "build_rehearsal.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(
        sys, "argv", ["x", "--data", str(data), "--out", str(private / "models" / "r.npz")]
    )
    (private / "models").mkdir()
    return mod.main(), private / "models" / "r.npz"


def test_rehearsal_build_refuses_pages_without_the_synthetic_signature(tmp_path, monkeypatch):
    code, out = _build(tmp_path, monkeypatch, with_signature=False, fonts=True)
    assert code == 1 and not out.exists()


def test_rehearsal_build_refuses_when_the_font_set_is_unknown(tmp_path, monkeypatch):
    code, out = _build(tmp_path, monkeypatch, with_signature=True, fonts=False)
    assert code == 1 and not out.exists()


def test_rehearsal_build_records_the_input_datasets_when_the_signature_is_valid(
    tmp_path, monkeypatch
):
    import json

    code, out = _build(tmp_path, monkeypatch, with_signature=True, fonts=True)
    assert code == 0 and out.exists()
    with np.load(out, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
    assert meta["sources"] == [{"dataset": "ds", "pages": 1, "seed_min": 1, "seed_max": 1}]
