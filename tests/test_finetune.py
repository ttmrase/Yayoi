"""追加学習(L2-SP)と初期重みの二系統(素 / 適応済み)の選択のテスト(小さなランダム重み)。"""

from __future__ import annotations

import numpy as np
import pytest

from yayoi_ocr.recognize.parseq.engine import MODEL_FILES
from yayoi_ocr.recognize.variants import (
    MIN_CONFIRMED_PAGES,
    PageCounts,
    VariantScore,
    available_variants,
    choose_variant,
    count_page,
    default_variant,
    score_counts,
    score_variant,
    variant_dir,
)


def _touch_models(d, widths=(256, 384)):
    d.mkdir(parents=True, exist_ok=True)
    for w in widths:
        (d / MODEL_FILES[w]).write_bytes(b"x")


def test_variant_dirs_and_defaults(tmp_path):
    assert variant_dir("base", tmp_path) == tmp_path
    assert variant_dir("adapted", tmp_path) == tmp_path / "adapted"
    assert available_variants(tmp_path) == [] and default_variant(tmp_path) == "base"
    _touch_models(tmp_path)
    assert available_variants(tmp_path) == ["base"] and default_variant(tmp_path) == "base"
    _touch_models(tmp_path / "adapted", widths=(256,))  # 片方の幅しか無いものは使えない
    assert available_variants(tmp_path) == ["base"]
    _touch_models(tmp_path / "adapted")
    assert available_variants(tmp_path) == ["adapted", "base"]
    assert default_variant(tmp_path) == "adapted"  # 合成データでの評価が良いほうが既定


def test_choose_variant_waits_for_confirmed_pages_then_compares(tmp_path):
    _touch_models(tmp_path)
    _touch_models(tmp_path / "adapted")
    worse_adapted = [
        VariantScore("adapted", 0.90, 0.0, score_variant(0.90, 0.0)),
        VariantScore("base", 0.93, 0.0, score_variant(0.93, 0.0)),
    ]
    # 確定ページが足りないうちは比較しない
    assert choose_variant(MIN_CONFIRMED_PAGES - 1, worse_adapted, tmp_path) == "adapted"
    assert choose_variant(MIN_CONFIRMED_PAGES, worse_adapted, tmp_path) == "base"
    assert choose_variant(MIN_CONFIRMED_PAGES, None, tmp_path) == "adapted"
    tie = [VariantScore("base", 0.9, 0.0, 0.9), VariantScore("adapted", 0.9, 0.0, 0.9)]
    assert choose_variant(MIN_CONFIRMED_PAGES, tie, tmp_path) == "adapted"  # 同点は既定の優先順


def test_score_variant_punishes_misses_harder_than_errors():
    # 正解率が少し高くても、見逃しが多い側は負ける
    assert score_variant(0.95, 0.02) < score_variant(0.92, 0.0)
    assert score_variant(0.9, 0.0) == pytest.approx(0.9)


def test_count_page_and_score_counts():
    from types import SimpleNamespace

    def cell(value, review=False):
        return SimpleNamespace(value=value, review=review)

    def conf(resolved, kind="text"):
        return SimpleNamespace(resolved=resolved, kind=kind)

    result = SimpleNamespace(
        cells=[cell("a"), cell("b"), cell("c", review=True), cell("x"), cell("")]
    )
    confirmed = [conf("a"), conf("B"), conf("C"), conf("", "blank"), conf("", "blank")]
    c = count_page(result, confirmed)
    # 全 5 セル。空欄でない 3 つのうち正解は 1。見逃しは 'b'(誤り・確認なし)と 'x'(誤り・確認なし)の 2、'c' は確認済みなので数えない
    assert (c.n_cells, c.n_text, c.n_correct, c.n_miss) == (5, 3, 1, 2)
    s = score_counts("adapted", c + PageCounts(5, 5, 5, 0))
    assert s.accuracy == pytest.approx(6 / 8) and s.miss_rate == pytest.approx(0.2)
    assert s.score == pytest.approx(score_variant(s.accuracy, s.miss_rate))


torch = pytest.importorskip("torch")

from yayoi_ocr.recognize.parseq.finetune import FinetuneConfig, finetune  # noqa: E402
from yayoi_ocr.recognize.parseq.model import PARSeq  # noqa: E402

CHARSET = " 0123456789.-あいうえお"


def _tiny(seed: int = 0) -> PARSeq:
    torch.manual_seed(seed)
    m = PARSeq(
        num_tokens=len(CHARSET) + 3,
        max_label_length=6,
        img_size=(24, 64),
        patch_size=(4, 8),
        embed_dim=32,
        enc_num_heads=2,
        enc_mlp_ratio=2,
        enc_depth=3,
        dec_num_heads=4,
        dec_mlp_ratio=2,
        dec_depth=1,
    )
    return m


def _samples(n: int = 24):
    rng = np.random.default_rng(1)
    texts = ["12", "345", "あい", "6.7", "-9"]
    out = []
    for i in range(n):
        t = texts[i % len(texts)]
        img = np.full((30, 20 + 15 * len(t)), 255, np.uint8)
        # 文字の種類ごとに決まった濃淡の縞を描く(学習できる単純な手がかり)
        for k, ch in enumerate(t):
            img[5:25, 4 + 15 * k : 12 + 15 * k] = 30 + (ord(ch) * 7) % 150
        img = np.clip(img.astype(int) + rng.integers(-5, 5, img.shape), 0, 255).astype(np.uint8)
        out.append((img, t))
    return out


def _drift(a: PARSeq, b: PARSeq) -> float:
    return float(
        sum(((p - q) ** 2).sum() for p, q in zip(a.parameters(), b.parameters(), strict=True))
    )


def test_finetune_reduces_loss_and_l2sp_limits_drift():
    samples = _samples()
    results = {}
    for l2sp in (0.0, 50.0):
        m = _tiny()
        m0 = _tiny()
        h1 = finetune(
            m, CHARSET, samples, FinetuneConfig(width=64, epochs=1, batch=8, lr=3e-3, l2sp=l2sp)
        )
        h2 = finetune(
            m, CHARSET, samples, FinetuneConfig(width=64, epochs=4, batch=8, lr=3e-3, l2sp=l2sp)
        )
        results[l2sp] = (h1, h2, _drift(m, m0))
    free, anchored = results[0.0], results[50.0]
    assert free[1]["ar"] < free[0]["ar"]  # 学習が進めば損失は下がる
    assert anchored[2] < free[2]  # L2-SP は元の重みからのずれを抑える


def test_finetune_freezes_front_blocks_and_skips_unknown_chars():
    m = _tiny()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    finetune(
        m,
        CHARSET,
        [*_samples(8), (np.full((30, 40), 255, np.uint8), "☺")],  # 文字集合に無い正解は捨てる
        FinetuneConfig(width=64, epochs=1, batch=4, frozen_blocks=2, lr=1e-2),
    )
    after = m.state_dict()
    frozen = [
        k for k in before if k.startswith("encoder.blocks.0.") or k.startswith("encoder.blocks.1.")
    ]
    assert frozen and all(torch.equal(before[k], after[k]) for k in frozen)
    trained = [k for k in before if k.startswith("encoder.blocks.2.")]
    assert any(not torch.equal(before[k], after[k]) for k in trained)
    with pytest.raises(ValueError):
        finetune(m, CHARSET, [(np.full((30, 40), 255, np.uint8), "☺")], FinetuneConfig(width=64))
