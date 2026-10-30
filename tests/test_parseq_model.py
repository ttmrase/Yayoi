"""PARSeq の自作実装とエンジンの単体テスト(小さなランダム重み。モデルの重みは不要)。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from yayoi_ocr.models import ColumnKind

pytestmark = pytest.mark.ml

torch = pytest.importorskip("torch")

from yayoi_ocr.recognize.parseq.charset import column_charset, normalize_width  # noqa: E402
from yayoi_ocr.recognize.parseq.engine import EMBED_BINS, ParseqEngine  # noqa: E402
from yayoi_ocr.recognize.parseq.model import PARSeq  # noqa: E402
from yayoi_ocr.recognize.parseq.preprocess import ndl_preprocess  # noqa: E402

CHARSET = " 0123456789R.〃,-あいうえお現金"


@pytest.fixture(autouse=True)
def _no_grad():
    with torch.inference_mode():
        yield


def _tiny(width: int = 64, max_len: int = 6, seed: int = 0) -> PARSeq:
    torch.manual_seed(seed)
    m = PARSeq(
        num_tokens=len(CHARSET) + 3,
        max_label_length=max_len,
        img_size=(24, width),
        patch_size=(4, 8),
        embed_dim=32,
        enc_num_heads=2,
        enc_mlp_ratio=2,
        enc_depth=2,
        dec_num_heads=4,
        dec_mlp_ratio=2,
        dec_depth=1,
    )
    with torch.no_grad():
        for p in m.parameters():  # LayerNorm の初期値のままだと出力が似通うので、全体を少し揺らす
            p.add_(torch.randn_like(p) * 0.3)
    return m.eval()


def _crop(seed: int, w: int = 90, h: int = 30) -> np.ndarray:
    return (np.random.default_rng(seed).random((h, w, 3)) * 255).astype(np.uint8)


def test_preprocess_matches_ndl_convention():
    img = np.full((10, 20, 3), 127.5, np.float32).astype(np.uint8)
    x = ndl_preprocess(img, 64, 24)
    assert x.shape == (1, 3, 24, 64) and x.dtype == np.float32
    assert abs(float(x.mean())) < 0.01
    tall = np.zeros((50, 10, 3), np.uint8)
    assert ndl_preprocess(tall, 64, 24, rotate_tall=True).shape == (1, 3, 24, 64)


def test_normalize_width():
    assert normalize_width("１２，３４５－") == "12,345-"
    assert normalize_width("ＲＡ．ｂ") == "RA.b"
    assert normalize_width("ー") == "ー"  # 長音は触らない


def test_column_charset():
    assert column_charset(ColumnKind.DATE) == set("0123456789R.〃 ")
    assert column_charset("amount") == set("0123456789,-")
    assert column_charset(ColumnKind.DESCRIPTION) is None
    assert column_charset("title") is None  # 列以外の欄は制限なし


def test_forward_shape_and_vocab_mask():
    m = _tiny()
    x = torch.from_numpy(np.concatenate([ndl_preprocess(_crop(i), 64) for i in range(3)]))
    logits = m(x)
    assert logits.shape == (3, 7, len(CHARSET) + 1)
    mask = torch.zeros(len(CHARSET) + 1, dtype=torch.bool)
    mask[0] = True
    mask[[CHARSET.index("1") + 1, CHARSET.index("2") + 1]] = True
    ids = set(m(x, vocab_mask=mask).argmax(-1).flatten().tolist())
    assert ids <= {0, CHARSET.index("1") + 1, CHARSET.index("2") + 1}
    row_mask = mask.expand(3, -1)  # 行ごとのマスク
    assert torch.equal(m(x, vocab_mask=row_mask).argmax(-1), m(x, vocab_mask=mask).argmax(-1))


def test_batch_independence():
    m = _tiny()
    xs = [torch.from_numpy(ndl_preprocess(_crop(i), 64)) for i in range(4)]
    batch = m(torch.cat(xs))
    for i, x in enumerate(xs):
        assert torch.allclose(m(x)[0], batch[i], atol=1e-4)


def test_beam_matches_scoring_and_greedy():
    m = _tiny(seed=1)
    x = torch.from_numpy(np.concatenate([ndl_preprocess(_crop(i), 64) for i in range(3)]))
    mem = m.encode(x)
    toks, scores = m.beam_search(mem, 5)
    assert toks.shape[:2] == (3, 5)
    assert (scores[:, :-1] >= scores[:, 1:]).all()  # 降順
    # ビームの対数尤度 = 同じ系列を教師強制で採点した値
    for b in range(3):
        for k in range(5):
            ids = []
            for t in toks[b, k].tolist():
                if t == 0:
                    break
                ids.append(t)
            lp = m.score_logprob(mem[b : b + 1], [ids])[0, 0]
            assert math.isclose(float(lp), float(scores[b, k]), abs_tol=1e-3)
    # beam=1 は AR の貪欲デコード(反復修正なし)と同じ
    toks1, _ = m.beam_search(mem, 1)
    greedy = m(x, refine=False).argmax(-1)
    for b in range(3):
        n = int((greedy[b] == 0).int().argmax()) if (greedy[b] == 0).any() else greedy.shape[1]
        n = min(n, m.max_label_length)  # ビームは最大長で EOS を強制する
        assert toks1[b, 0, :n].tolist() == greedy[b, :n].tolist()


def test_score_with_vocab_mask_is_minus_inf_for_disallowed():
    m = _tiny()
    x = torch.from_numpy(ndl_preprocess(_crop(0), 64))
    mem = m.encode(x)
    mask = torch.zeros(len(CHARSET) + 1, dtype=torch.bool)
    mask[0] = True
    mask[CHARSET.index("1") + 1] = True
    ok = m.score_logprob(mem, [[CHARSET.index("1") + 1]], mask)[0, 0]
    bad = m.score_logprob(mem, [[CHARSET.index("2") + 1]], mask)[0, 0]
    assert math.isfinite(float(ok)) and float(bad) == float("-inf")


@pytest.fixture(scope="module")
def engine() -> ParseqEngine:
    return ParseqEngine.from_models({256: _tiny(256, 8), 384: _tiny(384, 10, seed=2)}, CHARSET)


def test_engine_recognize_candidates_are_consistent(engine):
    out = engine.recognize(_crop(3), "account", beam=5)
    assert out.width == 256 and len(out.candidates) >= 1
    best = out.best
    assert best.source == "greedy"
    for c in out.candidates:
        assert len(c.char_confidences) == c.n_tokens == len(c.tokens)
        assert all(0.0 <= p <= 1.0 for p in c.char_confidences) and 0.0 <= c.eos_prob <= 1.0
        assert math.isclose(
            math.exp(c.logprob), math.prod(c.char_confidences) * c.eos_prob, rel_tol=1e-3
        )
        # score() は認識時の対数尤度と同じ定義
        s = engine.score(_crop(3), c.text, "account")
        assert math.isclose(s.logprob, c.logprob, abs_tol=1e-3)
    assert len({c.text for c in out.candidates}) == len(out.candidates)  # 重複なし


def test_engine_column_mask_and_width(engine):
    crop = _crop(4)
    d = engine.recognize(crop, ColumnKind.DATE, beam=3)
    a = engine.recognize(crop, ColumnKind.AMOUNT, beam=3)
    assert all(set(c.text) <= set("0123456789R.〃 ") for c in d.candidates)
    assert all(set(c.text) <= set("0123456789,-") for c in a.candidates)
    assert engine.recognize(crop, ColumnKind.DESCRIPTION, beam=1).width == 384
    # マスクで許可されない文字を含む文字列の採点は -inf、文字集合外の文字も -inf
    assert engine.score(crop, "あ", ColumnKind.AMOUNT).logprob == float("-inf")
    assert engine.score(crop, "漢", None).logprob == float("-inf")


def test_engine_batch_equals_single(engine):
    crops = [_crop(i, w=60 + 10 * i) for i in range(5)]
    cols = ["date", "account", "description", "amount", "description"]
    batch = engine.recognize_many(crops, cols, beam=3, with_embedding=True)
    for c, col, b in zip(crops, cols, batch, strict=True):
        single = engine.recognize(c, col, beam=3, with_embedding=True)
        assert [x.text for x in b.candidates] == [x.text for x in single.candidates]
        assert np.allclose(b.embedding, single.embedding, atol=1e-4)
        assert math.isclose(float(np.linalg.norm(b.embedding)), 1.0, rel_tol=1e-4)


def test_engine_embeddings_are_normalized(engine):
    e = engine.embed_many([_crop(i) for i in range(3)], width=256)
    assert e.shape == (3, 32 * EMBED_BINS)
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-4)


def test_engine_refine_modes_are_selectable():
    with pytest.raises(ValueError):
        ParseqEngine(refine_mode="bogus")
    crop = _crop(9)
    models = {256: _tiny(256, 8), 384: _tiny(384, 10, seed=2)}
    texts = {}
    for mode in ("onnx", "cloze", "none"):
        eng = ParseqEngine.from_models(models, CHARSET)
        eng.refine_mode = mode
        texts[mode] = eng.recognize(crop, "account", beam=1).best.text
    # 反復修正なしは、自己回帰の貪欲デコード(beam=1 の探索)と同じ文字列になる
    eng = ParseqEngine.from_models(models, CHARSET)
    eng.refine_mode = "none"
    ar = eng.recognize(crop, "account", beam=1).best
    assert ar.text == texts["none"]
    assert all(isinstance(t, str) for t in texts.values())
