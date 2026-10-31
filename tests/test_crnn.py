"""数字専用 CRNN-CTC(R2)の単体テスト(torch が無い環境では飛ばす)。"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import cv2
import numpy as np
import torch
from torch import nn

from yayoi_ocr.recognize.crnn import (
    ALPHABET,
    CRNN,
    N_CLASSES,
    collate,
    encode,
    greedy_decode,
    prepare,
)


def render(text: str, rng: np.random.Generator) -> np.ndarray:
    img = np.full((60, 40 + 26 * len(text)), 255, np.uint8)
    cv2.putText(img, text, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 20, 2, cv2.LINE_AA)
    return np.clip(img.astype(np.int16) + rng.integers(-6, 6, img.shape), 0, 255).astype(np.uint8)


def test_prepare_keeps_the_aspect_ratio_and_inverts():
    img = np.full((80, 400), 255, np.uint8)
    img[:, :200] = 0
    a = prepare(img, 40)
    assert (
        a.shape == (40, 200) and a[:, :10].mean() > 0.9 and a[:, -10:].mean() < 0.1
    )  # インク = 1、紙 = 0


def test_collate_pads_to_the_widest_and_lengths_follow_the_width():
    x, w = collate([np.zeros((40, 64), np.float32), np.zeros((40, 100), np.float32)])
    assert x.shape == (2, 1, 40, 100) and w.tolist() == [64, 100]
    assert CRNN.out_lengths(w).tolist() == [16, 25]
    assert CRNN()(x).shape == (25, 2, N_CLASSES)


def test_encode_ignores_characters_outside_the_alphabet():
    assert encode("R8 7.1x") == [ALPHABET.index(c) + 1 for c in "R8 7.1"]


def test_a_tiny_run_learns_to_read_digits():
    torch.set_num_threads(
        1
    )  # 他の計算と同時に走るとき、スレッドの取り合いで極端に遅くなる(認識ワーカーと同じ理由)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    texts = [
        "".join(rng.choice(list("0123456789"), size=int(rng.integers(2, 5)))) for _ in range(160)
    ]
    imgs = [prepare(render(t, rng)) for t in texts]
    model = CRNN()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    first = last = 0.0
    for _ in range(90):
        idx = rng.integers(0, len(texts), 12)
        x, w = collate([imgs[i] for i in idx])
        tg = [encode(texts[i]) for i in idx]
        loss = ctc(
            model(x),
            torch.tensor([t for ts in tg for t in ts]),
            CRNN.out_lengths(w),
            torch.tensor([len(t) for t in tg]),
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or loss.item()
        last = loss.item()
    assert last < 0.7 * first  # 学習が進む(損失が下がる)
    # 読めるようになるまでには数百バッチ以上かかる(精度は scripts/exp_r2_ctc.py の実験で測る)。ここでは復号が動くことだけを見る
    model.eval()
    with torch.no_grad():
        x, w = collate(imgs[:8])
        out = greedy_decode(model(x), CRNN.out_lengths(w), "0123456789")
    assert len(out) == 8 and all(set(s) <= set("0123456789") for s, _ in out)


def test_decode_respects_the_allowed_characters():
    torch.manual_seed(0)
    lp = torch.randn(30, 1, N_CLASSES).log_softmax(-1)
    s, _ = greedy_decode(lp, torch.tensor([30]), "0123456789")[0]
    assert set(s) <= set("0123456789")
