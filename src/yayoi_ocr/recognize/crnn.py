"""数字専用の小型 CRNN-CTC(R2)。日付・金額の列を、PARSeq(R1)とは別系統の誤りで読むための第 2 の認識器。

- 入力: 高さ `HEIGHT`(32〜48)に縮めたグレー画像(縦横比を保つ。幅は可変、CTC なので文字数に合わせて伸びる)。
- 構成: 畳み込み 5 段(幅は 1/4、高さは 1 まで縮める)→ 双方向 LSTM 2 層 → 文字ごとの確率(CTC、blank = 0)。約 100 万パラメータ。
- 文字集合は日付・金額に出る字だけ(`ALPHABET`): 数字・「.」「,」「-」「R」と空白。列ごとの制限は復号時のマスクで行う。
- 追加学習ではなく、書き手のデータで**全層を学習する**ことを想定する(`scripts/exp_r2_ctc.py` の方法検証で、R1 との誤りの独立性と精度を測る)。
- 推論は CPU で 1 セルあたり数 ms。重みは `torch.save(state_dict)` のテンソルのみ(pickle の任意オブジェクトを含めない)。
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn

ALPHABET = " .,-R0123456789"  # blank は 0 番(CTC)。文字は 1 番から
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(ALPHABET)}
N_CLASSES = len(ALPHABET) + 1
COLUMN_CHARS = {
    "date": " .R0123456789",
    "amount": ",-0123456789",
}
HEIGHT = 40
MIN_W, MAX_W = 32, 400


def prepare(img: np.ndarray, height: int = HEIGHT) -> np.ndarray:
    """グレー画像(紙 = 明るい)→ 高さ `height` に縮めた float32 [0, 1](インク = 1、紙 = 0)。"""
    h, w = img.shape[:2]
    nw = int(np.clip(round(w * height / max(h, 1)), MIN_W, MAX_W))
    small = cv2.resize(img, (nw, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return 1.0 - small


def collate(imgs: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """幅の違う画像を右に 0(紙)でそろえる。返り値: (B, 1, H, Wmax)、各画像の幅。"""
    widths = torch.tensor([a.shape[1] for a in imgs])
    out = np.zeros((len(imgs), 1, imgs[0].shape[0], int(widths.max())), np.float32)
    for i, a in enumerate(imgs):
        out[i, 0, :, : a.shape[1]] = a
    return torch.from_numpy(out), widths


def encode(text: str) -> list[int]:
    return [CHAR_TO_ID[c] for c in text if c in CHAR_TO_ID]


def _block(cin: int, cout: int, pool: tuple[int, int] | None) -> list[nn.Module]:
    layers: list[nn.Module] = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if pool is not None:
        layers.append(nn.MaxPool2d(pool))
    return layers


class CRNN(nn.Module):
    def __init__(self, height: int = HEIGHT, hidden: int = 128) -> None:
        super().__init__()
        pools = [(2, 2), (2, 2), (2, 1), (2, 1), (2, 1)]
        chans = [1, 32, 64, 128, 128, 192]
        layers: list[nn.Module] = []
        h = height
        for i, p in enumerate(pools):
            layers += _block(chans[i], chans[i + 1], p)
            h //= p[0]
        self.cnn = nn.Sequential(*layers)
        self.feat_h = max(h, 1)
        self.rnn = nn.LSTM(192 * self.feat_h, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(2 * hidden, N_CLASSES)

    @staticmethod
    def out_lengths(widths: torch.Tensor) -> torch.Tensor:
        return torch.clamp(widths // 4, min=1)  # 幅は (2, 2) の 2 段で 1/4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) → 対数確率 (T, B, クラス数)(CTCLoss の形)。"""
        f = self.cnn(x)  # (B, C, h', W')
        b, c, h, w = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, w, c * h)
        y, _ = self.rnn(f)
        return self.fc(y).log_softmax(-1).permute(1, 0, 2)


def greedy_decode(logp: torch.Tensor, lengths: torch.Tensor, allowed: str | None = None) -> list[tuple[str, float]]:
    """貪欲な CTC 復号。`allowed` の文字だけを許す(それ以外の確率は無かったことにする)。返り値: (文字列, 経路の確率の目安)。"""
    lp = logp.clone()
    if allowed is not None:
        mask = torch.full((N_CLASSES,), float("-inf"))
        mask[0] = 0.0
        for c in allowed:
            mask[CHAR_TO_ID[c]] = 0.0
        lp = lp + mask
        lp = lp - torch.logsumexp(lp, dim=-1, keepdim=True)
    best = lp.argmax(-1)  # (T, B)
    top = lp.max(-1).values
    out = []
    for b in range(lp.shape[1]):
        t = int(lengths[b])
        path = best[:t, b].tolist()
        chars, prev = [], 0
        for k in path:
            if k != prev and k != 0:
                chars.append(ALPHABET[k - 1])
            prev = k
        out.append(("".join(chars), float(top[:t, b].sum())))
    return out
