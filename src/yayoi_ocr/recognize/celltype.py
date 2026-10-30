"""セル種別分類器(2.1): blank / ditto(〃)/ arrow(矢印)/ wave(波線で省略)/ text。

入力は 2 つ:
- 画像: 罫線除去済みのセル全体を高さ 64 px・幅 448 px にしたグレー(`crops.classifier_input`)。
- 統計: Phase 1 のインク統計(ink_ratio・faint_only・ditto_likely・blank・インクの高さ/幅の比)と、列の種類・行の文脈
  (金額欄が空欄か)。レイアウト段階の行文脈ルール(金額欄が空欄の行は他列の薄いインクを空欄とみなす等)を、
  学習可能な特徴として取り込む。

小さな CNN(約 10 万パラメータ)で、合成の複数書き手データで事前学習する。ファインチューニングできるよう、
`CellTypeNet` の重みだけを `state_dict` として保存・読み込みする(学習・推論とも PyTorch)。
確率は温度スケーリングで較正する(`temperature`)。「迷ったら確認」のため、最大確率を確信度として使う。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from ..models import ColumnKind

CLASSES = ("blank", "ditto", "arrow", "wave", "text")
COLUMNS = (ColumnKind.DATE, ColumnKind.ACCOUNT, ColumnKind.DESCRIPTION, ColumnKind.AMOUNT)
N_STATS = 12


def kind_to_class(kind: str) -> int:
    """合成データの正解の書かれ方 → 分類のクラス番号(elided = wave)。"""
    return CLASSES.index({"elided": "wave"}.get(kind, kind))


def stats_vector(
    column: str,
    ink_ratio: float,
    flags: Sequence[str],
    faint_only: bool,
    ink_h_frac: float,
    ink_w_frac: float,
    amount_blank: bool,
) -> np.ndarray:
    """Phase 1 の統計 + 列 + 行の文脈 → 固定長の特徴(N_STATS)。"""
    v = np.zeros(N_STATS, np.float32)
    v[0] = min(1.0, ink_ratio * 8.0)  # 本体のインクの割合(0.12 で飽和)
    v[1] = float(faint_only)
    v[2] = float("ditto_likely" in flags)
    v[3] = float("blank" in flags)
    v[4] = float("overflow" in flags)
    v[5] = ink_h_frac
    v[6] = ink_w_frac
    v[7] = float(amount_blank)
    v[8 + [c.value for c in COLUMNS].index(column)] = 1.0
    return v


class CellTypeNet(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES), n_stats: int = N_STATS) -> None:
        super().__init__()

        def block(i: int, o: int, pool: tuple[int, int]) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
            )

        self.features = nn.Sequential(
            block(1, 16, (2, 2)),  # 32 x 224
            block(16, 32, (2, 2)),  # 16 x 112
            block(32, 64, (2, 2)),  # 8 x 56
            block(64, 96, (2, 2)),  # 4 x 28
        )
        self.pool_avg = nn.AdaptiveAvgPool2d((1, 7))  # 横方向の位置(左・中・右)を粗く残す
        self.pool_max = nn.AdaptiveMaxPool2d((1, 1))
        self.stats = nn.Sequential(nn.Linear(n_stats, 32), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Linear(96 * 7 + 96 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: Tensor, stats: Tensor) -> Tensor:
        """x: (N, 1, 64, 448) ink = 1 / paper = 0 の float。stats: (N, N_STATS)。logits (N, n_classes)。"""
        f = self.features(x)
        z = torch.cat(
            [self.pool_avg(f).flatten(1), self.pool_max(f).flatten(1), self.stats(stats)], dim=1
        )
        return self.head(z)


def to_tensor(clf_images: np.ndarray) -> Tensor:
    """(N, 64, 448) uint8(紙 = 255)→ (N, 1, 64, 448) float(インク = 1)。"""
    x = torch.from_numpy(clf_images).float().div_(255.0)
    return (1.0 - x).unsqueeze(1)


@dataclass
class CellTypePrediction:
    cls: str
    probs: dict[str, float]

    @property
    def confidence(self) -> float:
        return self.probs[self.cls]


class CellTypeClassifier:
    """学習済みの分類器(推論)。温度スケーリングで確率を較正する。"""

    def __init__(self, net: CellTypeNet, temperature: float = 1.0, device: str | None = None):
        self.device = torch.device(device or "cpu")
        self.net = net.eval().to(self.device)
        self.temperature = temperature

    @torch.inference_mode()
    def predict_proba(
        self, clf_images: np.ndarray, stats: np.ndarray, batch: int = 256
    ) -> np.ndarray:
        out = []
        for s in range(0, len(clf_images), batch):
            x = to_tensor(clf_images[s : s + batch]).to(self.device)
            st = torch.from_numpy(stats[s : s + batch]).to(self.device)
            logits = self.net(x, st) / self.temperature
            out.append(F.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(CLASSES)), np.float32)

    def predict(self, clf_images: np.ndarray, stats: np.ndarray) -> list[CellTypePrediction]:
        p = self.predict_proba(clf_images, stats)
        return [
            CellTypePrediction(
                CLASSES[int(row.argmax())], dict(zip(CLASSES, row.tolist(), strict=True))
            )
            for row in p
        ]

    # ---- 保存・読み込み(テンソルと基本型だけ。weights_only で読める)
    def save(self, path: str | Path, extra: dict | None = None) -> None:
        torch.save(
            {
                "format": 1,
                "classes": list(CLASSES),
                "temperature": float(self.temperature),
                "state_dict": self.net.state_dict(),
                "extra": json.dumps(extra or {}, ensure_ascii=False),
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> CellTypeClassifier:
        d = torch.load(Path(path), map_location="cpu", weights_only=True)
        if d.get("format") != 1 or list(d["classes"]) != list(CLASSES):
            raise ValueError(f"{path} は未対応の分類器ファイルです")
        net = CellTypeNet()
        net.load_state_dict(d["state_dict"])
        return cls(net, float(d["temperature"]), device)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, iters: int = 200) -> float:
    """検証データのロジットから、負の対数尤度を最小にする温度を求める(温度スケーリング)。"""
    lg = torch.from_numpy(logits).float()
    y = torch.from_numpy(labels).long()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())
