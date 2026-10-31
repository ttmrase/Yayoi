"""PARSeq の追加学習(ドメイン適応・Phase 4 の学習ループ共通)。

- 学習するのはエンコーダの後ろ側のブロック・デコーダ・出力層(前段は凍結)。損失は自己回帰 + 反復修正。
- **忘却の防止**: 元の重みからのずれへの L2 ペナルティ(L2-SP: λ Σ ‖θ − θ0‖²)。さらに、呼び出し側が一般語彙の
  サンプル(リハーサル)を混ぜる。θ0 は学習開始時の重み(= ONNX 由来の素の重み)。
- 画像は強い拡張(`augment.augment_gray`)をかけて NDL と同じ前処理にかける。
- 学習結果の重みは呼び出し側が保存する(リポジトリの外)。
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .augment import augment_gray
from .model import PARSeq
from .preprocess import ndl_preprocess


@dataclass
class FinetuneConfig:
    width: int
    epochs: int = 3
    batch: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.01
    frozen_blocks: int = 8
    aug_strength: float = 1.0
    l2sp: float = 0.1  # 元の重みからのずれ(二乗和)への係数。0 で無効
    refine_mask: str = "onnx"
    seed: int = 0
    device: str = "cpu"  # "cuda" なら GPU で学習する(終わったら重みは CPU に戻す)。GPU の経路は試用版では未検証


def finetune(
    model: PARSeq,
    charset: str,
    samples: Sequence[tuple[np.ndarray, str]],
    cfg: FinetuneConfig,
    log: Callable[[str], None] = print,
) -> dict[str, float]:
    """samples: (認識用グレー画像, 正解文字列)。model を直接更新する。履歴(最終エポックの損失・ずれ)を返す。"""
    tok = {ch: i + 1 for i, ch in enumerate(charset)}
    data = [
        (img, text)
        for img, text in samples
        if 0 < len(text) <= model.max_label_length and all(ch in tok for ch in text)
    ]
    if not data:
        raise ValueError("学習できるサンプルがありません")
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    dev = torch.device(cfg.device)
    model.to(dev)
    model.refine_mask = cfg.refine_mask
    params = model.freeze_for_finetune(cfg.frozen_blocks)
    anchor = [p.detach().clone() for p in params]  # θ0(L2-SP の基準)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * math.ceil(len(data) / cfg.batch)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda t: min(1.0, (t + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, t / steps)))),
    )
    model.train()
    t0 = time.time()
    hist: dict[str, float] = {}
    for ep in range(cfg.epochs):
        perm = rng.permutation(len(data))
        tot_ar = tot_ref = tot_sp = 0.0
        n = 0
        for b in range(0, len(perm), cfg.batch):
            items = [data[i] for i in perm[b : b + cfg.batch]]
            x = np.concatenate(
                [
                    ndl_preprocess(augment_gray(img, rng, cfg.aug_strength), cfg.width)
                    for img, _ in items
                ]
            )
            ids = [[tok[ch] for ch in text] for _, text in items]
            ar, ref, tgt = model.training_logits(torch.from_numpy(x).to(dev), ids)
            v = ar.shape[-1]
            l_ar = F.cross_entropy(ar.reshape(-1, v), tgt.reshape(-1), ignore_index=model.pad_id)
            l_ref = F.cross_entropy(ref.reshape(-1, v), tgt.reshape(-1), ignore_index=model.pad_id)
            sp = sum(((p - a) ** 2).sum() for p, a in zip(params, anchor, strict=True))
            loss = l_ar + l_ref + cfg.l2sp * sp
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tot_ar += l_ar.item() * len(items)
            tot_ref += l_ref.item() * len(items)
            tot_sp += float(sp.detach()) * len(items)
            n += len(items)
        hist = {"ar": tot_ar / n, "refine": tot_ref / n, "drift_sq": tot_sp / n}
        log(
            f"[幅 {cfg.width}] epoch {ep + 1}/{cfg.epochs} ar {hist['ar']:.4f} refine {hist['refine']:.4f} "
            f"元の重みからのずれ(二乗和) {hist['drift_sq']:.3f} ({time.time() - t0:.0f}s)"
        )
    model.to("cpu").eval()
    return hist
