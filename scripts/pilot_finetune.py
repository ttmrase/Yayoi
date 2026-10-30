#!/usr/bin/env python3
"""帳票ドメインへの軽いファインチューニングの予備実験(報告2 の「改善の見込み」用)。

    python scripts/pilot_finetune.py --train DIR [DIR ...] --out-dir MODELS_DIR [--width 256]
        [--columns date account amount] [--n-per-col 2000] [--epochs 3] [--lr 1e-4] [--frozen-blocks 8]

- 学習データ: 学習用の書き手・フォントだけ(評価用の書き手・未知フォントは使わない)。合成のみ。
- 学習対象: エンコーダの後ろ側のブロック・デコーダ・出力層。低い学習率、強い拡張。損失は自己回帰 + 反復修正。
- 出力: MODELS_DIR に微調整後の重み(.pt)。もう一方の幅は元の重みをコピーする。評価は
  scripts/eval_recognize.py --models-dir MODELS_DIR(未使用の書き手・未知フォント)で行う。
- 実行は開発用(PyTorch)。重みはリポジトリの外に置く。
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

from yayoi_ocr.config import assert_safe_output_dir, models_dir
from yayoi_ocr.recognize.grammar import norm_text
from yayoi_ocr.recognize.parseq.augment import augment_gray
from yayoi_ocr.recognize.parseq.engine import MODEL_FILES
from yayoi_ocr.recognize.parseq.preprocess import ndl_preprocess
from yayoi_ocr.recognize.parseq.weights import load_converted, save_converted
from yayoi_ocr.synth.dataset import list_pages, load_page


def collect(dirs: list[str], columns: list[str], n_per_col: int, seed: int):
    """学習データ: (画像, 正解文字列(空白除去), 列)。列ごとに n_per_col 件(ページをまたいで抽出)。"""
    rng = np.random.default_rng(seed)
    pools: dict[str, list] = {c: [] for c in columns}
    for d in dirs:
        for p in list_pages(d):
            page = load_page(p)
            for i, c in enumerate(page.cells):
                if c.kind == "text" and c.column in pools and i in page.rec:
                    pools[c.column].append((page.rec[i], norm_text(c.raw), c.column))
    out = []
    for c in columns:
        pool = pools[c]
        idx = rng.permutation(len(pool))[:n_per_col]
        out += [pool[i] for i in idx]
    rng.shuffle(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--columns", nargs="+", default=["date", "account", "amount"])
    ap.add_argument("--n-per-col", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--frozen-blocks", type=int, default=8)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--refine-mask", choices=["onnx", "cloze"], default="onnx")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    out_dir = assert_safe_output_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = models_dir()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model, charset, source = load_converted(src / MODEL_FILES[args.width])
    model.refine_mask = args.refine_mask
    tok = {ch: i + 1 for i, ch in enumerate(charset)}
    data = collect(args.train, args.columns, args.n_per_col, args.seed)
    data = [
        d for d in data if all(ch in tok for ch in d[1]) and 0 < len(d[1]) <= model.max_label_length
    ]
    print(f"学習サンプル {len(data)} 件 列={args.columns} 幅={args.width}", flush=True)

    params = model.freeze_for_finetune(args.frozen_blocks)
    print("学習するパラメータ数", sum(p.numel() for p in params), flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(data) / args.batch)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda t: min(1.0, (t + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, t / steps)))),
    )
    model.train()
    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        perm = rng.permutation(len(data))
        tot_ar = tot_ref = 0.0
        n = 0
        for b in range(0, len(perm), args.batch):
            items = [data[i] for i in perm[b : b + args.batch]]
            x = np.concatenate(
                [
                    ndl_preprocess(augment_gray(img, rng, args.strength), args.width)
                    for img, _, _ in items
                ]
            )
            ids = [[tok[ch] for ch in text] for _, text, _ in items]
            ar, ref, tgt = model.training_logits(torch.from_numpy(x), ids)
            v = ar.shape[-1]
            l_ar = F.cross_entropy(ar.reshape(-1, v), tgt.reshape(-1), ignore_index=model.pad_id)
            l_ref = F.cross_entropy(ref.reshape(-1, v), tgt.reshape(-1), ignore_index=model.pad_id)
            loss = l_ar + l_ref
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            tot_ar += l_ar.item() * len(items)
            tot_ref += l_ref.item() * len(items)
            n += len(items)
            if step % 20 == 0:
                print(
                    f"  step {step}/{steps} ar {tot_ar / n:.4f} refine {tot_ref / n:.4f} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
        print(
            f"epoch {ep + 1}/{args.epochs} ar {tot_ar / n:.4f} refine {tot_ref / n:.4f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    model.eval()
    # 出力: 微調整した幅は新しい重み、もう一方の幅は元の重みをコピー
    for w, fn in MODEL_FILES.items():
        if w != args.width:
            shutil.copy(src / fn, out_dir / fn)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    save_converted(
        out_dir / MODEL_FILES[args.width],
        sd,
        charset,
        args.width,
        {
            **source,
            "finetuned": f"synthetic pilot: columns={args.columns} n={len(data)} epochs={args.epochs}",
        },
    )
    print("saved", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
