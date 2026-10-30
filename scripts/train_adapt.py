#!/usr/bin/env python3
"""帳票ドメインへの適応学習(合成の多書き手データ。幅 256 と 384 の両方)。

    python scripts/train_adapt.py --train DIR [DIR ...] --out-dir MODELS_DIR/adapted
        [--l2sp 0.1] [--epochs 3] [--n256 2500] [--n384 6000]

- 学習データ: 学習用の書き手・フォントのみ(評価用の書き手・未知フォント yusei・klee は使わない)。崩しの強さは 0.5〜1.0 を混ぜる
  (build_synth_cells.py --strength-range)。幅 256 = 日付・科目・金額(列ごとに n256 件)、幅 384 = 摘要(n384 件。一般語彙を含む)。
- 忘却の防止: 元の重みからのずれへの L2-SP(--l2sp)と、一般語彙の摘要(リハーサル)。
- 出力: MODELS_DIR/adapted/ に、幅 256・384 の重み(.pt)。素の重みは変更しない。評価は eval_recognize.py --models-dir で行う。
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from yayoi_ocr.config import assert_safe_output_dir, models_dir
from yayoi_ocr.recognize.grammar import norm_text
from yayoi_ocr.recognize.parseq.engine import MODEL_FILES
from yayoi_ocr.recognize.parseq.finetune import FinetuneConfig, finetune
from yayoi_ocr.recognize.parseq.weights import load_converted, save_converted
from yayoi_ocr.synth.dataset import list_pages, load_page


def collect(
    dirs: list[str], columns: list[str], n_per_col: int, seed: int
) -> list[tuple[np.ndarray, str]]:
    rng = np.random.default_rng(seed)
    pools: dict[str, list] = {c: [] for c in columns}
    for d in dirs:
        for p in list_pages(d):
            page = load_page(p)
            for i, c in enumerate(page.cells):
                if c.kind == "text" and c.column in pools and i in page.rec:
                    pools[c.column].append((page.rec[i], norm_text(c.raw)))
    out: list[tuple[np.ndarray, str]] = []
    for c in columns:
        idx = rng.permutation(len(pools[c]))[:n_per_col]
        out += [pools[c][i] for i in idx]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--widths", type=int, nargs="+", default=[256, 384])
    ap.add_argument("--n256", type=int, default=2500, help="幅 256: 列(日付・科目・金額)ごとの件数")
    ap.add_argument("--n384", type=int, default=6000, help="幅 384: 摘要の件数")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--l2sp", type=float, default=0.1)
    ap.add_argument("--frozen-blocks", type=int, default=8)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    out_dir = assert_safe_output_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = models_dir()
    for width in args.widths:
        model, charset, source = load_converted(src / MODEL_FILES[width])
        if width == 256:
            samples = collect(args.train, ["date", "account", "amount"], args.n256, args.seed)
        else:
            samples = collect(args.train, ["description"], args.n384, args.seed)
        print(f"[幅 {width}] 学習サンプル {len(samples)} 件", flush=True)
        cfg = FinetuneConfig(
            width=width,
            epochs=args.epochs,
            lr=args.lr,
            frozen_blocks=args.frozen_blocks,
            aug_strength=args.strength,
            l2sp=args.l2sp,
            seed=args.seed,
        )
        hist = finetune(model, charset, samples, cfg, log=lambda s: print(s, flush=True))
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        save_converted(
            out_dir / MODEL_FILES[width],
            sd,
            charset,
            width,
            {
                **source,
                "finetuned": f"synthetic domain adaptation: n={len(samples)} epochs={args.epochs} "
                f"l2sp={args.l2sp} lr={args.lr} frozen_blocks={args.frozen_blocks}",
                "final_loss": f"ar={hist['ar']:.4f} refine={hist['refine']:.4f}",
            },
        )
        print(f"saved {out_dir / MODEL_FILES[width]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
