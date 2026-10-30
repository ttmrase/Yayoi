#!/usr/bin/env python3
"""セル種別分類器(blank / ditto / arrow / wave / text)を合成データで学習する。

    python scripts/train_celltype.py --train DIR [DIR ...] --calib DIR --out PATH [--epochs 8]

- データは scripts/build_synth_cells.py が作る(書き手 ID・フォントを学習用/較正用/評価用で分けておく)。
- 較正用データで温度スケーリングして確率を較正し、較正後の精度とクラスごとの再現率を表示する。
- 出力(重み)は $YAYOI_PRIVATE_DATA/models/ など **リポジトリの外**に置く。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

from yayoi_ocr.config import assert_safe_output_dir
from yayoi_ocr.recognize.celltype import (
    CLASSES,
    CellTypeClassifier,
    CellTypeNet,
    fit_temperature,
    kind_to_class,
    stats_vector,
    to_tensor,
)
from yayoi_ocr.synth.dataset import list_pages, load_page


def load_dir(dirs: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ディレクトリ群 → (画像 (N,64,448) uint8, 統計 (N,12), ラベル (N,))。"""
    xs, ss, ys = [], [], []
    for d in dirs:
        for p in list_pages(d):
            page = load_page(p)
            for i, c in enumerate(page.cells):
                amount_blank = page.cell(c.row, 3).blank
                xs.append(page.clf[i])
                ss.append(
                    stats_vector(
                        c.column,
                        c.ink_ratio,
                        c.flags,
                        c.faint_only,
                        c.ink_h_frac,
                        c.ink_w_frac,
                        amount_blank,
                    )
                )
                ys.append(kind_to_class(c.kind))
    return np.stack(xs), np.stack(ss), np.array(ys, np.int64)


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """x: (N,1,64,448) ink=1。水平・垂直のずれ、濃さ・コントラストの揺れ、雑音。"""
    n = x.shape[0]
    out = torch.empty_like(x)
    dx = torch.randint(-12, 13, (n,), generator=gen)
    dy = torch.randint(-4, 5, (n,), generator=gen)
    for i in range(n):
        out[i] = torch.roll(x[i], shifts=(int(dy[i]), int(dx[i])), dims=(1, 2))
    gain = 0.75 + 0.5 * torch.rand((n, 1, 1, 1), generator=gen)
    out = out * gain
    out = out + 0.02 * torch.randn(out.shape, generator=gen)
    return out.clamp_(0.0, 1.0)


def evaluate(clf_net: CellTypeNet, x: np.ndarray, s: np.ndarray, y: np.ndarray, batch: int = 256):
    clf_net.eval()
    logits = []
    with torch.inference_mode():
        for b in range(0, len(x), batch):
            logits.append(clf_net(to_tensor(x[b : b + batch]), torch.from_numpy(s[b : b + batch])))
    lg = torch.cat(logits).numpy()
    return lg, float((lg.argmax(1) == y).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--calib", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    assert_safe_output_dir(out.parent)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    x, s, y = load_dir(args.train)
    xc, sc, yc = load_dir(args.calib)
    print(
        f"train {len(y)} cells / calib {len(yc)} cells ({time.time() - t0:.0f}s) "
        f"class counts: {dict(zip(CLASSES, np.bincount(y, minlength=len(CLASSES)).tolist(), strict=True))}"
    )
    net = CellTypeNet()
    print("params:", sum(p.numel() for p in net.parameters()))
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float64)
    w = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32)
    w = w / w.mean()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(y) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(y), generator=gen).numpy()
        tot, n = 0.0, 0
        for b in range(0, len(perm), args.batch):
            idx = np.sort(perm[b : b + args.batch])
            xb = augment(to_tensor(x[idx]), gen)
            sb = torch.from_numpy(s[idx])
            yb = torch.from_numpy(y[idx])
            loss = F.cross_entropy(net(xb, sb), yb, weight=w, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)
            n += len(idx)
        _, acc = evaluate(net, xc, sc, yc)
        print(
            f"epoch {ep + 1}/{args.epochs} loss {tot / n:.4f} calib acc {acc:.4f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    logits, acc = evaluate(net, xc, sc, yc)
    temp = fit_temperature(logits, yc)
    clf = CellTypeClassifier(net, temp)
    clf.save(out, {"train": args.train, "calib": args.calib, "epochs": args.epochs})
    pred = logits.argmax(1)
    print(f"saved {out}  temperature {temp:.3f}  calib acc {acc:.4f}")
    print("per-class recall / precision (calib):")
    for k, name in enumerate(CLASSES):
        tp = int(((pred == k) & (yc == k)).sum())
        rec = tp / max(1, int((yc == k).sum()))
        prec = tp / max(1, int((pred == k).sum()))
        print(f"  {name:6s} n={int((yc == k).sum()):6d} recall={rec:.4f} precision={prec:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
