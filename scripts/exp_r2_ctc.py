#!/usr/bin/env python3
"""Phase 4 方法検証 (a): 数字専用 CRNN-CTC(R2)を、書き手のデータで全層学習したときの精度の曲線。

    python scripts/exp_r2_ctc.py --data DIR --out EVAL_DIR [--writers W ...] [--levels 20 50 100 200]

各書き手について、先頭 N ページ(N ∈ --levels)の日付・金額のセル(書かれた通りの文字列を正解とする)で R2 を**最初から**学習し、
続く --eval-pages ページ(未使用)の日付・金額のセルを認識して JSON Lines に保存する(`r2_<書き手>.jsonl`)。学習の量は --steps(バッチ数)で固定し、
N を増やしてもデータの多様さだけが増える(R1 の天井推定の Arm A と同じ考え方)。R1 との比較・一致の分析は `scripts/analyze_r2.py`。
合成データのみ。出力はリポジトリの外に置くこと。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from yayoi_ocr.config import assert_safe_output_dir
from yayoi_ocr.recognize.crnn import (
    COLUMN_CHARS,
    CRNN,
    HEIGHT,
    collate,
    encode,
    greedy_decode,
    prepare,
)
from yayoi_ocr.synth.dataset import list_pages, load_page

COLUMNS = ("date", "amount")


def cells_of(page) -> list[tuple[np.ndarray, str, str, int, int]]:
    out = []
    for c in page.cells:
        if c.column in COLUMNS and c.kind == "text" and c.raw:
            img = page.rec_image(c.row, c.col)
            if img is not None:
                out.append((img, c.raw, c.column, c.row, c.col))
    return out


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape
    ang = rng.normal(0, 1.5)
    shear = rng.normal(0, 0.06)
    sx = rng.uniform(0.9, 1.15)
    sy = rng.uniform(0.9, 1.1)
    m = np.array(
        [
            [sx * np.cos(np.radians(ang)), shear + np.sin(np.radians(ang)), 0],
            [-np.sin(np.radians(ang)), sy, 0],
        ],
        np.float32,
    )
    m[0, 2] = (w - m[0, 0] * w - m[0, 1] * h) / 2 + rng.normal(0, 1.5)
    m[1, 2] = (h - m[1, 0] * w - m[1, 1] * h) / 2 + rng.normal(0, 1.5)
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.4:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    f = out.astype(np.float32)
    f = (f - 128) * rng.uniform(0.8, 1.2) + 128 + rng.normal(0, 8)
    f += rng.normal(0, rng.uniform(0, 5), f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


def train(samples, steps: int, batch: int, lr: float, height: int, seed: int) -> CRNN:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = CRNN(height)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    model.train()
    for step in range(steps):
        idx = rng.integers(0, len(samples), batch)
        imgs = [prepare(augment(samples[i][0], rng), height) for i in idx]
        x, widths = collate(imgs)
        targets = [encode(samples[i][1]) for i in idx]
        tgt = torch.tensor([t for ts in targets for t in ts], dtype=torch.long)
        tlen = torch.tensor([len(t) for t in targets], dtype=torch.long)
        logp = model(x)
        loss = ctc(logp, tgt, CRNN.out_lengths(widths), tlen)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        if (step + 1) % 500 == 0:
            print(f"    step {step + 1}/{steps} loss {loss.item():.3f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def recognize(model: CRNN, cells, height: int) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for i in range(0, len(cells), 64):
        chunk = cells[i : i + 64]
        x, widths = collate([prepare(c[0], height) for c in chunk])
        lp = model(x)
        lens = CRNN.out_lengths(widths)
        for k, c in enumerate(chunk):  # 列ごとに許す字を制限して復号する
            out.append(greedy_decode(lp[:, k : k + 1], lens[k : k + 1], COLUMN_CHARS[c[2]])[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--writers", type=int, nargs="*", default=[])
    ap.add_argument("--levels", type=int, nargs="+", default=[20, 50, 100, 200])
    ap.add_argument("--train-pages", type=int, default=200)
    ap.add_argument("--eval-pages", type=int, default=12)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--height", type=int, default=HEIGHT)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out_dir = assert_safe_output_dir(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths_by_writer: dict[int, list[Path]] = defaultdict(list)
    for path in list_pages(args.data):
        with np.load(path, allow_pickle=False) as z:
            paths_by_writer[int(json.loads(str(z["meta"]))["page"]["writer_id"])].append(path)
    for w in args.writers or sorted(paths_by_writer):
        pages = sorted((load_page(p) for p in paths_by_writer[w]), key=lambda p: p.meta["seed"])
        pool, evalp = (
            pages[: args.train_pages],
            pages[args.train_pages : args.train_pages + args.eval_pages],
        )
        if len(evalp) < args.eval_pages:
            print(f"書き手 {w}: ページ不足でスキップ")
            continue
        eval_cells = [(cell, pg.meta["seed"]) for pg in evalp for cell in cells_of(pg)]
        with open(out_dir / f"r2_{w}.jsonl", "w", encoding="utf-8") as sink:
            for n in args.levels:
                samples = [c for pg in pool[:n] for c in cells_of(pg)]
                t0 = time.time()
                print(
                    f"書き手 {w} N={n}: 学習 {len(samples)} セル, {args.steps} バッチ", flush=True
                )
                model = train(
                    samples, args.steps, args.batch, args.lr, args.height, seed=w % 1000 + n
                )
                secs = time.time() - t0
                preds = recognize(model, [c for c, _ in eval_cells], args.height)
                for ((_img, raw, column, row, col), page), (pred, conf) in zip(
                    eval_cells, preds, strict=True
                ):
                    sink.write(
                        json.dumps(
                            {
                                "writer": w,
                                "N": n,
                                "page": page,
                                "row": row,
                                "col": col,
                                "column": column,
                                "gt_raw": raw,
                                "pred_raw": pred,
                                "conf": conf,
                                "train_cells": len(samples),
                                "train_seconds": secs,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                sink.flush()
                print(f"  N={n}: 学習 {secs:.0f} 秒", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
