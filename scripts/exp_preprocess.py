#!/usr/bin/env python3
"""R1 の入力前処理の切り分け実験(合成データ・種別は正解を与える)。

    python scripts/exp_preprocess.py --data DIR [DIR ...] [--limit 25] [--min-aspects 3 5 8]

比較する前処理(認識用グレー画像に適用してから、エンジンが (幅, 24) にリサイズする):
- stretch: NDLOCR-Lite と同じ(縦横比を無視して幅いっぱいに引き伸ばす)。現在の既定。
- minasp-r: 幅が「高さ x r」に満たないとき、右側を背景(紙)で埋めて幅を高さ x r にしてからリサイズ(最小縦横比 r)。
- fixed-aspect: 縦横比を保って高さ 24 にし、右側を背景で幅いっぱいまで埋める(長い文字列は縮める)。
列ごとの値の正解率(text セルのみ)と、要確認なし(見逃し)の件数を、暫定の確信度のまま比較する。
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np
import torch

from yayoi_ocr.recognize.evaluate import judge, summarize
from yayoi_ocr.recognize.inputs import from_page_data
from yayoi_ocr.recognize.page import JobContext, PageRecognizer
from yayoi_ocr.recognize.parseq.engine import ParseqEngine
from yayoi_ocr.synth.content import ACCOUNTS, TITLES
from yayoi_ocr.synth.dataset import list_pages, load_page

WIDTH = {"date": 256, "account": 256, "amount": 256, "description": 384}


def min_aspect(r: float):
    def f(g: np.ndarray, col: str) -> np.ndarray:
        h, w = g.shape
        need = round(h * r)
        if w >= need:
            return g
        return cv2.copyMakeBorder(g, 0, 0, 0, need - w, cv2.BORDER_CONSTANT, value=255)

    return f


def fixed_aspect(g: np.ndarray, col: str) -> np.ndarray:
    h, w = g.shape
    wt = WIDTH[col]
    nw = max(1, round(w * 24 / h))
    r = cv2.resize(g, (min(nw, wt), 24), interpolation=cv2.INTER_AREA)
    out = np.full((24, wt), 255, np.uint8)
    out[:, : r.shape[1]] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--min-aspects", type=float, nargs="*", default=[3, 5, 8])
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    variants = {"stretch(NDL)": lambda g, c: g}
    for r in args.min_aspects:
        variants[f"minasp-{r:g}"] = min_aspect(r)
    variants["fixed-aspect"] = fixed_aspect
    eng = ParseqEngine()
    rec = PageRecognizer(eng)
    ctx = JobContext(accounts=list(ACCOUNTS), titles=list(TITLES))
    for d in args.data:
        pages = [load_page(p) for p in list_pages(d)[: args.limit]]
        print(f"\n===== {d} ({len(pages)} ページ) text セルの値の正解率 / 見逃し数 =====")
        for name, fn in variants.items():
            rows = []
            for pg in pages:
                inp = from_page_data(pg)
                for c in inp.cells:
                    if c.rec is not None:
                        c.rec = fn(c.rec, c.column)
                res = rec.recognize_page(inp, ctx, [c.kind for c in pg.cells])
                rows += [judge(r, c, False) for r, c in zip(res.cells, pg.cells, strict=True)]
            parts = []
            for col in WIDTH:
                s = summarize([j for j in rows if j.column == col and j.gt_kind == "text"])["all"]
                parts.append(f"{col}: {s['accuracy']:.3f}/{int(s['miss'])}")
            print(f"{name:14s} " + "  ".join(parts), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
