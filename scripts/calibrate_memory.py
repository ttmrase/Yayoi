#!/usr/bin/env python3
"""見覚えメモリの類似度しきい値の較正(合成データ・エンコーダの埋め込みだけを使うので高速)。

    python scripts/calibrate_memory.py --data DIR [--max-writers 6] [--k 5]

同じ書き手の連続ページを順に処理し、各 text セルについて、それまでに確定したセルから最近傍を引き、
(類似度, 最近傍の確定テキストが正解と一致したか)を列ごとに集める。類似度のしきい値ごとに、
最近傍(上位 k の票の最多)の適合率と被覆率(しきい値以上の近傍があるセルの割合)を表示する。
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np
import torch

from yayoi_ocr.recognize.memory import RecognitionMemory
from yayoi_ocr.recognize.parseq.engine import ParseqEngine
from yayoi_ocr.synth.dataset import list_pages, load_page

THRESHOLDS = (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-writers", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    eng = ParseqEngine()
    pages = [load_page(p) for p in list_pages(args.data)]
    by_writer: dict[int, list] = defaultdict(list)
    for pg in pages:
        by_writer[pg.meta["writer_id"]].append(pg)
    writers = sorted(by_writer)[: args.max_writers or None]
    rec: dict[str, list[tuple[float, bool, int]]] = defaultdict(
        list
    )  # 列 → (類似度, 一致, 蓄積件数)
    for w in writers:
        mem = RecognitionMemory(k=args.k, sim_threshold=0.0)
        for pg in sorted(by_writer[w], key=lambda p: p.meta["seed"]):
            idx = [i for i, c in enumerate(pg.cells) if c.kind == "text" and i in pg.rec]
            cols = [pg.cells[i].column for i in idx]
            embs = np.zeros((len(idx), 192), np.float32)
            for width in (256, 384):
                sel = [j for j, c in enumerate(cols) if eng.default_width(c) == width]
                if sel:
                    crops = [np.stack([pg.rec[idx[j]]] * 3, axis=-1) for j in sel]
                    embs[sel] = eng.embed_many(crops, width)
            for j, i in enumerate(idx):
                c = pg.cells[i]
                if mem.size(c.column):
                    nb = mem.query(c.column, embs[j])
                    votes: dict[str, float] = defaultdict(float)
                    for n in nb:
                        votes[n.text] += n.sim
                    top = max(votes, key=lambda t: votes[t])
                    rec[c.column].append((nb[0].sim, top == c.raw, mem.size(c.column)))
            for j, i in enumerate(idx):  # ページの確定後に追加(同じページの内容は参照しない)
                mem.add(pg.cells[i].column, embs[j], pg.cells[i].raw)
    print("最近傍(上位 k の類似度つき多数決)の適合率 / 被覆率(列 x 類似度のしきい値)")
    print(f"{'列':12s} {'n':>6s} " + " ".join(f"{t:>11.2f}" for t in THRESHOLDS))
    for col in ("date", "account", "description", "amount"):
        r = rec[col]
        if not r:
            continue
        sims = np.array([x[0] for x in r])
        ok = np.array([x[1] for x in r])
        cells = []
        for t in THRESHOLDS:
            m = sims >= t
            cells.append(f"{ok[m].mean() if m.any() else float('nan'):.2f}/{m.mean():.2f}")
        print(f"{col:12s} {len(r):6d} " + " ".join(f"{c:>11s}" for c in cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
