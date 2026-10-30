#!/usr/bin/env python3
"""セル種別分類器の評価(混同行列・クラス別の再現率/適合率・較正誤差・確信度が低い誤りの割合)。

python scripts/eval_celltype.py --classifier PATH --data DIR [DIR ...]
"""

from __future__ import annotations

import argparse
import itertools
import sys

import numpy as np

from yayoi_ocr.recognize.celltype import CLASSES, CellTypeClassifier, kind_to_class
from yayoi_ocr.recognize.inputs import from_page_data
from yayoi_ocr.synth.dataset import list_pages, load_page


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--min-conf", type=float, default=0.7, help="これ未満の確率は確認に回す")
    args = ap.parse_args()
    clf = CellTypeClassifier.load(args.classifier)
    for d in args.data:
        ys, ps = [], []
        for p in list_pages(d):
            page = load_page(p)
            inp = from_page_data(page)
            x = np.stack([c.clf for c in inp.cells])
            s = np.stack([c.stats for c in inp.cells])
            ps.append(clf.predict_proba(x, s))
            ys += [kind_to_class(c.kind) for c in page.cells]
        prob = np.concatenate(ps)
        y = np.array(ys)
        pred = prob.argmax(1)
        conf = prob.max(1)
        k = len(CLASSES)
        cm = np.zeros((k, k), int)
        for a, b in zip(y, pred, strict=True):
            cm[a, b] += 1
        print(f"\n===== {d}  cells={len(y)} accuracy={np.mean(pred == y):.4f} =====")
        print("混同行列(行 = 正解, 列 = 判定):", " ".join(f"{c:>7s}" for c in CLASSES))
        for i, c in enumerate(CLASSES):
            print(f"  {c:7s}", " ".join(f"{cm[i, j]:7d}" for j in range(k)))
        for i, c in enumerate(CLASSES):
            tp = cm[i, i]
            print(
                f"  {c:7s} 再現率 {tp / max(1, cm[i].sum()):.4f}  適合率 {tp / max(1, cm[:, i].sum()):.4f}"
            )
        wrong = pred != y
        low = conf < args.min_conf
        print(
            f"誤り {int(wrong.sum())} 件のうち確率 < {args.min_conf} (確認に回る): "
            f"{int((wrong & low).sum())} 件 → 黙った誤り {int((wrong & ~low).sum())} 件 "
            f"({(wrong & ~low).sum() / len(y):.4%})"
        )
        # 較正誤差(ECE, 10 区間)
        bins = np.linspace(0, 1, 11)
        ece = 0.0
        for lo, hi in itertools.pairwise(bins):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
        print(f"ECE {ece:.4f}")
        silent = {}
        for a, b in zip(y[wrong & ~low], pred[wrong & ~low], strict=True):
            silent[f"{CLASSES[a]}->{CLASSES[b]}"] = silent.get(f"{CLASSES[a]}->{CLASSES[b]}", 0) + 1
        if silent:
            print("黙った誤りの内訳:", dict(sorted(silent.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
