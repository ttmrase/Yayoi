#!/usr/bin/env python3
"""確信度の較正が未使用の書き手でも見逃し率の目標を守るかを、書き手で分けた交差検証で調べる。

    python scripts/check_calibration.py --dump CALIB.jsonl [--folds 5] [--alpha 0.05]

しきい値は、較正に使った同じデータで選ぶと見逃し率が楽観的に出る(選択バイアス)。ここでは書き手を --folds 個に分け、
残りの書き手で較正(`fit_confidence`)→ 取り分けた書き手の見逃し率を測る。列ごとに、全 fold をまとめた実測の見逃し率と、
その Clopper-Pearson 95% 上側限界、目標、要確認率(空欄を除く)を表示する。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import numpy as np

from yayoi_ocr.recognize.confidence import (
    cp_upper,
    fit_confidence,
    group_of,
    samples_from_dump,
)

TARGETS = {"amount": 0.001, "date": 0.005, "account": 0.005, "description": 0.005}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", nargs="+", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--l2", type=float, default=3.0)
    args = ap.parse_args()
    rows = []
    for d in args.dump:
        with open(d, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh]
    samples = samples_from_dump(rows)
    writers = sorted({r["writer"] for r in rows})
    fold_of = {w: i % args.folds for i, w in enumerate(writers)}
    folds = np.array([fold_of[r["writer"]] for r in rows])
    cols = np.array([r["column"] for r in rows])
    miss = Counter()
    wrong_all = Counter()
    n_cells = Counter()
    reviewed = Counter()
    nonblank = Counter()
    for f in range(args.folds):
        tr = [s for s, k in zip(samples, folds, strict=True) if k != f]
        tr_cells = Counter(c for c, k in zip(cols, folds, strict=True) if k != f)
        model = fit_confidence(
            tr,
            dict(tr_cells),
            TARGETS,
            alpha=args.alpha,
            l2=args.l2,
        )
        for s, r, k in zip(samples, rows, folds, strict=True):
            if k != f:
                continue
            g = model.groups.get(group_of(s.column, s.kind))
            review = s.hard or g is None or g.prob(s.x) < g.threshold
            n_cells[s.column] += 1
            if r["gt_kind"] != "blank":
                nonblank[s.column] += 1
                reviewed[s.column] += int(review)
            if not s.correct:
                wrong_all[s.column] += 1
                miss[s.column] += int(not review)
    print(
        f"書き手 {len(writers)} 人を {args.folds} fold に分けた交差検証(較正 {1 - 1 / args.folds:.0%} → 残りで測定)"
    )
    print(
        f"{'列':12s} {'セル数':>7s} {'誤り':>6s} {'見逃し':>6s} {'見逃し率':>8s} {'95%上限':>8s} {'目標':>6s} {'要確認率(nb)':>12s}"
    )
    for col in ("date", "account", "description", "amount"):
        n, k = n_cells[col], miss[col]
        print(
            f"{col:12s} {n:7d} {wrong_all[col]:6d} {k:6d} {100 * k / max(n, 1):7.3f}% "
            f"{100 * cp_upper(k, n, args.alpha):7.3f}% {100 * TARGETS[col]:5.1f}% "
            f"{reviewed[col] / max(nonblank[col], 1):12.3f}"
        )
    tot_n, tot_k = sum(n_cells.values()), sum(miss.values())
    print(
        f"{'全体':12s} {tot_n:7d} {sum(wrong_all.values()):6d} {tot_k:6d} {100 * tot_k / max(tot_n, 1):7.3f}% "
        f"{100 * cp_upper(tot_k, tot_n, args.alpha):7.3f}%"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
