#!/usr/bin/env python3
"""学習曲線(scripts/sim_learning.py --dump の出力)を、条件 x ページ段階 x 列で要約する。

    python scripts/summarize_curve.py static=A.jsonl dict=B.jsonl mem=C.jsonl [--stages 1-1,2-3,4-7,8-12]

書き手ごとの連続ページの通し番号で段階に分け、列ごとに(空欄を除いた)値の正解率・要確認率と、
見逃し(誤り ∧ 要確認なし)÷ 全セルの率とその Clopper-Pearson 95% 上側限界を表示する。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

from yayoi_ocr.recognize.confidence import cp_upper

COLUMNS = ("date", "account", "description", "amount")


def load(path: str) -> tuple[list[dict], dict[tuple[int, int], int]]:
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    pages: dict[int, set[int]] = defaultdict(set)
    for r in rows:
        pages[r["writer"]].add(r["page"])
    rank = {(w, p): i for w, ps in pages.items() for i, p in enumerate(sorted(ps), 1)}
    return rows, rank


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="名前=ダンプ")
    ap.add_argument("--stages", default="1-1,2-3,4-7,8-12")
    args = ap.parse_args()
    stages = [tuple(int(x) for x in g.split("-")) for g in args.stages.split(",")]
    for spec in args.runs:
        name, _, path = spec.partition("=")
        rows, rank = load(path)
        print(f"\n===== {name}  ({len(rows) // 116} ページ)")
        print(f"{'列':12s} " + " | ".join(f"p{a}-{b}".center(34) for a, b in stages))
        for col in COLUMNS:
            cells = []
            for a, b in stages:
                rs = [
                    r
                    for r in rows
                    if r["column"] == col and a <= rank[(r["writer"], r["page"])] <= b
                ]
                nb = [r for r in rs if r["gt_kind"] != "blank"]
                miss = sum((not r["correct"]) and (not r["review"]) for r in rs)
                acc = sum(r["correct"] for r in nb) / max(len(nb), 1)
                rev = sum(r["review"] for r in nb) / max(len(nb), 1)
                cells.append(
                    f"acc {acc:.3f} rev {rev:.2f} miss {miss:2d}/{len(rs):4d} "
                    f"({100 * cp_upper(miss, len(rs)):.2f}%)"
                )
            print(f"{col:12s} " + " | ".join(c.center(34) for c in cells))
        allrs = rows
        miss = sum((not r["correct"]) and (not r["review"]) for r in allrs)
        print(
            f"全体の見逃し {miss}/{len(allrs)} = {100 * miss / len(allrs):.3f}% "
            f"(95%上限 {100 * cp_upper(miss, len(allrs)):.3f}%)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
