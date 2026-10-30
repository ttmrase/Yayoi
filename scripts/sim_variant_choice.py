#!/usr/bin/env python3
"""初期重みの二系統(素 / 適応済み)の自動選択のシミュレーション。

    python scripts/sim_variant_choice.py --dump base=BASE.jsonl adapted=ADAPTED.jsonl [--pages 3]

--dump: 同じデータセットを 2 種類の重みで認識した scripts/eval_recognize.py --dump の出力(書き手ごとの連続ページ)。
各書き手について、連続する `--pages` ページ(= 利用者が確定したページ)でそれぞれの成績を数え、
`variants.choose_variant`(アプリが使う選択)で選んだ重みが、その書き手の残りのページで本当に良かったほうかを調べる。
選択が外れた場合の損失(残りのページでの点数の差)も出す。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

from yayoi_ocr.recognize.variants import (
    PageCounts,
    choose_variant,
    default_variant,
    score_counts,
)


def load(path: str) -> dict[int, dict[int, PageCounts]]:
    """writer → page(seed) → PageCounts"""
    out: dict[int, dict[int, PageCounts]] = defaultdict(lambda: defaultdict(PageCounts))
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            c = out[r["writer"]][r["page"]]
            c.n_cells += 1
            if r["gt_kind"] != "blank":
                c.n_text += 1
                c.n_correct += int(r["correct"])
            c.n_miss += int(not r["correct"] and not r["review"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", nargs="+", required=True, help="名前=ダンプ のペア(2 つ)")
    ap.add_argument("--pages", type=int, default=3)
    args = ap.parse_args()
    runs = {}
    for spec in args.dump:
        name, _, path = spec.partition("=")
        runs[name] = load(path)
    names = list(runs)
    if len(names) != 2:
        print("ダンプは 2 つ(名前=パス)を指定してください", file=sys.stderr)
        return 2
    default = default_variant()
    default = default if default in names else names[0]
    n_trials = n_right = 0
    regret = 0.0
    first_trials = first_right = 0
    for w in sorted(runs[names[0]]):
        pages = sorted(runs[names[0]][w])
        if len(pages) < args.pages + 3 or w not in runs[names[1]]:
            continue
        for start in range(0, len(pages) - args.pages - 2):
            conf, rest = pages[start : start + args.pages], pages[start + args.pages :]
            sc_conf = []
            sc_rest = {}
            for n in names:
                cc, rr = PageCounts(), PageCounts()
                for p in conf:
                    cc = cc + runs[n][w][p]
                for p in rest:
                    rr = rr + runs[n][w][p]
                sc_conf.append(score_counts(n, cc))
                sc_rest[n] = score_counts(n, rr).score
            chosen = choose_variant(args.pages, sc_conf)
            best = max(names, key=lambda n: sc_rest[n])
            n_trials += 1
            n_right += int(chosen == best or abs(sc_rest[chosen] - sc_rest[best]) < 1e-9)
            regret += sc_rest[best] - sc_rest[chosen]
            if start == 0:
                first_trials += 1
                first_right += int(chosen == best)
    if not n_trials:
        print("試行がありません(ページ数が足りない)")
        return 1
    print(
        f"確定 {args.pages} ページでの選択: {n_trials} 試行(書き手 x 開始位置)のうち、残りのページで良いほうを選べたのは "
        f"{n_right / n_trials:.0%}。選択ミスによる点数の損失(平均) {regret / n_trials:.4f}"
    )
    print(f"  先頭の {args.pages} ページだけで見ると {first_right}/{first_trials} 書き手")
    for n in names:
        allc = PageCounts()
        for w in runs[n]:
            for p in runs[n][w]:
                allc = allc + runs[n][w][p]
        s = score_counts(n, allc)
        print(
            f"  {n:8s} 全体: 正解率 {s.accuracy:.3f} 見逃し率 {s.miss_rate:.4f} 点数 {s.score:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
