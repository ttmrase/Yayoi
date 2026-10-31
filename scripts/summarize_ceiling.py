#!/usr/bin/env python3
"""書き手 1 人あたりの確定ページ数 N の天井の見積り(writer_adapt_curve.py の出力)を、N × 列の表に集計する。

    python scripts/summarize_ceiling.py EVAL_DIR [EVAL_DIR2 ...] [--per-writer]

各 N について、同じ書き手の未使用ページの非空欄セルの値の正解率・要確認率(出荷の較正のまま)・見逃し率(全セル中、誤り ∧ 要確認なし)を列別に出す。
`--per-writer` は書き手ごとの 日付+金額 の正解率も出す(書き手による差を見る)。値の文字列は出さない(合成データだが、件数と割合だけ)。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

COLUMNS = ("date", "account", "description", "amount")


def load(dirs: list[str]) -> list[dict]:
    rows: list[dict] = []
    for d in dirs:
        for f in sorted(Path(d).glob("wa_*.jsonl")):
            with open(f, encoding="utf-8") as fh:
                rows += [json.loads(line) for line in fh]
    return [r for r in rows if r.get("set", "own") == "own"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--per-writer", action="store_true")
    args = ap.parse_args()
    rows = load(args.dirs)
    levels = sorted({r["N"] for r in rows})
    print(
        f"書き手 {len({r['writer'] for r in rows})} 人、セル {len(rows)} 件({', '.join(args.dirs)})"
    )
    print("\n値の正解率 / 要確認率(空欄を除く)")
    print(
        "N      " + "  ".join(f"{c:>13}" for c in (*COLUMNS, "date+amount")) + "   見逃し(全セル)"
    )
    for n in levels:
        rs = [r for r in rows if r["N"] == n]
        cells = []
        for cols in [(c,) for c in COLUMNS] + [("date", "amount")]:
            x = [r for r in rs if r["column"] in cols and r["gt_kind"] != "blank"]
            acc = sum(r["correct"] for r in x) / max(1, len(x))
            rev = sum(r["review"] for r in x) / max(1, len(x))
            cells.append(f"{acc:.3f} / {rev:.2f}")
        miss = sum((not r["correct"]) and (not r["review"]) for r in rs) / max(1, len(rs))
        print(f"{n:<6} " + "  ".join(f"{c:>13}" for c in cells) + f"   {miss * 100:.2f}%")
    if args.per_writer:
        print("\n書き手ごとの 日付+金額 の正解率")
        writers = sorted({r["writer"] for r in rows})
        print("N      " + "  ".join(f"{w:>9}" for w in writers))
        for n in levels:
            out = []
            for w in writers:
                x = [
                    r
                    for r in rows
                    if r["N"] == n
                    and r["writer"] == w
                    and r["column"] in ("date", "amount")
                    and r["gt_kind"] != "blank"
                ]
                out.append(f"{sum(r['correct'] for r in x) / max(1, len(x)):.3f}")
            print(f"{n:<6} " + "  ".join(f"{c:>9}" for c in out))
    # 学習時間
    by_n: dict[int, list[float]] = defaultdict(list)
    for d in args.dirs:
        for f in Path(d).glob("timing_*.json"):
            for t in json.loads(f.read_text(encoding="utf-8")):
                by_n[t["N"]].append(sum(v["seconds"] for v in t["train"].values()))
    if by_n:
        print(
            "\n追加学習の時間(1 スレッド、幅 256+384 の合計、秒の平均): "
            + ", ".join(f"N={n}: {sum(v) / len(v):.0f}" for n, v in sorted(by_n.items()))
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
