#!/usr/bin/env python3
"""R2(数字専用 CRNN-CTC)と R1(PARSeq)の比較と、一致による確信度の分析(Phase 4 方法検証の (a)(d))。

    python scripts/analyze_r2.py --r2 R2_DIR [R2_DIR ...] --r1 R1_DIR [R1_DIR ...]

R2_DIR: `scripts/exp_r2_ctc.py` の出力(r2_*.jsonl)。R1_DIR: `scripts/writer_adapt_curve.py` の出力(wa_*.jsonl、同じ書き手・同じ N の追加学習済み R1)。
比較は**書かれた通りの文字列を正規化した水準**で行う(日付 = (年, 月, 日)・金額 = 整数。年の継承などの文脈の解決は含めない)。
  ① 正解率(日付・金額)の N による曲線: R1 の第 1 候補・R1 の最終の値(文脈の解決つき)・R2。
  ② R1 と R2 の一致: 一致するセルの割合、一致したセルの中での誤り率、不一致が R1 の誤りをどれだけ拾うか。
  ③ 要確認の削減の見込み: R1 が要確認にしたセルのうち「R1 と R2 が一致」したものの割合(確認から外せる分)と、その中の誤り(外した場合に増える見逃し)。
値の文字列は出さない(件数と割合だけ)。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from yayoi_ocr.recognize.grammar import parse_amount, repair_date

COLUMNS = ("date", "amount")


def canon(column: str, s: str):
    s = s.strip()
    if column == "date":
        v = repair_date(s)
        return (v[0].era_year, v[0].month, v[0].day) if v else None
    return parse_amount(s)


def canon_value(column: str, value: str):
    """R1 の最終の値(日付 'R8.7.11' / 金額 '2350')を、同じ水準の正規形にする。年を書かない日付は gt 側の年の有無で比べる。"""
    return canon(column, value)


def load(dirs, pattern):
    rows = []
    for d in dirs:
        for f in sorted(Path(d).glob(pattern)):
            with open(f, encoding="utf-8") as fh:
                rows += [json.loads(line) for line in fh]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r2", nargs="+", required=True)
    ap.add_argument("--r1", nargs="+", required=True)
    args = ap.parse_args()
    r2 = load(args.r2, "r2_*.jsonl")
    r1 = [
        r
        for r in load(args.r1, "wa_*.jsonl")
        if r.get("set", "own") == "own" and r["column"] in COLUMNS
    ]
    levels = sorted({r["N"] for r in r2})
    key = lambda r: (r["writer"], r["N"], r["page"], r["row"], r["col"])  # noqa: E731
    r1_by = {key(r): r for r in r1}
    n_cells_by = defaultdict(int)  # (N) → 日付・金額の全セル(空欄を含む)
    for r in r1:
        n_cells_by[r["N"]] += 1

    def raw_ok(column, gt_raw, text):
        a, b = canon(column, gt_raw), canon(column, text)
        return a is not None and a == b

    print(f"R2: {len(r2)} 件({len({r['writer'] for r in r2})} 人)、R1: {len(r1)} 件")
    print("\n① 正解率(書かれた通りの水準。日付+金額、空欄を除く)")
    print(
        f"{'N':<5}{'R1 第1候補':>12}{'R1 最終(文脈つき)':>18}{'R2':>8}{'  R2 日付':>10}{'R2 金額':>9}  セル数"
    )
    for n in [0, *levels]:
        rows = [r for r in r1 if r["N"] == n and r["gt_kind"] == "text"]
        if not rows:
            continue
        a1 = sum(raw_ok(r["column"], r["gt_raw"], r["r1_text"]) for r in rows) / len(rows)
        a1f = sum(r["correct"] for r in rows) / len(rows)
        s2 = [r for r in r2 if r["N"] == n]
        if s2:
            a2 = sum(raw_ok(r["column"], r["gt_raw"], r["pred_raw"]) for r in s2) / len(s2)
            d = [r for r in s2 if r["column"] == "date"]
            m = [r for r in s2 if r["column"] == "amount"]
            ad = sum(raw_ok("date", r["gt_raw"], r["pred_raw"]) for r in d) / max(1, len(d))
            am = sum(raw_ok("amount", r["gt_raw"], r["pred_raw"]) for r in m) / max(1, len(m))
            print(f"{n:<5}{a1:>12.3f}{a1f:>18.3f}{a2:>8.3f}{ad:>10.3f}{am:>9.3f}  {len(rows)}")
        else:
            print(f"{n:<5}{a1:>12.3f}{a1f:>18.3f}{'—':>8}")

    print("\n② R1 と R2 の一致(日付+金額)")
    print(
        f"{'N':<5}{'一致の割合':>10}{'一致内の誤り率':>14}{'不一致内の R1 誤り率':>20}{'R1 の誤りのうち不一致':>20}"
    )
    for n in levels:
        agree = dis = agree_wrong = dis_wrong = 0
        for r in (r for r in r2 if r["N"] == n):
            q = r1_by.get((r["writer"], n, r["page"], r["row"], r["col"]))
            if q is None:
                continue
            same = (
                canon(r["column"], q["r1_text"]) == canon(r["column"], r["pred_raw"])
                and canon(r["column"], r["pred_raw"]) is not None
            )
            wrong = not raw_ok(r["column"], r["gt_raw"], q["r1_text"])
            if same:
                agree += 1
                agree_wrong += wrong
            else:
                dis += 1
                dis_wrong += wrong
        tot = agree + dis
        if tot:
            print(
                f"{n:<5}{agree / tot:>10.3f}{agree_wrong / max(1, agree):>14.4f}{dis_wrong / max(1, dis):>20.3f}"
                f"{dis_wrong / max(1, dis_wrong + agree_wrong):>20.3f}"
            )

    print("\n③ 要確認の削減の見込み(R1 が要確認にしたセルのうち、R1 と R2 が一致したもの)")
    print(
        f"{'N':<5}{'R1 の要確認':>11}{'うち一致':>9}{'一致のうち正解':>14}{'外した場合の見逃し増(全セル比)':>26}{'現状の見逃し(全セル比)':>22}"
    )
    for n in levels:
        flagged = flagged_agree = fa_wrong = 0
        for r in (r for r in r2 if r["N"] == n):
            q = r1_by.get((r["writer"], n, r["page"], r["row"], r["col"]))
            if q is None or not q["review"]:
                continue
            flagged += 1
            same = (
                canon(r["column"], q["r1_text"]) == canon(r["column"], r["pred_raw"])
                and canon(r["column"], r["pred_raw"]) is not None
            )
            if same:
                flagged_agree += 1
                fa_wrong += not raw_ok(r["column"], r["gt_raw"], q["r1_text"])
        miss_now = sum((not r["correct"]) and (not r["review"]) for r in r1 if r["N"] == n)
        tot = max(1, n_cells_by[n])
        print(
            f"{n:<5}{flagged:>11}{flagged_agree / max(1, flagged):>9.3f}{1 - fa_wrong / max(1, flagged_agree):>14.4f}"
            f"{fa_wrong / tot * 100:>25.3f}%{miss_now / tot * 100:>21.3f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
