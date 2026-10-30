#!/usr/bin/env python3
"""認識の誤りの内訳: 列別・正解の文字種別・誤りの型(置換・脱落・挿入)・要確認が付かなかった誤り。

    python scripts/analyze_errors.py DUMP.jsonl [DUMP2.jsonl ...] [--by flags]

DUMP は scripts/eval_recognize.py --dump の出力(合成データのみ)。R1 の第 1 候補の文字列(r1_text)と、
書かれた通りの正解(gt_raw)を、列ごとの正規化(日付・金額は空白を除く、科目・摘要は NFKC + 空白除去)のうえで
文字単位に整列して数える。値のレベル(pred_value と gt_resolved)の誤りは別に集計する。
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter, defaultdict


def char_class(ch: str) -> str:
    if ch.isdigit():
        return "digit"
    if ch in "R":
        return "R"
    if ch in ".":
        return "dot"
    if ch in ",":
        return "comma"
    if ch in "-ー−―":
        return "dash"
    if ch == "〃":
        return "ditto"
    code = ord(ch)
    if 0x3040 <= code <= 0x309F:
        return "hiragana"
    if 0x30A0 <= code <= 0x30FF:
        return "katakana"
    if 0x4E00 <= code <= 0x9FFF:
        return "kanji"
    if ch.isalpha():
        return "latin"
    return "symbol"


def norm(column: str, s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return "".join(s.split())


def align(a: str, b: str):
    """正解 a と認識 b のレーベンシュタイン整列 → [(op, 正解の文字 | None, 認識の文字 | None)]。"""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(
                d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1])
            )
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            ops.append(("ok" if a[i - 1] == b[j - 1] else "sub", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", a[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, b[j - 1]))
            j -= 1
    return ops[::-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--by", choices=["column", "flags"], default="column")
    args = ap.parse_args()
    rows = []
    for d in args.dumps:
        with open(d, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh]
    text = [r for r in rows if r["gt_kind"] == "text" and r["pred_kind"] == "text"]
    print(f"text セル {len(text)} 件(正解・判定とも text)")
    # ---- 文字レベルの誤り(R1 の第 1 候補 vs 書かれた通りの正解)
    per_col: dict[str, Counter] = defaultdict(Counter)
    per_class: dict[str, Counter] = defaultdict(Counter)
    subs: dict[str, Counter] = defaultdict(Counter)
    for r in text:
        a, b = norm(r["column"], r["gt_raw"]), norm(r["column"], r["r1_text"])
        for op, x, y in align(a, b):
            per_col[r["column"]][op] += 1
            if x is not None:
                per_class[char_class(x)][op if op != "ins" else "ok"] += 1
            if op == "ins":
                per_class["(挿入)"]["ins"] += 1
            if op == "sub":
                subs[r["column"]][(x, y)] += 1
    print("\n== 文字レベル(列別): ok / sub(置換)/ del(脱落)/ ins(挿入) ==")
    for col in ("date", "account", "description", "amount"):
        c = per_col[col]
        tot = c["ok"] + c["sub"] + c["del"]
        print(
            f"{col:12s} 正解文字 {tot:6d}  一致 {c['ok'] / max(1, tot):.3f}  置換 {c['sub'] / max(1, tot):.3f}"
            f"  脱落 {c['del'] / max(1, tot):.3f}  挿入(対正解文字数) {c['ins'] / max(1, tot):.3f}"
        )
    print("\n== 正解の文字種別ごとの誤り ==")
    for k, c in sorted(per_class.items(), key=lambda kv: -sum(kv[1].values())):
        tot = c["ok"] + c["sub"] + c["del"]
        if k == "(挿入)":
            print(f"{k:10s} 挿入 {c['ins']}")
            continue
        print(
            f"{k:10s} n={tot:6d}  一致 {c['ok'] / max(1, tot):.3f}  置換 {c['sub'] / max(1, tot):.3f}  脱落 {c['del'] / max(1, tot):.3f}"
        )
    print("\n== 多い置換(列別 上位 8) ==")
    for col in ("date", "amount", "account", "description"):
        top = subs[col].most_common(8)
        print(f"{col:12s}", "  ".join(f"{x}→{y}:{n}" for (x, y), n in top))
    # ---- 値レベルの誤りと、要確認が付かなかった誤り
    print("\n== 値レベル(列別・text セル): 誤り / うち要確認なし(見逃し) ==")
    for col in ("date", "account", "description", "amount"):
        rs = [r for r in text if r["column"] == col]
        wrong = [r for r in rs if not r["correct"]]
        miss = [r for r in wrong if not r.get("review", False)]
        print(
            f"{col:12s} n={len(rs):5d} 誤り {len(wrong):5d} ({len(wrong) / max(1, len(rs)):.3f})  見逃し {len(miss)}"
        )
    # ---- 混入要因別(はみ出し・注記・余計な点線)の正解率
    print("\n== 混入要因別の正解率(text セル) ==")
    for key in ("overflow_gt", "stray", "annotation"):
        on = [r for r in text if r.get(key)]
        off = [r for r in text if not r.get(key)]
        if on:
            print(
                f"{key:12s} あり n={len(on):5d} 正解率 {sum(r['correct'] for r in on) / len(on):.3f}   "
                f"なし n={len(off):5d} 正解率 {sum(r['correct'] for r in off) / max(1, len(off)):.3f}"
            )
    # ---- 種別の取り違え
    conf = Counter((r["gt_kind"], r["pred_kind"]) for r in rows if r["gt_kind"] != r["pred_kind"])
    print("\n== 種別の取り違え(正解→判定: 件数) ==", dict(conf.most_common(10)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
