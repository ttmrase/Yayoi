#!/usr/bin/env python3
"""2 つの重みの認識結果(eval_recognize.py --dump)を、列ごとに並べて比べる(素 → 適応済みの変化)。

    python scripts/compare_dumps.py base=BASE.jsonl adapted=ADAPTED.jsonl [--min-desc-char-gain 0.0]

セルは同じ並び(同じデータセットを同じ順に認識したもの)であること。表示:
- 値の正解率(正解が空欄でないセル)と、文字の一致率(R1 の第 1 候補と書かれた通りの正解の整列。正解の text セルのみ)
- 要確認率(空欄を除く)と見逃し(誤り ∧ 要確認なし)の件数
摘要の文字の一致率が素より下がっていたら、終了コード 1 で終わる(ドメイン適応の合格条件)。
"""

from __future__ import annotations

import argparse
import json
import sys

from yayoi_ocr.recognize.evaluate import align, norm

COLUMNS = ("date", "account", "description", "amount")


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def stats(rows: list[dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for col in COLUMNS:
        rs = [r for r in rows if r["column"] == col]
        nb = [r for r in rs if r["gt_kind"] != "blank"]
        text = [r for r in nb if r["gt_kind"] == "text" and r["pred_kind"] == "text"]
        ok = tot = 0
        for r in text:
            for op, _x, _y in align(norm(col, r["gt_raw"]), norm(col, r["r1_text"])):
                if op != "ins":
                    tot += 1
                    ok += op == "ok"
        out[col] = {
            "n": len(rs),
            "value_acc": sum(r["correct"] for r in nb) / max(1, len(nb)),
            "char_acc": ok / max(1, tot),
            "review": sum(r["review"] for r in nb) / max(1, len(nb)),
            "miss": sum((not r["correct"]) and (not r["review"]) for r in rs),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs=2, help="名前=パス を 2 つ(素, 適応済みの順)")
    ap.add_argument("--min-desc-char-gain", type=float, default=0.0)
    args = ap.parse_args()
    names, st = [], []
    for spec in args.dumps:
        name, _, path = spec.partition("=")
        names.append(name)
        st.append(stats(load(path)))
    print(f"{'列':12s} {'指標':8s} {names[0]:>9s} {names[1]:>9s} {'変化':>8s}")
    for col in COLUMNS:
        for key, label in (
            ("value_acc", "値の正解率"),
            ("char_acc", "文字一致率"),
            ("review", "要確認率"),
        ):
            a, b = st[0][col][key], st[1][col][key]
            print(f"{col:12s} {label:8s} {a:9.3f} {b:9.3f} {100 * (b - a):+7.1f}pt")
        print(
            f"{col:12s} {'見逃し':8s} {st[0][col]['miss']:9.0f} {st[1][col]['miss']:9.0f} "
            f"(全 {st[0][col]['n']:.0f} セル)"
        )
    gain = st[1]["description"]["char_acc"] - st[0]["description"]["char_acc"]
    ok = gain >= args.min_desc_char_gain
    print(
        f"\n摘要の文字一致率の変化 {100 * gain:+.1f}pt → {'合格' if ok else '不合格(素より低下)'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
