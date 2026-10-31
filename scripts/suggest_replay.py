#!/usr/bin/env python3
"""提案モードの効果の見積り: 報告 4 の追加学習の曲線(`writer_adapt_curve.py` の出力)を使って、確認の手間(見るセル数・直すセル数)を比べる。

    python scripts/suggest_replay.py EVAL_DIR [--levels 1 3 5 10 20] [--json OUT.json]

書き手 × N ごとに、その書き手の未使用ページ(12 ページ)で、N = 0(全体の重み = 親)と N = n(先頭 n ページで追加学習した世代 = 提案モードの世代)を、セルごとに突き合わせる:

- 親: 入力済みの値 = 親の値、要確認 = 親の判定。
- 提案モード: 入力済みの値 = 提案モードの世代の値、要確認 = 親の判定 OR 提案モードの判定 OR 値(または種別)が違う。
- 見るセル = 要確認のセル(非空欄)。直すセル = 要確認で、入力済みの値が誤りのセル。要確認でない誤り(見逃し)は、提案モードでは親の見逃しの部分集合になる(検算する)。

提案モードの世代の判定は、記録された判定(全体の較正のしきい値)を使う。値の内容は出力しない(件数だけ)。実データは使わない(合成の書き手)。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load(eval_dir: Path) -> dict[int, dict[int, dict[tuple, dict]]]:
    out: dict[int, dict[int, dict[tuple, dict]]] = defaultdict(lambda: defaultdict(dict))
    for f in sorted(eval_dir.glob("wa_*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("set") == "own":
                    out[r["writer"]][r["N"]][(r["page"], r["row"], r["col"])] = r
    return out


def compare(parent: dict[tuple, dict], cand: dict[tuple, dict]) -> dict:
    c = defaultdict(int)
    for k, p in parent.items():
        s = cand.get(k)
        if s is None or p["gt_kind"] == "blank":
            continue
        differ = p["pred_kind"] != s["pred_kind"] or p["pred_value"] != s["pred_value"]
        rev_p = bool(p["review"])
        rev_s = rev_p or bool(s["review"]) or differ
        ok_p, ok_s = bool(p["correct"]), bool(s["correct"])
        c["cells"] += 1
        c["review_parent"] += rev_p
        c["review_suggest"] += rev_s
        c["fix_parent"] += rev_p and not ok_p
        c["fix_suggest"] += rev_s and not ok_s
        c["miss_parent"] += (not rev_p) and not ok_p
        c["miss_suggest"] += (not rev_s) and not ok_s
        c["differ"] += differ
        c["differ_suggest_right"] += differ and ok_s and not ok_p
        c["differ_parent_right"] += differ and ok_p and not ok_s
        c["unreviewed_not_subset"] += (not rev_s) and (rev_p or differ)  # 検算: 常に 0
    return dict(c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    data = load(Path(args.eval_dir))
    rows = []
    print(
        "N | セル | 見る(親→提案) | 直す(親→提案) | 直す回数の変化 | 入力済みが正しい割合(見るセル中: 親→提案) | 見逃し(親→提案) | 提案が正しい/親が正しい(値が違うセル) | 検算"
    )
    for n in args.levels:
        tot: dict[str, int] = defaultdict(int)
        better_writers = 0
        for _w, by_n in sorted(data.items()):
            if 0 not in by_n or n not in by_n:
                continue
            c = compare(by_n[0], by_n[n])
            for k, v in c.items():
                tot[k] += v
            better_writers += c["fix_suggest"] < c["fix_parent"]
        t = dict(tot)
        t["writers_with_fewer_fixes"] = better_writers
        rows.append({"N": n, **t})
        rp, rs = t["review_parent"], t["review_suggest"]
        fp, fs = t["fix_parent"], t["fix_suggest"]
        print(
            f"{n} | {t['cells']} | {rp} → {rs} | {fp} → {fs} | {(fs - fp) / fp:+.1%} | "
            f"{1 - fp / rp:.1%} → {1 - fs / rs:.1%} | {t['miss_parent']} → {t['miss_suggest']} | "
            f"{t['differ_suggest_right']} / {t['differ_parent_right']} | 部分集合でない {t['unreviewed_not_subset']}・修正が減った書き手 {better_writers}"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
