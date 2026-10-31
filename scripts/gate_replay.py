#!/usr/bin/env python3
"""採用ゲートの感度の検証: 報告 4 の追加学習の曲線(`writer_adapt_curve.py` の出力)を、新しいゲートで再生する。

    python scripts/gate_replay.py EVAL_DIR [--levels 1 3 5 10 20] [--json OUT.json]

EVAL_DIR は書き手ごとの JSON Lines(`wa_<書き手>.jsonl`)。書き手 × N(先頭 N ページで追加学習)ごとに、その書き手の未使用ページ(12 ページ)を
「評価用ページ」、N = 0(全体の重み)を「親」、N = n を「新しい世代」とみなして、ゲートの判定を比べる:

- 旧ゲート: 見逃し率が悪化しない・正解率が下がらない・得がある・要確認率の増加 ≤ 2pt(件数だけ)。
- 新ゲート: クロスフィッティングした判定(各ページを除いて決めたしきい値)での件数 + 金額の見逃しが 1 件でもあれば不採用 + 列ごとの見逃し + 期待見逃しとのポアソン判定 + 較正の質(NLL)。

確信度は、記録された特徴量に全体の確信度モデル(`--confidence`、既定は適応済みの出荷設定)を当てて作り直す。同上の参照元による要確認は記録に無いので、
理由に ditto_source_uncertain があるセルは確認必須として扱う(近似)。出力は件数だけ。実データは使わない(合成の書き手)。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from yayoi_ocr.app import evalstats, gate, recalibrate
from yayoi_ocr.config import models_dir
from yayoi_ocr.recognize.confidence import (
    ALWAYS_REVIEW_KINDS,
    HARD_REASONS,
    ConfidenceModel,
    featurize,
    group_of,
)


def load(eval_dir: Path) -> dict[int, dict[int, list[dict]]]:
    """書き手 → N → 行(自分のページだけ)。"""
    out: dict[int, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for f in sorted(eval_dir.glob("wa_*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("set") == "own":
                    out[r["writer"]][r["N"]].append(r)
    return out


def to_pages(
    rows: list[dict], cal: ConfidenceModel
) -> tuple[list[list[evalstats.CellObs]], list[list[bool]]]:
    by_page: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_page[r["page"]].append(r)
    pages, flags = [], []
    for pid in sorted(by_page):
        obs, fl = [], []
        for r in sorted(by_page[pid], key=lambda x: (x["row"], x["col"])):
            kind = r["pred_kind"]
            g = cal.groups.get(group_of(r["column"], kind))
            cls_p = r["cls_probs"].get(kind, 1.0)
            p = (
                0.0
                if kind in ALWAYS_REVIEW_KINDS
                else (
                    r["cls_probs"].get(kind, 1.0)
                    if g is None
                    else g.prob(featurize(r["features"], cls_p))
                )
            )
            hard = (
                bool(HARD_REASONS & set(r["reasons"]))
                or bool(r.get("page_review"))
                or kind in ALWAYS_REVIEW_KINDS
            )
            obs.append(
                evalstats.CellObs(
                    r["column"], kind, float(p), bool(r["correct"]), hard,
                    page=len(pages), idx=r["row"] * 4 + r["col"], nonblank=r["gt_kind"] != "blank",
                )
            )  # fmt: skip
            fl.append(bool(r["review"]))
        pages.append(obs)
        flags.append(fl)
    return pages, flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    ap.add_argument("--confidence", default=str(models_dir() / "adapted" / "confidence.json"))
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    cal = ConfidenceModel.load(args.confidence)
    data = load(Path(args.eval_dir))
    summary = []
    for n in args.levels:
        row = {"N": n, "writers": 0, "harmful": 0, "old_adopt": 0, "new_adopt": 0,
               "circ_adopt": 0, "circ_adopt_harmful": 0, "old_adopt_harmful": 0, "new_adopt_harmful": 0, "miss_parent": 0, "miss_old_adopted": 0,
               "miss_new_adopted": 0, "reasons": defaultdict(int)}  # fmt: skip
        for _w, by_n in sorted(data.items()):
            if 0 not in by_n or n not in by_n:
                continue
            ppages, pflags = to_pages(by_n[0], cal)
            cpages, cflags = to_pages(by_n[n], cal)
            parent = evalstats.side_from(ppages, pflags)
            glob = evalstats.side_from(
                cpages, cflags
            )  # 全体の較正のしきい値のまま(実際に記録された判定)
            _final, _calibs, cf = recalibrate.cross_fit(cal, cpages, floor=None)
            cand = evalstats.side_from(cpages, cf)
            old = gate.decide(parent.total, glob.total)
            # 循環(同じ評価ページで決めたしきい値で、同じページの見逃しを数える)= クロスフィッティング導入前のやり方
            fin_flags = [evalstats.review_flags(pg, _final) for pg in cpages]
            circ = gate.decide(parent.total, evalstats.side_from(cpages, fin_flags).total)
            new = gate.decide(parent, cand)
            harmful = glob.total.n_miss > parent.total.n_miss
            row["writers"] += 1
            row["harmful"] += harmful
            row["miss_parent"] += parent.total.n_miss
            row["circ_adopt"] += circ.adopted
            row["circ_adopt_harmful"] += circ.adopted and harmful
            row["old_adopt"] += old.adopted
            row["new_adopt"] += new.adopted
            row["old_adopt_harmful"] += old.adopted and harmful
            row["new_adopt_harmful"] += new.adopted and harmful
            row["miss_old_adopted"] += glob.total.n_miss if old.adopted else parent.total.n_miss
            row["miss_new_adopted"] += cand.total.n_miss if new.adopted else parent.total.n_miss
            row["reasons"][new.reason] += 1
        row["reasons"] = dict(row["reasons"])
        summary.append(row)
    print(
        "N | 書き手 | 見逃しが増えた | 循環(同じページで較正)の採用(うち増えた) | 較正なしの旧ゲートの採用(うち増えた) | 新ゲートの採用(うち増えた) | 見逃し合計: 親 / 較正なしの旧で採用した場合 / 新で採用した場合 | 新ゲートの理由"
    )
    for r in summary:
        print(
            f"{r['N']} | {r['writers']} | {r['harmful']} | {r['circ_adopt']} ({r['circ_adopt_harmful']}) | {r['old_adopt']} ({r['old_adopt_harmful']}) | "
            f"{r['new_adopt']} ({r['new_adopt_harmful']}) | {r['miss_parent']} / {r['miss_old_adopted']} / {r['miss_new_adopted']} | {r['reasons']}"
        )
    if args.json:
        Path(args.json).write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
