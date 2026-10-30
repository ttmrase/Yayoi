#!/usr/bin/env python3
"""確信度モデル(列 x 種別のロジスティック回帰と、要確認のしきい値)を較正する。

    python scripts/fit_confidence.py --dump CALIB.jsonl [CALIB2.jsonl ...] --out confidence.json
        [--alpha 0.05] [--max-miss-amount 0.001] [--max-miss 0.005]
        [--margin 0.6] [--margins amount=0.4] [--column-dump description=OOV.jsonl]

- --dump: scripts/eval_recognize.py --dump の出力(較正用データセットで、学習済みの分類器を使って認識したもの)。
- 見逃し率の目標(列の全セルに対する割合): 金額 0.1%、その他 0.5%。しきい値は見逃し率の Clopper-Pearson 上側信頼限界(--alpha、既定 95%)が目標以下になるように決める。
- 余裕(--margin / --margins): 目標に掛ける係数。較正データと実運用の分布のずれ・書き手単位のまとまった誤り(同じ書き手の同種の誤りが続く)に備える。
  既定の出荷設定(docs/recognition.md): 0.6(金額 0.4)、摘要は辞書に無い語のデータだけで較正。
- 出力はリポジトリの外(例: $YAYOI_PRIVATE_DATA/models/)に置く。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from yayoi_ocr.config import assert_safe_output_dir
from yayoi_ocr.recognize.confidence import fit_confidence, replace_columns, samples_from_dump


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Clopper-Pearson の片側の有意水準(0.05 = 95% 上側信頼限界)",
    )
    ap.add_argument("--max-miss", type=float, default=0.005)
    ap.add_argument("--max-miss-amount", type=float, default=0.001)
    ap.add_argument("--l2", type=float, default=3.0)
    ap.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="見逃し率の目標に掛ける余裕(0.8 = 目標の 80% で較正する。実運用の分布のずれへの備え)",
    )
    ap.add_argument(
        "--margins",
        default="",
        help="列ごとの余裕の上書き(例: amount=0.4,description=0.6。指定の無い列は --margin)",
    )
    ap.add_argument(
        "--column-dump",
        action="append",
        default=[],
        metavar="列=パス[,パス...]",
        help="その列だけ、このダンプで較正する(例: description=calib_gen.jsonl。語彙外の摘要が多いデータ)。繰り返し指定できる",
    )
    ap.add_argument(
        "--credit-learned-features",
        action="store_true",
        help="見覚えメモリ・金額履歴の特徴にも重みを与える(出荷の較正では使わない。Phase 4 で確定データを使うとき用)",
    )
    args = ap.parse_args()

    def read(paths) -> list[dict]:
        out: list[dict] = []
        for d in paths:
            with open(d, encoding="utf-8") as fh:
                out += [json.loads(line) for line in fh]
        return out

    margins = {c: args.margin for c in ("date", "account", "description", "amount")}
    margins.update(
        {k: float(v) for k, _, v in (kv.partition("=") for kv in args.margins.split(",") if kv)}
    )
    targets = {
        c: m * (args.max_miss_amount if c == "amount" else args.max_miss)
        for c, m in margins.items()
    }

    def fit(rows: list[dict]):
        return fit_confidence(
            samples_from_dump(rows),
            dict(Counter(r["column"] for r in rows)),
            targets,
            default_max_miss=args.margin * args.max_miss,
            alpha=args.alpha,
            l2=args.l2,
            credit_learned_features=args.credit_learned_features,
        )

    rows = read(args.dump)
    model = fit(rows)
    for spec in args.column_dump:
        col, _, paths = spec.partition("=")
        model = replace_columns(model, fit(read(paths.split(","))), [col])
        print(f"列 {col} は {paths} だけで較正")
    out = Path(args.out)
    assert_safe_output_dir(out.parent)
    model.save(out)
    print(f"{len(rows)} cells -> {out}")
    for k, g in sorted(model.groups.items()):
        print(f"  {k:22s} n={g.n_train:6d} threshold={g.threshold:.4f} {'LR' if g.lr else 'const'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
