#!/usr/bin/env python3
"""確信度モデル(列 x 種別のロジスティック回帰と、要確認のしきい値)を較正する。

    python scripts/fit_confidence.py --dump CALIB.jsonl [CALIB2.jsonl ...] --out confidence.json
        [--margin 2.0] [--max-miss-amount 0.001] [--max-miss 0.005]

- --dump: scripts/eval_recognize.py --dump の出力(較正用データセットで、学習済みの分類器を使って認識したもの)。
- 見逃し率の目標(列の全セルに対する割合): 金額 0.1%、その他 0.5%。--margin は安全側の余裕(目標を margin で割った件数まで許す)。
- 出力はリポジトリの外(例: $YAYOI_PRIVATE_DATA/models/)に置く。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from yayoi_ocr.config import assert_safe_output_dir
from yayoi_ocr.recognize.confidence import fit_confidence, samples_from_dump


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--max-miss", type=float, default=0.005)
    ap.add_argument("--max-miss-amount", type=float, default=0.001)
    ap.add_argument("--l2", type=float, default=3.0)
    args = ap.parse_args()
    rows = []
    for d in args.dump:
        with open(d, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh]
    samples = samples_from_dump(rows)
    n_cells = Counter(r["column"] for r in rows)
    model = fit_confidence(
        samples,
        dict(n_cells),
        {
            "amount": args.max_miss_amount,
            "date": args.max_miss,
            "account": args.max_miss,
            "description": args.max_miss,
        },
        default_max_miss=args.max_miss,
        margin=args.margin,
        l2=args.l2,
    )
    out = Path(args.out)
    assert_safe_output_dir(out.parent)
    model.save(out)
    print(f"{len(rows)} cells -> {out}")
    for k, g in sorted(model.groups.items()):
        print(f"  {k:22s} n={g.n_train:6d} threshold={g.threshold:.4f} {'LR' if g.lr else 'const'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
