#!/usr/bin/env python3
"""追加学習のリハーサル用の合成セルの束 `rehearsal.npz` を作る(開発用。配布物の `models/` に同梱する)。

    python scripts/build_rehearsal.py --data DIR [DIR ...] --out MODELS_DIR/rehearsal.npz [--per-width 1200] [--seed 0]

DIR は `scripts/build_synth_cells.py` が作った合成データセット(他の書き手・一般語彙)。書かれた通りの正解(`raw`)を持つ text セルから、
幅 256(日付・科目・金額)と幅 384(摘要)をそれぞれ `--per-width` 件、列ごとに均等に無作為抽出する。合成データだけから作る(実データは使わない)。
出力はリポジトリの外に置くこと。
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from yayoi_ocr.config import assert_safe_output_dir
from yayoi_ocr.recognize.grammar import norm_text
from yayoi_ocr.recognize.labels import WIDTH_COLUMNS
from yayoi_ocr.recognize.rehearsal import save_rehearsal
from yayoi_ocr.synth.dataset import list_pages, load_page


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-width", type=int, default=1200)
    ap.add_argument("--pages-per-dir", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    assert_safe_output_dir(out.parent)
    rng = np.random.default_rng(args.seed)
    pool: dict[tuple[int, str], list[tuple[np.ndarray, str]]] = defaultdict(list)
    for d in args.data:
        paths = list_pages(d)
        for i in rng.permutation(len(paths))[: args.pages_per_dir]:
            pg = load_page(paths[int(i)])
            for idx, c in enumerate(pg.cells):
                if c.kind != "text" or idx not in pg.rec or not c.raw:
                    continue
                for width, cols in WIDTH_COLUMNS.items():
                    if c.column in cols:
                        pool[(width, c.column)].append((pg.rec[idx], norm_text(c.raw)))
    items = []
    for width, cols in WIDTH_COLUMNS.items():
        per_col = args.per_width // len(cols)
        for col in cols:
            cand = pool[(width, col)]
            if len(cand) < per_col:
                print(f"警告: 幅 {width} の {col} は {len(cand)} 件しかない", file=sys.stderr)
            for j in rng.permutation(len(cand))[:per_col]:
                img, text = cand[int(j)]
                items.append((width, col, img, text))
    save_rehearsal(out, items)
    print(f"saved {out} ({len(items)} 件, {out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
