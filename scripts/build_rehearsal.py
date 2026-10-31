#!/usr/bin/env python3
"""追加学習のリハーサル用の合成セルの束 `rehearsal.npz` を作る(開発用。配布物の `models/` に同梱する)。

    python scripts/build_rehearsal.py --data DIR [DIR ...] --out MODELS_DIR/rehearsal.npz [--per-width 1200] [--seed 0]

DIR は `scripts/build_synth_cells.py` が作った合成データセット(他の書き手・一般語彙)。書かれた通りの正解(`raw`)を持つ text セルから、
幅 256(日付・科目・金額)と幅 384(摘要)をそれぞれ `--per-width` 件、列ごとに均等に無作為抽出する。合成データだけから作る(実データは使わない)。
入力のページは、すべて**合成データの署名**(生成器が付ける整数の seed・writer_id と、OFL フォント名)を持たなければならない(`scripts/model_provenance.py` の C2 と同じ検査)。
一つでも署名のないページがあれば失敗する(実サンプル由来のデータが紛れ込まない)。束には、入力にしたデータセットの識別子(名前・使ったページ数・seed の範囲)を記録する。
出力はリポジトリの外に置くこと。
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from yayoi_ocr.config import assert_safe_output_dir, private_data_dir
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
    fonts = {p.stem for p in (private_data_dir() / "fonts").glob("*.ttf")}
    if not fonts:
        print(
            "OFL フォントの置き場($YAYOI_PRIVATE_DATA/fonts)が空です。署名を検査できません",
            file=sys.stderr,
        )
        return 1
    sources: list[dict] = []
    pool: dict[tuple[int, str], list[tuple[np.ndarray, str]]] = defaultdict(list)
    for d in args.data:
        paths = list_pages(d)
        seeds: list[int] = []
        for i in rng.permutation(len(paths))[: args.pages_per_dir]:
            pg = load_page(paths[int(i)])
            page = pg.meta.get("page", pg.meta)
            if not (
                isinstance(page.get("seed"), int)
                and isinstance(page.get("writer_id"), int)
                and page.get("font") in fonts
            ):
                print(
                    f"合成データの署名のないページがあります: {paths[int(i)].name}", file=sys.stderr
                )
                return 1
            seeds.append(page["seed"])
            for idx, c in enumerate(pg.cells):
                if c.kind != "text" or idx not in pg.rec or not c.raw:
                    continue
                for width, cols in WIDTH_COLUMNS.items():
                    if c.column in cols:
                        pool[(width, c.column)].append((pg.rec[idx], norm_text(c.raw)))
        sources.append(
            {
                "dataset": Path(d).name,
                "pages": len(seeds),
                "seed_min": min(seeds, default=0),
                "seed_max": max(seeds, default=0),
            }
        )
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
    save_rehearsal(out, items, sources)
    print(f"saved {out} ({len(items)} 件, {out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
