"""追加学習のリハーサル(忘れ防止)用の合成セルの束 `rehearsal.npz`。

利用者の確定セルだけで追加学習すると、他の書き手・一般的な語彙の認識を忘れやすい。そこで、学習のたびに、利用者のセル数の一定割合(既定 50%)の合成セル
(書かれた通りの正解つき)を混ぜる(`docs/writer_adaptation.md` で検証)。束は**合成データだけ**(手書き風フォントで描いた架空の内容。実データを含まない)で、
配布物の `models/` に同梱する(`scripts/build_rehearsal.py` が作る。出所は `docs/model_provenance.md`)。
許可リスト形式の `.npz`(pickle なし): `meta`(JSON)と `rec_<i>`(グレー画像)。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FILE = "rehearsal.npz"
FORMAT = 1


def save_rehearsal(
    path: str | Path,
    items: list[tuple[int, str, np.ndarray, str]],
    sources: list[dict] | None = None,
) -> None:
    """items: (幅, 列, グレー画像, 正解の文字列)。`sources` は入力にした合成データセットの識別子(名前・使ったページ数・seed の範囲。生成器の署名を検査した結果)。"""
    meta = {
        "format": FORMAT,
        "synthetic": True,
        "sources": sources or [],
        "cells": [{"width": w, "column": c, "text": t} for w, c, _, t in items],
    }
    arrays = {"meta": np.array(json.dumps(meta, ensure_ascii=False))}
    for i, (_, _, img, _) in enumerate(items):
        arrays[f"rec_{i}"] = np.ascontiguousarray(img, dtype=np.uint8)
    np.savez_compressed(Path(path), **arrays)


def load_rehearsal(path: str | Path) -> dict[int, list[tuple[np.ndarray, str]]]:
    """幅 → [(グレー画像, 正解の文字列)]。形式が違えば ValueError。"""
    out: dict[int, list[tuple[np.ndarray, str]]] = {}
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        if meta.get("format") != FORMAT:
            raise ValueError("rehearsal_format")
        for i, c in enumerate(meta["cells"]):
            img = z[f"rec_{i}"]
            if img.ndim != 2 or img.dtype != np.uint8:
                raise ValueError("rehearsal_image")
            out.setdefault(int(c["width"]), []).append((img, str(c["text"])))
    return out
