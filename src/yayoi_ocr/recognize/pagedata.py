"""認識の入力(`PageInput`: セルごとの認識用画像・種別分類の入力・統計)の保存と読み込み。

追加学習(確定したセルの画像で重みを更新)と、世代の採否の評価(評価用ページを親・新世代の両方で認識し直す)に使う。利用者の PC の中にだけ置く
(確定したセルの切り出し画像と同じ扱い。学習画面の「リセット(削除)」で消える)。許可リスト形式の `.npz`(pickle なし): `meta`(JSON)・`clf`・`stats`・`rec_<i>`・`title_rec`・`number_rec`。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .page import CellInput, PageInput

FORMAT = 1
N_CELLS = 116
CLF_SHAPE = (64, 448)


def save_page_input(path: str | Path, page: PageInput, prev_year: int | None = None) -> None:
    """途中までの書き込みを残さないよう、一時ファイルに書いてから置き換える。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "format": FORMAT,
        "prev_year": prev_year,
        "layout_status": page.layout_status,
        "layout_notes": list(page.layout_notes),
        "cells": [
            {
                "row": c.row,
                "col": c.col,
                "column": c.column,
                "flags": list(c.flags),
                "blank": bool(c.blank),
                "faint_only": bool(c.faint_only),
                "ink_h_frac": float(c.ink_h_frac),
                "has_rec": c.rec is not None,
            }
            for c in page.cells
        ],
    }
    arrays: dict[str, np.ndarray] = {
        "meta": np.array(json.dumps(meta)),
        "clf": np.stack([c.clf for c in page.cells]).astype(np.uint8),
        "stats": np.stack([c.stats for c in page.cells]).astype(np.float32),
    }
    for i, c in enumerate(page.cells):
        if c.rec is not None:
            arrays[f"rec_{i}"] = np.ascontiguousarray(c.rec, dtype=np.uint8)
    if page.title_rec is not None:
        arrays["title_rec"] = np.ascontiguousarray(page.title_rec, dtype=np.uint8)
    if page.number_rec is not None:
        arrays["number_rec"] = np.ascontiguousarray(page.number_rec, dtype=np.uint8)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def load_page_input(path: str | Path) -> tuple[PageInput, int | None]:
    """保存した `PageInput` と、そのページを認識したときの前のページの最後の年を返す。形式が違えば ValueError。"""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        if meta.get("format") != FORMAT:
            raise ValueError("pagedata_format")
        clf, stats = z["clf"], z["stats"]
        n = len(meta["cells"])
        if n != N_CELLS or clf.shape != (n, *CLF_SHAPE) or stats.shape[0] != n:
            raise ValueError("pagedata_shape")
        cells = []
        for i, m in enumerate(meta["cells"]):
            rec = z[f"rec_{i}"] if m["has_rec"] else None
            if rec is not None and (rec.ndim != 2 or rec.dtype != np.uint8):
                raise ValueError("pagedata_rec")
            cells.append(
                CellInput(
                    row=int(m["row"]),
                    col=int(m["col"]),
                    column=str(m["column"]),
                    rec=rec,
                    clf=clf[i],
                    stats=stats[i],
                    flags=[str(f) for f in m["flags"]],
                    blank=bool(m["blank"]),
                    faint_only=bool(m["faint_only"]),
                    ink_h_frac=float(m["ink_h_frac"]),
                )
            )
        page = PageInput(
            cells,
            z["title_rec"] if "title_rec" in z.files else None,
            z["number_rec"] if "number_rec" in z.files else None,
            str(meta["layout_status"]),
            [str(s) for s in meta["layout_notes"]],
        )
    py = meta.get("prev_year")
    return page, None if py is None else int(py)
