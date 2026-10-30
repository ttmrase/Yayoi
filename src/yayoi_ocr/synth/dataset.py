"""合成帳票のセル単位データセット: 生成 → レイアウト解析 → セル画像・統計・正解をページごとに保存する。

認識・分類器・後処理の学習と評価で共通に使う(レイアウト解析は 1 ページ数秒かかるので、結果をキャッシュする)。
保存先は **リポジトリの外**(`$YAYOI_PRIVATE_DATA/datasets/…`)。合成データでも画像・正解はコミットしない。

1 ページ = 1 つの .npz:
- `meta`: JSON 文字列(ページのメタ情報とセルごとのメタ情報)
- `clf`: (N, 64, 448) uint8 … 分類器用(セル全体・罫線除去済み)
- `rec_<i>`: (H, W) uint8 … 認識(PARSeq)用のグレー画像(インクがあるセルだけ)
- `rec_title` / `rec_number`: タイトル欄・No. 欄の認識用グレー画像(インクがあれば)
pickle は使わない(`np.load(allow_pickle=False)` で読める)。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..config import assert_safe_output_dir
from ..layout.metrics import polygon_iou
from ..layout.pipeline import analyze_image
from ..models import CellFlag
from ..recognize.crops import classifier_input, recognition_image
from .generator import SynthConfig, generate_sample

COLUMNS = ("date", "account", "description", "amount")

# 劣化の組(評価の層別用)。None = 既定(全種類・解像度 80〜400 dpi)
PRESETS: dict[str, dict] = {
    "clean": {"degrade": False, "dpi": 200.0},
    # 200 dpi 固定で、低解像度以外の劣化をすべてかける(典型的なスキャン・スマホ撮影の目安)
    "typical": {
        "degradations": (
            "rotation",
            "perspective",
            "curve",
            "lighting",
            "contrast",
            "blur",
            "noise",
            "jpeg",
            "cut",
        ),
    },
    "default": {},  # 全種類(80〜400 dpi の対数一様)。最悪側の目安
}


def preset_config(name: str, **overrides) -> SynthConfig:
    """プリセット名(clean / typical / default / 劣化名 1 種類)→ SynthConfig(seed 等は overrides)。"""
    from .generator import ALL_DEGRADATIONS

    if name in PRESETS:
        kw = dict(PRESETS[name])
    elif name in ALL_DEGRADATIONS:
        kw = {"degradations": (name,)}
        if name != "lowres":
            kw["dpi"] = 200.0
    else:
        raise ValueError(f"未知のプリセット: {name}")
    kw.update(overrides)
    return SynthConfig(**kw)


@dataclass
class CellRecord:
    row: int
    col: int
    column: str
    kind: str  # 正解の書かれ方: blank / text / ditto / arrow / elided
    raw: str  # 書かれた通りの文字列(〃・→・頭〜尾 を含む)
    resolved: str  # 展開後の値
    head: str = ""  # elided: 頭
    tail: str = ""  # elided: 尾
    annotation: str = ""
    overflow_gt: bool = False  # 正解: セルからはみ出して書いた
    stray: bool = False  # 正解: 余計な点・線がある
    iou: float = 1.0  # レイアウトのセル境界と正解の IoU
    ink_px: int = 0
    ink_ratio: float = 0.0
    blank: bool = False
    faint_only: bool = False
    flags: list[str] = field(default_factory=list)  # レイアウトのフラグ(blank / ditto_likely / …)
    ink_h_frac: float = 0.0
    ink_w_frac: float = 0.0
    has_rec: bool = False


@dataclass
class PageData:
    meta: dict
    cells: list[CellRecord]
    clf: np.ndarray  # (N, 64, 448) uint8
    rec: dict[int, np.ndarray]  # セル番号(row * 4 + col)→ グレー画像
    fields: dict[str, np.ndarray] = field(default_factory=dict)  # "title" / "number" → グレー画像

    def cell(self, row: int, col: int) -> CellRecord:
        return self.cells[row * len(COLUMNS) + col]

    def rec_image(self, row: int, col: int) -> np.ndarray | None:
        """認識用のグレー画像(2 次元 uint8)。インクの無いセルは None。"""
        return self.rec.get(row * len(COLUMNS) + col)


def build_page(cfg: SynthConfig) -> PageData:
    """1 ページを生成してレイアウト解析し、セルごとのデータにする。"""
    s = generate_sample(cfg)
    res = analyze_image(s.image)
    row_h = round(float(np.median(np.diff(res.tpl.row_edges))) * res.tpl.canon_width)
    cells: list[CellRecord] = []
    clf = np.zeros((29 * len(COLUMNS), 64, 448), np.uint8)
    rec: dict[int, np.ndarray] = {}
    for r in range(29):
        for c, col in enumerate(COLUMNS):
            plan = s.plan.rows[r][col]
            crop = res.crops[(r, c)]
            lc = res.page.rows[r].cells[c]
            ri = recognition_image(crop, row_h)
            idx = r * len(COLUMNS) + c
            clf[idx] = classifier_input(crop)
            if ri is not None:
                rec[idx] = ri.image[:, :, 0].copy()
            cells.append(
                CellRecord(
                    row=r,
                    col=c,
                    column=col,
                    kind=plan.kind,
                    raw=plan.raw,
                    resolved=plan.resolved,
                    head=plan.head,
                    tail=plan.tail,
                    annotation=plan.annotation,
                    overflow_gt=plan.overflow,
                    stray=plan.stray,
                    iou=float(polygon_iou(np.array(lc.polygon), s.cell_polys[(r, c)])),
                    ink_px=int(crop.ink_px),
                    ink_ratio=float(crop.ink_ratio),
                    blank=bool(crop.blank),
                    faint_only=bool(crop.faint_only),
                    flags=sorted(f.value for f in lc.flags if isinstance(f, CellFlag)),
                    ink_h_frac=float(ri.ink_height_frac) if ri else 0.0,
                    ink_w_frac=float(ri.ink_width_frac) if ri else 0.0,
                    has_rec=ri is not None,
                )
            )
    fields: dict[str, np.ndarray] = {}
    for name, fc in (("title", res.title_crop), ("number", res.number_crop)):
        if fc is not None and fc.gray is not None:
            fri = recognition_image(fc, round(fc.gray.shape[0] / 1.5))
            if fri is not None:
                fields[name] = fri.image[:, :, 0].copy()
    meta = {
        "seed": cfg.seed,
        "writer_id": s.writer.writer_id,
        "font": Path(s.writer.font_path).stem,
        "dpi": float(s.meta.get("dpi", 0.0)),
        "degradations": list(cfg.degradations) if cfg.degradations is not None else "all",
        "degrade": cfg.degrade,
        "strength": cfg.degrade_strength,
        "visible_fraction": float(s.visible_fraction),
        "status": res.page.status.value,
        "notes": list(res.page.notes),
        "title": s.plan.title,
        "number": s.plan.number,
        "row_h_px": row_h,
    }
    return PageData(meta, cells, clf, rec, fields)


def save_page(path: str | Path, page: PageData) -> None:
    p = Path(path)
    assert_safe_output_dir(p.parent)
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"rec_{i}": im for i, im in page.rec.items()}
    arrays.update({f"rec_{k}": im for k, im in page.fields.items()})
    meta = json.dumps(
        {"page": page.meta, "cells": [asdict(c) for c in page.cells]}, ensure_ascii=False
    )
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(meta), clf=page.clf, **arrays)
    tmp.replace(p)


def load_page(path: str | Path) -> PageData:
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        rec = {int(k[4:]): z[k] for k in z.files if k.startswith("rec_") and k[4:].isdigit()}
        fields = {k[4:]: z[k] for k in z.files if k in ("rec_title", "rec_number")}
        clf = z["clf"]
    cells = [CellRecord(**c) for c in meta["cells"]]
    return PageData(meta["page"], cells, clf, rec, fields)


def list_pages(directory: str | Path) -> list[Path]:
    return sorted(Path(directory).glob("page_*.npz"))
