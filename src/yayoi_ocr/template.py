"""様式テンプレート(form_v1)。

座標系は「表の左上の隅(見出し帯の下辺の左端)を原点、表の幅を 1.0」とした等方の単位。
x は右向き、y は下向き。見出し帯・タイトル欄・No.欄は表の上(y が負)にある。
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from .models import ColumnKind


class Box(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0


class ColumnSpec(BaseModel):
    kind: ColumnKind
    x0: float
    x1: float


class FormTemplate(BaseModel):
    name: str
    version: int = 1
    description: str = ""
    n_rows: int
    # 列: 左から順に隣接していること(x1 == 次の列の x0)
    columns: list[ColumnSpec]
    # 行の境界の y(n_rows + 1 個)。row_edges[0] = 0(表の上端 = 帯の下辺)
    row_edges: list[float]
    band: Box  # 黒い見出し帯(y < 0)
    title_box: Box  # ページ上部の手書きの勘定科目名(ページタイトル)
    number_box: Box  # 右上の「No.」
    page_box: Box  # 正準ページ画像として出力する範囲
    canon_width: int = Field(default=2000, description="正準ページ画像での表の幅(px)")

    @model_validator(mode="after")
    def _check(self) -> FormTemplate:
        if len(self.row_edges) != self.n_rows + 1:
            raise ValueError("row_edges は n_rows + 1 個必要です")
        if abs(self.row_edges[0]) > 1e-9:
            raise ValueError("row_edges[0] は 0 である必要があります")
        if any(b <= a for a, b in zip(self.row_edges, self.row_edges[1:], strict=False)):
            raise ValueError("row_edges は単調増加である必要があります")
        for a, b in zip(self.columns, self.columns[1:], strict=False):
            if abs(a.x1 - b.x0) > 1e-9:
                raise ValueError("列は隣接している必要があります")
        if abs(self.columns[0].x0) > 1e-9 or abs(self.columns[-1].x1 - 1.0) > 1e-9:
            raise ValueError("列は 0.0〜1.0 を覆う必要があります")
        return self

    @property
    def vline_x(self) -> list[float]:
        """縦罫線の x(左端・列境界・右端)。"""
        return [self.columns[0].x0, *[c.x1 for c in self.columns]]

    @property
    def hline_y(self) -> list[float]:
        """横罫線の y(上端 = 帯の下辺、各行の下辺)。"""
        return list(self.row_edges)

    @property
    def table_height(self) -> float:
        return self.row_edges[-1]

    def cell_box(self, row: int, col: int) -> Box:
        c = self.columns[col]
        return Box(x0=c.x0, y0=self.row_edges[row], x1=c.x1, y1=self.row_edges[row + 1])


_TEMPLATE_DIR = "templates"


@lru_cache(maxsize=8)
def load_template(name: str = "form_v1") -> FormTemplate:
    """同梱のテンプレート(src/yayoi_ocr/templates/<name>.json)を読む。"""
    res = resources.files("yayoi_ocr").joinpath(_TEMPLATE_DIR, f"{name}.json")
    return FormTemplate.model_validate_json(res.read_text(encoding="utf-8"))


def load_template_file(path: str | Path) -> FormTemplate:
    return FormTemplate.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
