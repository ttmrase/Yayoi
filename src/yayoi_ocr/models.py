"""ドメインモデル: Document → Page → Row → Cell。

座標系:
- `bbox` … 位置合わせ後の正準ページ座標(様式テンプレートの単位を canon_width px に拡大したもの)
- `polygon` … 入力画像(向き補正前の元画像)上の座標。確認画面での重ね表示に使う
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ColumnKind(StrEnum):
    DATE = "date"
    ACCOUNT = "account"
    DESCRIPTION = "description"
    AMOUNT = "amount"


class CellFlag(StrEnum):
    BLANK = "blank"  # 空欄
    DITTO = "ditto"  # 同上記号(〃 など)
    ARROW = "arrow"  # 矢印・線(「→」「⌐」等、下の行へ続く意味)
    NEEDS_REVIEW = "needs_review"  # 要確認(確信度が低い・文法違反など)
    USER_CORRECTED = "user_corrected"  # ユーザーが修正済み
    LAYOUT_UNCERTAIN = "layout_uncertain"  # 罫線の検出に自信がない
    OVERFLOW = "overflow"  # セルからはみ出した記入あり


class CellStatus(StrEnum):
    UNCONFIRMED = "unconfirmed"  # 未確認
    CONFIRMED = "confirmed"  # 確定(学習に使ってよい)


class PageStatus(StrEnum):
    OK = "ok"
    NEEDS_REVIEW = "needs_review"  # レイアウト検出に自信がない → 確認画面で手動補正


class BBox(BaseModel):
    x: float
    y: float
    w: float
    h: float

    @property
    def x1(self) -> float:
        return self.x + self.w

    @property
    def y1(self) -> float:
        return self.y + self.h


Point = tuple[float, float]


class TextField(BaseModel):
    """認識対象の 1 領域(セル・タイトル・No. 共通)。"""

    bbox: BBox | None = None
    polygon: list[Point] = Field(default_factory=list)
    raw_text: str = ""  # 書かれた通りの文字列(「〃」等の記号を含む)
    resolved_value: str | None = None  # 同上の展開・補完後の値
    char_confidences: list[float] = Field(default_factory=list)
    flags: set[CellFlag] = Field(default_factory=set)
    status: CellStatus = CellStatus.UNCONFIRMED
    ink_ratio: float = 0.0  # 罫線除去後のインク量(空欄判定の根拠)


class Cell(TextField):
    column: ColumnKind


class Row(BaseModel):
    index: int
    cells: list[Cell]

    def cell(self, column: ColumnKind) -> Cell:
        for c in self.cells:
            if c.column == column:
                return c
        raise KeyError(column)


class ImageRef(BaseModel):
    """画像そのものはモデルに持たず、参照だけ保持する(画像は private データ領域)。"""

    path: str | None = None
    sha256: str | None = None
    width: int
    height: int
    source_page_index: int = 0  # PDF / 複数ページ TIFF 内のページ番号
    rotation_applied: int = 0  # 向き補正で適用した回転(度, 時計回り 0/90/180/270)


class Page(BaseModel):
    index: int
    title: TextField = Field(default_factory=TextField)
    number: TextField = Field(default_factory=TextField)  # 右上の「No.」
    image: ImageRef
    rows: list[Row] = Field(default_factory=list)
    status: PageStatus = PageStatus.OK
    notes: list[str] = Field(default_factory=list)  # 要確認の理由など(実データの値は入れない)


class Document(BaseModel):
    id: str
    pages: list[Page] = Field(default_factory=list)
