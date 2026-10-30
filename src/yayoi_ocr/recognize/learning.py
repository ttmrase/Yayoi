"""確定したセルを「見覚え」に反映する(推論側の学習)。

確定(利用者が確認した)ページの各セルから、次のページの認識が使う状態を更新する:
- 見覚えメモリ: 書かれた通りのテキストの確定セル(text)の画像埋め込みと確定テキスト。
- 摘要の頻度辞書: 展開後の摘要(〃・矢印・波線も展開後の値で数える)。
- 科目の語彙: 展開後の科目名。
- 年: ページの最後の行の年(次のページの先頭行が年を省略したときに継承する)。
Phase 3 の確認画面が「確定」を押したときにも、同じ関数を使う。
"""

from __future__ import annotations

from dataclasses import dataclass

from .grammar import parse_date, repair_date
from .page import JobContext, PageResult


@dataclass
class ConfirmedCell:
    row: int
    col: int
    column: str
    kind: str  # blank / text / ditto / arrow / elided(または wave)
    raw: str  # 書かれた通りの文字列
    resolved: str  # 展開後の値(日付は 'R7.11.19'、金額は '3456')


def confirm_page(ctx: JobContext, result: PageResult, cells: list[ConfirmedCell]) -> int:
    """確定セルで ctx を更新する。メモリに入れた件数を返す。"""
    n_mem = 0
    last_year: int | None = None
    for r, c in zip(result.cells, cells, strict=True):
        if c.kind == "blank":
            continue
        if c.column == "description" and c.resolved:
            ctx.dictionary[c.resolved] = ctx.dictionary.get(c.resolved, 0) + 1
        if c.column == "account" and c.resolved and c.resolved not in ctx.accounts:
            ctx.accounts.append(c.resolved)
        if c.column == "date" and c.resolved:
            d = parse_date(c.resolved)
            if d is not None and d.era_year is not None:
                last_year = d.era_year
        if (
            ctx.memory is not None
            and c.kind == "text"
            and c.raw
            and r.out is not None
            and r.out.embedding is not None
        ):
            ctx.memory.add(c.column, r.out.embedding, c.raw)
            n_mem += 1
    if last_year is not None:
        ctx.prev_year = last_year
    return n_mem


__all__ = ["ConfirmedCell", "confirm_page", "repair_date"]
