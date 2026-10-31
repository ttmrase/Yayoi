"""確定したセルを「見覚え」に反映する(推論側の学習)。

確定(利用者が確認した)ページの各セルから、次のページの認識が使う状態を更新する:
- 見覚えメモリ: 書かれた通りのテキストの確定セル(text)の画像埋め込みと確定テキスト。
- 摘要の頻度辞書: 展開後の摘要(〃・矢印・波線も展開後の値で数える)。
- 科目の語彙: 展開後の科目名。
- 年: ページの最後の行の年(次のページの先頭行が年を省略したときに継承する)。
Phase 3 の確認画面が「確定」を押したときにも、同じ関数を使う。
"""

from __future__ import annotations

from .grammar import repair_date
from .page import JobContext, PageResult
from .state import ConfirmedCell, update_from_confirmed


def confirm_page(ctx: JobContext, result: PageResult, cells: list[ConfirmedCell]) -> int:
    """確定セルで ctx を更新する(状態の更新 + 見覚えメモリへの追加)。メモリに入れた件数を返す。"""
    update_from_confirmed(ctx, cells)
    n_mem = 0
    for r, c in zip(result.cells, cells, strict=True):
        if c.kind == "blank":
            continue
        if (
            ctx.memory is not None
            and c.kind == "text"
            and c.raw
            and r.out is not None
            and r.out.embedding is not None
        ):
            ctx.memory.add(c.column, r.out.embedding, c.raw)
            n_mem += 1
    return n_mem


__all__ = ["ConfirmedCell", "confirm_page", "repair_date", "update_from_confirmed"]
