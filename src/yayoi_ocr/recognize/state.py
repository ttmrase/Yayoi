"""確定セルから作る学習状態(画像を持たない部分)。torch を読み込まない(アプリの CI・認識を使わない起動でも使えるように)。

`recognize/learning.py`(見覚えメモリを含む)と `app/learned.py`(確定済みページから毎回作り直す)が共有する。
状態は `accounts`・`dictionary`・`amount_history`・`prev_year` を持つ任意のオブジェクト(`JobContext` や `app.learned.LearnedState`)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .grammar import norm_text, parse_date


@dataclass
class ConfirmedCell:
    row: int
    col: int
    column: str
    kind: str  # blank / text / ditto / arrow / elided(または wave)
    raw: str  # 書かれた通りの文字列
    resolved: str  # 展開後の値(日付は 'R7.11.19'、金額は '3456')


class LearnedLike(Protocol):
    accounts: list[str]
    dictionary: dict[str, int]
    amount_history: dict[str, dict[int, int]]
    prev_year: int | None


def update_from_confirmed(ctx: LearnedLike, cells: list[ConfirmedCell]) -> None:
    """確定セルから、画像を持たない学習状態(摘要の頻度辞書・科目の語彙・金額の履歴・年)を更新する。
    画面のアプリは、確定済みページからこの状態を作り直す(何度呼んでも同じ結果になるよう、呼び出し側が空の状態から始める)。"""
    last_year: int | None = None
    n = 4  # 列数(日付・科目・摘要・金額)
    for row in range(len(cells) // n):
        rc = cells[row * n : (row + 1) * n]
        amount = rc[3]
        if amount.kind != "blank" and amount.resolved.isdigit():
            v = int(amount.resolved)
            keys = []
            if rc[2].resolved:
                keys.append(f"d:{norm_text(rc[2].resolved)}")
            if rc[1].resolved:
                keys.append(f"a:{rc[1].resolved}")
            for k in keys:
                h = ctx.amount_history.setdefault(k, {})
                h[v] = h.get(v, 0) + 1
    for c in cells:
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
    if last_year is not None:
        ctx.prev_year = last_year
