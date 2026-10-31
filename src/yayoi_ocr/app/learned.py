"""確定済みページから作る「学習状態」(摘要の頻度辞書・科目の語彙・金額の履歴・タイトル)。

確定したページの値から、次のページの認識が使う状態を**毎回作り直す**(確定の取り消し・再確定でも二重に数えない)。画像の特徴(見覚えメモリ)は
Phase 3 では扱わない(確信度への寄与を 0 にする方針。Phase 4 で、書き手の確定データから較正してから使う)。
プロファイルごとに分ける(ジョブが属するプロファイルの確定ページだけ)。語彙の初期値は設定(支払手段・タイトル)。
torch を読み込まない(認識のワーカーが `JobContext` に詰め替える)。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from ..recognize.state import ConfirmedCell, update_from_confirmed
from .retention import learned_since
from .settings import Settings

MAX_ACCOUNTS = 60


@dataclass
class LearnedState:
    accounts: list[str] = field(default_factory=list)  # 科目(支払手段)の閉じた語彙
    titles: list[str] = field(default_factory=list)  # ページのタイトルの語彙
    dictionary: dict[str, int] = field(default_factory=dict)  # 摘要の頻度辞書
    amount_history: dict[str, dict[int, int]] = field(default_factory=dict)  # 金額の履歴
    prev_year: int | None = None


def confirmed_cells(con: sqlite3.Connection, page_id: int) -> list[ConfirmedCell]:
    """確定したページのセル(行優先)。値は確定値(直していなければ認識の解決値)。"""
    rows = con.execute(
        "SELECT row, col, column_kind, kind, recognized_text, resolved_value, confirmed_value"
        " FROM cells WHERE page_id = ? ORDER BY row, col",
        (page_id,),
    ).fetchall()
    return [
        ConfirmedCell(
            r["row"],
            r["col"],
            r["column_kind"],
            r["kind"],
            r["recognized_text"],
            r["resolved_value"] if r["confirmed_value"] is None else r["confirmed_value"],
        )
        for r in rows
    ]


def confirmed_pages(con: sqlite3.Connection, profile_id: int) -> list[sqlite3.Row]:
    """学習に使える確定ページ(学習画面の「リセット」より後に確定したもの)を、確定した順(古い → 新しい)に。"""
    return con.execute(
        "SELECT p.id, p.title, p.confirmed_at, p.generation FROM pages p JOIN jobs j ON j.id = p.job_id"
        " WHERE j.profile_id = ? AND p.status = 'confirmed' AND COALESCE(p.confirmed_at, 0) >= ?"
        " ORDER BY COALESCE(p.confirmed_at, 0), p.id",
        (profile_id, learned_since(con)),
    ).fetchall()


def build_state(
    con: sqlite3.Connection,
    profile_id: int,
    settings: Settings,
    only_pages: set[int] | None = None,
) -> LearnedState:
    """`only_pages` があれば、そのページだけから作る(世代の評価で、学習用ページだけの文脈を作る)。"""
    ctx = LearnedState(
        accounts=[p.name for p in settings.payments],
        titles=[t.name for t in settings.titles],
    )
    pages = con.execute(
        "SELECT p.id, p.title FROM pages p JOIN jobs j ON j.id = p.job_id"
        " WHERE j.profile_id = ? AND p.status = 'confirmed' AND COALESCE(p.confirmed_at, 0) >= ?"
        " ORDER BY j.id, p.idx",
        (
            profile_id,
            learned_since(con),
        ),  # 学習画面の「リセット」より前に確定したページは、学習に使わない
    ).fetchall()
    for pg in pages:
        if only_pages is not None and pg["id"] not in only_pages:
            continue
        update_from_confirmed(ctx, confirmed_cells(con, pg["id"]))
        if pg["title"] and pg["title"] not in ctx.titles:
            ctx.titles.append(pg["title"])
    ctx.accounts = ctx.accounts[:MAX_ACCOUNTS]
    ctx.prev_year = (
        None  # 年はジョブの中の前のページから引き継ぐ(runner)。他のジョブからは持ち込まない
    )
    return ctx
