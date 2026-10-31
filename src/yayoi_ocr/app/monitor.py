"""採用した世代の監視と自動ロールバック(実行時の安全網)。設計は `docs/phase4_design.md`。

採用の判定(`gate.py`)は、少数の評価ページでの統計なので、見逃しの悪化を見逃しうる。そこで、採用した世代で認識したページが確定するたびに、
**確定時に「要確認でなかったのに修正されたセル」(観測した見逃し)**を世代ごと・列ごとに数え、次のいずれかに当たったら、自動で親の世代に戻す:

- **金額で、観測した見逃しが 1 件以上**(`halted_amount_miss`)。
- **全体で、観測した見逃し率が、Clopper-Pearson の下側限界(片側 95%)で目標(0.5%)を超えた**(`halted_miss_rate`)。分母は、その世代で認識して確定したページの全セル。

止めた世代は `halted`(安全のため停止。戻せない)にし、使用中だったら親の世代に戻す。学習画面に理由を出し、画面の上部に通知を出す(`gens.notices`)。
次の追加学習は、停止の時点より後に確定したデータで判断する(確定データを増やしてから再挑戦する)。
観測した見逃しは、利用者が要確認でないセルまで見て直した分だけなので、実際の見逃しはこれ以上ある(下限)。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ..recognize.confidence import cp_lower
from . import gens

TARGET_MISS = 0.005  # 全体の見逃し率の目標
ALPHA = 0.05
CORRECTED = ("corrected", "candidate", "expanded_edit")
R_AMOUNT = "halted_amount_miss"
R_RATE = "halted_miss_rate"


def observed(con: sqlite3.Connection, gen_id: int) -> dict:
    """この世代で認識して確定したページの、観測した見逃し(要確認でなかったのに直したセル)の列ごとの件数と、全セル数・ページ数。値は含まない。
    提案モードで認識したページは、親(この世代)の判定と値で数える(提案モードの値が違うセルは要確認になっているが、親の見逃しとして数える)。"""
    q = ",".join("?" * len(CORRECTED))
    cols: dict[str, int] = {}
    n_cells = 0
    for r in con.execute(
        "SELECT c.column_kind AS col, COUNT(*) AS n,"
        " SUM(CASE WHEN c.parent_review IS NULL"
        f"  THEN (c.kind != 'blank' AND c.review = 0 AND c.edit_kind IN ({q}))"
        # 提案モードで認識したページ: 親の世代が要確認にしなかったのに、確定値が親の値と違うセルを、親の見逃しとして数える
        # (利用者が実際に受ける見逃しは親の部分集合だが、親の世代の質の監視は、親の判定と値で行う)
        "  ELSE (c.parent_review = 0 AND (c.kind != 'blank' OR COALESCE(c.parent_value, '') != '')"
        "        AND COALESCE(c.confirmed_value, c.resolved_value) != COALESCE(c.parent_value, c.resolved_value)) END) AS k"
        " FROM cells c JOIN pages p ON p.id = c.page_id"
        " WHERE p.generation = ? AND p.status = 'confirmed' GROUP BY c.column_kind",
        (*CORRECTED, gen_id),
    ):
        cols[r["col"]] = int(r["k"] or 0)
        n_cells += int(r["n"])
    n_pages = int(
        con.execute(
            "SELECT COUNT(*) FROM pages WHERE generation = ? AND status = 'confirmed'", (gen_id,)
        ).fetchone()[0]
    )
    return {
        "by_column": cols,
        "k_total": sum(cols.values()),
        "n_cells": n_cells,
        "n_pages": n_pages,
    }


def decide(obs: dict) -> str | None:
    """停止すべきなら理由コード。"""
    if obs["by_column"].get("amount", 0) >= 1:
        return R_AMOUNT
    if cp_lower(obs["k_total"], obs["n_cells"], ALPHA) > TARGET_MISS:
        return R_RATE
    return None


def check_after_confirm(con: sqlite3.Connection, models_root: Path, page_id: int) -> dict | None:
    """ページを確定した直後に呼ぶ。そのページを認識した世代が、使用中の追加学習した世代で、停止の条件に当たったら止める(止めた世代の情報)。"""
    row = con.execute(
        "SELECT g.* FROM pages p JOIN generations g ON g.id = p.generation WHERE p.id = ?",
        (page_id,),
    ).fetchone()
    if row is None or row["kind"] != "trained" or row["status"] != "active":
        return None
    obs = observed(con, row["id"])
    code = decide(obs)
    if code is None:
        return None
    last = con.execute(
        "SELECT MAX(p.confirmed_at) FROM pages p JOIN jobs j ON j.id = p.job_id"
        " WHERE j.profile_id = ? AND p.status = 'confirmed'",
        (row["profile_id"],),
    ).fetchone()[0]
    info = {
        "k_total": obs["k_total"],
        "k_amount": obs["by_column"].get("amount", 0),
        "n_cells": obs["n_cells"],
        "n_pages": obs["n_pages"],
        "cutoff": float(last or 0.0),
    }
    return gens.halt(con, models_root, row["id"], code, info)
