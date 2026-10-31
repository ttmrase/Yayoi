"""行ごとの税区分(確認画面での上書き)と、摘要ごとの記憶。

- 行の借方税区分は、**行ごとの指定 > 摘要の記憶からの提案 > 科目ごとの既定**の順で決まる。`source` で区別する:
  `user`(利用者が選んだ)/ `suggested`(記憶からの提案。画面に「提案」と出す)/ `default`(科目ごとの既定)。
- ページを確定すると、(1) 提案のままの行は提案の値を固定する(`row_tax` に `suggested` で保存。以後、記憶が変わっても出力は変わらない)、
  (2) 利用者が選んだ行は、その摘要(`tax.normalize_description`)と税区分の対応をプロファイルに記憶する。
  選んだ税区分が科目ごとの既定と同じなら、記憶していた対応を消す(提案を断った)。再オープンすると固定した提案は外れ、また動的に提案される。
- 記憶は設定画面で編集・削除できる(`/api/tax-memory`)。摘要の文字列を保存するので、学習画面の精度レポートには含めない。
- 免税事業者は税区分がすべて「対象外」なので、行の指定も記憶も使わない。
"""

from __future__ import annotations

import sqlite3
import time

from . import settings as app_settings
from . import tax
from .db import DEFAULT_PROFILE_ID, transaction
from .store import StoreError, _bump, _require_open

SOURCE_USER = "user"
SOURCE_SUGGESTED = "suggested"
SOURCE_DEFAULT = "default"
AMOUNT_COL, DESC_COL = 3, 2


def account_default(s: app_settings.Settings, title: str | None) -> str:
    """ページのタイトル → 借方科目 → 科目ごとの既定の税区分。分からなければ空。"""
    t = next((x for x in s.titles if x.name == (title or "").strip()), None)
    if t is None or not t.debit_account:
        return ""
    return next((a.tax_category for a in s.default_tax if a.account == t.debit_account), "")


def _value(c: sqlite3.Row) -> str:
    return (c["resolved_value"] if c["confirmed_value"] is None else c["confirmed_value"]).strip()


def _rows(con: sqlite3.Connection, page_id: int) -> dict[int, str]:
    """金額のある行 → 摘要(確定値、または現在の値)。"""
    cells = {
        (r["row"], r["col"]): r
        for r in con.execute("SELECT * FROM cells WHERE page_id = ?", (page_id,))
    }
    return {
        row: _value(cells[(row, DESC_COL)]) if (row, DESC_COL) in cells else ""
        for (row, col), c in sorted(cells.items())
        if col == AMOUNT_COL and _value(c)
    }


def _profile_id(con: sqlite3.Connection, p: sqlite3.Row) -> int:
    r = con.execute("SELECT profile_id FROM jobs WHERE id = ?", (p["job_id"],)).fetchone()
    return int(r["profile_id"]) if r else DEFAULT_PROFILE_ID


def _memory_for(con: sqlite3.Connection, profile_id: int, keys: set[str]) -> dict[str, sqlite3.Row]:
    if not keys:
        return {}
    ks = sorted(keys)
    q = ",".join("?" * len(ks))
    return {
        r["description_key"]: r
        for r in con.execute(
            f"SELECT * FROM tax_memory WHERE profile_id = ? AND description_key IN ({q})",
            [profile_id, *ks],
        )
    }


def page_rows(con: sqlite3.Connection, p: sqlite3.Row) -> dict:
    """確認画面に渡す、行ごとの借方税区分(金額のある行)。免税事業者は `enabled: False`。"""
    s = app_settings.load(con)
    if s.taxpayer == "exempt":
        return {"enabled": False, "options": [tax.EXEMPT], "rows": []}
    allowed = tax.categories(s.taxpayer, s.tax_scheme, s.tax_method)
    default = account_default(s, p["title"])
    rows = _rows(con, p["id"])
    overrides = {
        r["row"]: r for r in con.execute("SELECT * FROM row_tax WHERE page_id = ?", (p["id"],))
    }
    keys = {tax.normalize_description(d) for d in rows.values() if d}
    memory = _memory_for(con, _profile_id(con, p), keys)
    out = []
    for row, desc in rows.items():
        ov = overrides.get(row)
        if ov is not None:
            category, source = ov["tax_category"], ov["source"]
        else:
            m = memory.get(tax.normalize_description(desc)) if desc else None
            if m is not None and m["tax_category"] in allowed and m["tax_category"] != default:
                category, source = m["tax_category"], SOURCE_SUGGESTED
            else:
                category, source = default, SOURCE_DEFAULT
        out.append({"row": row, "category": category, "source": source, "default": default})
    return {"enabled": True, "options": allowed, "rows": out}


def set_row_tax(con: sqlite3.Connection, page_id: int, row: int, category: str | None) -> None:
    """行の借方税区分を利用者が選ぶ(None で指定を外す)。許可リストにあるものだけ。確定済みのページは再オープンしてから。"""
    with transaction(con):
        _require_open(con, page_id)
        s = app_settings.load(con)
        if s.taxpayer == "exempt":
            raise StoreError(409, "tax_exempt")
        if row not in _rows(con, page_id):
            raise StoreError(404, "row_not_found")
        if category is None:
            con.execute("DELETE FROM row_tax WHERE page_id = ? AND row = ?", (page_id, row))
        else:
            if category not in tax.categories(s.taxpayer, s.tax_scheme, s.tax_method):
                raise StoreError(422, "tax_category_not_allowed")
            con.execute(
                "INSERT INTO row_tax (page_id, row, tax_category, source) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(page_id, row) DO UPDATE SET tax_category = excluded.tax_category,"
                " source = excluded.source",
                (page_id, row, category, SOURCE_USER),
            )
        _bump(con, page_id)


def on_confirm(con: sqlite3.Connection, p: sqlite3.Row) -> None:
    """確定: 提案のままの行は値を固定し、利用者が選んだ行は摘要との対応を記憶する(`confirm_page` のトランザクションの中)。"""
    s = app_settings.load(con)
    if s.taxpayer == "exempt":
        return
    data = page_rows(con, p)
    rows = _rows(con, p["id"])
    profile_id = _profile_id(con, p)
    now = time.time()
    for r in data["rows"]:
        desc = rows.get(r["row"], "")
        key = tax.normalize_description(desc)
        if r["source"] == SOURCE_SUGGESTED:
            con.execute(
                "INSERT OR REPLACE INTO row_tax (page_id, row, tax_category, source) VALUES (?, ?, ?, ?)",
                (p["id"], r["row"], r["category"], SOURCE_SUGGESTED),
            )
            if key:
                con.execute(
                    "UPDATE tax_memory SET n_used = n_used + 1, updated_at = ? WHERE profile_id = ? AND description_key = ?",
                    (now, profile_id, key),
                )
        elif r["source"] == SOURCE_USER and key:
            if r["category"] == r["default"]:  # 既定と同じ = 記憶していた対応を断った
                con.execute(
                    "DELETE FROM tax_memory WHERE profile_id = ? AND description_key = ?",
                    (profile_id, key),
                )
            else:
                con.execute(
                    "INSERT INTO tax_memory (profile_id, description_key, display, tax_category, n_used, updated_at)"
                    " VALUES (?, ?, ?, ?, 1, ?)"
                    " ON CONFLICT(profile_id, description_key) DO UPDATE SET display = excluded.display,"
                    " tax_category = excluded.tax_category, n_used = n_used + 1, updated_at = excluded.updated_at",
                    (profile_id, key, desc, r["category"], now),
                )


def on_reopen(con: sqlite3.Connection, page_id: int) -> None:
    """再オープン: 確定のときに固定した提案を外す(利用者が選んだ行は残す)。"""
    con.execute("DELETE FROM row_tax WHERE page_id = ? AND source = ?", (page_id, SOURCE_SUGGESTED))


# ------------------------------------------------------------------ 記憶の編集


def memory_list(con: sqlite3.Connection, profile_id: int = DEFAULT_PROFILE_ID) -> dict:
    s = app_settings.load(con)
    allowed = tax.categories(s.taxpayer, s.tax_scheme, s.tax_method)
    entries = [
        {
            "id": r["id"],
            "description": r["display"],
            "tax_category": r["tax_category"],
            "n_used": r["n_used"],
            "updated_at": r["updated_at"],
            "allowed": r["tax_category"] in allowed,
        }
        for r in con.execute(
            "SELECT * FROM tax_memory WHERE profile_id = ? ORDER BY updated_at DESC, id DESC",
            (profile_id,),
        )
    ]
    return {"entries": entries, "options": allowed, "exempt": s.taxpayer == "exempt"}


def memory_update(con: sqlite3.Connection, mem_id: int, category: str) -> None:
    s = app_settings.load(con)
    if category not in tax.categories(s.taxpayer, s.tax_scheme, s.tax_method):
        raise StoreError(422, "tax_category_not_allowed")
    with transaction(con):
        cur = con.execute(
            "UPDATE tax_memory SET tax_category = ?, updated_at = ? WHERE id = ?",
            (category, time.time(), mem_id),
        )
        if cur.rowcount == 0:
            raise StoreError(404, "memory_not_found")


def memory_delete(con: sqlite3.Connection, mem_id: int) -> None:
    with transaction(con):
        cur = con.execute("DELETE FROM tax_memory WHERE id = ?", (mem_id,))
        if cur.rowcount == 0:
            raise StoreError(404, "memory_not_found")


def memory_clear(con: sqlite3.Connection, profile_id: int = DEFAULT_PROFILE_ID) -> int:
    with transaction(con):
        return con.execute("DELETE FROM tax_memory WHERE profile_id = ?", (profile_id,)).rowcount


def overrides_for_export(con: sqlite3.Connection, page_id: int) -> dict[int, str]:
    """出力に使う行ごとの税区分(利用者が選んだもの、確定のときに固定した提案)。"""
    return {
        r["row"]: r["tax_category"]
        for r in con.execute("SELECT row, tax_category FROM row_tax WHERE page_id = ?", (page_id,))
    }
