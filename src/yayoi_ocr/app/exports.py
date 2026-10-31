"""出力(確定したページ → CSV)。検証・重複出力の警告・履歴・ファイル。

**形式は暫定**(`PROVISIONAL`)。弥生のインポート形式(列・必須項目・税区分の値・日付の書き方)は Phase 5 で確定するので、形式は `Format` として
差し替えられる作りにしてある(`FORMATS`)。出力の検証・重複の判定・履歴は形式に依存しない。
実データの値(確定値・ファイル名)はログに出さない。履歴にも値は載せない(日時・形式・ページ数・行数・ファイルのハッシュ・設定のスナップショット)。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..recognize.grammar import parse_amount, parse_date
from .db import transaction
from .settings import Settings, readiness
from .settings import load as load_settings
from .store import StoreError
from .validate import ERROR_LABELS, validate_value

ENCODING = "cp932"
NEWLINE = "\r\n"

ISSUE_LABELS = {
    "page_not_confirmed": "確定していないページです",
    "title_missing": "ページのタイトルが未入力です",
    "title_unmapped": "タイトルが設定(タイトル → 借方科目)にありません",
    "debit_account_missing": "タイトルに対する借方科目が未設定です",
    "account_unmapped": "科目(支払手段)が設定にありません",
    "credit_account_missing": "支払手段に対する貸方科目が未設定です",
    "no_rows": "出力できる行がありません",
    **{k: v for k, v in ERROR_LABELS.items()},
}


@dataclass
class Row:
    """出力する仕訳 1 行(確定値から作る)。"""

    page_id: int
    row: int
    date: str  # 西暦 YYYY/MM/DD
    debit: str
    debit_sub: str
    credit: str
    credit_sub: str
    description: str
    amount: int
    tax: str


@dataclass(frozen=True)
class Format:
    id: str
    label: str
    provisional: bool
    header: tuple[str, ...]
    to_fields: Callable[[Row], list[str]]


def _provisional_fields(r: Row) -> list[str]:
    return [
        r.date,
        r.debit,
        r.debit_sub,
        r.credit,
        r.credit_sub,
        r.description,
        str(r.amount),
        r.tax,
    ]


FORMATS: dict[str, Format] = {
    "provisional_v0": Format(
        "provisional_v0",
        "仮の形式(弥生のインポート形式は Phase 5 で確定します)",
        True,
        ("日付", "借方科目", "借方補助科目", "貸方科目", "貸方補助科目", "摘要", "金額", "税区分"),
        _provisional_fields,
    ),
}
DEFAULT_FORMAT = "provisional_v0"


def get_format(fid: str) -> Format:
    f = FORMATS.get(fid)
    if f is None:
        raise StoreError(422, "unknown_format")
    return f


# ------------------------------------------------------------------ 確定値の取り出し


def _page_cells(con: sqlite3.Connection, page_id: int) -> dict[tuple[int, int], sqlite3.Row]:
    return {
        (r["row"], r["col"]): r
        for r in con.execute("SELECT * FROM cells WHERE page_id = ?", (page_id,))
    }


def _value(c: sqlite3.Row) -> str:
    return (c["resolved_value"] if c["confirmed_value"] is None else c["confirmed_value"]).strip()


def page_content_hash(con: sqlite3.Connection, page: sqlite3.Row) -> str:
    """ページの画像(ファイルのハッシュ)+ 確定値 + タイトルのハッシュ。同じものを 2 回出力していないかの判定に使う。"""
    cells = [
        [r["row"], r["col"], _value(r)]
        for r in con.execute(
            "SELECT row, col, confirmed_value, resolved_value FROM cells WHERE page_id = ?"
            " ORDER BY row, col",
            (page["id"],),
        )
    ]
    blob = json.dumps(
        [page["image_sha256"], page["source_index"], page["title"], cells], ensure_ascii=False
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def collect(
    con: sqlite3.Connection, page_ids: list[int], s: Settings
) -> tuple[list[Row], list[dict]]:
    """出力する行と、検証の指摘(ページ・セルへのリンクつき。値は含まない)。"""
    payments = {p.name: p for p in s.payments}
    titles = {t.name: t for t in s.titles}
    tax = {a.account: a.tax_category for a in s.default_tax}
    rows: list[Row] = []
    issues: list[dict] = []

    def issue(code: str, page_id: int, cell: sqlite3.Row | None = None, row: int | None = None):
        issues.append(
            {
                "code": code,
                "label": ISSUE_LABELS[code],
                "page_id": page_id,
                "cell_id": None if cell is None else cell["id"],
                "row": row if cell is None else cell["row"],
            }
        )

    ordered = con.execute(
        f"SELECT p.* FROM pages p JOIN jobs j ON j.id = p.job_id WHERE p.id IN ({','.join('?' * len(page_ids))})"
        " ORDER BY j.id, p.idx",
        page_ids,
    ).fetchall()
    if len(ordered) != len(set(page_ids)):
        raise StoreError(404, "page_not_found")
    for pg in ordered:
        pid = pg["id"]
        if pg["status"] != "confirmed":
            issue("page_not_confirmed", pid)
            continue
        title = (pg["title"] or "").strip()
        debit = debit_sub = ""
        if not title:
            issue("title_missing", pid)
        elif title not in titles:
            issue("title_unmapped", pid)
        elif not titles[title].debit_account:
            issue("debit_account_missing", pid)
        else:
            debit, debit_sub = titles[title].debit_account, titles[title].debit_sub
        cells = _page_cells(con, pid)
        n_before = len(rows)
        for r in sorted({k[0] for k in cells}):
            amount_cell = cells.get((r, 3))
            if amount_cell is None or not _value(amount_cell):
                continue
            ok = True
            vals = {
                col: cells[(r, c)]
                for c, col in enumerate(("date", "account", "description", "amount"))
            }
            for col, cell in vals.items():
                for code in validate_value(col, _value(cell), True):
                    issue(code, pid, cell)
                    ok = False
            acct = _value(vals["account"])
            pay = payments.get(acct)
            if acct and pay is None:
                issue("account_unmapped", pid, vals["account"])
                ok = False
            elif pay is not None and not pay.credit_account:
                issue("credit_account_missing", pid, vals["account"])
                ok = False
            d = parse_date(_value(vals["date"]))
            amt = parse_amount(_value(vals["amount"]))
            if not ok or not debit or d is None or d.era_year is None or amt is None or pay is None:
                continue
            rows.append(
                Row(
                    pid,
                    r,
                    f"{d.western_year()}/{d.month:02d}/{d.day:02d}",
                    debit,
                    debit_sub,
                    pay.credit_account,
                    pay.credit_sub,
                    _value(vals["description"]),
                    amt,
                    tax.get(debit, ""),
                )
            )
        if len(rows) == n_before and not any(i["page_id"] == pid for i in issues):
            issue("no_rows", pid)
    return rows, issues


# ------------------------------------------------------------------ 重複


def exported_hashes(con: sqlite3.Connection) -> dict[str, list[int]]:
    """ページの内容のハッシュ → それを出力した出力 ID のリスト。"""
    out: dict[str, list[int]] = {}
    for r in con.execute("SELECT export_id, content_sha256 FROM export_pages ORDER BY export_id"):
        out.setdefault(r["content_sha256"], []).append(r["export_id"])
    return out


def duplicates(con: sqlite3.Connection, page_ids: list[int]) -> list[dict]:
    seen = exported_hashes(con)
    out = []
    for pid in page_ids:
        pg = con.execute("SELECT * FROM pages WHERE id = ?", (pid,)).fetchone()
        if pg is None or pg["status"] != "confirmed":
            continue
        ex = seen.get(page_content_hash(con, pg))
        if ex:
            out.append({"page_id": pid, "export_ids": ex})
    return out


def candidates(con: sqlite3.Connection) -> list[dict]:
    """出力の対象にできるページ(確定済み)と、出力済みか。"""
    seen = exported_hashes(con)
    out = []
    for pg in con.execute(
        "SELECT p.*, j.name AS job_name FROM pages p JOIN jobs j ON j.id = p.job_id"
        " WHERE p.status = 'confirmed' ORDER BY j.id, p.idx"
    ).fetchall():
        out.append(
            {
                "id": pg["id"],
                "job_id": pg["job_id"],
                "job_name": pg["job_name"],
                "idx": pg["idx"],
                "title": pg["title"],
                "exported": page_content_hash(con, pg) in seen,
            }
        )
    return out


# ------------------------------------------------------------------ 検証・出力・履歴


def validate(con: sqlite3.Connection, page_ids: list[int], fmt: str = DEFAULT_FORMAT) -> dict:
    get_format(fmt)
    if not page_ids:
        raise StoreError(422, "no_pages")
    s = load_settings(con)
    rows, issues = collect(con, page_ids, s)
    return {
        "n_rows": len(rows),
        "issues": issues,
        "duplicates": duplicates(con, page_ids),
        "missing_settings": readiness(s),
        "provisional": get_format(fmt).provisional,
    }


def render(fmt: Format, rows: list[Row]) -> bytes:
    lines = [",".join(_q(x) for x in fmt.header)] + [
        ",".join(_q(x) for x in fmt.to_fields(r)) for r in rows
    ]
    return (NEWLINE.join(lines) + NEWLINE).encode(ENCODING)


def _q(v: str) -> str:
    return '"' + v.replace('"', '""') + '"' if any(ch in v for ch in ',"\r\n') else v


def run_export(
    con: sqlite3.Connection,
    exports_dir: Path,
    page_ids: list[int],
    fmt_id: str = DEFAULT_FORMAT,
    accept_duplicates: bool = False,
) -> dict:
    fmt = get_format(fmt_id)
    if not page_ids:
        raise StoreError(422, "no_pages")
    s = load_settings(con)
    rows, issues = collect(con, page_ids, s)
    if issues or readiness(s):
        raise StoreError(
            409, "validation_failed", {"issues": issues, "missing_settings": readiness(s)}
        )
    dups = duplicates(con, page_ids)
    if dups and not accept_duplicates:
        raise StoreError(409, "duplicate_pages", {"duplicates": dups})
    data = render(fmt, rows)
    sha = hashlib.sha256(data).hexdigest()
    now = time.time()
    exports_dir.mkdir(parents=True, exist_ok=True)
    with transaction(con):
        cur = con.execute(
            "INSERT INTO exports (format, format_label, created_at, n_pages, n_rows, file_sha256, settings_snapshot)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                fmt.id,
                fmt.label,
                now,
                len({r.page_id for r in rows}),
                len(rows),
                sha,
                s.model_dump_json(),
            ),
        )
        export_id = int(cur.lastrowid)
        name = f"export_{export_id:04d}_{time.strftime('%Y%m%d_%H%M%S', time.localtime(now))}.csv"
        (exports_dir / name).write_bytes(data)
        con.execute("UPDATE exports SET file_name = ? WHERE id = ?", (name, export_id))
        for pid in sorted({r.page_id for r in rows}):
            pg = con.execute("SELECT * FROM pages WHERE id = ?", (pid,)).fetchone()
            con.execute(
                "INSERT INTO export_pages (export_id, page_id, content_sha256) VALUES (?, ?, ?)",
                (export_id, pid, page_content_hash(con, pg)),
            )
    return get_export(con, export_id)


def _export_dict(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"],
        "format": r["format"],
        "format_label": r["format_label"],
        "created_at": r["created_at"],
        "n_pages": r["n_pages"],
        "n_rows": r["n_rows"],
        "file_sha256": r["file_sha256"],
        "file_name": r["file_name"],
    }


def get_export(con: sqlite3.Connection, export_id: int) -> dict:
    r = con.execute("SELECT * FROM exports WHERE id = ?", (export_id,)).fetchone()
    if r is None:
        raise StoreError(404, "export_not_found")
    return _export_dict(r)


def list_exports(con: sqlite3.Connection) -> list[dict]:
    return [_export_dict(r) for r in con.execute("SELECT * FROM exports ORDER BY id DESC")]


def export_file(con: sqlite3.Connection, exports_dir: Path, export_id: int) -> Path:
    e = get_export(con, export_id)
    path = (exports_dir / (e["file_name"] or "")).resolve()
    if exports_dir.resolve() not in path.parents or not path.is_file():
        raise StoreError(404, "export_file_not_found")
    return path
