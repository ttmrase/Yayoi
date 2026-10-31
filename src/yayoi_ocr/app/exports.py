"""出力(確定したページ → 弥生に取り込む CSV)。検証・行単位の重複警告・履歴・ファイル。仕様は `docs/yayoi_export.md`。

形式(`FORMATS`):
- `yayoi_import`(既定): 弥生の仕訳日記帳へのインポート形式(1 仕訳 = 1 行、25 項目、Shift_JIS、CRLF、ヘッダーなし)。
- `smart_torihiki`: スマート取引取込の CSV(日付・入金・出金・摘要。支払手段ごとにファイルを分け、200KB 未満に分割)。勘定科目は指定できない(タイトルは失われる)。

出力する値は書き換えない(数式対策の `'` の付加などはしない)。出力前の検証で、エラー(出力不可)と注意(出力は止めない)を、セルへのリンクつきで返す。
実データの値(確定値・ファイル名)はログに出さない。履歴にも値は載せない(日時・形式・ファイル名・ページ数・行数・ファイルのハッシュ・設定のスナップショット)。
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..recognize.grammar import parse_amount, parse_date
from .db import transaction
from .settings import Settings, readiness
from .settings import load as load_settings
from .store import StoreError
from .validate import ERROR_LABELS, validate_value

ENCODING = "cp932"
NEWLINE = "\r\n"
MAX_DESC_BYTES_YAYOI = 64  # 弥生のインポート: 摘要は cp932 で 64 バイトを超える部分が切り捨てられる
MAX_DESC_CHARS_SMART = 80  # スマート取引取込: 摘要は 80 文字を超える部分が切り捨てられる
SMART_FILE_LIMIT = 190_000  # スマート取引取込は 1 ファイル 200KB 未満(余裕を持って分割する)
FORMULA_PREFIX = ("=", "+", "-", "@", "\t")  # Excel で開くと数式として解釈され得る摘要の先頭

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"

ISSUE_LABELS = {
    "page_not_confirmed": "確定していないページです",
    "title_missing": "ページのタイトルが未入力です",
    "title_unmapped": "タイトルが設定(タイトル → 借方科目)にありません",
    "debit_account_missing": "タイトルに対する借方科目が未設定です",
    "account_unmapped": "科目(支払手段)が設定にありません",
    "credit_account_missing": "支払手段に対する貸方科目が未設定です",
    "tax_category_missing": "借方科目の既定の税区分が未設定です(借方税区分は必須です)",
    "no_rows": "出力できる行がありません",
    "description_too_long": "摘要が長すぎます(弥生で切り捨てられます)",
    "formula_like": "摘要の先頭が = + - @ TAB のため、Excel で開くと数式として扱われる可能性があります(弥生への取り込みには影響しません。CSV は Excel で開かず、弥生で直接取り込んでください)",
    "exclusive_tax_amount_empty": "税抜経理の設定ですが、借方税金額は空で出力されます(弥生側で税額を確認してください)",
    "title_lost": "スマート取引取込では勘定科目を指定できないため、ページのタイトル(借方の科目)は失われます(取り込み時に取引手段を選び、相手科目は弥生が摘要から推測します)",
    **dict(ERROR_LABELS),
}
WARNING_CODES = {"formula_like", "exclusive_tax_amount_empty", "title_lost"}


@dataclass
class Row:
    """出力する取引 1 行(確定値から作る)。"""

    page_id: int
    row: int
    date: str  # 西暦 yyyy/mm/dd
    debit: str
    debit_sub: str
    debit_tax: str
    credit: str
    credit_sub: str
    payment: str  # 帳票の「科目」欄に書かれた支払手段の名前(スマート取引取込のファイル分けに使う)
    description: str
    amount: int
    content_sha256: str = ""
    cell_ids: dict[str, int] = field(default_factory=dict)  # 列 → セル ID(エラーのリンク用)


@dataclass
class OutFile:
    name: str  # 拡張子つきのファイル名(出力 ID などを付ける前の素名)
    data: bytes
    n_rows: int
    payment: str | None = None


@dataclass(frozen=True)
class Format:
    id: str
    label: str
    notes: tuple[str, ...]  # 出力画面に表示する公式の注意事項
    check: Callable[[list[Row], Settings], list[dict]]  # 形式固有の注意・エラー(issue の辞書)
    build: Callable[[list[Row], Settings], list[OutFile]]


# ------------------------------------------------------------------ CSV の書式


def _q(v: str) -> str:
    """カンマ・ダブルクォート・改行を含む項目はダブルクォートで囲む(空の項目もカンマは残る)。値は書き換えない。"""
    return '"' + v.replace('"', '""') + '"' if any(ch in v for ch in ',"\r\n') else v


def _encode(text: str) -> bytes:
    try:
        return text.encode(ENCODING)
    except (
        UnicodeEncodeError
    ) as exc:  # 検証で弾いているはずの経路。万一でも 500 にせず、分かる形で断る
        raise StoreError(409, "cp932_unencodable") from exc


def _line(fields: list[str]) -> str:
    return ",".join(_q(x) for x in fields) + NEWLINE


def yayoi_fields(r: Row, s: Settings) -> list[str]:
    """弥生インポート形式の 1 行(A〜Y の 25 項目)。"""
    tax = "対象外" if s.taxpayer == "exempt" else r.debit_tax
    amt = str(r.amount)
    return [
        "2000",  # A 識別フラグ
        "",  # B 伝票No.
        "",  # C 決算
        r.date,  # D 取引日付
        r.debit,  # E 借方勘定科目
        r.debit_sub,  # F 借方補助科目
        "",  # G 借方部門(青色申告ではインポートできない)
        tax,  # H 借方税区分(必須)
        amt,  # I 借方金額
        "",  # J 借方税金額(税込経理の前提)
        r.credit,  # K 貸方勘定科目
        r.credit_sub,  # L 貸方補助科目
        "",  # M 貸方部門
        "対象外",  # N 貸方税区分
        amt,  # O 貸方金額
        "",  # P 貸方税金額
        r.description,  # Q 摘要
        "",  # R 番号
        "",  # S 期日
        "0",  # T タイプ
        "",  # U 生成元
        "",  # V 仕訳メモ
        "",  # W 付箋1
        "",  # X 付箋2
        "no",  # Y 調整
    ]


def _issue(code: str, page_id: int, cell_id: int | None = None, row: int | None = None) -> dict:
    return {
        "code": code,
        "label": ISSUE_LABELS[code],
        "severity": SEVERITY_WARNING if code in WARNING_CODES else SEVERITY_ERROR,
        "page_id": page_id,
        "cell_id": cell_id,
        "row": row,
    }


def _check_yayoi(rows: list[Row], s: Settings) -> list[dict]:
    out: list[dict] = []
    if s.tax_method == "exclusive" and rows:
        out.append(_issue("exclusive_tax_amount_empty", rows[0].page_id))
    for r in rows:
        if len(r.description.encode(ENCODING, errors="ignore")) > MAX_DESC_BYTES_YAYOI:
            out.append(
                _issue("description_too_long", r.page_id, r.cell_ids.get("description"), r.row)
            )
    return out


def _check_smart(rows: list[Row], s: Settings) -> list[dict]:
    out: list[dict] = []
    if rows:
        out.append(_issue("title_lost", rows[0].page_id))
    for r in rows:
        if len(r.description) > MAX_DESC_CHARS_SMART:
            out.append(
                _issue("description_too_long", r.page_id, r.cell_ids.get("description"), r.row)
            )
    return out


def _build_yayoi(rows: list[Row], s: Settings) -> list[OutFile]:
    text = "".join(_line(yayoi_fields(r, s)) for r in rows)
    return [OutFile("yayoi_import.csv", _encode(text), len(rows))]


_BAD_FILENAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def safe_name(name: str) -> str:
    """ファイル名に使える形にする(Windows で使えない文字を除く)。"""
    n = _BAD_FILENAME.sub("_", name).strip(" .") or "payment"
    return n[:40]


def _smart_file(pay: str, part: int, lines: list[str]) -> OutFile:
    return OutFile(f"smart_{safe_name(pay)}_{part}.csv", _encode("".join(lines)), len(lines), pay)


def _build_smart(rows: list[Row], s: Settings) -> list[OutFile]:
    """支払手段ごとにファイルを分け、1 ファイル 200KB 未満に分割する。列は 日付・入金(空)・出金・摘要(全行が同じ列数)。"""
    out: list[OutFile] = []
    by_pay: dict[str, list[Row]] = {}
    for r in rows:
        by_pay.setdefault(r.payment, []).append(r)
    for pay, rs in by_pay.items():
        part, size, lines = 1, 0, []
        for r in rs:
            line = _line([r.date, "", str(r.amount), r.description])
            b = len(_encode(line))
            if lines and size + b >= SMART_FILE_LIMIT:
                out.append(_smart_file(pay, part, lines))
                part, size, lines = part + 1, 0, []
            lines.append(line)
            size += b
        if lines:
            out.append(_smart_file(pay, part, lines))
    return out


FORMATS: dict[str, Format] = {
    "yayoi_import": Format(
        "yayoi_import",
        "弥生インポート形式(仕訳日記帳へのインポート)",
        (
            "科目・補助科目・税区分の名前が弥生の登録と一致しないと「マッチングリスト」が出ます(補助科目は新規作成される可能性があります)。",
            "同じデータを再インポートすると、弥生は警告なしで重複登録します。このアプリは出力した行を記録し、再出力のときに警告します。",
            "インポート後は元に戻せません。事前に弥生でバックアップを取ってください。",
            "会計期間外の日付は取り込めません。",
            "出力した CSV は Excel で開かず、弥生で直接取り込んでください(Excel で開くと、先頭が = + - @ の摘要が数式として扱われる恐れがあります)。",
        ),
        _check_yayoi,
        _build_yayoi,
    ),
    "smart_torihiki": Format(
        "smart_torihiki",
        "スマート取引取込(CSV ファイル取込)",
        (
            "勘定科目は CSV で指定できません。取り込むときに「取引手段」(科目 + 補助科目)を 1 つ選び、相手の科目は弥生が摘要から推測します。",
            "そのため、支払手段ごとにファイルを分けて出力します(ファイル名に支払手段の名前が入ります)。ページのタイトル(借方の科目)は失われます。",
            "1 ファイル 200KB 未満(超える場合は分割します)。日付は西暦 4 桁で出力します。",
            "出力した CSV は Excel で開かず、弥生で直接取り込んでください。",
        ),
        _check_smart,
        _build_smart,
    ),
}
DEFAULT_FORMAT = "yayoi_import"


def get_format(fid: str) -> Format:
    f = FORMATS.get(fid)
    if f is None:
        raise StoreError(422, "unknown_format")
    return f


# ------------------------------------------------------------------ 確定値の取り出し


def _value(c: sqlite3.Row) -> str:
    return (c["resolved_value"] if c["confirmed_value"] is None else c["confirmed_value"]).strip()


def _enc_ok(text: str) -> bool:
    try:
        text.encode(ENCODING)
    except UnicodeEncodeError:
        return False
    return True


def row_hash(page: sqlite3.Row, title: str, row: int, values: list[str]) -> str:
    """行の内容のハッシュ(入力ファイルのハッシュ + ページ番号 + タイトル + 行番号 + 確定値)。同じ行を 2 回出力していないかの判定に使う。"""
    blob = json.dumps(
        [page["image_sha256"], page["source_index"], title, row, values], ensure_ascii=False
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def page_content_hash(con: sqlite3.Connection, page: sqlite3.Row) -> str:
    """ページ全体の内容のハッシュ(出力の履歴の互換と、元画像の保管の判定に使う)。"""
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
    """出力する行と、検証の指摘(エラー・注意。ページ・セルへのリンクつき。値は含まない)。"""
    payments = {p.name: p for p in s.payments}
    titles = {t.name: t for t in s.titles}
    tax = {a.account: a.tax_category for a in s.default_tax}
    rows: list[Row] = []
    issues: list[dict] = []

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
            issues.append(_issue("page_not_confirmed", pid))
            continue
        title = (pg["title"] or "").strip()
        debit = debit_sub = debit_tax = ""
        page_ok = True
        if not title:
            issues.append(_issue("title_missing", pid))
            page_ok = False
        elif title not in titles:
            issues.append(_issue("title_unmapped", pid))
            page_ok = False
        elif not titles[title].debit_account:
            issues.append(_issue("debit_account_missing", pid))
            page_ok = False
        else:
            debit, debit_sub = titles[title].debit_account, titles[title].debit_sub
            debit_tax = tax.get(debit, "")
            if s.taxpayer == "taxable" and not debit_tax:  # 借方税区分(H)は必須
                issues.append(_issue("tax_category_missing", pid))
                page_ok = False
        cells = {
            (r["row"], r["col"]): r
            for r in con.execute("SELECT * FROM cells WHERE page_id = ?", (pid,))
        }
        n_before = len(issues)
        n_rows_before = len(rows)
        for r in sorted({k[0] for k in cells}):
            amount_cell = cells.get((r, 3))
            if amount_cell is None or not _value(amount_cell):
                continue
            vals = {
                col: cells[(r, c)]
                for c, col in enumerate(("date", "account", "description", "amount"))
            }
            ok = page_ok
            for col, cell in vals.items():
                text = _value(cell)
                for code in validate_value(col, text, True):
                    issues.append(_issue(code, pid, cell["id"], r))
                    ok = False
                if not _enc_ok(
                    text
                ):  # cp932 で表現できない文字(どの列でも。出力の encode で例外にしない)
                    if not any(
                        i["code"] == "cp932_unencodable" and i["cell_id"] == cell["id"]
                        for i in issues
                    ):
                        issues.append(_issue("cp932_unencodable", pid, cell["id"], r))
                    ok = False
            acct = _value(vals["account"])
            pay = payments.get(acct)
            if acct and pay is None:
                issues.append(_issue("account_unmapped", pid, vals["account"]["id"], r))
                ok = False
            elif pay is not None and not pay.credit_account:
                issues.append(_issue("credit_account_missing", pid, vals["account"]["id"], r))
                ok = False
            d = parse_date(_value(vals["date"]))
            amt = parse_amount(_value(vals["amount"]))
            if not ok or d is None or d.era_year is None or amt is None or pay is None:
                continue
            desc = _value(vals["description"])
            if desc.startswith(FORMULA_PREFIX):  # 注意だけ(値は書き換えない)
                issues.append(_issue("formula_like", pid, vals["description"]["id"], r))
            rows.append(
                Row(
                    pid,
                    r,
                    f"{d.western_year()}/{d.month:02d}/{d.day:02d}",
                    debit,
                    debit_sub,
                    debit_tax,
                    pay.credit_account,
                    pay.credit_sub,
                    acct,
                    desc,
                    amt,
                    row_hash(pg, title, r, [_value(vals[c]) for c in vals]),
                    {c: vals[c]["id"] for c in vals},
                )
            )
        if len(rows) == n_rows_before and len(issues) == n_before:
            issues.append(_issue("no_rows", pid))
    return rows, issues


# ------------------------------------------------------------------ 重複(行単位)


def exported_row_hashes(con: sqlite3.Connection) -> dict[str, list[int]]:
    """行の内容のハッシュ → それを出力した出力 ID のリスト。"""
    out: dict[str, list[int]] = {}
    for r in con.execute("SELECT export_id, content_sha256 FROM export_rows ORDER BY export_id"):
        out.setdefault(r["content_sha256"], []).append(r["export_id"])
    return out


def duplicates(con: sqlite3.Connection, rows: list[Row]) -> list[dict]:
    """すでに出力した行を含むページ(ページごとの行数と、出力 ID)。"""
    seen = exported_row_hashes(con)
    by_page: dict[int, dict] = {}
    for r in rows:
        ex = seen.get(r.content_sha256)
        if ex:
            d = by_page.setdefault(r.page_id, {"page_id": r.page_id, "n_rows": 0, "export_ids": []})
            d["n_rows"] += 1
            d["export_ids"] = sorted({*d["export_ids"], *ex})
    return list(by_page.values())


def candidates(con: sqlite3.Connection) -> list[dict]:
    """出力の対象にできるページ(確定済み)と、行ごとの出力済みの数。"""
    s = load_settings(con)
    seen = exported_row_hashes(con)
    out = []
    for pg in con.execute(
        "SELECT p.*, j.name AS job_name FROM pages p JOIN jobs j ON j.id = p.job_id"
        " WHERE p.status = 'confirmed' ORDER BY j.id, p.idx"
    ).fetchall():
        rows, _ = collect(con, [pg["id"]], s)
        n_done = sum(1 for r in rows if r.content_sha256 in seen)
        n_all = _count_rows(con, pg["id"])
        out.append(
            {
                "id": pg["id"],
                "job_id": pg["job_id"],
                "job_name": pg["job_name"],
                "idx": pg["idx"],
                "title": pg["title"],
                "n_rows": n_all,
                "n_exported": n_done,
                "exported": n_all > 0 and n_done >= n_all,
            }
        )
    return out


def _count_rows(con: sqlite3.Connection, page_id: int) -> int:
    return int(
        con.execute(
            "SELECT COUNT(*) FROM cells WHERE page_id = ? AND col = 3 AND TRIM(COALESCE(confirmed_value, resolved_value)) != ''",
            (page_id,),
        ).fetchone()[0]
    )


# ------------------------------------------------------------------ 検証・出力・履歴


def validate(con: sqlite3.Connection, page_ids: list[int], fmt: str = DEFAULT_FORMAT) -> dict:
    f = get_format(fmt)
    if not page_ids:
        raise StoreError(422, "no_pages")
    s = load_settings(con)
    rows, issues = collect(con, page_ids, s)
    issues = issues + f.check(rows, s)
    return {
        "n_rows": len(rows),
        "issues": issues,
        "errors": [i for i in issues if i["severity"] == SEVERITY_ERROR],
        "warnings": [i for i in issues if i["severity"] == SEVERITY_WARNING],
        "duplicates": duplicates(con, rows),
        "missing_settings": readiness(s),
        "notes": list(f.notes),
        "format": f.id,
    }


def run_export(
    con: sqlite3.Connection,
    exports_dir: Path,
    page_ids: list[int],
    fmt_id: str = DEFAULT_FORMAT,
    accept_duplicates: bool = False,
) -> dict:
    f = get_format(fmt_id)
    if not page_ids:
        raise StoreError(422, "no_pages")
    s = load_settings(con)
    rows, issues = collect(con, page_ids, s)
    issues = issues + f.check(rows, s)
    errors = [i for i in issues if i["severity"] == SEVERITY_ERROR]
    if errors or readiness(s):
        raise StoreError(
            409, "validation_failed", {"issues": issues, "missing_settings": readiness(s)}
        )
    dups = duplicates(con, rows)
    if dups and not accept_duplicates:
        raise StoreError(409, "duplicate_pages", {"duplicates": dups})
    files = f.build(rows, s)
    now = time.time()
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    exports_dir.mkdir(parents=True, exist_ok=True)
    combined = hashlib.sha256()
    with transaction(con):
        cur = con.execute(
            "INSERT INTO exports (format, format_label, created_at, n_pages, n_rows, file_sha256, settings_snapshot)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f.id,
                f.label,
                now,
                len({r.page_id for r in rows}),
                len(rows),
                "",
                s.model_dump_json(),
            ),
        )
        export_id = int(cur.lastrowid)
        for i, of in enumerate(files):
            name = f"export_{export_id:04d}_{stamp}_{of.name}"
            (exports_dir / name).write_bytes(of.data)
            sha = hashlib.sha256(of.data).hexdigest()
            combined.update(sha.encode())
            con.execute(
                "INSERT INTO export_files (export_id, idx, name, sha256, n_rows, payment) VALUES (?, ?, ?, ?, ?, ?)",
                (export_id, i, name, sha, of.n_rows, of.payment),
            )
        con.execute(
            "UPDATE exports SET file_name = ?, file_sha256 = ? WHERE id = ?",
            (f"export_{export_id:04d}_{stamp}_{files[0].name}", combined.hexdigest(), export_id),
        )
        for r in rows:
            con.execute(
                "INSERT INTO export_rows (export_id, page_id, row, content_sha256) VALUES (?, ?, ?, ?)",
                (export_id, r.page_id, r.row, r.content_sha256),
            )
        for pid in sorted({r.page_id for r in rows}):
            pg = con.execute("SELECT * FROM pages WHERE id = ?", (pid,)).fetchone()
            con.execute(
                "INSERT INTO export_pages (export_id, page_id, content_sha256) VALUES (?, ?, ?)",
                (export_id, pid, page_content_hash(con, pg)),
            )
    return get_export(con, export_id)


def _export_dict(con: sqlite3.Connection, r: sqlite3.Row) -> dict:
    files = [
        {
            "idx": f["idx"],
            "name": f["name"],
            "sha256": f["sha256"],
            "n_rows": f["n_rows"],
            "payment": f["payment"],
        }
        for f in con.execute(
            "SELECT * FROM export_files WHERE export_id = ? ORDER BY idx", (r["id"],)
        )
    ]
    return {
        "id": r["id"],
        "format": r["format"],
        "format_label": r["format_label"],
        "created_at": r["created_at"],
        "n_pages": r["n_pages"],
        "n_rows": r["n_rows"],
        "file_sha256": r["file_sha256"],
        "file_name": r["file_name"],
        "files": files,
    }


def get_export(con: sqlite3.Connection, export_id: int) -> dict:
    r = con.execute("SELECT * FROM exports WHERE id = ?", (export_id,)).fetchone()
    if r is None:
        raise StoreError(404, "export_not_found")
    return _export_dict(con, r)


def list_exports(con: sqlite3.Connection) -> list[dict]:
    return [_export_dict(con, r) for r in con.execute("SELECT * FROM exports ORDER BY id DESC")]


def export_file(con: sqlite3.Connection, exports_dir: Path, export_id: int, idx: int = 0) -> Path:
    e = get_export(con, export_id)
    if not 0 <= idx < len(e["files"]):
        raise StoreError(404, "export_file_not_found")
    path = (exports_dir / e["files"][idx]["name"]).resolve()
    if exports_dir.resolve() not in path.parents or not path.is_file():
        raise StoreError(404, "export_file_not_found")
    return path
