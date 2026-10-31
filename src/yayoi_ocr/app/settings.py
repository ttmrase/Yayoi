"""設定(支払手段 → 貸方・タイトル → 借方・事業者区分など)の形式・検証・保存。

保存先は SQLite の `settings`(キー `app` に JSON)。弥生のインポート形式(`docs/yayoi_export.md`)で必要な値を持つ: 事業者区分(課税/免税)・課税方式(本則/簡易)・経理方式(税込/税抜)、
支払手段 → 貸方科目・補助科目、タイトル → 借方科目・補助科目、科目ごとの既定の税区分。科目・補助科目・税区分の名前は**弥生の登録名と完全に一致させる**(利用者が入力した文字列をそのまま出力する)。
保存の検証は「データが壊れない」ための最低限(重複・長さ・cp932)。出力に足りないもの(支払手段が 1 つもない等)は出力の検証が指摘する。
"""

from __future__ import annotations

import json
import sqlite3
from typing import Literal

from pydantic import BaseModel, Field

from .db import transaction
from .store import StoreError

KEY = "app"
MAX_NAME = 20  # 弥生の科目名・補助科目名の長さの目安(全角 20 文字)
MAX_ITEMS = 100

# 税区分の入力候補(目安。弥生の登録名と完全に一致する必要がある。正式名称の一覧は docs/yayoi_export.md の注記のとおり未確認)
TAX_SUGGESTIONS = (
    "課対仕入込10%",
    "課対仕入込8%(軽)",
    "課対仕入込8%",
    "課対仕入10%",
    "課対仕入8%(軽)",
    "課対仕入8%",
    "対象外",
)
MAX_ACCOUNT_BYTES = 24  # 勘定科目・補助科目(cp932 のバイト数)。弥生の借方勘定科目の上限


class PaymentMap(BaseModel):
    """帳票の「科目」欄に書かれる支払手段 → 貸方科目・補助科目。"""

    name: str = Field(max_length=MAX_NAME)
    credit_account: str = Field(default="", max_length=MAX_NAME)
    credit_sub: str = Field(default="", max_length=MAX_NAME)


class TitleMap(BaseModel):
    """帳票の上部のタイトル(勘定科目名)→ 借方科目・補助科目。"""

    name: str = Field(max_length=MAX_NAME)
    debit_account: str = Field(default="", max_length=MAX_NAME)
    debit_sub: str = Field(default="", max_length=MAX_NAME)


class AccountTax(BaseModel):
    """借方科目ごとの既定の税区分(弥生の登録名と完全に一致させる。税率を書かない課税区分は保存を拒否する)。"""

    account: str = Field(max_length=MAX_NAME)
    tax_category: str = Field(default="", max_length=40)


class Settings(BaseModel):
    taxpayer: Literal["taxable", "exempt"] = (
        "taxable"  # 事業者区分(課税事業者 / 免税事業者 = すべて「対象外」)
    )
    tax_scheme: Literal["general", "simplified"] = "general"  # 課税方式(本則 / 簡易)
    tax_method: Literal["inclusive", "exclusive"] = (
        "inclusive"  # 経理方式(税込 / 税抜。税抜は借方税金額が空になるので警告)
    )
    wizard_done: bool = False  # 初期設定ウィザードを終えた(スキップを含む)
    payments: list[PaymentMap] = Field(default_factory=list, max_length=MAX_ITEMS)
    titles: list[TitleMap] = Field(default_factory=list, max_length=MAX_ITEMS)
    default_tax: list[AccountTax] = Field(default_factory=list, max_length=MAX_ITEMS)
    # 取り込んだ元画像(実帳票)の保管: keep = 残す(既定)/ delete_after_export = そのファイルの全ページを出力したら削除する
    original_retention: Literal["keep", "delete_after_export"] = "keep"


def _encodable(s: str) -> bool:
    try:
        s.encode("cp932")
    except UnicodeEncodeError:
        return False
    return True


def problems(s: Settings) -> list[dict]:
    """保存を断る問題(重複・空の名前・インポートできない文字)。`path` は画面が該当の入力欄を示すのに使う。"""
    out: list[dict] = []

    def check(path: str, value: str, required: bool = False) -> None:
        if required and not value.strip():
            out.append({"path": path, "code": "empty"})
        elif value != value.strip():
            out.append({"path": path, "code": "surrounding_space"})
        elif not _encodable(value):
            out.append({"path": path, "code": "cp932_unencodable"})

    def check_bytes(path: str, value: str) -> None:
        if _encodable(value) and len(value.encode("cp932")) > MAX_ACCOUNT_BYTES:
            out.append({"path": path, "code": "too_long_bytes"})

    for i, it in enumerate(s.default_tax):
        c = it.tax_category
        # 税率を書かない課税の税区分は、弥生が税率 5% として取り込んでしまう。% は半角
        if "％" in c:
            out.append({"path": f"default_tax[{i}].tax_category", "code": "tax_percent_fullwidth"})
        elif c.startswith("課") and "%" not in c:
            out.append({"path": f"default_tax[{i}].tax_category", "code": "tax_rate_missing"})
    for i, p in enumerate(s.payments):
        check_bytes(f"payments[{i}].credit_account", p.credit_account)
        check_bytes(f"payments[{i}].credit_sub", p.credit_sub)
    for i, t in enumerate(s.titles):
        check_bytes(f"titles[{i}].debit_account", t.debit_account)
        check_bytes(f"titles[{i}].debit_sub", t.debit_sub)

    for group, items, fields in (
        ("payments", s.payments, ("name", "credit_account", "credit_sub")),
        ("titles", s.titles, ("name", "debit_account", "debit_sub")),
        ("default_tax", s.default_tax, ("account", "tax_category")),
    ):
        seen: set[str] = set()
        for i, it in enumerate(items):
            for f in fields:
                check(f"{group}[{i}].{f}", getattr(it, f), required=f in ("name", "account"))
            key = getattr(it, fields[0])
            if key in seen:
                out.append({"path": f"{group}[{i}].{fields[0]}", "code": "duplicate"})
            seen.add(key)
    return out


def load(con: sqlite3.Connection) -> Settings:
    r = con.execute("SELECT value FROM settings WHERE key = ?", (KEY,)).fetchone()
    if r is None:
        return Settings()
    try:
        return Settings.model_validate(json.loads(r["value"]))
    except ValueError:  # 壊れた設定は既定に戻す(出力の前に画面で気づける)
        return Settings()


def save(con: sqlite3.Connection, s: Settings) -> None:
    errs = problems(s)
    if errs:
        raise StoreError(422, "invalid_settings", {"problems": errs})
    with transaction(con):
        con.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (KEY, s.model_dump_json()),
        )


def readiness(s: Settings) -> list[str]:
    """出力に足りない設定(コードの一覧)。保存は拒否しない。"""
    out: list[str] = []
    if not s.payments:
        out.append("no_payments")
    elif any(not p.credit_account for p in s.payments):
        out.append("payment_without_credit_account")
    if not s.titles:
        out.append("no_titles")
    elif any(not t.debit_account for t in s.titles):
        out.append("title_without_debit_account")
    if s.taxpayer == "taxable":  # 借方税区分(H)は必須。免税事業者はすべて「対象外」
        tax = {a.account: a.tax_category for a in s.default_tax}
        if any(t.debit_account and not tax.get(t.debit_account) for t in s.titles):
            out.append("debit_account_without_tax_category")
    return out
