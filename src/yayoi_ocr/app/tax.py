"""税区分(弥生の記述形式)の許可リストと、税抜経理の税額計算。仕様と出典は `docs/yayoi_export.md`。

このアプリは経費(支払)の帳票専用なので、借方の税区分は「仕入・不課税・非課税」の系統だけに限る。弥生の登録名と完全に一致させる必要があるため、
許可リストにない名前(税率を書かない「課対仕入」など)は設定の保存と出力の検証で拒否する。
 (A) 本則課税 × 税込経理: 課対/非対/共対 仕入込{10%,軽減8%,8%,5%}、非課仕入、対外仕入、対象外
 (B) 本則課税 × 税抜経理: 課対/非対/共対 仕入{内,外,別}{10%,軽減8%,8%,5%}、非課仕入、対外仕入、対象外
 (C) 簡易課税 × 税込経理: 課対仕入込{…}、非課仕入、対外仕入、対象外
 (D) 簡易課税 × 税抜経理: 課対仕入{内,外,別}{…}、非課仕入、対外仕入、対象外
 (E) 免税事業者: すべて「対象外」(選択肢を出さない)
貸方(現金・預金・未払金など)は常に「対象外」。`%` は半角。8%(軽減でないもの)と 5% は旧税率だが、過去の日付の取引に備えて残す。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

Taxpayer = Literal["taxable", "exempt"]
Scheme = Literal["general", "simplified"]
Method = Literal["inclusive", "exclusive"]
Rounding = Literal["floor", "round", "ceil"]

RATES = ("10%", "軽減8%", "8%", "5%")  # 10% を先頭(既定)
CALC = ("内", "外", "別")  # 税抜経理の税計算区分: 内税 / 外税 / 別記(内税を先頭 = 既定)
KINDS_GENERAL = ("課対", "非対", "共対")  # 課税売上対応 / 非課税売上対応 / 共通対応
KINDS_SIMPLIFIED = ("課対",)
NO_RATE = ("非課仕入", "対外仕入", "対象外")  # 税率がなくても正しい名称
EXEMPT = "対象外"
ROUNDING_LABEL = {"floor": "切り捨て", "round": "四捨五入", "ceil": "切り上げ"}

_RATE_RE = re.compile(r"(?:軽減)?(\d+)%$")


def categories(taxpayer: Taxpayer, scheme: Scheme, method: Method) -> list[str]:
    """その事業者・課税方式・経理方式で選べる借方税区分(許可リスト)。免税事業者は「対象外」だけ。"""
    if taxpayer == "exempt":
        return [EXEMPT]
    kinds = KINDS_GENERAL if scheme == "general" else KINDS_SIMPLIFIED
    out: list[str] = []
    for kind in kinds:
        for rate in RATES:
            if method == "inclusive":
                out.append(f"{kind}仕入込{rate}")
    if method == "exclusive":
        for kind in kinds:
            for calc in CALC:
                for rate in RATES:
                    out.append(f"{kind}仕入{calc}{rate}")
    return [*out, *NO_RATE]


def all_options() -> dict[str, list[str]]:
    """画面に渡す、課税方式 × 経理方式ごとの許可リスト(課税事業者の場合。免税事業者は「対象外」だけ)。"""
    return {
        f"{scheme}_{method}": categories("taxable", scheme, method)
        for scheme in ("general", "simplified")
        for method in ("inclusive", "exclusive")
    }


def default_category(taxpayer: Taxpayer, scheme: Scheme, method: Method) -> str:
    """設定の入力欄の既定の候補。税込経理は「課対仕入込10%」、税抜経理は内税の「課対仕入内10%」(director 指示)。"""
    return categories(taxpayer, scheme, method)[0]


@dataclass(frozen=True)
class TaxInfo:
    kind: str  # 課対 / 非対 / 共対 / ""(非課仕入・対外仕入・対象外)
    calc: str  # 込 / 内 / 外 / 別 / ""
    rate: int | None  # 10 / 8 / 5。税率のない区分は None


def parse_category(category: str) -> TaxInfo | None:
    """許可リストの形式の税区分を分解する。形式が合わなければ None。"""
    if category in NO_RATE:
        return TaxInfo("", "", None)
    m = re.fullmatch(r"(課対|非対|共対)仕入(込|内|外|別)((?:軽減)?\d+%)", category)
    if not m:
        return None
    rm = _RATE_RE.search(m.group(3))
    return TaxInfo(m.group(1), m.group(2), int(rm.group(1)) if rm else None)


def has_rate_but_wrong(category: str) -> bool:
    """「課…仕入」の形で税率を書いていない(弥生は税率 5% として取り込む)。"""
    return bool(re.fullmatch(r"(課対|非対|共対)仕入(込|内|外|別)?", category))


def is_manual_tax(category: str) -> bool:
    """税抜経理の外税・別記: 税額は手入力の扱い(借方税金額を空で出力する)。"""
    info = parse_category(category)
    return info is not None and info.calc in ("外", "別")


def tax_amount(amount: int, category: str, rounding: Rounding = "floor") -> int | None:
    """税抜経理で内税の行の借方税金額 = 金額 × 税率 ÷ (100 + 税率)(端数処理は設定)。内税でなければ None(空で出力する)。"""
    info = parse_category(category)
    if info is None or info.calc != "内" or info.rate is None:
        return None
    num, den = amount * info.rate, 100 + info.rate
    if rounding == "ceil":
        return -(-num // den)
    if rounding == "round":  # 四捨五入(0.5 は切り上げ)
        return (2 * num + den) // (2 * den)
    return num // den


def normalize_description(text: str) -> str:
    """摘要の記憶のキー: NFKC に正規化し、空白(全角含む)を除いて小文字にする。値の中身は書き換えず、照合のためだけに使う。"""
    t = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", "", t).casefold()
