"""確定値の検証(確認画面の「ページを確定できる条件」。UI でも、サーバの確定でも同じ関数を使う)。

返すのは理由コード(実データの値は含めない)。認識側の検証(`recognize/page.py`)と同じ規則:
cp932 で表現できること、摘要は 64 バイト・80 文字以内、日付と金額は文法に合うこと。
"""

from __future__ import annotations

from ..recognize.grammar import parse_amount, parse_date

E_DATE = "date_grammar"
E_YEAR = "year_missing"
E_AMOUNT = "amount_grammar"
E_CP932 = "cp932_unencodable"
E_LONG = "too_long"
E_ACCOUNT = "account_empty"

ERROR_LABELS = {
    E_DATE: "日付の形式が正しくありません",
    E_YEAR: "年が決まっていません(令和の年を入力してください)",
    E_AMOUNT: "金額は 1 以上の数字で入力してください",
    E_CP932: "インポートできない文字が含まれています",
    E_LONG: "摘要が長すぎます(64 バイト以内)",
    E_ACCOUNT: "科目が空です",
}


def validate_value(column: str, value: str, row_nonblank: bool = True) -> list[str]:
    """セル 1 つの確定値の検証。空欄(空文字)は、行が空でなければ科目だけを誤りとする。"""
    v = value.strip()
    if not v:
        return [E_ACCOUNT] if (column == "account" and row_nonblank) else []
    if column == "date":
        d = parse_date(v)
        if d is None:
            return [E_DATE]
        return [] if d.era_year is not None else [E_YEAR]
    if column == "amount":
        n = parse_amount(v)
        return [] if (n is not None and n > 0) else [E_AMOUNT]
    if column == "description":
        try:
            n_bytes = len(v.encode("cp932"))
        except UnicodeEncodeError:
            return [E_CP932]
        return [E_LONG] if (n_bytes > 64 or len(v) > 80) else []
    return []
