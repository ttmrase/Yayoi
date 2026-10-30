"""PARSeq の文字集合と、列ごとの許可文字(デコード時に logits をマスクする)。

NDLmoji.yaml(ndlocr-lite の src/config、7,141 字。`model.charset_train` の順序がトークン ID を決める)は
リポジトリに含めず、`$YAYOI_PRIVATE_DATA/models/` に置く(`scripts/fetch_models.py` が取得)。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ...config import private_data_dir
from ...models import ColumnKind

NDL_CHARSET_FILE_HINT = "NDLmoji.yaml"

# 全角 → 半角(認識後に正規化する)
_ZEN = "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
_HAN = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_ZEN2HAN = {ord(z): h for z, h in zip(_ZEN, _HAN, strict=True)}
_ZEN2HAN.update({ord("，"): ",", ord("．"): ".", ord("－"): "-", ord("ー"): "ー"})  # 長音は触らない
_ZEN2HAN.update({ord("　"): " "})


def normalize_width(text: str) -> str:
    """全角の数字・英字・一部の記号を半角にする(列を問わず認識後に適用)。"""
    return text.translate(_ZEN2HAN)


def load_charset(path: str | Path | None = None) -> str:
    """NDLmoji.yaml の charset_train(トークン ID の順序)を読む。"""
    p = Path(path) if path else private_data_dir() / "models" / NDL_CHARSET_FILE_HINT
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["model"]["charset_train"]


_DATE_CHARS = set("0123456789R.〃 ")
_AMOUNT_CHARS = set("0123456789,-")


def column_charset(kind: ColumnKind | str | None) -> set[str] | None:
    """列の種類ごとの許可文字(None = 全文字)。日付は 0-9 R . 〃 空白、金額は 0-9 , -。"""
    if kind is not None and not isinstance(kind, ColumnKind):
        try:
            kind = ColumnKind(kind)
        except ValueError:  # タイトル・注記など列以外の欄は制限なし
            return None
    if kind == ColumnKind.DATE:
        return set(_DATE_CHARS)
    if kind == ColumnKind.AMOUNT:
        return set(_AMOUNT_CHARS)
    return None
