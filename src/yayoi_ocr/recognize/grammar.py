"""日付・金額の文法と、認識文字列からの候補生成(エンジン非依存の純粋関数)。

- 日付: `^(R\\d{1,2}\\s?)?\\d{1,2}\\.\\d{1,2}$`(年は省略可)。月 1〜12、日は暦どおり。〃 は別扱い(同上)。
- 金額: `^\\d{1,3}(,\\d{3})*-?$` または `^\\d+-?$`。値は 0 より大きい。末尾の「-」は付いても付かなくてもよい。

認識文字列は崩れる(ピリオド・カンマの欠落、先頭の余計な線)ので、「文字列 → 文法に合う値の候補」へ修復する。
値ごとに、書かれ方(表記ゆれ)を列挙しておき、エンジンでは表記ごとの尤度を足し合わせて
その値の尤度とする(周辺化)。例: 3456 円は「3,456-」「3456-」「3,456」「3456」。
"""

from __future__ import annotations

import calendar
import re
import unicodedata
from dataclasses import dataclass

REIWA_BASE = 2018  # 令和 n 年 = 2018 + n 年
MAX_ERA_YEAR = 30  # 令和 30 年(2048 年)まで


def norm_text(s: str) -> str:
    """比較用の正規化: NFKC・空白をすべて除去(摘要・科目の比較)。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))


# ---------------------------------------------------------------- 日付


@dataclass(frozen=True, order=True)
class DateValue:
    era_year: int | None  # 令和の年(R7 → 7)。年が書かれていなければ None
    month: int
    day: int

    @property
    def explicit_year(self) -> bool:
        return self.era_year is not None

    def format(self) -> str:
        """内部表記(正解データの resolved と同じ): 'R7.11.19'。年が無ければ '11.19'。"""
        md = f"{self.month}.{self.day}"
        return md if self.era_year is None else f"R{self.era_year}.{md}"

    def with_year(self, era_year: int | None) -> DateValue:
        return DateValue(era_year, self.month, self.day)

    def western_year(self) -> int | None:
        return None if self.era_year is None else REIWA_BASE + self.era_year


def is_valid_date(era_year: int | None, month: int, day: int) -> bool:
    if not 1 <= month <= 12 or day < 1:
        return False
    if era_year is not None and not 1 <= era_year <= MAX_ERA_YEAR:
        return False
    year = REIWA_BASE + era_year if era_year is not None else 2024  # 年不明は閏年として日数を数える
    return day <= calendar.monthrange(year, month)[1]


_DATE_NOYEAR_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})$")
_DATE_DOTTED_YEAR_RE = re.compile(r"^R(\d{1,2})\.(\d{1,2})\.(\d{1,2})$")
_DATE_R_RE = re.compile(r"^R(\d+)( ?)(\d*)\.(\d+)$")


def parse_date(text: str) -> DateValue | None:
    """文法どおりの日付だけを受理する(厳密)。スペースなしの 'R711.19' は桁の切れ目を試して、有効な読みを採る。"""
    t = text.strip()
    m = _DATE_DOTTED_YEAR_RE.match(t)
    if m:
        y, mo, d = (int(g) for g in m.groups())
        return _mk(y, mo, d)
    m = _DATE_NOYEAR_RE.match(t)
    if m:
        return _mk(None, int(m.group(1)), int(m.group(2)))
    m = _DATE_R_RE.match(t)
    if not m:
        return None
    g1, space, g3, day = m.groups()
    if space and g3:
        options = [(g1, g3)]
    else:
        digits = g1 + g3
        options = [(digits[:k], digits[k:]) for k in (1, 2)]
    for y, mo in options:
        if len(y) <= 2 and 1 <= len(mo) <= 2 and len(day) <= 2:
            v = _mk(int(y), int(mo), int(day))
            if v is not None:
                return v
    return None


def date_surfaces(v: DateValue) -> list[str]:
    """日付の書かれ方(表記ゆれ)。月日の区切りは「.」「なし」「空白」(ピリオドが見えない・小さいことが多い)、
    年があれば年と月日の間も「空白」「なし」「.」(例: 'R7 11.19' 'R711.19' 'R7.11.19' 'R7 1119')。"""
    mds = [f"{v.month}{sep}{v.day}" for sep in (".", "", " ")]
    if v.era_year is None:
        return mds
    return [f"R{v.era_year}{ysep}{md}" for ysep in (" ", "", ".") for md in mds]


def repair_date(text: str) -> list[DateValue]:
    """認識文字列 → 文法に合う日付の候補(ピリオド欠落・年と月の桁の切れ目の曖昧さを展開する)。"""
    t = unicodedata.normalize("NFKC", text).strip()
    out: list[DateValue] = []

    def add(v: DateValue | None) -> None:
        if v is not None and v not in out:
            out.append(v)

    add(parse_date(t))
    compact = re.sub(r"\s+", "", t)
    has_r = compact.startswith("R")
    body = compact[1:] if has_r else compact
    if not re.fullmatch(r"[0-9.]+", body) or not body:
        return out
    dots = body.count(".")
    if has_r and dots == 2:  # R7.11.19 / R7.1119 など(parse_date が拾えない崩れは下の分割で)
        parts = body.split(".")
        if all(p.isdigit() for p in parts):
            add(_mk(int(parts[0]), int(parts[1]), int(parts[2])))
    if dots == 1:
        left, right = body.split(".")
        if not left.isdigit() or not right.isdigit():
            return out
        if has_r:  # 年と月が連続している: 桁の切れ目を全部試す
            for k in (1, 2):
                if len(left) > k and len(left) - k <= 2 and len(right) <= 2:
                    add(_mk(int(left[:k]), int(left[k:]), int(right)))
        elif len(left) <= 2 and len(right) <= 2:
            add(_mk(None, int(left), int(right)))
    elif dots == 0 and body.isdigit():  # ピリオドが欠落: 年(R のとき)・月・日の桁の切れ目を試す
        n = len(body)
        for ky in (1, 2) if has_r else (0,):
            for km in (1, 2):
                kd = n - ky - km
                if kd in (1, 2):
                    y = int(body[:ky]) if ky else None
                    add(_mk(y, int(body[ky : ky + km]), int(body[ky + km :])))
    return out


def _mk(y: int | None, mo: int, d: int) -> DateValue | None:
    return DateValue(y, mo, d) if is_valid_date(y, mo, d) else None


# ---------------------------------------------------------------- 金額

_AMOUNT_COMMA_RE = re.compile(r"^\d{1,3}(,\d{3})+-?$")
_AMOUNT_PLAIN_RE = re.compile(r"^\d+-?$")
MAX_AMOUNT_DIGITS = 9


def parse_amount(text: str) -> int | None:
    """文法どおりの金額だけを受理する(厳密)。値は 0 より大きい。"""
    t = text.strip()
    if _AMOUNT_COMMA_RE.match(t) or _AMOUNT_PLAIN_RE.match(t):
        digits = t.replace(",", "").rstrip("-")
        if len(digits) <= MAX_AMOUNT_DIGITS and int(digits) > 0:
            return int(digits)
    return None


def amount_surfaces(v: int) -> list[str]:
    """金額の書かれ方(表記ゆれ): カンマあり/なし x 末尾の「-」あり/なし。"""
    plain, comma = str(v), f"{v:,}"
    out = [f"{comma}-", f"{plain}-", comma, plain]
    return list(dict.fromkeys(out))


def repair_amount(text: str) -> list[int]:
    """認識文字列 → 文法に合う金額の候補。先頭・末尾の余計な線やカンマ(罫線のかすれ・取り違え)を削った版も候補にする。"""
    t = unicodedata.normalize("NFKC", text).strip()
    out: list[int] = []

    def add(v: int | None) -> None:
        if v is not None and v not in out:
            out.append(v)

    add(parse_amount(t))
    for variant in (t.lstrip(",-"), t.rstrip(","), t.lstrip(",-").rstrip(",")):
        add(parse_amount(variant))
    # カンマの位置が崩れている(3 桁区切りでない)が数字は読めている場合: 数字だけにした値も候補にする
    digits = re.sub(r"[,\-]", "", t.lstrip(",-"))
    if digits.isdigit() and not out:
        add(parse_amount(digits))
    return out
