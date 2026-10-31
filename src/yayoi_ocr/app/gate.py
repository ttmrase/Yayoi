"""追加学習した世代の採用ゲート(純粋関数。torch なしで試験できる)。設計は `docs/phase4_design.md`。

評価用ページ(学習に使わなかった、確定した順で新しい側のページ)を**親(使用中の世代)**と**新しい世代**の両方で認識し直し、確定値と比べた件数
(`EvalCounts`)から採否を決める。新しい世代は次の**すべて**を満たすときだけ採用する:

1. 見逃し率が悪化しない(新 ≤ 親。見逃し = 誤り かつ 要確認なし。全セル中)。
2. 正解率が下がらない(新 ≥ 親。非空欄セルの値の正解率)。
3. 得がある(正解率が上がる、または要確認率が下がる。同じなら採用しない)。
4. 要確認率が増えすぎない(新 ≤ 親 + 2 pt。見逃しを抑えただけで確認の手間が増えた世代を採らない)。
5. 評価が成り立つ(評価の非空欄セルが 150 件以上)。足りないときは採否を決めず「見送り」(次の機会に回す)。

理由はコードだけ(値は入れない)。学習画面とログに出す。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

from ..recognize.grammar import norm_text

MIN_EVAL_CELLS = 150  # 評価に使う非空欄セルの最小数
MAX_REVIEW_INCREASE = 0.02  # 要確認率(非空欄セル中)の増加の上限
MIN_PAGES = 6  # 学習(4)+ 評価(2)に最低限必要な確定ページ数
MIN_EVAL_PAGES = 2
MAX_EVAL_PAGES = 10
EVAL_FRACTION = 0.25

# 理由コード(学習画面の文言は web/src/trainingText.ts と対応させる)
R_ADOPTED_ACCURACY = "adopted_accuracy_up"
R_ADOPTED_REVIEW = "adopted_review_down"
R_MISS_WORSE = "miss_rate_worse"
R_ACCURACY_DROPPED = "accuracy_dropped"
R_NO_GAIN = "no_gain"
R_REVIEW_INCREASED = "review_rate_increased"
R_EVAL_TOO_SMALL = "eval_too_small"


@dataclass(frozen=True)
class EvalCounts:
    """評価用ページでの 1 つの重みの成績(件数)。ページごとに数えて足し合わせる。"""

    n_cells: int  # 空欄を含む全セル
    n_text: int  # 確定が空欄でないセル
    n_correct: int  # そのうち、認識した値が確定値と一致したもの
    n_review: int  # そのうち、要確認が付いたもの
    n_miss: int  # 見逃し(認識した値が確定値と違うのに、要確認が付かなかった全セル)

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_text if self.n_text else 0.0

    @property
    def review_rate(self) -> float:
        return self.n_review / self.n_text if self.n_text else 0.0

    @property
    def miss_rate(self) -> float:
        return self.n_miss / self.n_cells if self.n_cells else 0.0

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "accuracy": self.accuracy,
            "review_rate": self.review_rate,
            "miss_rate": self.miss_rate,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EvalCounts:
        return cls(*(int(d[k]) for k in ("n_cells", "n_text", "n_correct", "n_review", "n_miss")))


@dataclass(frozen=True)
class Decision:
    outcome: Literal["adopt", "reject", "defer"]
    reason: str

    @property
    def adopted(self) -> bool:
        return self.outcome == "adopt"


def decide(
    parent: EvalCounts,
    candidate: EvalCounts,
    *,
    min_eval_cells: int = MIN_EVAL_CELLS,
    max_review_increase: float = MAX_REVIEW_INCREASE,
) -> Decision:
    """親と新しい世代の成績から、採用・不採用・見送りを決める。条件の順に最初に当たった理由を返す。"""
    if parent.n_text < min_eval_cells or candidate.n_text < min_eval_cells:
        return Decision("defer", R_EVAL_TOO_SMALL)
    if candidate.miss_rate > parent.miss_rate + 1e-12:
        return Decision("reject", R_MISS_WORSE)
    if candidate.accuracy < parent.accuracy - 1e-12:
        return Decision("reject", R_ACCURACY_DROPPED)
    if candidate.review_rate > parent.review_rate + max_review_increase + 1e-12:
        return Decision("reject", R_REVIEW_INCREASED)
    if candidate.accuracy > parent.accuracy + 1e-12:
        return Decision("adopt", R_ADOPTED_ACCURACY)
    if candidate.review_rate < parent.review_rate - 1e-12:
        return Decision("adopt", R_ADOPTED_REVIEW)
    return Decision("reject", R_NO_GAIN)


def split_pages(
    ordered_page_ids: list[int],
    *,
    min_pages: int = MIN_PAGES,
    fraction: float = EVAL_FRACTION,
    min_eval: int = MIN_EVAL_PAGES,
    max_eval: int = MAX_EVAL_PAGES,
) -> tuple[list[int], list[int]] | None:
    """確定した順(古い → 新しい)のページを、学習用と評価用に分ける。**新しい側から** `fraction`(最低 `min_eval`・最大 `max_eval` ページ)を評価用にする。
    ページ数が足りなければ None。次の実行では評価用だったページも学習に入る(窓が動く)。"""
    n = len(ordered_page_ids)
    if n < min_pages:
        return None
    n_eval = max(min_eval, min(max_eval, math.ceil(n * fraction)))
    return ordered_page_ids[: n - n_eval], ordered_page_ids[n - n_eval :]


class _Recognized(Protocol):
    column: str
    value: str
    review: bool


class _Confirmed(Protocol):
    column: str
    kind: str
    resolved: str


TEXT_COLUMNS = ("account", "description")  # 比較のとき NFKC・空白除去で正規化する列


def same_value(column: str, recognized: str, confirmed: str) -> bool:
    """認識した値が確定値と一致するか(科目・摘要は NFKC・空白を除いて比べる。日付・金額は文字列のまま)。"""
    if column in TEXT_COLUMNS:
        return norm_text(recognized) == norm_text(confirmed)
    return recognized == confirmed


def count_cells(
    recognized: Sequence[_Recognized], confirmed: Sequence[_Confirmed]
) -> dict[str, EvalCounts]:
    """1 ページ(行優先の全セル)の認識結果を確定値と突き合わせた件数。列ごと + 全体("all")。"""
    acc: dict[str, list[int]] = {}
    for r, t in zip(recognized, confirmed, strict=True):
        ok = same_value(t.column, r.value, t.resolved)
        nonblank = t.kind != "blank"
        row = [
            1,
            int(nonblank),
            int(nonblank and ok),
            int(nonblank and r.review),
            int(not ok and not r.review),
        ]
        for key in (t.column, "all"):
            a = acc.setdefault(key, [0] * 5)
            for i, v in enumerate(row):
                a[i] += v
    return {k: EvalCounts(*v) for k, v in acc.items()}


def add_counts(a: dict[str, EvalCounts], b: dict[str, EvalCounts]) -> dict[str, EvalCounts]:
    """ページごとの件数を足し合わせる。"""
    out = dict(a)
    for k, c in b.items():
        o = out.get(k)
        out[k] = (
            c
            if o is None
            else EvalCounts(
                o.n_cells + c.n_cells,
                o.n_text + c.n_text,
                o.n_correct + c.n_correct,
                o.n_review + c.n_review,
                o.n_miss + c.n_miss,
            )
        )
    return out


def total_counts(pages: Iterable[dict[str, EvalCounts]]) -> dict[str, EvalCounts]:
    out: dict[str, EvalCounts] = {}
    for p in pages:
        out = add_counts(out, p)
    return out
