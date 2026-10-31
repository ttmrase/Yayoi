"""追加学習した世代の採用ゲート(純粋関数。torch なしで試験できる)。設計は `docs/phase4_design.md`。

評価用ページ(学習に使わなかった、確定した順で新しい側のページ)を**親(使用中の世代)**と**新しい世代**の両方で認識し直し、確定値と比べた成績(`EvalSide`)から採否を決める。
新しい世代の成績は、**クロスフィッティング**で数える: 各評価ページの要確認の判定に使うしきい値は、そのページを除いた評価ページ(と全体の較正)から決めたもの
(`evalstats.cross_fit`)。同じページに合わせたしきい値で同じページの見逃しを数えると、見逃しの基準がほぼ必ず通ってしまうため。

評価ページは最大 10 ページ(非空欄のセルで数百件)で、期待される見逃しは 1 件前後なので、「見逃しの件数が悪化しない」だけでは 0 件と 0 件の比較になりやすい。
そこで、感度の高い指標を加える。新しい世代は次の**すべて**を満たすときだけ採用する(上から順に見て、最初に当たった理由を返す):

0. 評価が成り立つ(評価の非空欄セルが 150 件以上)。足りないときは採否を決めず「見送り」(次の機会に回す)。
1. **金額は、評価セル中に要確認でない誤りが 1 件でもあれば不採用**(`amount_miss`)。
2. 見逃しが悪化しない(全体と列ごとに、新 ≤ 親。見逃し = 誤り かつ 要確認なし)(`miss_rate_worse`)。
3. **自信のある誤りが、予測より有意に多くない**: 要確認でないセルの予測した誤り確率の合計(期待される見逃し件数)に対して、観測した見逃しがポアソン分布の上側確率で
   有意に多くない(全体 5%・列ごと 1%)(`overconfident`)。
4. **較正の質が悪化しない**: 評価セルの正誤の確率に対する負の対数尤度(NLL)が親より悪化しない(`calibration_worse`)。
5. 正解率が下がらない(`accuracy_dropped`)。
6. 要確認率が増えすぎない(新 ≤ 親 + 2 pt)(`review_rate_increased`)。
7. 得がある(正解率が上がる、または要確認率が下がる。同じなら採用しない)(`no_gain`)。

理由はコードだけ(値は入れない)。学習画面とログに出す。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Literal

from ..recognize.grammar import norm_text

MIN_EVAL_CELLS = 150  # 評価に使う非空欄セルの最小数
MAX_REVIEW_INCREASE = 0.02  # 要確認率(非空欄セル中)の増加の上限
ALPHA_TOTAL = 0.05  # 観測した見逃しが予測より有意に多いと見なす水準(全体)
ALPHA_COLUMN = 0.01  # 同(列ごと。検定の数が多いので厳しく)
MIN_PAGES = 6  # 学習(4)+ 評価(2)に最低限必要な確定ページ数
MIN_EVAL_PAGES = 2
MAX_EVAL_PAGES = 10
EVAL_FRACTION = 0.25

# 理由コード(学習画面の文言は web/src/trainingText.ts と対応させる)
R_ADOPTED_ACCURACY = "adopted_accuracy_up"
R_ADOPTED_REVIEW = "adopted_review_down"
R_AMOUNT_MISS = "amount_miss"
R_MISS_WORSE = "miss_rate_worse"
R_OVERCONFIDENT = "overconfident"
R_CALIBRATION_WORSE = "calibration_worse"
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
class Calib:
    """確信度の較正の質(評価セルのうち、確認必須の理由のないもの)。"""

    n: int  # 評価したセル数
    nll: float  # 正誤に対する負の対数尤度(平均)
    brier: float  # ブライアスコア(平均)
    expected_miss: float  # 要確認でないセルの (1 - 確信度) の合計 = 予測した見逃しの件数
    observed_miss: int  # 要確認でない誤りの件数

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Calib:
        return cls(
            int(d["n"]),
            float(d["nll"]),
            float(d["brier"]),
            float(d["expected_miss"]),
            int(d["observed_miss"]),
        )


@dataclass(frozen=True)
class EvalSide:
    """親または新しい世代の評価成績: 全体 + 列ごとの件数と、較正の質。"""

    total: EvalCounts
    calib: Calib | None = None
    by_column: dict[str, EvalCounts] = field(default_factory=dict)
    calib_by_column: dict[str, Calib] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            **self.total.to_dict(),
            "calib": None if self.calib is None else self.calib.to_dict(),
            "by_column": {
                col: {
                    **c.to_dict(),
                    "calib": (
                        None
                        if col not in self.calib_by_column
                        else self.calib_by_column[col].to_dict()
                    ),
                }
                for col, c in self.by_column.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> EvalSide:
        by, cal = {}, {}
        for col, v in (d.get("by_column") or {}).items():
            by[col] = EvalCounts.from_dict(v)
            if v.get("calib"):
                cal[col] = Calib.from_dict(v["calib"])
        return cls(
            EvalCounts.from_dict(d),
            Calib.from_dict(d["calib"]) if d.get("calib") else None,
            by,
            cal,
        )


@dataclass(frozen=True)
class Decision:
    outcome: Literal["adopt", "reject", "defer"]
    reason: str

    @property
    def adopted(self) -> bool:
        return self.outcome == "adopt"


def poisson_sf(k: int, lam: float) -> float:
    """P(X ≥ k), X ~ Poisson(lam)。期待される見逃しの件数 lam に対して、k 件以上観測する確率。"""
    if k <= 0:
        return 1.0
    if lam <= 0.0:
        return 0.0
    # 1 - P(X ≤ k-1)。k は小さい(見逃しの件数)ので直接足す
    term = math.exp(-lam)
    cdf = term
    for i in range(1, k):
        term *= lam / i
        cdf += term
    return max(0.0, 1.0 - cdf)


def _side(x: EvalSide | EvalCounts) -> EvalSide:
    return x if isinstance(x, EvalSide) else EvalSide(x)


def decide(
    parent: EvalSide | EvalCounts,
    candidate: EvalSide | EvalCounts,
    *,
    min_eval_cells: int = MIN_EVAL_CELLS,
    max_review_increase: float = MAX_REVIEW_INCREASE,
    alpha_total: float = ALPHA_TOTAL,
    alpha_column: float = ALPHA_COLUMN,
) -> Decision:
    """親と新しい世代の成績から、採用・不採用・見送りを決める。条件の順に最初に当たった理由を返す(モジュールの説明を参照)。"""
    p, c = _side(parent), _side(candidate)
    pt, ct = p.total, c.total
    if pt.n_text < min_eval_cells or ct.n_text < min_eval_cells:
        return Decision("defer", R_EVAL_TOO_SMALL)
    # 1. 金額は、要確認でない誤りが 1 件でもあれば不採用
    amount = c.by_column.get("amount")
    if (amount is not None and amount.n_miss >= 1) or (
        amount is None
        and "amount" in c.calib_by_column
        and c.calib_by_column["amount"].observed_miss >= 1
    ):
        return Decision("reject", R_AMOUNT_MISS)
    # 2. 見逃しが悪化しない(全体と列ごと)
    if ct.miss_rate > pt.miss_rate + 1e-12:
        return Decision("reject", R_MISS_WORSE)
    for col, cc in c.by_column.items():
        pc = p.by_column.get(col)
        if pc is not None and cc.n_miss > pc.n_miss:
            return Decision("reject", R_MISS_WORSE)
    # 3. 自信のある誤りが、予測より有意に多くない(ポアソンの上側確率)
    if c.calib is not None and (
        poisson_sf(c.calib.observed_miss, c.calib.expected_miss) <= alpha_total
    ):
        return Decision("reject", R_OVERCONFIDENT)
    for cb in c.calib_by_column.values():
        if poisson_sf(cb.observed_miss, cb.expected_miss) <= alpha_column:
            return Decision("reject", R_OVERCONFIDENT)
    # 4. 較正の質が悪化しない
    if p.calib is not None and c.calib is not None and c.calib.nll > p.calib.nll + 1e-9:
        return Decision("reject", R_CALIBRATION_WORSE)
    # 5〜7. 正解率・要確認率・得
    if ct.accuracy < pt.accuracy - 1e-12:
        return Decision("reject", R_ACCURACY_DROPPED)
    if ct.review_rate > pt.review_rate + max_review_increase + 1e-12:
        return Decision("reject", R_REVIEW_INCREASED)
    if ct.accuracy > pt.accuracy + 1e-12:
        return Decision("adopt", R_ADOPTED_ACCURACY)
    if ct.review_rate < pt.review_rate - 1e-12:
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


TEXT_COLUMNS = ("account", "description")  # 比較のとき NFKC・空白除去で正規化する列


def same_value(column: str, recognized: str, confirmed: str) -> bool:
    """認識した値が確定値と一致するか(科目・摘要は NFKC・空白を除いて比べる。日付・金額は文字列のまま)。"""
    if column in TEXT_COLUMNS:
        return norm_text(recognized) == norm_text(confirmed)
    return recognized == confirmed
