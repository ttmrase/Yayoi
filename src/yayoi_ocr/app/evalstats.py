"""評価用ページの観測(セルごとの確信度・正誤)から、要確認の判定・件数・較正の質を数える(純 numpy。torch なしで試験できる)。

認識の結果は、確信度 p(しきい値に依存しない)と、確認必須の理由・同上の参照元から決まる。要確認の判定は p とそのしきい値の比較だけなので、
**しきい値を変えた場合の要確認の判定を、認識をやり直さずに再現できる**(`review_flags`。`PageRecognizer._finalize` と同じ規則。学習のワーカーが、実際の判定との一致を確かめる)。
これを使って、クロスフィッティング(`cross_fit`)をする: 各評価ページの判定に使うしきい値は、そのページを除いた評価ページ(と全体の較正)から決める。
同じページに合わせたしきい値で同じページの見逃しを数えると、見逃しの基準がほぼ必ず通ってしまう(循環)ため。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..recognize.confidence import ALWAYS_REVIEW_KINDS, ConfidenceModel, group_of
from .gate import Calib, EvalCounts, EvalSide

P_CLIP = 1e-4  # NLL の計算で確率を [P_CLIP, 1 - P_CLIP] に収める
CELLS_PER_PAGE_COLUMN = 29  # 1 ページの 1 列のセル数(行数)


@dataclass(frozen=True)
class CellObs:
    """評価用ページの 1 セルの観測(確定値は含まない)。先頭の 5 つが基本で、残りは要確認の判定の再現・件数の集計に使う。"""

    column: str
    kind: str  # 認識した種別(blank / text / ditto / arrow / wave)
    p: float  # 確信度(しきい値に依存しない)
    correct: bool  # 認識した値が確定値と一致した
    hard: bool  # 確認が必須の理由が付いている(同上の参照元の要確認による理由を除く)
    page: int = 0  # 評価ページの通し番号
    idx: int = -1  # ページ内のセル番号(row * 4 + col)。-1 は参照元を持たない
    nonblank: bool = True  # 確定値が空欄でない
    src: int | None = None  # 同上の参照元のセル番号


def threshold_of(model: ConfidenceModel, o: CellObs) -> float:
    if o.kind in ALWAYS_REVIEW_KINDS:
        return 1.1
    g = model.groups.get(group_of(o.column, o.kind))
    return model.default_threshold if g is None else g.threshold


def review_flags(obs: Sequence[CellObs], model: ConfidenceModel) -> list[bool]:
    """1 ページ分(セル番号順)の要確認の判定。`PageRecognizer._finalize` と同じ規則: 確信度がしきい値未満、確認必須の理由、同上の参照元が要確認。"""
    by_idx: dict[int, bool] = {}
    out: list[bool] = []
    for o in obs:
        r = bool(
            o.p < threshold_of(model, o)
            or o.hard
            or (o.src is not None and by_idx.get(o.src, False))
        )
        if o.idx >= 0:
            by_idx[o.idx] = r
        out.append(r)
    return out


def _counts(rows: Sequence[tuple[CellObs, bool]]) -> EvalCounts:
    n_text = sum(o.nonblank for o, _ in rows)
    return EvalCounts(
        n_cells=len(rows),
        n_text=n_text,
        n_correct=sum(o.nonblank and o.correct for o, _ in rows),
        n_review=sum(o.nonblank and r for o, r in rows),
        n_miss=sum((not o.correct) and not r for o, r in rows),
    )


def _calib(rows: Sequence[tuple[CellObs, bool]]) -> Calib:
    free = [o for o, _ in rows if not o.hard]
    nll = brier = 0.0
    for o in free:
        p = min(1 - P_CLIP, max(P_CLIP, o.p))
        y = 1.0 if o.correct else 0.0
        nll -= y * math.log(p) + (1 - y) * math.log(1 - p)
        brier += (p - y) ** 2
    n = len(free)
    passed = [o for o, r in rows if not r]
    return Calib(
        n=n,
        nll=nll / n if n else 0.0,
        brier=brier / n if n else 0.0,
        expected_miss=float(sum(1.0 - min(1.0, max(0.0, o.p)) for o in passed)),
        observed_miss=sum(not o.correct for o in passed),
    )


def side_from(pages: Sequence[Sequence[CellObs]], flags: Sequence[Sequence[bool]]) -> EvalSide:
    """評価ページごとの観測と要確認の判定から、全体と列ごとの成績(件数・較正の質)を作る。"""
    rows = [
        (o, f) for pg, fl in zip(pages, flags, strict=True) for o, f in zip(pg, fl, strict=True)
    ]
    cols = sorted({o.column for o, _ in rows})
    return EvalSide(
        total=_counts(rows),
        calib=_calib(rows),
        by_column={c: _counts([x for x in rows if x[0].column == c]) for c in cols},
        calib_by_column={c: _calib([x for x in rows if x[0].column == c]) for c in cols},
    )


def apply_floor(model: ConfidenceModel, floor: ConfidenceModel | None) -> ConfidenceModel:
    """`floor`(親の世代の確信度)より緩いしきい値を、`floor` のしきい値に引き上げた新しいモデル。群ごと。`floor` が無ければそのまま。"""
    if floor is None:
        return model
    import copy

    out = copy.deepcopy(model)
    for gk, g in out.groups.items():
        fg = floor.groups.get(gk)
        if fg is not None and g.lr is not None and fg.threshold > g.threshold:
            g.threshold = fg.threshold
    return out
