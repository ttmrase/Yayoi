"""認識の評価: 正解との照合(正規化)と集計。

指標(ディレクター指示):
- 見逃し率(miss)= 「間違っているのに要確認が付いていないセル」÷ 全セル。金額列は特に低く(目標 0.1% 以下、全体 0.5% 以下)。
- 要確認率(review)= 要確認が付いたセル ÷ 全セル(空欄を除いた分母も併記)。
- 「間違っている」の定義(正規化後に比較): 日付 = 年月日が一致(年は継承後)、金額 = 数値が一致、科目・摘要 = 空白を除いた
  NFKC 文字列が一致。空欄は種別(blank)が一致すれば正解。矢印・波線は候補に正解を含むか(常に確認が付く)。
- レイアウトのページが「要確認」の場合、そのページの全セルは利用者が位置を確認するので「確認済み扱い」とする
  (見逃しに数えない)。別途ページ数を報告する。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ..synth.dataset import CellRecord
from .grammar import norm_text
from .page import CellResult


def expected_value(c: CellRecord) -> str:
    """正解の値(比較用に正規化)。空欄は ''。"""
    if c.kind == "blank":
        return ""
    if c.column == "amount":
        return str(int(c.resolved)) if c.resolved.isdigit() else c.resolved
    if c.column == "date":
        return c.resolved
    return norm_text(c.resolved)


def predicted_value(r: CellResult) -> str:
    if r.kind == "blank":
        return ""
    if r.column in ("account", "description"):
        return norm_text(r.value)
    return r.value


@dataclass
class Judged:
    column: str
    gt_kind: str
    pred_kind: str
    correct: bool
    review: bool
    page_review: bool
    miss: bool
    kind_ok: bool
    in_candidates: bool  # 値が違っても、候補(上位 5)に正解があるか
    reasons: tuple[str, ...]
    confidence: float
    iou: float
    has_annotation: bool
    row: int


def judge(r: CellResult, c: CellRecord, page_review: bool = False) -> Judged:
    gt_kind = "wave" if c.kind == "elided" else c.kind
    exp = expected_value(c)
    pred = predicted_value(r)
    kind_ok = r.kind == gt_kind
    if gt_kind == "blank":
        correct = r.kind == "blank"
    elif gt_kind in ("arrow", "wave"):
        correct = kind_ok and pred == exp
    else:
        correct = (r.kind in ("text", "ditto")) and pred == exp
        if r.kind == "ditto" and gt_kind == "text":
            correct = (
                pred == exp
            )  # 同上と判定したが、書かれていたのは同じ値の文字(値が同じなら実害なし)
    cands = {predicted_value_of(c.column, v) for v, _ in r.candidates}
    in_cands = correct or exp in cands
    review = r.review or page_review
    return Judged(
        column=c.column,
        gt_kind=gt_kind,
        pred_kind=r.kind,
        correct=correct,
        review=review,
        page_review=page_review,
        miss=(not correct) and not review,
        kind_ok=kind_ok,
        in_candidates=in_cands,
        reasons=tuple(r.reasons),
        confidence=r.confidence,
        iou=c.iou,
        has_annotation=bool(c.annotation),
        row=c.row,
    )


def predicted_value_of(column: str, v: str) -> str:
    return norm_text(v) if column in ("account", "description") else v


def summarize(rows: list[Judged], by: str | None = None) -> dict[str, dict[str, float]]:
    """列ごと(+ 全体)の集計。by="gt_kind" などで層別。"""
    groups: dict[str, list[Judged]] = defaultdict(list)
    for j in rows:
        key = "all" if by is None else str(getattr(j, by))
        groups[key].append(j)
    out: dict[str, dict[str, float]] = {}
    for key, js in groups.items():
        out[key] = _stats(js)
    return out


def _stats(js: list[Judged]) -> dict[str, float]:
    n = len(js)
    nb = [j for j in js if j.gt_kind != "blank"]
    wrong = [j for j in js if not j.correct]
    miss = [j for j in js if j.miss]
    return {
        "n": n,
        "n_nonblank": len(nb),
        "accuracy": 1 - len(wrong) / n if n else float("nan"),
        "accuracy_nonblank": (sum(j.correct for j in nb) / len(nb)) if nb else float("nan"),
        "review_rate": sum(j.review for j in js) / n if n else float("nan"),
        "review_rate_nonblank": (sum(j.review for j in nb) / len(nb)) if nb else float("nan"),
        "miss": len(miss),
        "miss_rate": len(miss) / n if n else float("nan"),
        "miss_rate_nonblank": (len(miss) / len(nb)) if nb else float("nan"),
        "wrong_flagged": float(sum(1 for j in wrong if j.review)),
        "wrong": float(len(wrong)),
        "kind_acc": sum(j.kind_ok for j in js) / n if n else float("nan"),
    }


def format_table(summary: dict[str, dict[str, float]], title: str) -> str:
    head = f"{title:18s} {'n':>6s} {'nonblank':>8s} {'acc':>7s} {'acc(nb)':>8s} {'review':>7s} {'rev(nb)':>8s} {'miss':>5s} {'miss%':>7s} {'miss%(nb)':>9s}"
    lines = [head]
    for k, s in summary.items():
        lines.append(
            f"{k:18s} {int(s['n']):6d} {int(s['n_nonblank']):8d} {s['accuracy']:7.3f} "
            f"{s['accuracy_nonblank']:8.3f} {s['review_rate']:7.3f} {s['review_rate_nonblank']:8.3f} "
            f"{int(s['miss']):5d} {100 * s['miss_rate']:7.2f} {100 * s['miss_rate_nonblank']:9.2f}"
        )
    return "\n".join(lines)
