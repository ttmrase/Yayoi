"""提案モード: 安全基準には届かないが、評価用ページで親より正解率が有意に高い世代の値を、入力済みの値の提案に使う。

認識のたびに、**使用中の世代(親)**と**提案モードの世代**の両方で推論し、次のように合わせる:

- 確認画面で入力済みにする値(種別・解決した値・候補・展開)は、**提案モードの世代**のもの。
- 要確認の判定 = **親の判定 OR 提案モードの判定 OR 両者の値(または種別)が違う**。
- したがって、要確認でないセルは「親も要確認ではなく、かつ両者の値が一致したセル」だけ。その値は親と同じなので、**セルの見逃しの集合は、作りの上で親の見逃しの部分集合**になる
  (安全性は親以上)。値が違うセルは必ず要確認になり、親の値も候補に入れる(利用者が戻せる)。
- **金額の列で値が違うセルは、利用者が明示的に選ぶ(候補の選択か入力)まで確定できない**(`R_AMOUNT_CHOICE`。末尾「-」の 10 倍違いと同じ扱い)。違うセルは両方誤りのことも多く、
  誤った金額をそのまま Enter で通すのを避けるため。候補は [提案の値, 親の値, …] の順。
- 性質の主張は**セル**についてだけ。ページ単位の項目は親のものを使う: 年の継承の元になる `last_year` は親の結果から取る(次のページの日付の値が、親だけで認識した場合と変わらないように)。
  タイトルは提案モードの世代のものだが、候補として画面に出るだけで、確定は利用者の操作。
- 効果: 要確認のセルでも、最初から正しい値が入っている割合が上がり、修正の手間(上書きの回数)が減る。

`merge` はセルの属性(kind・value・review・reasons・candidates・confidence)だけを使う(torch に依存しない。試験は `SimpleNamespace` で行う)。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

R_DISAGREE = "suggest_disagreement"  # 親と提案モードの値が違うので要確認(確認画面に理由として出る)
R_AMOUNT_CHOICE = (
    "suggest_amount_disagreement"  # 金額で値が違う: 選ぶ(または入力する)まで確定できない
)
AMOUNT = "amount"
MAX_CANDIDATES = 8


class _Cell(Protocol):
    column: str
    kind: str
    value: str
    review: bool
    reasons: list[str]
    candidates: list[tuple[str, float]]
    confidence: float


def differs(parent: _Cell, suggest: _Cell) -> bool:
    return parent.kind != suggest.kind or parent.value != suggest.value


def merge(parent: Sequence[_Cell], suggest: Sequence[_Cell]) -> int:
    """提案モードの世代のセル(`suggest`)を、親のセル(`parent`)と合わせる(`suggest` を書き換える。`parent` は変えない)。値が違うセルの数を返す。"""
    n = 0
    for p, s in zip(parent, suggest, strict=True):
        d = differs(p, s)
        s.review = bool(p.review or s.review or d)
        if not d:
            continue
        n += 1
        if R_DISAGREE not in s.reasons:
            s.reasons.append(R_DISAGREE)
        if s.column == AMOUNT and (p.value or s.value):
            # 金額: 提案の値と親の値を候補の先頭に並べ、どちらかを選ぶ(または入力する)まで確定できないようにする
            if R_AMOUNT_CHOICE not in s.reasons:
                s.reasons.append(R_AMOUNT_CHOICE)
            head = [(v, c) for v, c in ((s.value, s.confidence), (p.value, p.confidence)) if v]
            rest = [(v, c) for v, c in s.candidates if v not in {h[0] for h in head}]
            s.candidates = [*head, *rest][:MAX_CANDIDATES]
        elif p.value and all(v != p.value for v, _ in s.candidates):
            # 親の値も候補に入れる(利用者が選んで戻せる)。先頭に置く
            s.candidates = [(p.value, float(p.confidence)), *s.candidates][:MAX_CANDIDATES]
    return n


def combine(parent_result, suggest_result):
    """ページの結果を合わせる(セルは `merge`)。返り値: (画面に出す結果, 次のページに引き継ぐ年, 親のセル)。
    画面に出す結果は提案モードの世代のもの。ただし年の継承の元(`last_year`)は親の結果から取る。"""
    merge(parent_result.cells, suggest_result.cells)
    return suggest_result, parent_result.last_year, parent_result.cells


def sign_test_p(better: int, worse: int) -> float:
    """片側の符号検定: 値が違うセルのうち、提案が正しい `better` 件・親が正しい `worse` 件(両方誤りは除く)で、P(提案が正しい件数 ≥ better | 公平)。"""
    n = better + worse
    if n == 0 or better <= worse:
        return 1.0
    # P(X ≥ better), X ~ Binomial(n, 1/2)
    tail = sum(math.comb(n, k) for k in range(better, n + 1))
    return min(1.0, tail / 2**n)


# 安全側の基準で採用されなかった理由(これだけが提案モードの対象。正解率・要確認率・得がない場合は対象外)
SAFETY_REASONS = frozenset({"amount_miss", "miss_rate_worse", "overconfident", "calibration_worse"})
ALPHA = 0.05
# 値が違うセル(提案が正 + 親が正)の最小件数。少ない標本の偶然(5 対 0 でも p = 0.031)で提案モードの世代が出ないようにする
MIN_DIFFER = 20


def eligible(reason: str, better: int, worse: int, alpha: float = ALPHA) -> bool:
    """不採用の理由が安全側の基準だけで、値が違うセルが十分にあり(`MIN_DIFFER`)、かつ正解率が親より有意に高い(符号検定の片側 p < alpha)。
    検定は追加学習の実行ごとに独立に行う(実行を重ねるほど偶然に通る確率は累積するが、提案モードは見逃しを親以下に保つので許容する)。"""
    return (
        reason in SAFETY_REASONS
        and better + worse >= MIN_DIFFER
        and sign_test_p(better, worse) < alpha
    )
