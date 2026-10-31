"""提案モード: 安全基準には届かないが、評価用ページで親より正解率が有意に高い世代の値を、入力済みの値の提案に使う。

認識のたびに、**使用中の世代(親)**と**提案モードの世代**の両方で推論し、次のように合わせる:

- 確認画面で入力済みにする値(種別・解決した値・候補・展開)は、**提案モードの世代**のもの。
- 要確認の判定 = **親の判定 OR 提案モードの判定 OR 両者の値(または種別)が違う**。
- したがって、要確認でないセルは「親も要確認ではなく、かつ両者の値が一致したセル」だけ。その値は親と同じなので、**見逃しの集合は、作りの上で親の見逃しの部分集合**になる
  (安全性は親以上)。値が違うセルは必ず要確認になり、親の値も候補に入れる(利用者が戻せる)。
- 効果: 要確認のセルでも、最初から正しい値が入っている割合が上がり、修正の手間(上書きの回数)が減る。

`merge` はセルの属性(kind・value・review・reasons・candidates・confidence)だけを使う(torch に依存しない。試験は `SimpleNamespace` で行う)。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

R_DISAGREE = "suggest_disagreement"  # 親と提案モードの値が違うので要確認(確認画面に理由として出る)
MAX_CANDIDATES = 8


class _Cell(Protocol):
    kind: str
    value: str
    review: bool
    reasons: list[str]
    candidates: list[tuple[str, float]]
    confidence: float


def differs(parent: _Cell, suggest: _Cell) -> bool:
    return parent.kind != suggest.kind or parent.value != suggest.value


def merge(parent: Sequence[_Cell], suggest: Sequence[_Cell]) -> int:
    """提案モードの世代のセル(`suggest`)を、親のセル(`parent`)と合わせる(`suggest` を書き換える)。値が違うセルの数を返す。"""
    n = 0
    for p, s in zip(parent, suggest, strict=True):
        d = differs(p, s)
        s.review = bool(p.review or s.review or d)
        if d:
            n += 1
            if R_DISAGREE not in s.reasons:
                s.reasons.append(R_DISAGREE)
            if p.value and all(v != p.value for v, _ in s.candidates):
                # 親の値も候補に入れる(利用者が選んで戻せる)。先頭に置く
                s.candidates = [(p.value, float(p.confidence)), *s.candidates][:MAX_CANDIDATES]
    return n


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


def eligible(reason: str, better: int, worse: int, alpha: float = ALPHA) -> bool:
    """不採用の理由が安全側の基準だけで、かつ正解率が親より有意に高い(符号検定の片側 p < alpha)。"""
    return reason in SAFETY_REASONS and sign_test_p(better, worse) < alpha
