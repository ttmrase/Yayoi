"""追加学習のラベル: 確定値(CSV に出す値)から、紙への書き方を推定して正解の文字列にする。

確認画面は書かれた通りの文字列を聞かない(確定値だけが分かる)。確定値と整合する書き方の集合(日付: 年の有無・区切りの違い、金額: 桁区切りと末尾の「-」の有無、
ほか: 確定値そのもの)のうち、今の重みで最も尤度の高いものを正解にする(自己ラベル。「確定値と整合する書き方の周辺尤度」)。
検証(`docs/writer_adaptation.md`)では、書かれた通りのラベルを使った場合とほぼ同じ精度だった。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .grammar import amount_surface_priors, date_surfaces, norm_text, parse_date

# 幅ごとに学習する列(幅 256 = 日付・科目・金額、幅 384 = 摘要)
WIDTH_COLUMNS: dict[int, tuple[str, ...]] = {
    256: ("date", "account", "amount"),
    384: ("description",),
}


def label_candidates(column: str, resolved: str) -> list[str]:
    """確定値と整合する、紙への書き方の集合(1 つ以上)。"""
    if column == "date":
        v = parse_date(resolved)
        if v is None:
            return [resolved]
        return date_surfaces(v) + date_surfaces(v.with_year(None))  # 年が書かれていない場合も含む
    if column == "amount" and resolved.isdigit():
        return list(amount_surface_priors(int(resolved)))
    return [norm_text(resolved)]


class Scorer(Protocol):
    def score_many(self, crop: np.ndarray, texts, column=None, *, width=None): ...


def choose_label(engine: Scorer | None, rec: np.ndarray, column: str, resolved: str) -> str | None:
    """学習の正解にする文字列。候補が複数あれば `engine` で最も尤度の高いものを選ぶ(`engine` が無ければ先頭)。空なら None。"""
    cands = [c for c in label_candidates(column, resolved) if c]
    if not cands:
        return None
    if len(cands) > 1 and engine is not None:
        crop = np.stack([rec] * 3, axis=-1)
        scores = engine.score_many(crop, cands, column)
        return max(zip(cands, scores, strict=True), key=lambda t: t[1].logprob)[0]
    return cands[0]
