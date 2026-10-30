"""列ごとの解決: R1 の候補 → 文法・語彙に合う値(と事後確率・確信度の特徴量)。

どの列も同じ考え方: 候補の値ごとに、書かれ方(表記ゆれ)の尤度をエンジンで採点して足し合わせ(周辺化)、
値の尤度とする。事後確率は候補集合の中での正規化(post_top, post_gap)と、モデルが与える絶対的な確率
(top_p = exp(L_top)、mass = 候補集合全体の確率質量)の両方を特徴量に出す。後者が小さいときは、
モデルが「文法に合わない文字列」に確率を割いている = 読めていない、という意味になる。

- 日付: 文法 + 暦。年は省略可(年の継承はページ段階 `page.py` で行う)。
- 金額: 文法(カンマ・末尾の「-」は任意)。値 > 0。
- 科目・タイトル: 閉じた語彙(設定の支払方法 + 過去の確定値)の中で最も尤度の高いもの。語彙外なら確認。
- 摘要: 過去に確定した値の頻度辞書 vs 自由認識。辞書に無い新しい記入は自由認識の結果を使う。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .grammar import (
    DateValue,
    amount_comma_ok,
    amount_surface_priors,
    dash_digit_alternatives,
    date_surfaces,
    norm_text,
    repair_amount,
    repair_date,
    ten_fold_pair,
)
from .parseq.engine import ParseqResult, RecognizeOutput

NEG_INF = float("-inf")


class Scorer(Protocol):
    """採点器(ParseqEngine)。テストでは差し替える。"""

    def score_output(
        self, out: RecognizeOutput, texts: Sequence[str], column: object = None
    ) -> list[ParseqResult]: ...


def logsumexp(xs: Sequence[float]) -> float:
    xs = [x for x in xs if x != NEG_INF]
    if not xs:
        return NEG_INF
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def _marginal_logps(
    scorer: Scorer,
    out: RecognizeOutput,
    values: Sequence,
    surfaces_of,
    column: object = None,
    priors_of=None,
    flat: list[float] | None = None,
) -> list[float]:
    """値ごとの対数尤度 = その値の表記ゆれの尤度の和(周辺化)。

    全表記を 1 回の呼び出しでまとめて採点する。複数の値が同じ表記を持つ場合(例: '111' は 1.11 とも 11.1 とも読める)
    は、その表記の確率を値の数で等分する(曖昧さが事後確率の差の小ささとして現れる)。
    """
    owners: dict[str, list[int]] = {}
    for i, v in enumerate(values):
        for s in surfaces_of(v):
            owners.setdefault(s, [])
            if i not in owners[s]:
                owners[s].append(i)
    texts = list(owners)
    scores = scorer.score_output(out, texts, column)
    per: list[list[float]] = [[] for _ in values]
    per_flat: list[list[float]] = [[] for _ in values]
    pri = [priors_of(v) for v in values] if priors_of else None
    for t, sc in zip(texts, scores, strict=True):
        share = math.log(len(owners[t]))
        for i in owners[t]:
            lw = math.log(max(pri[i].get(t, 1.0), 1e-12)) if pri else 0.0  # 表記の事前
            per[i].append(sc.logprob - share + lw)
            per_flat[i].append(sc.logprob - share)
    if flat is not None:  # 事前なし(画像の証拠だけ)の対数尤度も返す
        flat[:] = [logsumexp(p) for p in per_flat]
    return [logsumexp(p) for p in per]


MEMORY_BONUS = (
    3.0  # 票の割合 1.0 のとき加える対数尤度(見覚えメモリ。scripts/sim_learning.py で較正する)
)


def _mem_features(votes: dict | None, max_sim: float, chosen_share: float) -> dict[str, float]:
    if votes is None:
        return {}
    return {
        "mem_share": chosen_share,
        "mem_max_sim": max_sim,
        "mem_n": float(len(votes)),
    }


@dataclass
class FieldResult:
    value: str  # 解決した値(日付は 'R7.11.19'・'11.19'、金額は '3456'、文字列はそのまま)。空 = 決まらなかった
    text: str  # R1 の第 1 候補(書かれた通りに近い認識文字列)
    ok: bool  # 文法・語彙に合う値が得られた
    candidates: list[tuple[str, float]] = field(default_factory=list)  # (値, 事後確率) の降順
    features: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    obj: object = None  # DateValue など(ページ段階で使う)
    char_confidences: list[float] = field(default_factory=list)


def _posterior_features(logps: dict[str, float], prefix: str = "") -> dict[str, float]:
    """値 → 対数尤度 から、事後確率・差・質量などの特徴量。"""
    items = sorted(logps.items(), key=lambda kv: -kv[1])
    total = logsumexp([v for _, v in items])
    top = items[0][1]
    second = items[1][1] if len(items) > 1 else NEG_INF
    post_top = math.exp(top - total) if total != NEG_INF else 0.0
    post_second = math.exp(second - total) if second != NEG_INF else 0.0
    return {
        f"{prefix}top_logp": max(top, -50.0),
        f"{prefix}top_p": math.exp(max(top, -50.0)),
        f"{prefix}mass": math.exp(max(total, -50.0)) if total != NEG_INF else 0.0,
        f"{prefix}post_top": post_top,
        f"{prefix}post_gap": post_top - post_second,
        f"{prefix}n_cands": float(len(items)),
    }


def _ranked(logps: dict[str, float], k: int = 5) -> list[tuple[str, float]]:
    total = logsumexp(list(logps.values()))
    items = sorted(logps.items(), key=lambda kv: -kv[1])[:k]
    return [(v, math.exp(lp - total) if total != NEG_INF else 0.0) for v, lp in items]


# ---------------------------------------------------------------- 日付


def resolve_date(
    scorer: Scorer,
    out: RecognizeOutput,
    max_cands: int = 8,
    votes: dict[str, float] | None = None,
    max_sim: float = 0.0,
    mem_bonus: float | None = None,
) -> FieldResult:
    mb = (
        MEMORY_BONUS if mem_bonus is None else mem_bonus
    )  # 実行時の値(sim_learning.py が較正で変える)
    best = out.best
    cands: dict[DateValue, None] = {}
    for c in out.candidates[:max_cands]:
        for v in repair_date(c.text):
            cands.setdefault(v, None)
    vote_md: dict[tuple[int, int], float] = {}
    for t, w in (
        votes or {}
    ).items():  # 見覚えメモリ: 近い見た目の確定テキスト(年は継承なので月日で照合する)
        vs = repair_date(t)
        if vs:
            vote_md[(vs[0].month, vs[0].day)] = vote_md.get((vs[0].month, vs[0].day), 0.0) + w
            cands.setdefault(vs[0], None)
    feats = {"r1_logp": max(best.logprob, -50.0), "r1_min_char": best.min_char_confidence}
    if not cands:
        return FieldResult(
            "", best.text, False, [], feats, ["grammar_violation"], None, best.char_confidences
        )
    vals = list(cands)
    logps = dict(zip(vals, _marginal_logps(scorer, out, vals, date_surfaces, "date"), strict=True))
    if vote_md:
        logps = {v: lp + mb * vote_md.get((v.month, v.day), 0.0) for v, lp in logps.items()}
    named = {v.format(): lp for v, lp in logps.items()}
    feats.update(_posterior_features(named))
    top = max(logps, key=lambda v: logps[v])
    feats.update(_mem_features(votes, max_sim, vote_md.get((top.month, top.day), 0.0)))
    feats["direct"] = float(repair_date(best.text)[:1] == [top])
    feats["explicit_year"] = float(top.explicit_year)
    reasons = []
    if feats["top_p"] < 1e-3:
        reasons.append("grammar_violation")
    return FieldResult(
        top.format(), best.text, True, _ranked(named), feats, reasons, top, best.char_confidences
    )


# ---------------------------------------------------------------- 金額


def resolve_amount(
    scorer: Scorer,
    out: RecognizeOutput,
    max_cands: int = 8,
    votes: dict[str, float] | None = None,
    max_sim: float = 0.0,
    mem_bonus: float | None = None,
    extra_values: Sequence[int] = (),
    ambiguity_post: float = 0.10,
) -> FieldResult:
    """金額の解決。

    末尾の字形について「『-』である」「数字(0 など)である」の両方の仮説の値を候補に出し(`dash_digit_alternatives`)、
    表記の事前(末尾の「-」あり = 0.9)を掛けて周辺化する。10 倍違う(最後の 1 字の取り違え)候補が両方とも
    有力(事後確率 ≥ `ambiguity_post`)なら、必ず要確認(`dash_digit_ambiguity`)。
    `extra_values`: 履歴(同じ摘要・科目の過去の確定金額)から来る候補。R1 の尤度で採点するだけで、ボーナスは与えない。
    """
    mb = (
        MEMORY_BONUS if mem_bonus is None else mem_bonus
    )  # 実行時の値(sim_learning.py が較正で変える)
    best = out.best
    cands: dict[int, None] = {}
    for c in out.candidates[:max_cands]:
        for v in repair_amount(c.text):
            cands.setdefault(v, None)
    for c in out.candidates[:3]:  # 末尾の字形の仮説(「-」か数字か)は上位の読みから
        for v in dash_digit_alternatives(c.text):
            cands.setdefault(v, None)
    vote_v: dict[int, float] = {}
    for t, w in (votes or {}).items():
        vs = repair_amount(t)
        if vs:
            vote_v[vs[0]] = vote_v.get(vs[0], 0.0) + w
            cands.setdefault(vs[0], None)
    for v in extra_values:
        cands.setdefault(int(v), None)
    feats = {"r1_logp": max(best.logprob, -50.0), "r1_min_char": best.min_char_confidence}
    if not cands:
        return FieldResult(
            "", best.text, False, [], feats, ["grammar_violation"], None, best.char_confidences
        )
    vals = list(cands)
    flat: list[float] = []
    logps = dict(
        zip(
            vals,
            _marginal_logps(
                scorer, out, vals, lambda v: list(amount_surface_priors(v)), "amount",
                priors_of=amount_surface_priors, flat=flat,
            ),
            strict=True,
        )
    )  # fmt: skip
    if vote_v:
        logps = {v: lp + mb * vote_v.get(v, 0.0) for v, lp in logps.items()}
    named = {str(v): lp for v, lp in logps.items()}
    feats.update(_posterior_features(named))
    top = max(logps, key=lambda v: logps[v])
    feats.update(_mem_features(votes, max_sim, vote_v.get(top, 0.0)))
    direct = repair_amount(best.text)[:1] == [top]
    feats["direct"] = float(direct)
    reasons = []
    if feats["top_p"] < 1e-3:
        reasons.append("grammar_violation")
    # 10 倍違う候補が両方とも有力なら要確認。「有力」は表記の事前(末尾「-」が多い)を除いた、画像の証拠だけの事後で判断する
    # (事前を入れると 10 倍違う側が小さく見えて、曖昧さを見落とす)
    flat_named = {str(v): lp for v, lp in zip(vals, flat, strict=True)}
    strong = [int(v) for v, p_ in _ranked(flat_named, k=len(flat_named)) if p_ >= ambiguity_post]
    ambig = any(ten_fold_pair(a, b) for ia, a in enumerate(strong) for b in strong[ia + 1 :])
    feats["dash_ambig"] = float(ambig)
    if ambig:
        reasons.append("dash_digit_ambiguity")
    # 桁区切りのカンマの位置が桁数と整合しない読みを採ったときは要確認
    comma_bad = not amount_comma_ok(best.text)
    feats["comma_bad"] = float(comma_bad)
    if comma_bad and direct:
        reasons.append("comma_mismatch")
    return FieldResult(
        str(top), best.text, True, _ranked(named), feats, reasons, top, best.char_confidences
    )


# ---------------------------------------------------------------- 閉じた語彙(科目・タイトル)


def _surfaces(v: str) -> list[str]:
    """文字列の表記ゆれ: 原文と、空白を除いたもの(NDL のモデルは空白をほとんど出さない)。"""
    nospace = "".join(v.split())
    return [v] if nospace == v else [v, nospace]


def resolve_closed(
    scorer: Scorer,
    out: RecognizeOutput,
    vocab: Sequence[str],
    prior: dict[str, float] | None = None,
    oov_gap: float = 4.0,
    votes: dict[str, float] | None = None,
    max_sim: float = 0.0,
    mem_bonus: float | None = None,
) -> FieldResult:
    """語彙の中で最も尤度の高い値。自由認識より `oov_gap`(対数尤度)以上悪いときは語彙外の疑い。"""
    mb = (
        MEMORY_BONUS if mem_bonus is None else mem_bonus
    )  # 実行時の値(sim_learning.py が較正で変える)
    best = out.best
    feats = {"r1_logp": max(best.logprob, -50.0), "r1_min_char": best.min_char_confidence}
    if not vocab:
        return FieldResult(
            "", best.text, False, [], feats, ["oov_account"], None, best.char_confidences
        )
    texts: list[str] = []
    owner: list[int] = []
    for i, v in enumerate(vocab):
        for s in _surfaces(v):
            texts.append(s)
            owner.append(i)
    scores = scorer.score_output(out, texts)
    per: dict[int, list[float]] = {}
    for o, sc in zip(owner, scores, strict=True):
        per.setdefault(o, []).append(sc.logprob)
    logps = {}
    for i, v in enumerate(vocab):
        lp = logsumexp(per[i])
        if prior is not None and v in prior and prior[v] > 0:
            lp += math.log(prior[v])
        if votes:
            lp += mb * votes.get(v, 0.0)
        logps[v] = lp
    feats.update(_posterior_features(logps))
    top = max(logps, key=lambda v: logps[v])
    feats.update(_mem_features(votes, max_sim, (votes or {}).get(top, 0.0)))
    fit = logps[top] - best.logprob  # 自由認識に対する、語彙内の最良の尤度の差(≦ 0 が普通)
    feats["vocab_fit"] = max(min(fit, 5.0), -50.0)
    feats["free_agree"] = float(norm_text(best.text) == norm_text(top))
    reasons = []
    if fit < -oov_gap:
        reasons.append("oov_account")
    return FieldResult(
        top, best.text, True, _ranked(logps), feats, reasons, top, best.char_confidences
    )


# ---------------------------------------------------------------- 摘要


def resolve_description(
    scorer: Scorer,
    out: RecognizeOutput,
    dictionary: dict[str, int] | None = None,
    max_dict: int = 400,
    dict_weight: float = 0.9,
    free_prior_logp: float = -5.0,
    votes: dict[str, float] | None = None,
    max_sim: float = 0.0,
    mem_bonus: float | None = None,
) -> FieldResult:
    """過去に確定した摘要(頻度辞書)と自由認識の比較。

    事後 ∝ 画像の尤度 x 事前。辞書の事前は頻度 x `dict_weight`、辞書に無い自由認識の事前は `free_prior_logp`。
    """
    mb = (
        MEMORY_BONUS if mem_bonus is None else mem_bonus
    )  # 実行時の値(sim_learning.py が較正で変える)
    best = out.best
    feats = {"r1_logp": max(best.logprob, -50.0), "r1_min_char": best.min_char_confidence}
    logps: dict[str, float] = {}
    dict_keys: set[str] = set()
    if dictionary:
        total = float(sum(dictionary.values()))
        entries = sorted(dictionary, key=lambda k: -dictionary[k])[:max_dict]
        texts, owner = [], []
        for i, v in enumerate(entries):
            for s_ in _surfaces(v):
                texts.append(s_)
                owner.append(i)
        scores = scorer.score_output(out, texts)
        per: dict[int, list[float]] = {}
        for o, sc in zip(owner, scores, strict=True):
            per.setdefault(o, []).append(sc.logprob)
        for i, v in enumerate(entries):
            prior = max(dict_weight * dictionary[v] / total, 1e-12)
            logps[v] = logsumexp(per[i]) + math.log(prior)
            dict_keys.add(norm_text(v))
    # 自由認識(第 1 候補と他の候補)。辞書の値と(空白を除いて)同じ文字列は辞書側にまとめる
    for c in out.candidates[:5]:
        if c.text and c.text not in logps and norm_text(c.text) not in dict_keys:
            logps[c.text] = c.logprob + free_prior_logp
    if votes:  # 見覚えメモリ: 近い見た目の確定テキストを候補に加え、票の割合を事前として足す
        extra = [t for t in votes if t not in logps and norm_text(t) not in dict_keys]
        if extra:
            for t, r in zip(extra, scorer.score_output(out, extra, None), strict=True):
                logps[t] = r.logprob + free_prior_logp
        for t, w in votes.items():
            if t in logps:
                logps[t] += mb * w
    if not logps:
        return FieldResult(
            "", best.text, False, [], feats, ["no_valid_candidate"], None, best.char_confidences
        )
    feats.update(_posterior_features(logps))
    top = max(logps, key=lambda v: logps[v])
    feats.update(_mem_features(votes, max_sim, (votes or {}).get(top, 0.0)))
    feats["from_dict"] = float(bool(dictionary) and top in dictionary)
    feats["free_agree"] = float(norm_text(best.text) == norm_text(top))
    return FieldResult(top, best.text, True, _ranked(logps), feats, [], top, best.char_confidences)
