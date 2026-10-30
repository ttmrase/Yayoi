"""列ごとの解決(fields.py)のテスト: 採点器を差し替えて、候補の修復・周辺化・事後確率を確かめる。"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from yayoi_ocr.recognize.fields import (  # noqa: E402
    logsumexp,
    resolve_amount,
    resolve_closed,
    resolve_date,
    resolve_description,
)
from yayoi_ocr.recognize.parseq.engine import ParseqResult, RecognizeOutput  # noqa: E402


class FakeScorer:
    """文字列 → 対数尤度の表で採点する(表に無い文字列は -20)。"""

    def __init__(self, table: dict[str, float]):
        self.table = table

    def score_output(self, out, texts, column=None):
        return [ParseqResult(t, self.table.get(t, -20.0), [], 0.0, "score") for t in texts]


def _out(*texts: str, lp: float = -1.0) -> RecognizeOutput:
    cands = [
        ParseqResult(t, lp - i, [0.9] * len(t), 0.9, "greedy" if i == 0 else "beam", ())
        for i, t in enumerate(texts)
    ]
    return RecognizeOutput(cands, 256)


def test_logsumexp():
    assert math.isclose(logsumexp([math.log(0.2), math.log(0.3)]), math.log(0.5))
    assert logsumexp([float("-inf")]) == float("-inf")


def test_date_marginalizes_surfaces_and_repairs_missing_dot():
    sc = FakeScorer({"5.17": -0.2, "5.1.7": -3.0})
    r = resolve_date(sc, _out("517", "5.17"))
    assert r.ok and r.value == "5.17" and r.obj.month == 5
    assert r.features["post_top"] > 0.9 and r.features["top_p"] > 0.5


def test_date_ambiguous_split_has_low_gap():
    sc = FakeScorer({"1.11": -0.7, "11.1": -0.8})  # "111" は 1.11 とも 11.1 とも読める
    r = resolve_date(sc, _out("111"))
    assert {v for v, _ in r.candidates[:2]} == {"1.11", "11.1"}
    assert r.features["post_gap"] < 0.2  # 差が小さい → 確認に回る


def test_date_with_year_surfaces_are_summed():
    sc = FakeScorer({"R7 11.19": -2.0, "R711.19": -2.0, "R7.11.19": -2.0})
    r = resolve_date(sc, _out("R7 11.19"))
    assert r.value == "R7.11.19"
    # 他の表記(ドット無しなど)は -20 とみなされ、足し合わせた値はほぼ 3 表記ぶん
    assert math.isclose(r.features["top_logp"], -2.0 + math.log(3), abs_tol=1e-3)


def test_date_dotless_surface_makes_missing_dot_confident():
    # ピリオドが写っていない: 「918」が高確率。9.18 は「918」の表記で支えられ、9.8 は支えられない
    sc = FakeScorer({"918": -0.1, "9.18": -9.0, "9.8": -9.0, "98": -8.0})
    r = resolve_date(sc, _out("918", "98"))
    assert r.value == "9.18" and r.features["post_top"] > 0.99 and r.features["top_p"] > 0.8


def test_date_shared_surface_splits_probability():
    sc = FakeScorer({"111": -0.2})  # 1.11 とも 11.1 とも読める表記
    r = resolve_date(sc, _out("111"))
    assert abs(r.features["post_gap"]) < 1e-6  # 同点 → 確認に回る
    assert math.isclose(
        r.features["mass"], math.exp(-0.2), rel_tol=0.01
    )  # 質量は表記の確率(二重に数えない)


def test_date_grammar_violation():
    r = resolve_date(FakeScorer({}), _out("2.30", "13.1", "5."))
    assert not r.ok and r.value == "" and "grammar_violation" in r.reasons


def test_amount_values_and_stray_marks():
    sc = FakeScorer({"1,280-": -0.1, "1280-": -1.5, "1,280": -2.0, "1280": -2.5, "8-": -9.0})
    r = resolve_amount(sc, _out("-1,280-", "1280-"))
    assert r.ok and r.value == "1280" and r.features["post_top"] > 0.99
    assert r.features["direct"] == 1.0  # 第 1 候補(先頭の余計な線を除いたもの)が採用値と同じ


def test_amount_rejects_non_positive_and_garbage():
    r = resolve_amount(FakeScorer({}), _out("0", "-", ""))
    assert not r.ok and r.value == ""


def test_closed_vocab_picks_best_and_flags_oov():
    sc = FakeScorer({"現金": -0.5, "普通預金": -6.0})
    r = resolve_closed(sc, _out("現金", lp=-0.4), ["現金", "普通預金"])
    assert r.value == "現金" and not r.reasons and r.features["post_top"] > 0.99
    # 自由認識のほうが語彙内のどれよりずっとよい → 語彙外の疑い
    r2 = resolve_closed(
        FakeScorer({"現金": -12.0, "普通預金": -13.0}),
        _out("売掛金", lp=-0.3),
        ["現金", "普通預金"],
    )
    assert "oov_account" in r2.reasons


def test_closed_vocab_space_surfaces():
    sc = FakeScorer({"当座 預金": -9.0, "当座預金": -0.3})
    r = resolve_closed(sc, _out("当座預金", lp=-0.3), ["当座 預金"])
    assert r.value == "当座 預金" and math.isclose(
        r.features["top_logp"], logsumexp([-9.0, -0.3]), abs_tol=1e-6
    )


def test_description_dictionary_beats_free_when_close():
    d = {"電話代 8月分": 5, "切手代": 1}
    sc = FakeScorer({"電話代 8月分": -1.5, "電話代8月分": -0.6, "切手代": -9.0})
    r = resolve_description(sc, _out("電話代8月分", lp=-0.6), d)
    assert r.value == "電話代 8月分" and r.features["from_dict"] == 1.0


def test_description_new_text_when_not_in_dictionary():
    d = {"切手代": 3}
    sc = FakeScorer({"切手代": -15.0})
    r = resolve_description(sc, _out("新しい記入", lp=-0.8), d)
    assert r.value == "新しい記入" and r.features["from_dict"] == 0.0


def test_description_cold_start():
    r = resolve_description(FakeScorer({}), _out("消耗品代", "消耗品伐"), None)
    assert r.value == "消耗品代" and r.ok
