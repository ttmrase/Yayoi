"""ページ単位の解決(page.py)のテスト: ダミーのエンジンで、〃・年の継承・矢印/波線の候補・理由コードを確かめる。"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from yayoi_ocr.recognize.celltype import N_STATS
from yayoi_ocr.recognize.page import (
    CellInput,
    JobContext,
    PageInput,
    PageRecognizer,
    layout_reason_codes,
    matches_head_tail,
    split_wave,
)
from yayoi_ocr.recognize.parseq.engine import ParseqResult, RecognizeOutput

COLS = ("date", "account", "description", "amount")


class FakeEngine:
    """セルの画像の (0, 0) 画素に埋めた番号から、あらかじめ決めた認識結果を返す。"""

    def __init__(self, texts: dict[int, str]):
        self.texts = texts

    def recognize_many(self, crops, columns, beam=5, with_embedding=False, keep_memory=False):
        outs = []
        for c in crops:
            t = self.texts[int(c[0, 0, 0])]
            r = ParseqResult(t, -0.2, [0.95] * len(t), 0.9, "greedy", ())
            outs.append(RecognizeOutput([r], 256))
        return outs

    def recognize(self, crop, column, **kw):
        return self.recognize_many([crop], [column])[0]

    def score_output(self, out, texts, column=None):
        best = out.best.text
        return [ParseqResult(t, -0.2 if t == best else -15.0, [], 0.0, "score") for t in texts]


def make_page(rows: list[dict[str, str]]) -> tuple[PageInput, dict[int, str], list[str]]:
    """rows: 行ごとの {列: 'text:認識文字列' | 'ditto' | 'arrow:…' | 'wave:頭〜尾'}。空は空欄。"""
    cells, texts, kinds = [], {}, []
    for r in range(29):
        for c, col in enumerate(COLS):
            spec = rows[r].get(col, "") if r < len(rows) else ""
            idx = r * 4 + c
            if not spec:
                kind, rec, text = "blank", None, ""
            else:
                kind, _, text = spec.partition(":")
                rec = np.zeros((24, 40), np.uint8)
                rec[0, 0] = idx
                if kind == "ditto":
                    text = "〃"
            texts[idx] = text
            kinds.append(kind)
            cells.append(
                CellInput(
                    r, c, col, rec, np.zeros((64, 448), np.uint8), np.zeros(N_STATS, np.float32)
                )
            )
    return PageInput(cells), texts, kinds


def run(rows, ctx=None, **kw):
    page, texts, kinds = make_page(rows)
    rec = PageRecognizer(FakeEngine(texts), **kw)
    return rec.recognize_page(page, ctx or JobContext(accounts=["現金", "普通預金"]), kinds)


def test_ditto_year_inheritance_and_blank_rows():
    res = run(
        [
            {
                "date": "text:R8 8.21",
                "account": "text:現金",
                "description": "text:切手代",
                "amount": "text:500-",
            },
            {
                "date": "text:9.1",
                "account": "ditto",
                "description": "text:電話代",
                "amount": "text:1,200-",
            },
            {"date": "ditto", "account": "ditto", "description": "ditto", "amount": "text:300-"},
            {
                "date": "text:10.8",
                "account": "text:普通預金",
                "description": "text:会議費",
                "amount": "text:50-",
            },
        ]
    )
    assert [res.cell(r, 0).value for r in range(4)] == ["R8.8.21", "R8.9.1", "R8.9.1", "R8.10.8"]
    assert res.cell(1, 1).value == "現金" and res.cell(2, 2).value == "電話代"
    assert res.cell(0, 3).value == "500" and res.cell(1, 3).value == "1200"
    assert res.cell(5, 0).kind == "blank" and res.cell(5, 0).value == ""
    assert res.last_year == 8
    assert not any(c.review for c in res.cells[: 4 * 4])  # きれいな読みはすべて確認なし


def test_year_unknown_and_previous_page_year():
    rows = [{"date": "text:3.4", "amount": "text:100-"}]
    res = run(rows)
    assert "year_unknown" in res.cell(0, 0).reasons and res.cell(0, 0).review
    res2 = run(rows, JobContext(prev_year=7, accounts=["現金"]))
    assert res2.cell(0, 0).value == "R7.3.4" and not res2.cell(0, 0).review


def test_month_decreased_and_year_jump_are_flagged():
    res = run(
        [
            {"date": "text:R8 10.8", "amount": "text:100-"},
            {"date": "text:2.3", "amount": "text:100-"},  # 年が書かれていないのに月が戻る
            {"date": "text:R10 3.1", "amount": "text:100-"},  # 令和 8 → 10(1 年を超えて飛ぶ)
        ]
    )
    assert "month_decreased" in res.cell(1, 0).reasons and res.cell(1, 0).review
    assert "year_jump" in res.cell(2, 0).reasons and res.cell(2, 0).review


def test_dates_going_back_are_only_flagged_never_rewritten():
    """日付は単調とは限らない(戻る・年をまたいで遡ることがある)。順序は要確認の理由にするだけで、値を変えない。"""
    res = run(
        [
            {"date": "text:R8 10.8", "amount": "text:100-"},
            {"date": "text:R8 3.2", "amount": "text:100-"},  # 同じ年のまま月が戻る
            {"date": "text:R7 12.30", "amount": "text:100-"},  # 前の年へ遡る(年をまたいで戻る)
            {
                "date": "text:2.3",
                "amount": "text:100-",
            },  # 年を書かない行は、直前の年(R7)を継承するだけ
        ]
    )
    assert [res.cell(r, 0).value for r in range(4)] == ["R8.10.8", "R8.3.2", "R7.12.30", "R7.2.3"]
    assert "year_jump" in res.cell(2, 0).reasons and "month_decreased" not in res.cell(1, 0).reasons
    assert all(res.cell(r, 0).value != "" for r in range(4))
    # 値のぶんだけ、候補も順序で並べ替えない(認識の第 1 候補がそのまま値になる)
    assert res.cell(1, 0).value == "R8.3.2"


def test_ditto_in_first_row_and_gap():
    res = run([{"account": "ditto"}, {}, {}, {"account": "text:現金"}, {}, {"account": "ditto"}])
    assert "ditto_first_row" in res.cell(0, 1).reasons and res.cell(0, 1).review
    assert "ditto_gap" in res.cell(5, 1).reasons  # 間に空行がある


def test_ditto_inherits_review_from_uncertain_source():
    res = run([{"date": "text:R8 8.21", "description": "text:切手代"}, {"description": "ditto"}])
    assert not res.cell(1, 2).review
    # 参照元が要確認(語彙に無い科目)なら、〃も要確認
    res = run([{"account": "text:売掛金"}, {"account": "ditto"}], JobContext(accounts=["現金"]))
    assert res.cell(0, 1).review and "ditto_source_uncertain" in res.cell(1, 1).reasons
    assert res.cell(1, 1).review


def test_grammar_violation_is_always_review():
    res = run([{"date": "text:13.45", "amount": "text:abc"}])
    assert res.cell(0, 0).review and "grammar_violation" in res.cell(0, 0).reasons
    assert res.cell(0, 3).review and res.cell(0, 3).value == ""


def test_arrow_candidates_rank_by_account_then_amount_then_proximity():
    res = run(
        [
            {"account": "text:現金", "description": "text:切手代", "amount": "text:500-"},
            {"account": "text:普通預金", "description": "text:電話代", "amount": "text:700-"},
            {"account": "text:普通預金", "description": "text:会議費", "amount": "text:900-"},
            {"account": "ditto", "description": "arrow:→", "amount": "text:700-"},
        ]
    )
    c = res.cell(3, 2)
    assert c.kind == "arrow" and c.review and "arrow_candidate" in c.reasons
    # 科目(普通預金)が一致する行のうち、金額も一致する電話代が最上位。近さだけなら会議費
    assert c.value == "電話代" and [v for v, _ in c.candidates][:2] == ["電話代", "会議費"]


def test_wave_candidates_from_page_then_dictionary():
    ctx = JobContext(
        accounts=["現金"], dictionary={"テスト機構 検査記録 閲覧 取扱手数料": 2, "他の摘要": 5}
    )
    res = run(
        [
            {"description": "text:サンプル協会 定期点検 立会い 作業 手数料", "amount": "text:100-"},
            {"description": "wave:サンプル協会〜手数料", "amount": "text:100-"},
            {"description": "wave:テスト機構〜取扱手数料", "amount": "text:100-"},
        ],
        ctx,
    )
    w1, w2 = res.cell(1, 2), res.cell(2, 2)
    assert (
        w1.kind == "wave" and w1.review and w1.value == "サンプル協会 定期点検 立会い 作業 手数料"
    )
    assert w2.value == "テスト機構 検査記録 閲覧 取扱手数料" and "wave_candidate" in w2.reasons


def test_wave_helpers():
    assert split_wave("サンプル協会〜手数料") == ("サンプル協会", "手数料")
    assert split_wave("ab~cd") == ("ab", "cd")
    assert split_wave("あいうえ") == ("あい", "うえ")  # 波線の記号が無ければ前半・後半
    assert matches_head_tail("サンプル協会 定期点検 手数料", "サンプル協会", "手数料")
    assert not matches_head_tail("サンプル協会手数料", "サンプル協会", "手数料")  # 間が無い
    assert not matches_head_tail("別 協会 手数料", "サンプル協会", "手数料")


def test_cp932_and_length_validation():
    res = run(
        [
            {"description": "text:切手代 ☺", "amount": "text:1-"},
            {"description": "text:" + "あ" * 40},
        ]
    )
    assert "cp932_unencodable" in res.cell(0, 2).reasons and res.cell(0, 2).review
    assert "too_long" in res.cell(1, 2).reasons  # 80 バイト(64 バイト超)


def test_layout_reason_codes():
    notes = [
        "見出し帯が画像の端で切れている(表の隅が写っていない可能性)",
        "横罫線の検出率が低い(70%)",
        "縦罫線の検出率が低い(60%)",
        "表の外枠の検出に自信がない",
        "罫線の交点の整合性が低い",
        "表が画像からはみ出している(撮影範囲が足りない可能性)",
        "未知の注意書き",
    ]
    assert layout_reason_codes(notes) == [
        "band_clipped",
        "ruled_lines_low",
        "outer_frame_missing",
        "intersection_inconsistent",
        "table_cut",
        "layout_other",
    ]


def test_page_review_forces_review_on_all_cells():
    page, texts, kinds = make_page([{"date": "text:R8 8.21", "amount": "text:500-"}])
    page.layout_status = "needs_review"
    res = PageRecognizer(FakeEngine(texts)).recognize_page(page, JobContext(), kinds)
    assert res.cell(0, 0).review and "page_needs_review" in res.cell(0, 0).reasons


def test_row_consistency_rules_force_review_regardless_of_classifier_confidence():
    res = run(
        [
            {
                "date": "text:R8 8.21",
                "account": "text:現金",
                "description": "",
                "amount": "text:500-",
            },
            {
                "date": "text:R8 8.22",
                "account": "text:現金",
                "description": "text:切手代",
                "amount": "",
            },
            {
                "date": "text:R8 8.23",
                "account": "text:現金",
                "description": "text:切手代",
                "amount": "text:5-",
            },
        ]
    )
    d = res.cell(0, 2)  # 金額あり・摘要が blank と判定
    assert d.kind == "blank" and d.review and "row_inconsistent" in d.reasons
    a = res.cell(1, 3)  # 他列に記入があるのに金額が blank
    assert a.kind == "blank" and a.review and "row_inconsistent" in a.reasons
    assert not res.cell(2, 2).review and "row_inconsistent" not in res.cell(2, 0).reasons
    assert not res.cell(5, 3).review  # 行全体が空欄なら何も起きない


def test_amount_history_feature_marks_repeat_transactions():
    ctx = JobContext(
        accounts=["現金"], amount_history={"d:切手代": {500: 2}, "a:現金": {500: 1, 900: 3}}
    )
    res = run(
        [
            {"account": "text:現金", "description": "text:切手代", "amount": "text:500-"},
            {"account": "text:現金", "description": "text:切手代", "amount": "text:800-"},
        ],
        ctx,
    )
    hit, miss = res.cell(0, 3), res.cell(1, 3)
    assert hit.features["hist_desc"] == 1.0 and hit.features["hist_acct"] == 1.0
    assert (
        miss.features["hist_desc"] == 0.0 and miss.features["hist_acct"] == 0.0
    )  # 不一致は 0(下げる特徴は無い)


def test_dash_digit_ambiguity_reason_reaches_cell():
    # FakeScorer は R1 の第 1 候補の表記だけを尤もらしいとするので、曖昧さの有無は表記の事前で決まる:
    # 「5000」と読まれた金額は、末尾が「-」の仮説(500)と 10 倍違う候補として並ぶ
    res = run([{"account": "text:現金", "description": "text:切手代", "amount": "text:5000"}])
    c = res.cell(0, 3)
    assert c.value in ("500", "5000") and "dash_ambig" in c.features
