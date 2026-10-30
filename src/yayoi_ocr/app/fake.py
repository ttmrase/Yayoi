"""テスト・デモ用の「認識結果」の合成(開発専用。`--dev` のときだけ API から使える)。

本物の認識器の代わりに、正解(truth)を持つページを作り、決まった割合で誤りを混ぜた認識値と、要確認フラグ・展開(〃・年)・
10 倍違いの金額を付ける。切り出し画像は OpenCV で文字を描いただけの簡素なもの(フォントを使わないので CI の web ジョブでも動く)。
torch を使わない。実データは使わない。
"""

from __future__ import annotations

import cv2
import numpy as np

from .store import CellSpec, PageSpec

COLUMNS = ("date", "account", "description", "amount")
ACCOUNTS = ["現金", "普通預金", "未払金", "売掛金"]
DESCRIPTIONS = [
    "事務用品費",
    "新聞代",
    "旅費交通費",
    "会議費",
    "消耗品費",
    "水道光熱費",
    "郵送料",
    "書籍代",
]
ASCII_FALLBACK = {c: f"{i:02d}" for i, c in enumerate(ACCOUNTS + DESCRIPTIONS)}


def _png(text: str, width: int, height: int = 40) -> bytes:
    img = np.full((height, width, 3), 255, np.uint8)
    label = text if text.isascii() else ASCII_FALLBACK.get(text, "#" + str(abs(hash(text)) % 100))
    cv2.putText(
        img, label, (6, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA
    )
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _wrong_digit(s: str, rng: np.random.Generator) -> str:
    idx = [i for i, ch in enumerate(s) if ch.isdigit()]
    if not idx:
        return s
    i = int(rng.choice(idx))
    d = str((int(s[i]) + int(rng.integers(1, 9))) % 10)
    return s[:i] + d + s[i + 1 :]


def fake_page(
    seed: int,
    n_rows: int = 29,
    fill_rows: int = 12,
    error_rate: float = 0.25,
    review_rate: float = 0.9,
    ambiguous_rows: tuple[int, ...] = (3,),
    layout_status: str = "ok",
) -> tuple[PageSpec, dict[tuple[int, int], str]]:
    """1 ページ分の認識結果と、正解(セル (行, 列) → 確定すべき値)。"""
    rng = np.random.default_rng(seed)
    truth: dict[tuple[int, int], str] = {}
    cells: list[CellSpec] = []
    year, month, day = 8, int(rng.integers(1, 12)), int(rng.integers(1, 20))
    prev_account = prev_desc = ""
    for r in range(n_rows):
        if r >= fill_rows:
            for c, col in enumerate(COLUMNS):
                cells.append(CellSpec(r, c, col, "blank", crop=_png("", 140)))
                truth[(r, c)] = ""
            continue
        day += int(rng.integers(0, 3))
        if day > 28:
            day, month = 1, month + 1
        month = min(month, 12)
        explicit_year = r == 0
        date_true = f"R{year}.{month}.{day}"
        amount = int(rng.choice([300, 500, 980, 1200, 3500, 4800, 12000]))
        account_true = ACCOUNTS[int(rng.integers(len(ACCOUNTS)))]
        desc_true = DESCRIPTIONS[int(rng.integers(len(DESCRIPTIONS)))]
        ditto_acct = r > 0 and prev_account != "" and rng.random() < 0.4
        ditto_desc = r > 0 and prev_desc != "" and rng.random() < 0.3
        if ditto_acct:
            account_true = prev_account
        if ditto_desc:
            desc_true = prev_desc
        prev_account, prev_desc = account_true, desc_true
        specs: list[tuple[str, str, str, dict | None, list[tuple[str, float]], list[str]]] = []
        # 日付(2 行目以降は年を書かない = 年の継承の展開)
        date_rec = date_true if explicit_year else f"{month}.{day}"
        date_shown = date_true if rng.random() > error_rate else _wrong_digit(date_true, rng)
        specs.append(
            (
                "text",
                date_rec,
                date_shown,
                None if explicit_year else {"kind": "year", "source": [r - 1, 0]},
                [],
                [],
            )
        )
        # 科目・摘要
        for text, ditto in ((account_true, ditto_acct), (desc_true, ditto_desc)):
            if ditto:
                specs.append(
                    ("ditto", "〃", text, {"kind": "ditto", "source": [r - 1, len(specs)]}, [], [])
                )
            else:
                shown = text if rng.random() > error_rate else text[:-1] + "?"
                specs.append(("text", text, shown, None, [], []))
        # 金額(指定の行は「500」か「5000」か曖昧)
        if r in ambiguous_rows:
            amount = 500
            specs.append(
                (
                    "text",
                    "500-",
                    "5000",
                    None,
                    [("5000", 0.5), ("500", 0.5)],
                    ["dash_digit_ambiguity"],
                )
            )
        else:
            shown = str(amount) if rng.random() > error_rate else _wrong_digit(str(amount), rng)
            specs.append(("text", f"{amount:,}-", shown, None, [], []))
        truths = [date_true, account_true, desc_true, str(amount)]
        for c, (col, (kind, rec, shown, exp, cands, extra)) in enumerate(
            zip(COLUMNS, specs, strict=True)
        ):
            truth[(r, c)] = truths[c]
            reasons = [*extra]
            review = rng.random() < review_rate or bool(extra)
            if review and not reasons:
                reasons.append("low_confidence")
            if exp and exp["kind"] == "ditto":
                reasons.append("ditto_source_uncertain") if review else None
            if not cands and shown != truths[c] and col in ("date", "amount"):
                cands = [(truths[c], 0.2)]  # 候補に正解が入っている例
            cells.append(
                CellSpec(
                    r,
                    c,
                    col,
                    kind,
                    rec,
                    shown,
                    exp,
                    reasons,
                    cands,
                    review,
                    crop=_png(truths[c], 360 if col == "description" else 220),
                )
            )
    if layout_status != "ok":
        for cs in cells:
            cs.reasons.append("page_needs_review")
            cs.review = True
    return PageSpec(
        cells, layout_status, ["ruled_lines_low"] if layout_status != "ok" else []
    ), truth
