"""ページ単位の認識: セル種別の判定 → R1 → 列ごとの解決 → 同上・矢印・波線・年の継承 → 検証・理由コード。

入力は `PageInput`(セルごとの認識用画像・分類器用画像・Phase 1 の統計)。評価では合成データセット
(`synth.dataset.PageData`)から、実運用では layout の `LayoutResult` から作る(`inputs.py`)。

出力の `CellResult` は、値(resolved)・確信度の特徴量・要確認の理由コード・候補を持つ。確信度の統合(ロジスティック
回帰)は `confidence.py` がこの特徴量から行う。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .celltype import CLASSES, CellTypeClassifier
from .confidence import HARD_REASONS
from .fields import (
    FieldResult,
    resolve_amount,
    resolve_closed,
    resolve_date,
    resolve_description,
)
from .grammar import DateValue, is_valid_date, norm_text
from .parseq.engine import ParseqEngine, RecognizeOutput

COLUMNS = ("date", "account", "description", "amount")
N_ROWS = 29

# 要確認の理由コード(Phase 6 の統計にも使う。実データの値は入れない)
R_BAND_CLIPPED = "band_clipped"
R_BAND_NOT_FOUND = "band_not_found"
R_OUTER_FRAME = "outer_frame_missing"
R_LINES_LOW = "ruled_lines_low"
R_INTERSECTION = "intersection_inconsistent"
R_TABLE_CUT = "table_cut"
R_LAYOUT_OTHER = "layout_other"
R_LAYOUT_CELL = "layout_uncertain"
R_FAINT = "faint_ink"
R_GRAMMAR = "grammar_violation"
R_NO_CAND = "no_valid_candidate"
R_OOV = "oov_account"
R_DITTO_FIRST = "ditto_first_row"
R_DITTO_GAP = "ditto_gap"
R_DITTO_UNCERTAIN = "ditto_source_uncertain"
R_ARROW = "arrow_candidate"
R_WAVE = "wave_candidate"
R_ANNOTATION = "annotation_present"
R_YEAR_UNKNOWN = "year_unknown"
R_MONTH_DEC = "month_decreased"
R_YEAR_JUMP = "year_jump"
R_ROW_INCONSISTENT = "row_inconsistent"
R_CP932 = "cp932_unencodable"
R_TOO_LONG = "too_long"
R_CLASS_UNCERTAIN = "class_uncertain"
R_LOW_CONF = "low_confidence"
R_PAGE_REVIEW = "page_needs_review"

_NOTE_CODES = [
    ("見出し帯を検出できなかった", R_BAND_NOT_FOUND),
    ("見出し帯が画像の端", R_BAND_CLIPPED),
    ("外枠", R_OUTER_FRAME),
    ("横罫線", R_LINES_LOW),
    ("縦罫線", R_LINES_LOW),
    ("交点", R_INTERSECTION),
    ("はみ出", R_TABLE_CUT),
]


def layout_reason_codes(notes: Sequence[str]) -> list[str]:
    out: list[str] = []
    for n in notes:
        code = next((c for k, c in _NOTE_CODES if k in n), R_LAYOUT_OTHER)
        if code not in out:
            out.append(code)
    return out


@dataclass
class CellInput:
    row: int
    col: int
    column: str
    rec: np.ndarray | None  # 認識用グレー(2 次元 uint8)。インクの無いセルは None
    clf: np.ndarray  # (64, 448) uint8
    stats: np.ndarray  # celltype.stats_vector
    flags: list[str] = field(default_factory=list)
    blank: bool = False
    faint_only: bool = False
    ink_h_frac: float = 0.0


@dataclass
class PageInput:
    cells: list[CellInput]  # 行優先(row * 4 + col)
    title_rec: np.ndarray | None = None
    number_rec: np.ndarray | None = None
    layout_status: str = "ok"
    layout_notes: list[str] = field(default_factory=list)

    def cell(self, row: int, col: int) -> CellInput:
        return self.cells[row * len(COLUMNS) + col]


@dataclass
class CellResult:
    row: int
    col: int
    column: str
    kind: str  # 判定した種別 blank / ditto / arrow / wave / text
    text: str = ""  # 認識した文字列(text)/ "〃" / "→" / "頭〜尾"
    value: str = ""  # 解決した値(日付 'R7.11.19'、金額 '3456'、文字列。blank は "")
    ok: bool = True
    candidates: list[tuple[str, float]] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    cls_probs: dict[str, float] = field(default_factory=dict)
    char_confidences: list[float] = field(default_factory=list)
    obj: object = None  # DateValue など
    annotation: str = ""
    annotation_candidates: list[str] = field(default_factory=list)
    confidence: float = 1.0
    review: bool = False
    source: tuple[int, int] | None = None  # 同上・矢印の参照元(行, 列)
    out: RecognizeOutput | None = field(default=None, repr=False)  # 見覚えメモリ用の埋め込み等


@dataclass
class PageResult:
    cells: list[CellResult]
    title: FieldResult | None = None
    number: str = ""
    layout_status: str = "ok"
    layout_reasons: list[str] = field(default_factory=list)
    last_year: int | None = None  # このページの最後の行の年(次のページが継承する)

    def cell(self, row: int, col: int) -> CellResult:
        return self.cells[row * len(COLUMNS) + col]


@dataclass
class JobContext:
    """同じ仕事(連続するページ)で引き継ぐ状態と、語彙。"""

    prev_year: int | None = None  # 前のページ(同じ No.)の最後の年
    accounts: list[str] = field(default_factory=list)  # 科目(支払方法)の閉じた語彙
    titles: list[str] = field(default_factory=list)  # タイトル(勘定科目名)の語彙
    dictionary: dict[str, int] = field(default_factory=dict)  # 摘要の頻度辞書(過去の確定値)
    account_prior: dict[str, float] | None = None


@dataclass
class RecognizerConfig:
    beam: int = 5
    beam_numeric: int = 8
    cls_min_conf: float = 0.7  # 種別の確率がこれ未満なら class_uncertain(確認)
    run_r1_min_p: float = 0.15  # text/wave の確率がこれ以上なら R1 を走らせる
    review_conf: float = 0.9  # 確信度がこれ未満なら確認(confidence.py で較正した値に置き換える)


Calibrator = Callable[[CellResult], float]


def default_confidence(r: CellResult) -> float:
    """較正前の暫定の確信度: 種別の確率 x 候補集合内の事後確率(confidence.py のロジスティック回帰に置き換える)。"""
    p = r.cls_probs.get(r.kind, 1.0)
    if r.kind == "text":
        p *= r.features.get("post_top", 0.0)
    return p


class PageRecognizer:
    def __init__(
        self,
        engine: ParseqEngine,
        classifier: CellTypeClassifier | None = None,
        config: RecognizerConfig | None = None,
        calibrator: Calibrator | None = None,
    ) -> None:
        self.engine = engine
        self.classifier = classifier
        self.cfg = config or RecognizerConfig()
        self.calibrator = calibrator or default_confidence

    # ------------------------------------------------------------------ 種別

    def classify(self, page: PageInput, oracle: Sequence[str] | None = None) -> np.ndarray:
        """(116, 5) の確率。oracle(正解の種別)があればそれを 1.0 にする(R1 以降の切り分け評価用)。"""
        n = len(page.cells)
        if oracle is not None:
            p = np.zeros((n, len(CLASSES)), np.float32)
            for i, k in enumerate(oracle):
                p[i, CLASSES.index({"elided": "wave"}.get(k, k))] = 1.0
            return p
        if self.classifier is not None:
            x = np.stack([c.clf for c in page.cells])
            s = np.stack([c.stats for c in page.cells])
            return self.classifier.predict_proba(x, s)
        # 分類器なし: Phase 1 の判定(空欄・〃の可能性・それ以外は text)
        p = np.full((n, len(CLASSES)), 0.02, np.float32)
        for i, c in enumerate(page.cells):
            k = "blank" if c.blank else "ditto" if "ditto_likely" in c.flags else "text"
            p[i, CLASSES.index(k)] = 0.9
        return p

    # ------------------------------------------------------------------ ページ

    def recognize_page(
        self,
        page: PageInput,
        ctx: JobContext | None = None,
        oracle_kinds: Sequence[str] | None = None,
    ) -> PageResult:
        ctx = ctx or JobContext()
        probs = self.classify(page, oracle_kinds)
        results: list[CellResult] = []
        for i, c in enumerate(page.cells):
            pr = {k: float(probs[i, j]) for j, k in enumerate(CLASSES)}
            kind = max(pr, key=lambda k: pr[k])
            results.append(CellResult(c.row, c.col, c.column, kind, cls_probs=pr))
        # R1 を走らせるセル(text / wave と判定、または迷っていて画像がある)
        run = [
            i
            for i, (c, r) in enumerate(zip(page.cells, results, strict=True))
            if c.rec is not None
            and (r.cls_probs["text"] >= self.cfg.run_r1_min_p or r.cls_probs["wave"] >= 0.5)
        ]
        outs = self._recognize(page, run)
        for i in run:
            self._resolve_cell(results[i], page.cells[i], outs[i], ctx)
        # タイトル・No.
        title = self._resolve_title(page, ctx)
        self._page_pass(page, results, ctx)
        layout_reasons = layout_reason_codes(page.layout_notes)
        for r, c in zip(results, page.cells, strict=True):
            self._finalize(r, c, page, results)
        last_year = None
        for r in reversed(results):
            if r.column == "date" and isinstance(r.obj, DateValue) and r.obj.era_year is not None:
                last_year = r.obj.era_year
                break
        return PageResult(results, title, "", page.layout_status, layout_reasons, last_year)

    # ------------------------------------------------------------------ R1

    def _recognize(self, page: PageInput, idx: list[int]) -> dict[int, RecognizeOutput]:
        if not idx:
            return {}
        crops, cols, ks = [], [], []
        for i in idx:
            c = page.cells[i]
            assert c.rec is not None
            crops.append(np.stack([c.rec] * 3, axis=-1))
            cols.append(c.column)
            ks.append(i)
        outs: dict[int, RecognizeOutput] = {}
        numeric = [j for j, col in enumerate(cols) if col in ("date", "amount")]
        other = [j for j, col in enumerate(cols) if col not in ("date", "amount")]
        for group, beam in ((numeric, self.cfg.beam_numeric), (other, self.cfg.beam)):
            if not group:
                continue
            res = self.engine.recognize_many(
                [crops[j] for j in group],
                [cols[j] for j in group],
                beam=beam,
                with_embedding=True,
                keep_memory=True,
            )
            for j, o in zip(group, res, strict=True):
                outs[ks[j]] = o
        return outs

    # ------------------------------------------------------------------ セルの解決

    def _resolve_cell(
        self, r: CellResult, c: CellInput, out: RecognizeOutput, ctx: JobContext
    ) -> None:
        r.out = out
        if r.kind == "wave":
            r.text = out.best.text
            r.features = {"r1_logp": max(out.best.logprob, -50.0)}
            return  # 候補は _page_pass で(同じページの記入・辞書から)作る
        if r.kind != "text":
            return
        fr = self._field(c.column, out, ctx)
        r.text = fr.text
        r.value = fr.value
        r.ok = fr.ok
        r.candidates = fr.candidates
        r.features = dict(fr.features)
        r.reasons = list(fr.reasons)
        r.obj = fr.obj
        r.char_confidences = fr.char_confidences

    def _field(self, column: str, out: RecognizeOutput, ctx: JobContext) -> FieldResult:
        if column == "date":
            return resolve_date(self.engine, out)
        if column == "amount":
            return resolve_amount(self.engine, out)
        if column == "account":
            return resolve_closed(self.engine, out, ctx.accounts, ctx.account_prior)
        return resolve_description(self.engine, out, ctx.dictionary)

    def _resolve_title(self, page: PageInput, ctx: JobContext) -> FieldResult | None:
        if page.title_rec is None or not ctx.titles:
            return None
        out = self.engine.recognize(
            np.stack([page.title_rec] * 3, axis=-1), "title", beam=3, keep_memory=True
        )
        return resolve_closed(self.engine, out, ctx.titles)

    # ------------------------------------------------------------------ ページ全体の解決(上から下へ)

    def _page_pass(self, page: PageInput, results: list[CellResult], ctx: JobContext) -> None:
        n_cols = len(COLUMNS)
        last: dict[int, CellResult] = {}  # 列 → 直近の非空欄セル
        last_row: dict[int, int] = {}
        cur_year = ctx.prev_year
        prev_date: DateValue | None = None
        for row in range(N_ROWS):
            for col in range(n_cols):
                r = results[row * n_cols + col]
                if r.kind == "blank":
                    continue
                if r.kind == "ditto":
                    src = last.get(col)
                    r.text = "〃"
                    if src is None:
                        r.value, r.ok = "", False
                        r.reasons.append(R_DITTO_FIRST)
                    else:
                        r.value, r.obj, r.ok = src.value, src.obj, src.ok
                        r.candidates = list(src.candidates)
                        r.source = (last_row[col], col)
                        if row - last_row[col] > 1:
                            r.reasons.append(R_DITTO_GAP)
                    if r.column == "date" and isinstance(r.obj, DateValue):
                        prev_date = r.obj
                    last[col], last_row[col] = r, row
                    continue
                if r.kind == "arrow":
                    r.text = "→"
                    self._resolve_arrow(page, results, r, row)
                    last[col], last_row[col] = r, row
                    continue
                if r.kind == "wave":
                    self._resolve_wave(results, r, row, ctx)
                    last[col], last_row[col] = r, row
                    continue
                # text
                if r.column == "date" and isinstance(r.obj, DateValue):
                    d: DateValue = r.obj
                    if d.era_year is not None:
                        if cur_year is not None and not 0 <= d.era_year - cur_year <= 1:
                            r.reasons.append(R_YEAR_JUMP)
                        cur_year = d.era_year
                    else:
                        if cur_year is None:
                            r.reasons.append(R_YEAR_UNKNOWN)
                        else:
                            if prev_date is not None and d.month < prev_date.month:
                                r.reasons.append(R_MONTH_DEC)
                            d = d.with_year(cur_year)
                            if not is_valid_date(d.era_year, d.month, d.day):
                                r.reasons.append(R_GRAMMAR)
                            r.obj = d
                            r.value = d.format()
                    prev_date = r.obj if isinstance(r.obj, DateValue) else prev_date
                if r.value or r.kind == "text":
                    last[col], last_row[col] = r, row

    def _resolve_arrow(
        self, page: PageInput, results: list[CellResult], r: CellResult, row: int
    ) -> None:
        """矢印の指す摘要の候補: 同じページの上の行の記入(科目の一致 > 金額の一致 > 近さ)。常に確認。"""
        n_cols = len(COLUMNS)
        acc = results[row * n_cols + 1]
        amt = results[row * n_cols + 3]
        cands: list[tuple[float, int, str]] = []
        seen: set[str] = set()
        for rr in range(row - 1, -1, -1):
            d = results[rr * n_cols + 2]
            if d.kind not in ("text", "ditto", "wave") or not d.value:
                continue
            key = norm_text(d.value)
            if key in seen:
                continue
            seen.add(key)
            a = results[rr * n_cols + 1]
            m = results[rr * n_cols + 3]
            score = 2.0 * float(bool(acc.value) and a.value == acc.value)
            score += 1.0 * float(bool(amt.value) and m.value == amt.value)
            cands.append((score - 0.01 * (row - rr), rr, d.value))
        cands.sort(key=lambda t: -t[0])
        r.candidates = [(v, 1.0 / (i + 1)) for i, (_, _, v) in enumerate(cands[:5])]
        r.value = cands[0][2] if cands else ""
        r.ok = bool(cands)
        r.reasons.append(R_ARROW)

    def _resolve_wave(
        self, results: list[CellResult], r: CellResult, row: int, ctx: JobContext
    ) -> None:
        """波線による省略「頭〜尾」の候補: 同じページ → 辞書(過去の確定値)の、頭で始まり尾で終わる長い摘要。常に確認。"""
        head, tail = split_wave(r.text)
        n_cols = len(COLUMNS)
        pool: list[str] = []
        for rr in range(row - 1, -1, -1):
            d = results[rr * n_cols + 2]
            if d.value and d.value not in pool and d.kind in ("text", "wave", "ditto"):
                pool.append(d.value)
        pool += [
            k for k in sorted(ctx.dictionary, key=lambda k: -ctx.dictionary[k]) if k not in pool
        ]
        hit = [v for v in pool if matches_head_tail(v, head, tail)]
        r.candidates = [(v, 1.0 / (i + 1)) for i, v in enumerate(hit[:5])]
        r.value = hit[0] if hit else r.text
        r.ok = bool(hit)
        r.reasons.append(R_WAVE)

    # ------------------------------------------------------------------ 仕上げ

    def _finalize(
        self,
        r: CellResult,
        c: CellInput,
        page: PageInput,
        results: list[CellResult],
    ) -> None:
        if "layout_uncertain" in c.flags:
            r.reasons.append(R_LAYOUT_CELL)
        if page.layout_status != "ok":
            r.reasons.append(R_PAGE_REVIEW)
        if c.faint_only and r.kind != "blank":
            r.reasons.append(R_FAINT)
        if r.cls_probs.get(r.kind, 1.0) < self.cfg.cls_min_conf:
            r.reasons.append(R_CLASS_UNCERTAIN)
        if r.kind == "text" and not r.ok and R_GRAMMAR not in r.reasons:
            r.reasons.append(R_NO_CAND)
        if r.column == "description" and r.value:
            try:
                n = len(r.value.encode("cp932"))
            except UnicodeEncodeError:
                r.reasons.append(R_CP932)
            else:
                if n > 64 or len(r.value) > 80:
                    r.reasons.append(R_TOO_LONG)
        if r.kind == "ditto" and r.source is not None:
            src = results[r.source[0] * len(COLUMNS) + r.source[1]]
            if src.review:  # 参照元が要確認なら、同上も確認に回す(参照元の修正で値が変わる)
                r.reasons.append(R_DITTO_UNCERTAIN)
        # 確信度の特徴量に、セルのレイアウト上の状況を加える
        r.features["ink_h_frac"] = float(c.stats[5])
        r.features["ink_w_frac"] = float(c.stats[6])
        r.features["faint"] = float(c.faint_only)
        r.features["overflow"] = float("overflow" in c.flags)
        if r.kind == "ditto" and r.source is not None:
            r.features["src_conf"] = results[r.source[0] * len(COLUMNS) + r.source[1]].confidence
        r.confidence = float(self.calibrator(r))
        thr_fn = getattr(self.calibrator, "threshold", None)
        thr = float(thr_fn(r)) if thr_fn else self.cfg.review_conf
        low = r.confidence < thr and (thr_fn is not None or r.kind != "blank")
        if low:
            r.reasons.append(R_LOW_CONF)
        r.review = low or bool(HARD_REASONS & set(r.reasons))


_WAVE_RE = re.compile(r"[〜~～ー\-—–]")


def split_wave(text: str) -> tuple[str, str]:
    """「頭〜尾」→ (頭, 尾)。波線の記号が見つからなければ、前半と後半に分ける。"""
    t = norm_text(text)
    m = list(_WAVE_RE.finditer(t))
    if m:
        k = m[0].start() if len(m) == 1 else m[len(m) // 2].start()
        return t[:k], t[k + 1 :]
    h = len(t) // 2
    return t[:h], t[h:]


def matches_head_tail(value: str, head: str, tail: str) -> bool:
    """value(正規化)が head で始まり tail で終わる(長さは head + tail より長い)。"""
    v = norm_text(value)
    return (
        len(v) >= len(head) + len(tail) + 1
        and v.startswith(head)
        and v.endswith(tail)
        and bool(head or tail)
    )


__all__ = [
    "CellInput",
    "CellResult",
    "JobContext",
    "PageInput",
    "PageRecognizer",
    "PageResult",
    "RecognizerConfig",
    "layout_reason_codes",
    "matches_head_tail",
    "split_wave",
]
