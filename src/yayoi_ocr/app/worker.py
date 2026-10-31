"""1 ページの認識(アップロードされたファイルの何ページ目か → セルの認識結果 `PageSpec`)。

- `FakeBackend`: 画像は読まずに合成の認識結果を返す(CI の web ジョブ・E2E 用。torch もフォントも使わない)。
- `RealBackend`: layout(位置合わせ・セル切り出し)→ セル種別 → PARSeq(R1)→ 列ごとの解決 → 確信度、を別プロセスで走らせる。
  別プロセスにするのは、サーバ(画面の応答)を認識の計算で止めないため、重い依存(torch)の常駐をそちらに閉じ込めるため、
  メモリ不足などで落ちてもアプリ本体が生き残るため。`spawn` で起動する(Windows と同じ挙動)。

どちらも例外は呼び出し側(runner)が受けてページを「失敗」にする。ログには実データの値を出さない。
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .store import CellSpec, PageSpec
from .suggest import R_AMOUNT_CHOICE

MAX_CANDIDATES = 5


@dataclass(frozen=True)
class PagePayload:
    """認識に必要な入力(別プロセスへ渡すので、画像そのものではなく置き場所を渡す)。"""

    path: str  # アップロードされたファイルの絶対パス(アプリのデータ置き場の中)
    source_index: int
    page_key: str  # ページを識別する文字列(合成の種に使う。値は含まない)
    prev_year: int | None = None  # 前のページの最後の年(年を書かない先頭行が継承する)
    accounts: tuple[str, ...] = ()  # 科目(支払方法)の閉じた語彙。空なら自由認識
    titles: tuple[str, ...] = ()  # ページのタイトル(勘定科目名)の語彙。空ならタイトルは認識しない
    dictionary: dict[str, int] | None = None  # 摘要の頻度辞書(確定済みページから)
    amount_history: dict[str, dict[int, int]] | None = (
        None  # 金額の履歴(同じ摘要・科目の過去の確定金額)
    )
    corners: tuple[tuple[float, float], ...] | None = (
        None  # 利用者が指定した表の 4 隅(元画像 px)。無ければ自動検出
    )
    weights_dir: str | None = None  # 使用中の世代の重み(Phase 4)。None なら出荷時の重み
    confidence: str | None = None  # その世代の確信度モデル。None なら出荷時のもの
    confidence_warm: bool = (
        False  # 採用直後の数ページ: 確信度モデルの warm_threshold(通常より厳しい)を使う
    )
    suggest_weights_dir: str | None = (
        None  # 提案モードの世代の重み(あれば、親と両方で推論し、入力済みの値は提案モードのものにする)
    )
    suggest_confidence: str | None = None  # その確信度モデル
    pagedata_path: str | None = (
        None  # 認識の入力(追加学習・世代の評価に使う)を保存する先。None なら保存しない
    )


@dataclass
class PageOutcome:
    spec: PageSpec
    last_year: int | None = None
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)
    suggested: bool = False  # 提案モードで認識した(入力済みの値は提案モードの世代のもの)


class Backend(Protocol):
    def process(self, payload: PagePayload) -> PageOutcome: ...

    def close(self) -> None: ...


# ------------------------------------------------------------------ 合成(開発・E2E)


FAKE_VIEW = (700, 990)  # (幅, 高さ)


def _attach_fake_view(spec: PageSpec, corners) -> None:
    """合成の「元画像」(灰色の紙に表の枠を描いただけ)と、その罫線情報を付ける。隅の指定があればそこに枠が描かれる。"""
    import cv2
    import numpy as np

    w, h = FAKE_VIEW
    img = np.full((h, w, 3), 225, np.uint8)
    tl, tr, br, bl = (
        [np.array(p, dtype=float) for p in corners]
        if corners
        else [np.array(p, dtype=float) for p in ((60, 130), (640, 130), (640, 930), (60, 930))]
    )
    cv2.polylines(img, [np.array([tl, tr, br, bl], np.int32)], True, (60, 60, 60), 2)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    spec.view_jpeg, spec.view_scale = buf.tobytes(), 1.0

    def lerp(a, b, t):
        return a + (b - a) * t

    hl = [
        [
            list(map(float, lerp(lerp(tl, bl, k / 29), lerp(tr, br, k / 29), x / 8)))
            for x in range(9)
        ]
        for k in range(30)
    ]
    vl = [
        [list(map(float, lerp(lerp(tl, tr, k / 4), lerp(bl, br, k / 4), y / 8))) for y in range(9)]
        for k in range(5)
    ]
    spec.layout_info = {
        "width": w,
        "height": h,
        "corners": [list(map(float, p)) for p in (tl, tr, br, bl)],
        "hfrac": [k / 29 for k in range(30)],
        "vfrac": [0.0, 0.13, 0.33, 0.78, 1.0],
        "hlines": hl,
        "vlines": vl,
    }
    spec.approved = bool(corners)


class FakeBackend:
    """合成の認識結果を返す。`delay` 秒だけ待つ(進捗表示のテスト用)。`fail_next` 回だけ失敗する(失敗の表示・再試行のテスト用)。"""

    def __init__(
        self,
        delay: float = 0.0,
        fill_rows: int = 10,
        error_rate: float = 0.15,
        review_rate: float = 0.6,
    ) -> None:
        self.delay = delay
        self.fill_rows = fill_rows
        self.error_rate = error_rate
        self.review_rate = review_rate
        self.fail_next = 0
        self.layout_ok = (
            True  # False にすると、隅の指定が無いページはレイアウト要確認で返す(E2E 用)
        )

    def process(self, payload: PagePayload) -> PageOutcome:
        from .fake import fake_page

        t = time.time()
        if self.delay:
            time.sleep(self.delay)
        if self.fail_next > 0:
            self.fail_next -= 1
            raise RuntimeError("fake_failure")
        seed = int(hashlib.sha256(payload.page_key.encode()).hexdigest()[:8], 16)
        spec, _ = fake_page(
            seed,
            fill_rows=self.fill_rows,
            error_rate=self.error_rate,
            review_rate=self.review_rate,
            layout_status="ok" if payload.corners or self.layout_ok else "needs_review",
        )
        _attach_fake_view(spec, payload.corners)
        if payload.pagedata_path:  # 本物の入力の代わりの目印(MockTrainer は中身を読まない)
            Path(payload.pagedata_path).parent.mkdir(parents=True, exist_ok=True)
            Path(payload.pagedata_path).write_bytes(b"fake")
        suggested = False
        if payload.suggest_weights_dir:
            # 提案モードの模擬: 親と値が違うセルを要確認にする(5 セルに 1 つ。金額は明示の選択が要る)。親の判定は 10 セルに 1 つだけ「要確認でない」
            from .suggest import R_DISAGREE

            suggested = True
            for i, c in enumerate(spec.cells):
                c.parent_review = c.review
                if i % 5 == 0 and c.kind != "blank":
                    c.parent_review = i % 10 != 0
                    c.parent_value = c.resolved_value + "9"  # 親の値(提案の値と違う)
                    c.review = True
                    c.reasons.append(R_DISAGREE)
                    if c.column == "amount":
                        c.reasons.append(R_AMOUNT_CHOICE)
                        c.candidates = [(c.resolved_value, 0.9), (c.parent_value, 0.8)]
        return PageOutcome(spec, last_year=8, seconds=time.time() - t, suggested=suggested)

    def close(self) -> None:
        pass


# ------------------------------------------------------------------ 実装(別プロセス)

_RECOGNIZERS: dict[str, object] = {}  # 別プロセスの中だけで使う(重みを 1 回だけ読み込む)

ENV_THREADS = "YAYOI_TORCH_THREADS"


def worker_threads(cpu_count: int | None = None) -> int:
    """認識ワーカーの torch のスレッド数。論理コアの半分(2〜4)に抑える。

    全コア分のスレッドを張ると、画面(ブラウザ・サーバ)や他のアプリと取り合ってかえって遅くなる
    (4 コアで他の計算が動いているとき: 4 スレッド 34 秒/ページ、1〜2 スレッド 約 3〜4 秒/ページ。認識部分のみ)。
    環境変数 YAYOI_TORCH_THREADS で固定できる。
    """
    env = os.environ.get(ENV_THREADS)
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    n = cpu_count if cpu_count is not None else (os.cpu_count() or 2)
    return max(2, min(4, n // 2))


def _recognizer(
    variant: str,
    weights_dir: str | None = None,
    confidence: str | None = None,
    warm: bool = False,
):
    key = f"{variant}|{weights_dir or ''}|{confidence or ''}|{int(warm)}"
    if key not in _RECOGNIZERS:
        while (
            len(_RECOGNIZERS) >= 3
        ):  # 世代を切り替えても、読み込んだ重みを溜め込まない(古いものから捨てる)
            _RECOGNIZERS.pop(next(iter(_RECOGNIZERS)))
        import torch

        from ..config import models_dir
        from ..recognize import variants
        from ..recognize.celltype import CellTypeClassifier
        from ..recognize.confidence import ConfidenceModel
        from ..recognize.page import PageRecognizer

        torch.set_num_threads(worker_threads())
        if weights_dir:  # 利用者のデータから作った世代(重みの置き場が違う)
            from pathlib import Path

            from ..recognize.parseq.engine import ParseqEngine

            engine = ParseqEngine(models_dir=Path(weights_dir))
            conf_path = Path(confidence) if confidence else variants.confidence_path(variant)
        else:
            engine = variants.open_engine(variant)
            conf_path = variants.confidence_path(variant)
        clf_path = models_dir() / "celltype.pt"
        clf = CellTypeClassifier.load(clf_path) if clf_path.is_file() else None
        calibrator = ConfidenceModel.load(conf_path, warm=warm) if conf_path.is_file() else None
        _RECOGNIZERS[key] = PageRecognizer(engine, clf, calibrator=calibrator)
    return _RECOGNIZERS[key]


def _png(img) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("png_encode_failed")
    return buf.tobytes()


def _expansion(r, prev_row_with_date: int | None):
    """展開(〃・矢印・波線・年の継承)の記録。UI の「展開済み」バッジと、参照元を直したときの再展開に使う。"""
    from ..recognize.grammar import parse_date

    if r.source is not None and r.kind in ("ditto", "arrow", "wave"):
        return {"kind": r.kind, "source": [int(r.source[0]), int(r.source[1])]}
    if r.column == "date" and r.kind == "text" and r.value:
        d = parse_date(r.text)
        if d is not None and d.era_year is None:  # 年が書かれていないのに値には年がある = 継承した
            return {
                "kind": "year",
                "source": None if prev_row_with_date is None else [prev_row_with_date, 0],
            }
    return None


def _shown(r) -> str:
    """確認画面に入力済みで出す値(語彙が無い・語彙外の科目は、自由認識の文字列を初期値にする。要確認が付く)。"""
    if r.column == "account" and r.kind == "text" and not r.value:
        return r.text
    return r.value


def to_page_spec(result, layout, parent=None) -> PageSpec:
    """`PageResult`(認識)+ `LayoutResult`(切り出し画像)→ 保存用の `PageSpec`。
    `parent`: 提案モードのとき、親の世代の同じ順のセル(親の要確認の判定と、値が違うときの親の値を残す。親の見逃しの監視用)。"""
    cells: list[CellSpec] = []
    prev_date_row: int | None = None
    for i, r in enumerate(result.cells):
        crop = layout.crops[(r.row, r.col)]
        value = _shown(r)
        keep_shown = (
            R_AMOUNT_CHOICE in r.reasons
        )  # 金額の明示選択: 入力済みの値も候補に残す(選んで確定できるように)
        cands = [(v, float(p)) for v, p in r.candidates if keep_shown or v != value][
            :MAX_CANDIDATES
        ]
        parent_review: bool | None = None
        parent_value: str | None = None
        if parent is not None:
            pc = parent[i]
            parent_review = bool(pc.review)
            if pc.kind != r.kind or _shown(pc) != value:
                parent_value = _shown(pc)
        cells.append(
            CellSpec(
                row=r.row,
                col=r.col,
                column=r.column,
                kind=r.kind,
                recognized_text=r.text,
                resolved_value=value,
                expansion=_expansion(r, prev_date_row),
                reasons=list(dict.fromkeys(r.reasons)),
                candidates=cands,
                review=bool(r.review),
                crop=_png(crop.color),
                parent_review=parent_review,
                parent_value=parent_value,
            )
        )
        if r.column == "date" and r.kind != "blank" and r.value:
            prev_date_row = r.row
    return PageSpec(cells, result.layout_status, list(result.layout_reasons))


def process_page_real(payload: PagePayload, variant: str | None = None) -> PageOutcome:
    """別プロセスで呼ばれる(トップレベル関数 = pickle できる)。"""
    from ..ingest import load_page
    from ..layout.pipeline import analyze_image, overlay_info, view_jpeg
    from ..recognize import variants
    from ..recognize.inputs import from_layout
    from ..recognize.page import JobContext

    t = time.time()
    lp = load_page(payload.path, payload.source_index)
    layout = analyze_image(
        lp.image,
        source_page_index=lp.source_page_index,
        image_path=None,
        sha256=lp.sha256,
        corners=payload.corners,
    )
    rec = _recognizer(
        variant or variants.default_variant(),
        payload.weights_dir,
        payload.confidence,
        payload.confidence_warm,
    )

    def make_ctx() -> JobContext:
        return JobContext(
            prev_year=payload.prev_year,
            accounts=list(payload.accounts),
            titles=list(payload.titles),
            dictionary=dict(payload.dictionary or {}),
            amount_history=dict(payload.amount_history or {}),
        )

    page_input = from_layout(layout)
    if payload.pagedata_path:  # 追加学習と世代の評価のために、認識の入力を利用者の PC に残す
        from ..recognize.pagedata import save_page_input

        save_page_input(payload.pagedata_path, page_input, payload.prev_year)
    result = rec.recognize_page(page_input, make_ctx())
    suggested = False
    parent_cells = None
    last_year = result.last_year
    if payload.suggest_weights_dir:
        # 提案モード: 提案モードの世代でも推論し、入力済みの値はそちらにする。要確認 = 親 OR 提案 OR 値が違う(見逃しは親の部分集合)
        from . import suggest

        rec_s = _recognizer(
            variant or variants.default_variant(),
            payload.suggest_weights_dir,
            payload.suggest_confidence,
            False,
        )
        result_s = rec_s.recognize_page(page_input, make_ctx())
        # 年の継承の元(last_year)は親の結果を使う(次のページの日付が、親だけで認識した場合と変わらないように)。
        # 親のセルの判定と値は、親の見逃しの監視のためにセルに残す
        result, last_year, parent_cells = suggest.combine(result, result_s)
        suggested = True
    spec = to_page_spec(result, layout, parent=parent_cells)
    spec.view_jpeg, spec.view_scale = view_jpeg(layout)
    spec.layout_info = overlay_info(layout)
    spec.approved = payload.corners is not None
    if result.title is not None and result.title.value:
        spec.title_recognized = result.title.value
    return PageOutcome(spec, last_year=last_year, seconds=time.time() - t, suggested=suggested)


class RealBackend:
    """認識を別プロセス(1 つ)で走らせる。落ちたら次のページで作り直す。"""

    def __init__(self, variant: str | None = None) -> None:
        self.variant = variant
        self._pool: ProcessPoolExecutor | None = None

    def _get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn")
            )
        return self._pool

    def process(self, payload: PagePayload) -> PageOutcome:
        try:
            return self._get_pool().submit(process_page_real, payload, self.variant).result()
        except BrokenProcessPool:
            self._pool = None  # 次のページで作り直す
            raise

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None


def make_backend(name: str) -> Backend:
    if name == "fake":
        return FakeBackend()
    if name == "real":
        return RealBackend()
    raise ValueError(f"未知のバックエンド: {name}")


def upload_path(uploads_dir: Path, rel: str) -> Path:
    """DB の相対パスを、uploads ディレクトリの中に限った絶対パスにする(外へ出る値は拒否)。"""
    p = (uploads_dir / rel).resolve()
    if uploads_dir.resolve() not in p.parents:
        raise ValueError("upload_path_outside")
    return p
