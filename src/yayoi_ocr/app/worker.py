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
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .store import CellSpec, PageSpec

MAX_CANDIDATES = 5


@dataclass(frozen=True)
class PagePayload:
    """認識に必要な入力(別プロセスへ渡すので、画像そのものではなく置き場所を渡す)。"""

    path: str  # アップロードされたファイルの絶対パス(アプリのデータ置き場の中)
    source_index: int
    page_key: str  # ページを識別する文字列(合成の種に使う。値は含まない)
    prev_year: int | None = None  # 前のページの最後の年(年を書かない先頭行が継承する)
    accounts: tuple[str, ...] = ()  # 科目(支払方法)の閉じた語彙。空なら自由認識


@dataclass
class PageOutcome:
    spec: PageSpec
    last_year: int | None = None
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)


class Backend(Protocol):
    def process(self, payload: PagePayload) -> PageOutcome: ...

    def close(self) -> None: ...


# ------------------------------------------------------------------ 合成(開発・E2E)


class FakeBackend:
    """合成の認識結果を返す。`delay` 秒だけ待つ(進捗表示のテスト用)。`fail_next` 回だけ失敗する(失敗の表示・再試行のテスト用)。"""

    def __init__(self, delay: float = 0.0, fill_rows: int = 10) -> None:
        self.delay = delay
        self.fill_rows = fill_rows
        self.fail_next = 0

    def process(self, payload: PagePayload) -> PageOutcome:
        from .fake import fake_page

        t = time.time()
        if self.delay:
            time.sleep(self.delay)
        if self.fail_next > 0:
            self.fail_next -= 1
            raise RuntimeError("fake_failure")
        seed = int(hashlib.sha256(payload.page_key.encode()).hexdigest()[:8], 16)
        spec, _ = fake_page(seed, fill_rows=self.fill_rows, error_rate=0.15, review_rate=0.6)
        return PageOutcome(spec, last_year=8, seconds=time.time() - t)

    def close(self) -> None:
        pass


# ------------------------------------------------------------------ 実装(別プロセス)

_RECOGNIZERS: dict[str, object] = {}  # 別プロセスの中だけで使う(重みを 1 回だけ読み込む)


def _recognizer(variant: str):
    if variant not in _RECOGNIZERS:
        from ..config import models_dir
        from ..recognize import variants
        from ..recognize.celltype import CellTypeClassifier
        from ..recognize.confidence import ConfidenceModel
        from ..recognize.page import PageRecognizer

        engine = variants.open_engine(variant)
        clf_path = models_dir() / "celltype.pt"
        clf = CellTypeClassifier.load(clf_path) if clf_path.is_file() else None
        conf_path = variants.confidence_path(variant)
        calibrator = ConfidenceModel.load(conf_path) if conf_path.is_file() else None
        _RECOGNIZERS[variant] = PageRecognizer(engine, clf, calibrator=calibrator)
    return _RECOGNIZERS[variant]


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


def to_page_spec(result, layout) -> PageSpec:
    """`PageResult`(認識)+ `LayoutResult`(切り出し画像)→ 保存用の `PageSpec`。"""
    cells: list[CellSpec] = []
    prev_date_row: int | None = None
    for r in result.cells:
        crop = layout.crops[(r.row, r.col)]
        value = r.value
        if r.column == "account" and r.kind == "text" and not value:
            value = (
                r.text
            )  # 語彙が無い(または語彙外)のとき: 自由認識の文字列を初期値にする(要確認が付く)
        cands = [(v, float(p)) for v, p in r.candidates if v != value][:MAX_CANDIDATES]
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
            )
        )
        if r.column == "date" and r.kind != "blank" and r.value:
            prev_date_row = r.row
    return PageSpec(cells, result.layout_status, list(result.layout_reasons))


def process_page_real(payload: PagePayload, variant: str | None = None) -> PageOutcome:
    """別プロセスで呼ばれる(トップレベル関数 = pickle できる)。"""
    from ..ingest import load_page
    from ..layout.pipeline import analyze_image
    from ..recognize import variants
    from ..recognize.inputs import from_layout
    from ..recognize.page import JobContext

    t = time.time()
    lp = load_page(payload.path, payload.source_index)
    layout = analyze_image(
        lp.image, source_page_index=lp.source_page_index, image_path=None, sha256=lp.sha256
    )
    rec = _recognizer(variant or variants.default_variant())
    ctx = JobContext(prev_year=payload.prev_year, accounts=list(payload.accounts))
    result = rec.recognize_page(from_layout(layout), ctx)
    spec = to_page_spec(result, layout)
    return PageOutcome(spec, last_year=result.last_year, seconds=time.time() - t)


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
