"""取り込み: PDF / PNG / JPEG / TIFF(複数ページ)/ EXIF 回転 → BGR 画像のリスト。"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_RENDER_DPI = 300
MAX_PIXELS = 120_000_000  # 展開爆弾・誤入力の防止


@dataclass
class LoadedPage:
    image: np.ndarray  # BGR uint8
    source_path: str
    source_page_index: int = 0
    dpi_hint: float | None = None  # 取れた場合のみ(解像度非依存に作るので参考値)
    sha256: str | None = None  # 入力ファイル全体のハッシュ


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pil_to_bgr(pil: Image.Image) -> np.ndarray:
    if pil.mode in ("I;16", "I;16L", "I;16B", "I"):
        arr = np.asarray(pil).astype(np.float32)
        hi = arr.max() or 1.0
        pil = Image.fromarray((arr / hi * 255.0).astype(np.uint8))
    elif pil.mode in ("RGBA", "LA", "P", "PA"):
        rgba = pil.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))  # 透過は白紙扱い
        pil = Image.alpha_composite(bg, rgba)
    if pil.mode not in ("RGB", "L"):
        pil = pil.convert("RGB")
    arr = np.asarray(pil)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _check_size(w: int, h: int) -> None:
    if w * h > MAX_PIXELS:
        raise ValueError(f"画像が大きすぎます: {w}x{h}")


def _dpi_of(pil: Image.Image) -> float | None:
    dpi = pil.info.get("dpi")
    if isinstance(dpi, tuple) and dpi and float(dpi[0]) > 1:
        return float(dpi[0])
    return None


def _iter_raster(path: Path, only: int | None = None) -> Iterator[LoadedPage]:
    sha = file_sha256(path)
    with Image.open(path) as im:
        n = getattr(im, "n_frames", 1)
        if only is not None and not 0 <= only < n:
            raise IndexError(f"ページ {only} はありません(全 {n} ページ)")
        for idx in range(n) if only is None else [only]:
            im.seek(idx)
            _check_size(*im.size)
            frame = im.copy()
            frame.info.update(im.info)
            oriented = ImageOps.exif_transpose(frame) or frame  # EXIF の回転情報を反映
            yield LoadedPage(
                image=_pil_to_bgr(oriented),
                source_path=str(path),
                source_page_index=idx,
                dpi_hint=_dpi_of(frame),
                sha256=sha,
            )


def _iter_pdf(path: Path, only: int | None = None) -> Iterator[LoadedPage]:
    import pypdfium2 as pdfium  # 重いので遅延 import

    sha = file_sha256(path)
    pdf = pdfium.PdfDocument(str(path))
    try:
        if only is not None and not 0 <= only < len(pdf):
            raise IndexError(f"ページ {only} はありません(全 {len(pdf)} ページ)")
        for idx in range(len(pdf)) if only is None else [only]:
            page = pdf[idx]
            w_pt, h_pt = page.get_size()
            scale = PDF_RENDER_DPI / 72.0
            _check_size(int(w_pt * scale), int(h_pt * scale))
            bitmap = page.render(scale=scale, rotation=0)
            yield LoadedPage(
                image=_pil_to_bgr(bitmap.to_pil()),
                source_path=str(path),
                source_page_index=idx,
                dpi_hint=float(PDF_RENDER_DPI),
                sha256=sha,
            )
    finally:
        pdf.close()


def page_count(path: str | Path) -> int:
    """ページ数だけを数える(画像の展開はしない)。壊れたファイルは例外。"""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(p))
        try:
            return len(pdf)
        finally:
            pdf.close()
    if suffix in SUPPORTED_SUFFIXES:
        with Image.open(p) as im:
            _check_size(*im.size)  # ヘッダーの寸法だけ見る(全画素の展開はしない)
            return int(getattr(im, "n_frames", 1))
    raise ValueError(f"未対応の形式です: {p.suffix}")


def iter_pages(path: str | Path) -> Iterator[LoadedPage]:
    """ファイルを 1 ページずつ、必要になったときに BGR 画像へ展開する(複数ページの PDF でメモリを使い切らない)。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return _iter_pdf(p)
    if suffix in SUPPORTED_SUFFIXES:
        return _iter_raster(p)
    raise ValueError(f"未対応の形式です: {p.suffix}")


def load_page(path: str | Path, index: int) -> LoadedPage:
    """ファイルの `index` ページ目(0 始まり)だけを展開する(認識待ちのページを 1 枚ずつ処理するときに使う)。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return next(_iter_pdf(p, index))
    if suffix in SUPPORTED_SUFFIXES:
        return next(_iter_raster(p, index))
    raise ValueError(f"未対応の形式です: {p.suffix}")


def load_pages(path: str | Path) -> list[LoadedPage]:
    """ファイルを 1 ページずつの BGR 画像に分解して返す。"""
    return list(iter_pages(path))
