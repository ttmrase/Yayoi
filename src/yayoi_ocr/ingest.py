"""取り込み: PDF / PNG / JPEG / TIFF(複数ページ)/ EXIF 回転 → BGR 画像のリスト。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageSequence

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


def _load_raster(path: Path) -> list[LoadedPage]:
    pages: list[LoadedPage] = []
    sha = file_sha256(path)
    with Image.open(path) as im:
        n = getattr(im, "n_frames", 1)
        for idx, frame in enumerate(ImageSequence.Iterator(im)):
            _check_size(*frame.size)
            frame = frame.copy()
            frame.info.update(im.info)
            oriented = ImageOps.exif_transpose(frame) or frame  # EXIF の回転情報を反映
            pages.append(
                LoadedPage(
                    image=_pil_to_bgr(oriented),
                    source_path=str(path),
                    source_page_index=idx,
                    dpi_hint=_dpi_of(frame),
                    sha256=sha,
                )
            )
            if idx + 1 >= n:
                break
    return pages


def _load_pdf(path: Path) -> list[LoadedPage]:
    import pypdfium2 as pdfium  # 重いので遅延 import

    sha = file_sha256(path)
    pages: list[LoadedPage] = []
    pdf = pdfium.PdfDocument(str(path))
    try:
        for idx in range(len(pdf)):
            page = pdf[idx]
            w_pt, h_pt = page.get_size()
            scale = PDF_RENDER_DPI / 72.0
            _check_size(int(w_pt * scale), int(h_pt * scale))
            bitmap = page.render(scale=scale, rotation=0)
            pages.append(
                LoadedPage(
                    image=_pil_to_bgr(bitmap.to_pil()),
                    source_path=str(path),
                    source_page_index=idx,
                    dpi_hint=float(PDF_RENDER_DPI),
                    sha256=sha,
                )
            )
    finally:
        pdf.close()
    return pages


def load_pages(path: str | Path) -> list[LoadedPage]:
    """ファイルを 1 ページずつの BGR 画像に分解して返す。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(p)
    if suffix in SUPPORTED_SUFFIXES:
        return _load_raster(p)
    raise ValueError(f"未対応の形式です: {p.suffix}")
