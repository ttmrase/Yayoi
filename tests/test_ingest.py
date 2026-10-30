from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from yayoi_ocr.ingest import load_pages


def _img(w=64, h=48, seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


@pytest.mark.parametrize("ext", ["png", "jpg", "tif", "bmp"])
def test_single_raster(tmp_path, ext):
    p = tmp_path / f"a.{ext}"
    _img().save(p)
    pages = load_pages(p)
    assert len(pages) == 1
    assert pages[0].image.shape == (48, 64, 3) and pages[0].image.dtype == np.uint8
    assert pages[0].sha256


def test_exif_orientation_is_applied(tmp_path):
    im = _img(64, 48)
    exif = im.getexif()
    exif[0x0112] = 6  # 90° 回転して表示する指定
    p = tmp_path / "rot.jpg"
    im.save(p, exif=exif)
    pages = load_pages(p)
    assert pages[0].image.shape[:2] == (64, 48)


def test_multipage_tiff(tmp_path):
    p = tmp_path / "m.tif"
    a, b = _img(seed=1), _img(40, 30, seed=2)
    a.save(p, save_all=True, append_images=[b])
    pages = load_pages(p)
    assert [pg.image.shape[:2] for pg in pages] == [(48, 64), (30, 40)]
    assert [pg.source_page_index for pg in pages] == [0, 1]


def test_pdf_pages(tmp_path):
    p = tmp_path / "d.pdf"
    a, b = _img(200, 300, 1), _img(200, 300, 2)
    a.save(p, save_all=True, append_images=[b], resolution=72)
    pages = load_pages(p)
    assert len(pages) == 2
    assert pages[0].dpi_hint == 300.0
    h, w = pages[0].image.shape[:2]
    assert abs(w - 200 * 300 / 72) < 3 and abs(h - 300 * 300 / 72) < 3


def test_rgba_and_gray(tmp_path):
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(tmp_path / "t.png")
    assert load_pages(tmp_path / "t.png")[0].image.min() == 255  # 透過は白
    Image.new("L", (10, 10), 77).save(tmp_path / "g.png")
    assert load_pages(tmp_path / "g.png")[0].image.shape == (10, 10, 3)


def test_unsupported(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("x")
    with pytest.raises(ValueError):
        load_pages(p)
    with pytest.raises(FileNotFoundError):
        load_pages(tmp_path / "none.png")
