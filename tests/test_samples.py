"""実サンプル(YAYOI_PRIVATE_DATA/samples)があるときだけ走るレイアウトの構造チェック。

実サンプルは様式のレイアウト検証だけに使う。内容(書かれた文字)は読み取らず、検証もしない。
"""

from __future__ import annotations

import pytest

from yayoi_ocr.config import samples_dir
from yayoi_ocr.ingest import load_pages
from yayoi_ocr.layout.pipeline import analyze_image
from yayoi_ocr.models import CellFlag, PageStatus

pytestmark = pytest.mark.samples

_FILES = sorted(samples_dir().glob("*.jpg")) if samples_dir().is_dir() else []


@pytest.mark.skipif(not _FILES, reason="実サンプルなし")
@pytest.mark.parametrize("path", _FILES, ids=[p.stem for p in _FILES])
def test_real_samples_layout_ok(path):
    res = analyze_image(load_pages(path)[0].image)
    assert res.page.status == PageStatus.OK, res.page.notes
    assert len(res.page.rows) == 29 and all(len(r.cells) == 4 for r in res.page.rows)
    # 空欄だけの様式ではほとんどのセルが空欄、記入済みではほとんどが非空欄
    blanks = sum(CellFlag.BLANK in c.flags for r in res.page.rows for c in r.cells)
    if path.stem.startswith("blank"):
        assert blanks >= 85
    else:
        assert blanks <= 40
