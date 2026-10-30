from __future__ import annotations

import pytest

from yayoi_ocr.synth.fonts import find_japanese_fonts


def pytest_collection_modifyitems(config, items):
    if find_japanese_fonts():
        return
    skip = pytest.mark.skip(reason="日本語フォントが無いため合成帳票を生成できない")
    for item in items:
        if "needs_font" in item.keywords:
            item.add_marker(skip)
