"""ビルドしたフロントエンドに、外部の URL を読み込む記述が無いこと(CDN・Web フォント・外部スクリプトを使わない)。

`web/` を `npm run build` した後(`src/yayoi_ocr/app/static/` がある)だけ実行する。CI のフロントエンドのジョブで必ず通す。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "yayoi_ocr" / "app" / "static"
pytestmark = pytest.mark.skipif(
    not (STATIC / "index.html").is_file(), reason="フロントエンドが未ビルド(web/ で npm run build)"
)

URL = re.compile(r"https?://[A-Za-z0-9._~:/?#@!$&'()*+,;=%-]+")
# 名前空間の識別子と、React のエラー文言(説明の文字列。読み込みはしない)だけは URL の形で残る
ALLOWED_PREFIXES = (
    "http://www.w3.org/",
    "https://reactjs.org/docs/error-decoder.html",
    "https://react.dev/",
)


def test_html_and_css_reference_no_external_resources():
    for f in [STATIC / "index.html", *STATIC.rglob("*.css")]:
        text = f.read_text(encoding="utf-8")
        assert not URL.findall(text), f"{f.name}: 外部 URL がある"
        assert "@import" not in text and "//cdn" not in text


def test_javascript_contains_only_namespace_urls():
    found = set()
    for f in STATIC.rglob("*.js"):
        found |= set(URL.findall(f.read_text(encoding="utf-8")))
    bad = [u for u in found if not u.startswith(ALLOWED_PREFIXES)]
    assert not bad, f"外部 URL: {bad}"


def test_no_font_or_binary_assets_are_bundled_from_elsewhere():
    exts = {p.suffix.lower() for p in STATIC.rglob("*") if p.is_file()}
    assert exts <= {".html", ".js", ".css", ".svg", ".json", ".map"}, (
        exts
    )  # 画像・フォントは M0 では無し
