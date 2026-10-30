"""合成帳票用のフォント探索。フォントファイルはリポジトリに含めない。

- 手書き風フォント: `scripts/fetch_fonts.py` が `$YAYOI_PRIVATE_DATA/fonts`(または YAYOI_FONT_DIR)に
  取得した SIL OFL 1.1 のフォント(docs/fonts.md)。
- 印刷体(見出し帯の文字・手書きフォントが無いときの代替): システムの日本語フォント。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import fonts_dir

ENV_FONT_DIR = "YAYOI_FONT_DIR"

_SYSTEM_DIRS = [
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    "~/.fonts",
    "~/.local/share/fonts",
    "/Library/Fonts",
    "/System/Library/Fonts",
    "C:/Windows/Fonts",
]
_SYSTEM_NAME_HINTS = (
    "ipag", "ipamp", "ipam", "notosanscjk", "notoserifcjk", "sourcehan",
    "yugoth", "meiryo", "msgothic",
)  # fmt: skip
_SUFFIXES = {".ttf", ".otf", ".ttc"}


def render_glyph(font: ImageFont.FreeTypeFont, ch: str) -> np.ndarray | None:
    """1 文字を 0..1 の float32 配列で返す(空白など描画なしなら None)。"""
    left, top, right, bottom = font.getbbox(ch)
    w, h = right - left, bottom - top
    if w <= 0 or h <= 0:
        return None
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).text((-left, -top), ch, fill=255, font=font)
    return np.asarray(im, dtype=np.float32) / 255.0


def render_glyph_offset(
    font: ImageFont.FreeTypeFont, ch: str
) -> tuple[np.ndarray, int, int] | None:
    """字形と、描画原点(左上、アセンダ線)からの字形左上のオフセット (left, top)。
    ピリオドなどの小さい字をベースラインに揃えるために、オフセットを保持する。"""
    left, top, right, bottom = font.getbbox(ch)
    w, h = right - left, bottom - top
    if w <= 0 or h <= 0:
        return None
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).text((-left, -top), ch, fill=255, font=font)
    return np.asarray(im, dtype=np.float32) / 255.0, left, top


def can_render_japanese(path: Path) -> bool:
    try:
        font = ImageFont.truetype(str(path), 32)
        # 日本語の字形が実際に描ければ OK(未対応フォントは同じ「豆腐」になる)
        a, b = render_glyph(font, "日"), render_glyph(font, "鬱")
        return a is not None and b is not None and (a.shape != b.shape or not np.array_equal(a, b))
    except OSError:
        return False


def _scan(d: Path, hints: tuple[str, ...] | None) -> list[Path]:
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.rglob("*")):
        if p.suffix.lower() not in _SUFFIXES:
            continue
        if hints is not None and not any(h in p.name.lower() for h in hints):
            continue
        if can_render_japanese(p):
            out.append(p)
    return out


@lru_cache(maxsize=1)
def find_handwriting_fonts() -> list[Path]:
    """取得済みの手書き風フォント(OFL)。YAYOI_FONT_DIR 優先、なければ private の fonts。"""
    dirs: list[Path] = []
    env = os.environ.get(ENV_FONT_DIR)
    if env:
        dirs.append(Path(env).expanduser())
    dirs.append(fonts_dir())
    found: list[Path] = []
    for d in dirs:
        for p in _scan(d, None):
            if p not in found:
                found.append(p)
    return found


@lru_cache(maxsize=1)
def find_system_fonts() -> list[Path]:
    found: list[Path] = []
    for d in _SYSTEM_DIRS:
        for p in _scan(Path(d).expanduser(), _SYSTEM_NAME_HINTS):
            if p not in found:
                found.append(p)
    return found


def find_japanese_fonts() -> list[Path]:
    """日本語を描けるフォント(手書き風が先、システムが後)。見つからなければ空。"""
    return [*find_handwriting_fonts(), *find_system_fonts()]


def print_font() -> Path | None:
    """見出し帯の白抜き文字などに使う印刷体(丸ゴシック > システムのゴシック)。"""
    hw = find_handwriting_fonts()
    for p in hw:
        if "zenmarugothic-medium" in p.name.lower():
            return p
    sysf = find_system_fonts()
    if sysf:
        return sysf[0]
    return hw[0] if hw else None


def load_font(path: Path, size: float) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), max(8, int(size)))
