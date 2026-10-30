"""「書き手」のシミュレーション。

1 人の書き手 = 手書き風フォント 1 つ + 書き手ごとに固定した変形パラメータの組。
同じ書き手が書く字には一貫した癖が出る(傾き・太さ・大きさ・字間・ベースラインのゆらぎ・同上記号の形など)。
文字ごとのゆらぎ(弾性変形・回転・拡大率)は、書き手のパラメータで決まる大きさで毎回ランダムに出る。

後で「使うほど精度が上がるか」(学習ループ)を検証するために使う。実データは一切使わない。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageFont

from .fonts import find_handwriting_fonts, find_japanese_fonts, load_font, render_glyph_offset


@dataclass(frozen=True)
class Writer:
    writer_id: int
    font_path: str
    size_scale: float  # 文字の大きさ(行の高さに対する比)
    slant: float  # 傾き(せん断。正で右へ傾く)
    thickness: float  # 線の太さ(1.0 が標準)
    elastic: float  # 文字ごとの弾性変形の強さ(文字サイズ比)
    rot_sigma: float  # 文字ごとの回転ゆらぎ(度)
    scale_sigma: float  # 文字ごとの拡大率ゆらぎ
    spacing: float  # 字間(平均。文字幅に対する比)
    spacing_jitter: float  # 字間のゆらぎ
    baseline_jitter: float  # ベースラインの文字ごとのゆらぎ(文字サイズ比)
    baseline_drift: float  # ベースラインのゆるやかなドリフト
    line_tilt_deg: float  # 行全体の傾き(度)
    ink_level: float  # 筆圧(インクの濃さ。0=黒, 255=白)
    ink_jitter: float
    ditto_angle: float  # 同上記号(〃)の画線の傾き(度)
    ditto_len: float  # 同上: 画線の長さ(文字サイズ比)
    ditto_gap: float  # 同上: 2 本の間隔
    ditto_thick: float  # 同上: 太さ
    align_date: float  # 日付を右寄せにする傾向(0..1)

    def describe(self) -> dict:
        return asdict(self)


def _usable_fonts() -> list[Path]:
    """書き手が使うフォント。手書き風(OFL)があればそれだけ。印刷体は取得できないときの代替。"""
    fonts = find_handwriting_fonts() or find_japanese_fonts()
    if not fonts:
        raise RuntimeError(
            "日本語フォントが見つかりません。`python scripts/fetch_fonts.py` で取得するか、"
            "YAYOI_FONT_DIR を設定してください。"
        )
    return fonts


def make_writer(writer_id: int, fonts: list[Path] | None = None) -> Writer:
    """書き手 ID から決定的に書き手を作る(同じ ID なら常に同じ癖)。"""
    fonts = fonts or _usable_fonts()
    rng = np.random.default_rng(1_000_003 * int(writer_id) + 17)
    font = fonts[int(rng.integers(len(fonts)))]
    u = rng.uniform
    return Writer(
        writer_id=int(writer_id),
        font_path=str(font),
        size_scale=float(u(0.50, 0.78)),
        slant=float(u(-0.08, 0.32)),
        thickness=float(u(0.65, 1.5)),
        elastic=float(u(0.012, 0.05)),
        rot_sigma=float(u(1.0, 6.0)),
        scale_sigma=float(u(0.03, 0.10)),
        spacing=float(u(-0.07, 0.12)),
        spacing_jitter=float(u(0.02, 0.09)),
        baseline_jitter=float(u(0.01, 0.06)),
        baseline_drift=float(u(0.004, 0.03)),
        line_tilt_deg=float(u(-2.0, 3.0)),
        ink_level=float(u(35, 150)),
        ink_jitter=float(u(3, 14)),
        ditto_angle=float(u(10, 35)),
        ditto_len=float(u(0.28, 0.55)),
        ditto_gap=float(u(0.14, 0.34)),
        ditto_thick=float(u(0.7, 1.6)),
        align_date=float(u(0.2, 0.95)),
    )


# ---------------------------------------------------------------- 文字の描画


@lru_cache(maxsize=64)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return load_font(Path(path), size)


@lru_cache(maxsize=8192)
def _base_glyph(path: str, size: int, ch: str) -> tuple[np.ndarray, int, int] | None:
    return render_glyph_offset(_font(path, size), ch)


def _glyph_variant(
    base: np.ndarray, w: Writer, size: float, rng: np.random.Generator
) -> np.ndarray:
    """基本字形に、書き手の癖と文字ごとのゆらぎ(弾性変形・回転・拡大・太さ)を与える(0..1)。"""
    h0, w0 = base.shape
    pad = max(4, int(0.3 * size))
    canvas = np.zeros((h0 + 2 * pad, w0 + 2 * pad), dtype=np.float32)
    canvas[pad : pad + h0, pad : pad + w0] = base
    h, wd = canvas.shape
    # 弾性変形(低解像度のランダム変位場を拡大)
    amp = w.elastic * size
    if amp > 0.2:
        dx = cv2.resize(
            rng.normal(0, 1, (4, 4)).astype(np.float32), (wd, h), interpolation=cv2.INTER_CUBIC
        )
        dy = cv2.resize(
            rng.normal(0, 1, (4, 4)).astype(np.float32), (wd, h), interpolation=cv2.INTER_CUBIC
        )
        xs, ys = np.meshgrid(np.arange(wd, dtype=np.float32), np.arange(h, dtype=np.float32))
        canvas = cv2.remap(canvas, xs + dx * amp, ys + dy * amp, cv2.INTER_LINEAR)
    # 傾き(せん断)・回転・拡大(書き手の癖 + 文字ごとのゆらぎ)
    ang = float(rng.normal(0.0, w.rot_sigma))
    sc = float(np.exp(rng.normal(0.0, w.scale_sigma)))
    sx = sc * float(np.exp(rng.normal(0.0, 0.5 * w.scale_sigma)))
    c = np.array([wd / 2.0, h / 2.0])
    m = cv2.getRotationMatrix2D((c[0], c[1]), ang, 1.0)
    lin = np.array([[sx, 0.0], [0.0, sc]]) @ np.array([[1.0, -w.slant], [0.0, 1.0]])
    a = m[:, :2] @ lin
    t = c - a @ c
    canvas = cv2.warpAffine(canvas, np.hstack([a, t[:, None]]), (wd, h), flags=cv2.INTER_LINEAR)
    # 線の太さ(ぼかしてからしきい値)
    thick = w.thickness * float(np.exp(rng.normal(0.0, 0.08)))
    cut = float(np.clip(0.5 - 0.22 * (thick - 1.0), 0.25, 0.62))
    g = cv2.GaussianBlur(canvas, (0, 0), max(0.5, 0.018 * size))
    return np.clip((g - cut) / 0.3, 0.0, 1.0)


def stamp(img: np.ndarray, mask: np.ndarray, x: int, y: int, ink: float) -> None:
    """マスクを (x, y)(左上)に「暗くする」方向で合成(罫線の上でも筆跡が残る)。"""
    h, w = mask.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    m = mask[y0 - y : y1 - y, x0 - x : x1 - x]
    reg = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = reg * (1 - m) + np.minimum(reg, ink) * m


def _pen_ink(w: Writer, rng: np.random.Generator, ink: float | None = None) -> float:
    base = w.ink_level if ink is None else ink
    return float(np.clip(base + rng.normal(0, w.ink_jitter), 15, 200))


def text_width(
    text: str, w: Writer, size: float, rng: np.random.Generator
) -> tuple[list[float], float]:
    """各文字の送り幅(ゆらぎ込み)と合計。"""
    font = _font(w.font_path, int(size))
    advs = []
    for ch in text:
        if ch == " ":
            advs.append(0.32 * size)
        else:
            adv = font.getlength(ch) * (1.0 + w.spacing + float(rng.normal(0.0, w.spacing_jitter)))
            advs.append(max(0.2 * size, adv))
    return advs, float(sum(advs))


def draw_text(
    img: np.ndarray,
    text: str,
    box: tuple[float, float, float, float],
    rng: np.random.Generator,
    w: Writer,
    *,
    size: float,
    align: str = "left",
    jitter_y: float = 0.0,
    ink: float | None = None,
) -> tuple[float, float]:
    """`box`(x0, y0, x1, y1 px)の中に書き手の癖で文字列を書く。はみ出しは許す。
    戻り値: 書いた範囲の (x 開始, x 終了)。"""
    x0, y0, x1, y1 = box
    advs, total = text_width(text, w, size, rng)
    if align == "right":
        x = x1 - total
    elif align == "center":
        x = 0.5 * (x0 + x1) - total / 2
    else:
        x = x0
    start = x
    base = 0.5 * (y0 + y1) - 0.5 * size + jitter_y
    drift = 0.0
    tilt = np.tan(np.radians(w.line_tilt_deg))
    pen = _pen_ink(w, rng, ink)
    for ch, adv in zip(text, advs, strict=True):
        if ch != " ":
            bg = _base_glyph(w.font_path, int(size), ch)
            if bg is not None:
                g, left, top = bg
                drift += float(rng.normal(0.0, w.baseline_drift * size))
                var = _glyph_variant(g, w, size, rng)
                pad = max(4, int(0.3 * size))
                gx = round(x + left - pad)
                gy = round(
                    base
                    + top
                    + drift
                    + float(rng.normal(0, w.baseline_jitter * size))
                    + tilt * (x - start)
                    - pad
                )
                stamp(
                    img,
                    var,
                    gx,
                    gy,
                    float(np.clip(pen + rng.normal(0, 0.5 * w.ink_jitter), 15, 200)),
                )
        x += adv
    return start, x


# ---------------------------------------------------------------- 記号・線(書き手の癖つき)


def _stamp_polyline(img: np.ndarray, pts: np.ndarray, thick: float, ink: float) -> None:
    """折れ線をアンチエイリアスで描いて合成する。"""
    pad = int(thick) + 3
    lo = np.floor(pts.min(axis=0)).astype(int) - pad
    hi = np.ceil(pts.max(axis=0)).astype(int) + pad
    mask = np.zeros((hi[1] - lo[1], hi[0] - lo[0]), np.float32)
    q = np.round((pts - lo) * 16).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [q], False, 1.0, max(1, round(thick)), cv2.LINE_AA, shift=4)
    stamp(img, mask, int(lo[0]), int(lo[1]), ink)


def draw_ditto(
    img: np.ndarray, cx: float, cy: float, size: float, rng: np.random.Generator, w: Writer,
    ink: float | None = None,
) -> None:  # fmt: skip
    """〃(2 本の短い斜線)。書き手ごとに角度・長さ・間隔が一貫する。"""
    thick = max(1.2, 0.07 * size * w.ditto_thick)
    length = w.ditto_len * size * float(rng.uniform(0.85, 1.15))
    pen = _pen_ink(w, rng, ink)
    for k in (-1, 1):
        ang = np.radians(w.ditto_angle + float(rng.normal(0, 4.0)))
        x = cx + k * 0.5 * w.ditto_gap * size + float(rng.normal(0, 0.02 * size))
        y = cy + float(rng.normal(0, 0.04 * size))
        dx, dy = np.sin(ang) * length / 2, np.cos(ang) * length / 2
        bend = float(rng.normal(0, 0.04 * size))
        pts = np.array([[x + dx, y - dy], [x + bend, y], [x - dx, y + dy]])
        _stamp_polyline(img, pts, thick, pen)


def draw_arrow(
    img: np.ndarray, box: tuple[float, float, float, float], size: float, rng: np.random.Generator,
    w: Writer, ink: float | None = None,
) -> None:  # fmt: skip
    """矢印(→ や >>)。上の行を指す意味で使う。"""
    x0, y0, x1, y1 = box
    cy = 0.5 * (y0 + y1) + float(rng.normal(0, 0.05 * (y1 - y0)))
    xa = x0 + float(rng.uniform(0.02, 0.1)) * (x1 - x0)
    xb = xa + float(rng.uniform(0.03, 0.14)) * (x1 - x0)
    thick = max(1.2, 0.05 * size * w.thickness)
    pen = _pen_ink(w, rng, ink)
    style = int(rng.integers(0, 3))
    head = 0.28 * size
    if style == 0:  # → 形
        _stamp_polyline(img, np.array([[xa, cy], [xb, cy + float(rng.normal(0, 1.0))]]), thick, pen)
        _stamp_polyline(
            img,
            np.array([[xb - head, cy - 0.7 * head], [xb, cy], [xb - head, cy + 0.7 * head]]),
            thick,
            pen,
        )
    elif style == 1:  # >> 形
        for off in (0.0, 0.55 * head):
            _stamp_polyline(
                img,
                np.array(
                    [
                        [xa + off, cy - 0.7 * head],
                        [xa + head + off, cy],
                        [xa + off, cy + 0.7 * head],
                    ]
                ),
                thick, pen,
            )  # fmt: skip
    else:  # 長い線 + 小さな頭
        _stamp_polyline(img, np.array([[xa, cy], [xb + 0.1 * (x1 - x0), cy]]), thick, pen)
        _stamp_polyline(
            img,
            np.array([[xb + 0.1 * (x1 - x0) - head, cy - 0.6 * head], [xb + 0.1 * (x1 - x0), cy],
                      [xb + 0.1 * (x1 - x0) - head, cy + 0.6 * head]]),
            thick, pen,
        )  # fmt: skip


def draw_wave(
    img: np.ndarray,
    x0: float,
    x1: float,
    y: float,
    size: float,
    rng: np.random.Generator,
    w: Writer,
    ink: float | None = None,
) -> None:
    """波線(長い記入を省略するときの「〜〜〜」)。"""
    amp = float(rng.uniform(0.05, 0.14)) * size
    lam = float(rng.uniform(1.4, 3.2)) * size
    phase = float(rng.uniform(0, 2 * np.pi))
    n = max(4, int((x1 - x0) / 5))
    xs = np.linspace(x0, x1, n)
    ys = y + amp * np.sin(2 * np.pi * (xs - x0) / lam + phase) + np.cumsum(rng.normal(0, 0.15, n))
    _stamp_polyline(
        img, np.stack([xs, ys], axis=1), max(1.2, 0.04 * size * w.thickness), _pen_ink(w, rng, ink)
    )


def draw_strokes_noise(
    img: np.ndarray, box: tuple[float, float, float, float], rng: np.random.Generator, w: Writer
) -> None:
    """余計な点・短い線(ゴミ・書き損じの点)を少しだけ置く。"""
    x0, y0, x1, y1 = box
    n = int(rng.integers(1, 3))
    for _ in range(n):
        x, y = float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))
        ln = float(rng.uniform(2, 0.25 * (y1 - y0)))
        a = float(rng.uniform(0, np.pi))
        pts = np.array([[x, y], [x + ln * np.cos(a), y + ln * np.sin(a)]])
        _stamp_polyline(img, pts, 1.5, _pen_ink(w, rng, w.ink_level + 30))
