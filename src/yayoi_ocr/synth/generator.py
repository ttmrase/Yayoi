"""合成帳票の生成器(テスト・評価・学習用)。

テンプレートから空の様式を描き、「書き手」(writer.py)の癖で架空の取引データ(content.py)を書き込み、
撮影による劣化(射影変換・回転・紙のたわみ・照明むら・影・ぼかし・JPEG ノイズ・低解像度)を与える。
返すもの: 画像、正解のセル多角形(劣化後の画像座標)、正解 JSON(raw / resolved)。

開発中は実データの内容を一切扱わない。内容はすべて架空で、評価はすべてこの合成帳票で行う。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..groundtruth import GTPage
from ..template import FormTemplate, load_template
from .content import PagePlan, plan_page
from .fonts import find_japanese_fonts, load_font, print_font
from .writer import (
    Writer,
    draw_arrow,
    draw_ditto,
    draw_strokes_noise,
    draw_text,
    draw_wave,
    make_writer,
    select_fonts,
    text_width,
)

# 撮影による劣化の種類(要素別に切り替えて、種類ごとの検出精度を測れる)
ALL_DEGRADATIONS = (
    "rotation",  # 回転 ±10°
    "perspective",  # 遠近(斜めから撮影)
    "curve",  # 紙のたわみ
    "lighting",  # 照明むら・影・周辺減光
    "contrast",  # コントラスト低下
    "blur",  # ぼけ
    "noise",  # ノイズ
    "jpeg",  # JPEG 圧縮ノイズ
    "lowres",  # 低解像度(80〜400dpi。無ければ 200dpi 固定)
    "cut",  # 用紙の端・表の端が写っていない
)


@dataclass
class SynthConfig:
    seed: int = 0
    writer_id: int | None = None  # None なら seed から決める(同じ ID = 同じ書き手の癖)
    dpi: float | None = None  # None なら 80〜400 から選ぶ
    degrade: bool = True
    degradations: tuple[str, ...] | None = None  # None なら全種類(degrade=True のとき)
    degrade_strength: float = 1.0  # 劣化の強さ(要素別評価で 0.5/1.0/1.5 など)
    rotate90: int | None = None  # 0/90/180/270(時計回り)。None ならなし
    fill_rows: int | None = None  # 埋まる行数(None なら 10〜29 から)
    canvas_scale: float = 1500.0  # 白紙の描画解像度(px / 単位)
    cut_prob: float = 0.15  # 用紙の端が写っていない(表が画像から切れる)撮影の割合
    font_include: tuple[str, ...] | None = (
        None  # 書き手が使うフォント(ファイル名の部分一致。None = すべて)
    )
    font_exclude: tuple[str, ...] = ()  # 使わないフォント(学習用/評価用のフォント分離)


@dataclass
class SynthSample:
    image: np.ndarray  # BGR uint8
    cell_polys: dict[
        tuple[int, int], np.ndarray
    ]  # (行, 列) → 4 隅 (4,2) 画像座標(左上, 右上, 右下, 左下)
    title_poly: np.ndarray
    number_poly: np.ndarray
    plan: PagePlan
    writer: Writer
    meta: dict = field(default_factory=dict)
    visible_fraction: float = 1.0  # 表のうち画像内に写っている割合

    @property
    def texts(self) -> dict[tuple[int, int], str]:
        """(行, 列) → 書かれた通りの文字列(raw。空欄は "")。"""
        cols = ("date", "account", "description", "amount")
        return {
            (r, c): row[col].raw
            for r, row in enumerate(self.plan.rows)
            for c, col in enumerate(cols)
        }

    @property
    def title_text(self) -> str:
        return self.plan.title

    def ground_truth(self, image_name: str = "synthetic.png") -> GTPage:
        return self.plan.to_ground_truth(image_name)


# ---------------------------------------------------------------- 白紙の様式


def _unit_to_canvas(tpl: FormTemplate, scale: float):
    b = tpl.page_box

    def f(u, v):
        return (np.asarray(u) - b.x0) * scale, (np.asarray(v) - b.y0) * scale

    w = round((b.x1 - b.x0) * scale)
    h = round((b.y1 - b.y0) * scale)
    return f, (w, h)


def render_blank(
    tpl: FormTemplate, scale: float, rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """白紙の様式を float32 グレー(紙 = 255)で描く。"""
    to_px, (w, h) = _unit_to_canvas(tpl, scale)
    img = np.full((h, w), 255.0, dtype=np.float32)
    line_gray = float(rng.uniform(95, 150))
    th = max(1, round(0.0011 * scale))
    # 横罫線・縦罫線
    x0p, y0p = (float(a) for a in to_px(0.0, 0.0))
    x1p, y1p = (float(a) for a in to_px(1.0, tpl.table_height))
    for y in tpl.hline_y[1:]:
        yp = round(float(to_px(0, y)[1]))
        cv2.line(img, (int(x0p), yp), (int(x1p), yp), line_gray, th, cv2.LINE_AA)
    for x in tpl.vline_x:
        xp = round(float(to_px(x, 0)[0]))
        cv2.line(img, (xp, int(y0p)), (xp, int(y1p)), line_gray, th, cv2.LINE_AA)
    # 見出し帯(黒ベタ + 白抜き文字)
    bx0, by0 = (round(float(a)) for a in to_px(tpl.band.x0, tpl.band.y0))
    bx1, by1 = (round(float(a)) for a in to_px(tpl.band.x1, tpl.band.y1))
    cv2.rectangle(img, (bx0, by0), (bx1, by1), float(rng.uniform(15, 45)), -1)
    pf = print_font()
    if pf is not None:
        pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(pil)
        font = load_font(pf, 0.62 * (by1 - by0))
        for col, label in zip(tpl.columns, ["日付", "科目", "摘要", "金額"], strict=True):
            xx = round(float(to_px(col.x0, 0)[0])) + int(0.01 * scale)
            draw.text((xx, by0 + int(0.2 * (by1 - by0))), label, fill=235, font=font)
        img = np.asarray(pil).astype(np.float32)
    return img, {"to_px": to_px, "size": (w, h)}


# ---------------------------------------------------------------- 書き込み


def fill_form(
    img: np.ndarray,
    tpl: FormTemplate,
    to_px,
    rng: np.random.Generator,
    writer: Writer,
    plan: PagePlan,
) -> None:
    """計画(PagePlan)を、書き手の癖で様式に書き込む。"""
    row_h = float(np.median(np.diff(tpl.row_edges))) * float(to_px(0, 1)[1] - to_px(0, 0)[1])
    ditto_frac = 0.3 + 0.3 * (((writer.writer_id * 7919) % 100) / 100.0)  # 〃を置く位置の癖
    for r, row in enumerate(plan.rows):
        for col in tpl.columns:
            c = row[col.kind.value]
            if c.kind == "blank":
                continue
            x0, y0 = (float(a) for a in to_px(col.x0, tpl.row_edges[r]))
            x1, y1 = (float(a) for a in to_px(col.x1, tpl.row_edges[r + 1]))
            pad = 0.014 * (x1 - x0)
            box = (x0 + pad, y0, x1 - pad, y1)
            size = row_h * writer.size_scale * float(rng.uniform(0.88, 1.12))
            jy = float(rng.normal(0.08, 0.07)) * row_h
            if c.overflow:
                jy += float(rng.choice([-1, 1])) * 0.3 * row_h
                size *= 1.15
            kind = col.kind.value
            if c.kind == "ditto":
                if kind == "date":
                    cx = x0 + (0.75 + 0.1 * float(rng.random())) * (x1 - x0)
                else:
                    cx = x0 + (ditto_frac + 0.08 * float(rng.normal())) * (x1 - x0)
                draw_ditto(img, cx, 0.5 * (y0 + y1) + 0.06 * row_h + jy * 0.3, size, rng, writer)
            elif c.kind == "arrow":
                draw_arrow(img, box, size, rng, writer)
            elif c.kind == "elided":
                _, wh = text_width(c.head, writer, size, rng)
                _, wt = text_width(c.tail, writer, size, rng)
                avail = (box[2] - box[0]) - 0.35 * size
                if wh + wt > 0.8 * avail:
                    size *= 0.8 * avail / (wh + wt)
                    _, wh = text_width(c.head, writer, size, rng)
                    _, wt = text_width(c.tail, writer, size, rng)
                _, xe = draw_text(img, c.head, box, rng, writer, size=size, jitter_y=jy)
                xs, _ = draw_text(
                    img, c.tail, box, rng, writer, size=size, align="right", jitter_y=jy
                )
                wave_y = 0.5 * (y0 + y1) + 0.15 * size + jy
                draw_wave(img, xe + 0.2 * size, xs - 0.2 * size, wave_y, size, rng, writer)
            else:
                text = c.raw
                _, wtxt = text_width(text, writer, size, rng)
                avail = box[2] - box[0]
                if wtxt > 0.96 * avail and not c.overflow:
                    size *= 0.96 * avail / wtxt
                if kind == "amount":
                    align = "right"
                elif kind == "date":
                    align = "right" if rng.random() < writer.align_date else "left"
                else:
                    align = "left"
                draw_text(img, text, box, rng, writer, size=size, align=align, jitter_y=jy)
                if c.annotation:
                    asz = 0.3 * row_h
                    abox = (box[0], y0 - 0.08 * row_h, box[2], y0 + 0.4 * row_h)
                    draw_text(
                        img, c.annotation, abox, rng, writer, size=asz, ink=writer.ink_level + 25
                    )
                if c.stray:
                    draw_strokes_noise(
                        img,
                        (x0, y0 + 0.2 * row_h, x0 + 0.15 * (x1 - x0), y1 - 0.1 * row_h),
                        rng,
                        writer,
                    )
    # タイトル・No.
    tb = tpl.title_box
    tx0, ty0 = (float(a) for a in to_px(tb.x0 + 0.1 * tb.w, tb.y0 + 0.2 * tb.h))
    tx1, ty1 = (float(a) for a in to_px(tb.x1 - 0.1 * tb.w, tb.y1 - 0.1 * tb.h))
    draw_text(
        img, plan.title, (tx0, ty0, tx1, ty1), rng, writer, size=0.8 * (ty1 - ty0), align="center"
    )
    nb = tpl.number_box
    nx0, ny0 = (float(a) for a in to_px(nb.x0, nb.y0 + 0.2 * nb.h))
    nx1, ny1 = (float(a) for a in to_px(nb.x1, nb.y1 - 0.05 * nb.h))
    draw_text(
        img,
        f"No.{plan.number}",
        (nx0, ny0, nx1, ny1),
        rng,
        writer,
        size=0.6 * (ny1 - ny0),
        align="right",
    )


# ---------------------------------------------------------------- 撮影による劣化


def _smooth_field(
    shape: tuple[int, int], grid: tuple[int, int], amp: float, rng: np.random.Generator
):
    g = rng.normal(0, 1, grid).astype(np.float32)
    return amp * cv2.resize(g, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def _random_homography(
    cw: int,
    ch: int,
    out_w: int,
    out_h: int,
    rng: np.random.Generator,
    allow_cut: bool,
    *,
    rotate: bool = True,
    perspective: bool = True,
    strength: float = 1.0,
) -> np.ndarray:
    """用紙キャンバス → 出力画像の射影変換(回転 ±10°・遠近・拡大・平行移動)。"""
    ang = math.radians(float(rng.uniform(-10, 10)) * strength) if rotate else 0.0
    zoom_lo, zoom_hi = (1.02, 1.16) if allow_cut else (0.86, 1.0)
    zoom = float(rng.uniform(zoom_lo, zoom_hi))
    s = zoom * min(out_w / cw, out_h / ch) * (1.0 if allow_cut else 0.95)
    cx, cy = cw / 2, ch / 2
    # 中心合わせの相似変換
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    A = s * R
    t = np.array([out_w / 2, out_h / 2]) - A @ np.array([cx, cy])
    t += rng.normal(0, 0.012, 2) * np.array([out_w, out_h]) if allow_cut else 0
    M = np.eye(3)
    M[:2, :2] = A
    M[:2, 2] = t
    # 遠近(キーストーン): 出力座標で 4 隅の少しの変位
    corners = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float64)
    dst = (M @ np.vstack([corners.T, np.ones(4)]))[:2].T
    jitter = rng.normal(0, 0.009 * strength, (4, 2)) * np.array([out_w, out_h])
    if not perspective:
        jitter = jitter * 0.0
    dst = dst + jitter
    return cv2.getPerspectiveTransform(corners.astype(np.float32), dst.astype(np.float32)).astype(
        np.float64
    )


def _apply_lighting(img: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    h, w = img.shape[:2]
    low = _smooth_field((h, w), (3, 3), 0.12 * strength, rng)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gx, gy = rng.uniform(-0.2, 0.2, 2) * strength
    grad = gx * (xx / w - 0.5) + gy * (yy / h - 0.5)
    light = 1.0 + low + grad
    # 影(大きな暗い多角形をぼかして乗せる)
    if rng.random() < 0.6:
        shadow = np.zeros((h, w), np.float32)
        poly = np.array([[rng.uniform(0, w), rng.uniform(0, h)] for _ in range(4)], dtype=np.int32)
        cv2.fillConvexPoly(shadow, cv2.convexHull(poly), 1.0)
        shadow = cv2.GaussianBlur(shadow, (0, 0), 0.04 * max(h, w))
        light = light * (1.0 - float(rng.uniform(0.1, 0.35)) * strength * shadow)
    vign = 1.0 - 0.15 * strength * (((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) * 2)
    return img * np.clip(light * vign, 0.3, 1.3)[..., None]


def generate_sample(cfg: SynthConfig | None = None, tpl: FormTemplate | None = None) -> SynthSample:
    cfg = cfg or SynthConfig()
    tpl = tpl or load_template()
    rng = np.random.default_rng(cfg.seed)
    if not find_japanese_fonts():
        raise RuntimeError(
            "日本語フォントが見つかりません(`python scripts/fetch_fonts.py` を実行してください)"
        )
    fonts = (
        select_fonts(cfg.font_include, cfg.font_exclude)
        if (cfg.font_include or cfg.font_exclude)
        else None
    )
    writer = make_writer(cfg.writer_id if cfg.writer_id is not None else cfg.seed, fonts)

    scale = cfg.canvas_scale
    blank, info = render_blank(tpl, scale, rng)
    to_px = info["to_px"]
    cw, ch = info["size"]
    plan = plan_page(rng, tpl.n_rows, cfg.fill_rows)
    fill_form(blank, tpl, to_px, rng, writer, plan)

    use = set(ALL_DEGRADATIONS if cfg.degradations is None else cfg.degradations)
    if not cfg.degrade:
        use = set()
    st = cfg.degrade_strength
    if cfg.dpi is not None:
        dpi = cfg.dpi
    elif "lowres" in use:
        dpi = float(np.exp(rng.uniform(np.log(80), np.log(400))))
    else:
        dpi = 200.0
    out_w = round(8.27 * dpi * rng.uniform(0.9, 1.1))
    out_h = round(11.69 * dpi * rng.uniform(0.9, 1.1))
    out_w, out_h = max(out_w, 480), max(out_h, 640)

    # 紙のたわみ(正準キャンバス上の滑らかな変位場)
    curve_amp = float(rng.uniform(0.0, 0.007)) * st * scale if "curve" in use else 0.0
    dx = (
        _smooth_field((ch, cw), (4, 3), curve_amp, rng)
        if curve_amp
        else np.zeros((ch, cw), np.float32)
    )
    dy = (
        _smooth_field((ch, cw), (4, 3), curve_amp, rng)
        if curve_amp
        else np.zeros((ch, cw), np.float32)
    )

    if use & {"rotation", "perspective", "cut"}:
        Hm = _random_homography(
            cw,
            ch,
            out_w,
            out_h,
            rng,
            bool("cut" in use and rng.random() < cfg.cut_prob),
            rotate="rotation" in use,
            perspective="perspective" in use,
            strength=st,
        )
    else:
        s = min(out_w / cw, out_h / ch) * 0.95
        Hm = np.array([[s, 0, (out_w - s * cw) / 2], [0, s, (out_h - s * ch) / 2], [0, 0, 1.0]])

    # 出力画素 → キャンバス(射影の逆 → たわみの逆を不動点反復)
    ys, xs = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    Hi = np.linalg.inv(Hm)
    den = Hi[2, 0] * xs + Hi[2, 1] * ys + Hi[2, 2]
    qx = ((Hi[0, 0] * xs + Hi[0, 1] * ys + Hi[0, 2]) / den).astype(np.float32)
    qy = ((Hi[1, 0] * xs + Hi[1, 1] * ys + Hi[1, 2]) / den).astype(np.float32)
    px_, py_ = qx.copy(), qy.copy()
    if curve_amp:
        for _ in range(4):
            px_ = qx - cv2.remap(dx, px_, py_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            py_ = qy - cv2.remap(dy, px_, py_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    page = cv2.remap(
        blank, px_, py_, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=-1
    )
    inside = page >= 0
    # 背景(机)の色
    bg = float(rng.uniform(50, 150))
    gray = np.where(inside, page, bg).astype(np.float32)
    color = np.repeat(gray[..., None], 3, axis=2)
    tint = np.array([1.0, rng.uniform(0.96, 1.0), rng.uniform(0.9, 0.99)], dtype=np.float32)[
        ::-1
    ].copy()  # 紙の色味(BGR・暖色寄り)
    color = color * tint[None, None, :]

    if "lighting" in use:
        color = _apply_lighting(color, rng, float(rng.uniform(0.4, 1.0)) * st)
    if "contrast" in use:  # コントラスト低下
        gain = 1.0 - float(rng.uniform(0.0, 0.3)) * st
        color = 255 - (255 - color) * gain
    if "blur" in use:
        sigma = float(rng.uniform(0.0, 1.4)) * st * max(out_w / 900.0, 0.6)
        if sigma > 0.1:
            color = cv2.GaussianBlur(color, (0, 0), sigma)
    if "noise" in use:
        color = color + rng.normal(0, rng.uniform(0, 5) * st, color.shape).astype(np.float32)
    img8 = np.clip(color, 0, 255).astype(np.uint8)
    if "jpeg" in use:
        q = int(np.clip(92 - rng.uniform(0, 57) * st, 20, 95))
        ok, enc = cv2.imencode(".jpg", img8, [cv2.IMWRITE_JPEG_QUALITY, q])
        img8 = cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img8

    # 正解の多角形(キャンバス → たわみ → 射影)
    def fwd(pts_unit: np.ndarray) -> np.ndarray:
        px = np.stack(to_px(pts_unit[:, 0], pts_unit[:, 1]), axis=1).astype(np.float32)
        if curve_amp:
            ddx = cv2.remap(
                dx,
                px[:, 0:1].copy(),
                px[:, 1:2].copy(),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            ddy = cv2.remap(
                dy,
                px[:, 0:1].copy(),
                px[:, 1:2].copy(),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            px = px + np.concatenate([ddx, ddy], axis=1)
        p3 = np.hstack([px, np.ones((len(px), 1), np.float32)]).astype(np.float64) @ Hm.T
        return p3[:, :2] / p3[:, 2:3]

    def quad(box_x0, box_y0, box_x1, box_y1) -> np.ndarray:
        return fwd(
            np.array([[box_x0, box_y0], [box_x1, box_y0], [box_x1, box_y1], [box_x0, box_y1]])
        )

    polys = {}
    for r in range(tpl.n_rows):
        for c in range(len(tpl.columns)):
            b = tpl.cell_box(r, c)
            polys[(r, c)] = quad(b.x0, b.y0, b.x1, b.y1)
    tp = quad(tpl.title_box.x0, tpl.title_box.y0, tpl.title_box.x1, tpl.title_box.y1)
    npoly = quad(tpl.number_box.x0, tpl.number_box.y0, tpl.number_box.x1, tpl.number_box.y1)

    table = quad(0, 0, 1, tpl.table_height)
    vis = _visible_fraction(table, out_w, out_h)

    sample = SynthSample(
        image=img8,
        cell_polys=polys,
        title_poly=tp,
        number_poly=npoly,
        plan=plan,
        writer=writer,
        visible_fraction=vis,
        meta={
            "dpi": dpi,
            "seed": cfg.seed,
            "writer_id": writer.writer_id,
            "size": (out_w, out_h),
            "curve_amp_units": curve_amp / scale,
        },
    )
    if cfg.rotate90:
        sample = rotate_sample(sample, cfg.rotate90)
    return sample


def _visible_fraction(poly: np.ndarray, w: int, h: int) -> float:
    """多角形のうち画像 (0,0)-(w,h) の内側にある面積の割合。"""
    f = 4.0
    x0, y0 = np.floor(poly.min(axis=0) / f) - 1
    x1, y1 = np.ceil(poly.max(axis=0) / f) + 1
    m = np.zeros((int(y1 - y0), int(x1 - x0)), np.uint8)
    cv2.fillPoly(m, [np.round(poly / f - [x0, y0]).astype(np.int32)], 1)
    total = int(m.sum())
    if total == 0:
        return 0.0
    gx = (np.arange(m.shape[1]) + x0) * f
    gy = (np.arange(m.shape[0]) + y0) * f
    inside = ((gy >= 0) & (gy < h))[:, None] & ((gx >= 0) & (gx < w))[None, :]
    return float((m.astype(bool) & inside).sum() / total)


def rotate_sample(s: SynthSample, deg_cw: int) -> SynthSample:
    """画像と正解座標を時計回りに 90° 単位で回す(向き判定のテスト用)。"""
    deg_cw %= 360
    if deg_cw == 0:
        return s
    h, w = s.image.shape[:2]
    k = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[
        deg_cw
    ]
    img = cv2.rotate(s.image, k)

    def rot(p: np.ndarray) -> np.ndarray:
        x, y = p[:, 0], p[:, 1]
        if deg_cw == 90:
            return np.stack([h - y, x], axis=1)
        if deg_cw == 180:
            return np.stack([w - x, h - y], axis=1)
        return np.stack([y, w - x], axis=1)

    return SynthSample(
        image=img,
        cell_polys={k_: rot(v) for k_, v in s.cell_polys.items()},
        title_poly=rot(s.title_poly),
        number_poly=rot(s.number_poly),
        plan=s.plan,
        writer=s.writer,
        meta={**s.meta, "rotate90": deg_cw},
        visible_fraction=s.visible_fraction,
    )
