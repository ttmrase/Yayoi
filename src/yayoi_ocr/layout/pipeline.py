"""レイアウト解析の入口: 画像 1 枚 → Page(セルの座標・空欄判定)+ 切り出し画像。"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..models import (
    BBox,
    Cell,
    CellFlag,
    ColumnKind,
    ImageRef,
    Page,
    PageStatus,
    Row,
    TextField,
)
from ..preprocess import preprocess_for_detection
from ..template import FormTemplate, load_template
from .band import BandResult, find_header_band
from .cells import (
    BLANK_INK_PX,
    PAD_X_PX_FRAC,
    PAD_Y_FRAC,
    CellCrop,
    PageInk,
    analyze_page_ink,
    extract_region,
)
from .grid import GridFit, LineTrack, fit_grid
from .warp import PageGeometry, make_geometry, rectify

# 要確認の判定しきい値
MIN_BAND_CONFIDENCE = 0.45
MIN_GOOD_H = 0.80
MIN_GOOD_V = 0.80
MAX_RESID_UNITS = 0.02  # 交点の残差(表の幅に対する比)。紙のたわみ分は変位場が吸収する


@dataclass
class LayoutResult:
    page: Page
    tpl: FormTemplate
    geom: PageGeometry
    rect_color: np.ndarray  # 正準ページ画像(カラー)
    ink: PageInk
    band: BandResult | None
    fit: GridFit | None
    source: np.ndarray  # 元画像(重ね描き用)
    crops: dict[tuple[int, int], CellCrop] = field(default_factory=dict)
    title_crop: CellCrop | None = None
    number_crop: CellCrop | None = None

    @property
    def needs_review(self) -> bool:
        return self.page.status == PageStatus.NEEDS_REVIEW

    def cell_crop(self, row: int, col: int) -> CellCrop:
        return self.crops[(row, col)]


def _fallback_fit(flat_shape: tuple[int, int], tpl: FormTemplate) -> GridFit:
    """帯が見つからないとき: 画像いっぱいに表があると仮定した当て推量の位置合わせ。"""
    h, w = flat_shape
    tw = 0.9 * w
    H = np.array([[tw, 0, 0.05 * w], [0, tw, 0.10 * h], [0, 0, 1.0]])
    hl = [
        LineTrack(y, np.linspace(0.1, 0.9, 4), np.zeros(4), np.zeros(4), np.zeros(4, dtype=bool))
        for y in tpl.hline_y
    ]
    vl = [
        LineTrack(x, np.linspace(0.1, 1.2, 4), np.zeros(4), np.zeros(4), np.zeros(4, dtype=bool))
        for x in tpl.vline_x
    ]
    return GridFit(H, hl, vl, 0.0, 0.0, False, float("nan"), ["fallback"])


def _line_good_at(track: LineTrack, at: float) -> bool:
    if track.good.size == 0:
        return False
    i = int(np.argmin(np.abs(track.centers - at)))
    return bool(track.good[i])


def _table_inside_fraction(fit: GridFit, tpl: FormTemplate, work_shape: tuple[int, int]) -> float:
    """表の矩形のうち作業画像の内側にある面積の割合(大きくはみ出していれば撮影が切れている)。"""
    h, w = work_shape
    gx, gy = np.meshgrid(np.linspace(0, 1, 21), np.linspace(0, tpl.table_height, 41))
    pts = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
    p = fit.H @ pts
    x, y = p[0] / p[2], p[1] / p[2]
    return float(np.mean((x >= -2) & (x < w + 2) & (y >= -2) & (y < h + 2)))


def _apply_row_context(prep: list, tpl: FormTemplate) -> None:
    """行の文脈で、薄いインクだけのセルの扱いを決める(Phase 2 の分類器までの暫定)。

    - 金額欄が空欄の行は「行全体が空欄」とみなし、ほかの列の薄いインクは空欄にする(要確認なし)。
    - 金額欄に記入がある行は、日付・科目・摘要の薄いインクを「〃の可能性」として拾う。
    金額欄自体の薄いインクは高感度のまま(空欄とせず要確認)。
    """
    amount_idx = next(i for i, c in enumerate(tpl.columns) if c.kind == ColumnKind.AMOUNT)
    amount_filled = not prep[amount_idx][3].blank
    for i, (_, _, _, crop, _) in enumerate(prep):
        if i == amount_idx or not crop.faint_only:
            continue
        crop.flags.discard(CellFlag.NEEDS_REVIEW)
        if amount_filled:
            crop.flags.add(CellFlag.DITTO_LIKELY)
        else:
            crop.blank = True
            crop.flags.add(CellFlag.BLANK)


def analyze_image(
    bgr: np.ndarray,
    tpl: FormTemplate | None = None,
    *,
    source_page_index: int = 0,
    image_path: str | None = None,
    sha256: str | None = None,
) -> LayoutResult:
    """画像(BGR)からレイアウトを解析する。検出に自信がなくても例外は出さず「要確認」で返す。"""
    tpl = tpl or load_template()
    src_h, src_w = bgr.shape[:2]
    notes: list[str] = []
    flat, work_scale = preprocess_for_detection(bgr)

    band = find_header_band(flat)
    fit: GridFit | None = None
    status = PageStatus.OK
    if band is None or band.confidence < MIN_BAND_CONFIDENCE:
        notes.append("見出し帯を検出できなかった(または自信が低い)ため、位置を推定しました")
        status = PageStatus.NEEDS_REVIEW
        fit = _fallback_fit(flat.shape, tpl)
    else:
        try:
            fit = fit_grid(flat, band, tpl)
        except Exception as exc:  # 異常画像でも落とさず「要確認」にする
            notes.append(f"罫線の位置合わせに失敗: {type(exc).__name__}")
            status = PageStatus.NEEDS_REVIEW
            fit = _fallback_fit(flat.shape, tpl)
        else:
            if fit.good_h < MIN_GOOD_H:
                notes.append(f"横罫線の検出率が低い({fit.good_h:.0%})")
            if fit.good_v < MIN_GOOD_V:
                notes.append(f"縦罫線の検出率が低い({fit.good_v:.0%})")
            if not fit.outer_good:
                notes.append("表の外枠の検出に自信がない")
            if not np.isfinite(fit.rms_resid_px) or fit.rms_resid_px / 1200.0 > MAX_RESID_UNITS:
                notes.append("罫線の交点の整合性が低い")
            if _table_inside_fraction(fit, tpl, flat.shape) < 0.85:
                notes.append("表が画像からはみ出している(撮影範囲が足りない可能性)")
            if (
                fit.good_h < MIN_GOOD_H
                or fit.good_v < MIN_GOOD_V
                or not fit.outer_good
                or any(n.startswith(("罫線の交点", "表が画像")) for n in notes)
            ):
                status = PageStatus.NEEDS_REVIEW

    if band is not None and band.clipped and status == PageStatus.OK:
        notes.append("見出し帯が画像の端で切れている(表の隅が写っていない可能性)")
        status = PageStatus.NEEDS_REVIEW

    geom = make_geometry(fit, tpl, work_scale)
    rect = rectify(bgr, geom)
    pink = analyze_page_ink(rect, geom)

    n_cols = len(tpl.columns)
    crops: dict[tuple[int, int], CellCrop] = {}
    rows: list[Row] = []
    px_w = tpl.canon_width
    pad_x = round(PAD_X_PX_FRAC * px_w)
    row_h_px = round(float(np.median(np.diff(tpl.row_edges))) * px_w)
    pad_y = round(PAD_Y_FRAC * row_h_px)
    inner = max(3, round(0.0022 * px_w) + 2)
    wide_inner = max(8, round(0.012 * px_w)) + 3  # 罫線が不確実な側は内側の余裕を広げる
    blank_px = BLANK_INK_PX * (px_w / 2000.0) ** 2
    uncertain_page = status == PageStatus.NEEDS_REVIEW and fit.notes == ["fallback"]
    for r in range(tpl.n_rows):
        prep = []
        for c, col in enumerate(tpl.columns):
            box = tpl.cell_box(r, c)
            rect_px = geom.box_to_px(box)
            cy = 0.5 * (box.y0 + box.y1)
            cx = 0.5 * (box.x0 + box.x1)
            ok_l = _line_good_at(fit.vlines[c], cy)
            ok_r = _line_good_at(fit.vlines[c + 1], cy)
            ok_t = _line_good_at(fit.hlines[r], cx)
            ok_b = _line_good_at(fit.hlines[r + 1], cx)
            margins = (
                inner if ok_l else wide_inner,
                inner if ok_t else wide_inner,
                inner if ok_r else wide_inner,
                inner if ok_b else wide_inner,
            )
            crop = extract_region(
                rect, pink, rect_px, r * n_cols + c, (pad_x, pad_y), margins, blank_px
            )
            crops[(r, c)] = crop
            prep.append((col, box, rect_px, crop, ok_l and ok_r and ok_t and ok_b))
        _apply_row_context(prep, tpl)
        cells: list[Cell] = []
        for col, box, rect_px, crop, lines_ok in prep:
            flags = set(crop.flags)
            if uncertain_page or not lines_ok:
                flags.add(CellFlag.LAYOUT_UNCERTAIN)
            cells.append(
                Cell(
                    column=col.kind,
                    bbox=BBox(
                        x=rect_px[0],
                        y=rect_px[1],
                        w=rect_px[2] - rect_px[0],
                        h=rect_px[3] - rect_px[1],
                    ),
                    polygon=geom.box_polygon_source(box),
                    flags=flags,
                    ink_ratio=crop.ink_ratio,
                )
            )
        rows.append(Row(index=r, cells=cells))

    def field_of(box) -> tuple[TextField, CellCrop]:
        rect_px = geom.box_to_px(box)
        crop = extract_region(
            rect, pink, rect_px, None, (pad_x, pad_y // 2), (inner,) * 4, blank_px
        )
        flags = set(crop.flags)
        if uncertain_page:
            flags.add(CellFlag.LAYOUT_UNCERTAIN)
        tf = TextField(
            bbox=BBox(
                x=rect_px[0], y=rect_px[1], w=rect_px[2] - rect_px[0], h=rect_px[3] - rect_px[1]
            ),
            polygon=geom.box_polygon_source(box),
            flags=flags,
            ink_ratio=crop.ink_ratio,
        )
        return tf, crop

    title, title_crop = field_of(tpl.title_box)
    number, number_crop = field_of(tpl.number_box)
    page = Page(
        index=source_page_index,
        title=title,
        number=number,
        image=ImageRef(
            path=image_path,
            sha256=sha256,
            width=src_w,
            height=src_h,
            source_page_index=source_page_index,
            rotation_applied=band.rotation_cw if band else 0,
        ),
        rows=rows,
        status=status,
        notes=notes,
    )
    return LayoutResult(
        page=page,
        tpl=tpl,
        geom=geom,
        rect_color=rect,
        ink=pink,
        band=band,
        fit=fit,
        source=bgr,
        crops=crops,
        title_crop=title_crop,
        number_crop=number_crop,
    )


# ---------------------------------------------------------------- 検証用の重ね描き


def draw_overlay(result: LayoutResult, long_side: int = 1600) -> np.ndarray:
    """元画像の上にセルの多角形・帯・タイトル欄などを重ねた画像(デバッグ用)。"""
    img = result.source
    h, w = img.shape[:2]
    f = long_side / max(h, w)
    vis = cv2.resize(
        img,
        (round(w * f), round(h * f)),
        interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA,
    )
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    def poly(pts, color, th=1):
        arr = np.round(np.array(pts) * f).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [arr], True, color, th, cv2.LINE_AA)

    for row in result.page.rows:
        for cell in row.cells:
            if CellFlag.LAYOUT_UNCERTAIN in cell.flags:
                color = (0, 0, 255)
            elif CellFlag.BLANK in cell.flags:
                color = (200, 140, 40)  # 空欄: 青系
            elif CellFlag.OVERFLOW in cell.flags:
                color = (0, 140, 255)
            else:
                color = (0, 170, 0)
            poly(cell.polygon, color, 1)
    poly(result.page.title.polygon, (255, 0, 255), 2)
    poly(result.page.number.polygon, (255, 0, 255), 2)
    if result.band is not None:
        b = result.band
        k = 1.0 / result.geom.work_scale  # 作業画像 → 元画像
        poly(
            [p * k for p in (b.top_left, b.top_right, b.bottom_right, b.bottom_left)],
            (0, 255, 255),
            1,
        )
    # 行番号
    for row in result.page.rows:
        x, y = row.cells[0].polygon[0]
        cv2.putText(
            vis,
            str(row.index + 1),
            (int(x * f) + 2, int(y * f) + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )
    status = f"status={result.page.status.value}"
    cv2.putText(
        vis,
        status,
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255) if result.needs_review else (0, 128, 0),
        2,
    )
    return vis


def draw_rectified_overlay(result: LayoutResult) -> np.ndarray:
    """正準ページ画像に、セルの矩形とインク(赤)・空欄(青の塗り)を重ねる。"""
    vis = result.rect_color.copy()
    ink = result.ink.ink
    vis[ink] = (0.45 * vis[ink] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
    for crop in result.crops.values():
        x0, y0, x1, y1 = crop.cell_rect
        color = (200, 140, 40) if crop.blank else (0, 170, 0)
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 1)
    return vis
