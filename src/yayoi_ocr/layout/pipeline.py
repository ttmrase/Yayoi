"""レイアウト解析の入口: 画像 1 枚 → Page(セルの座標・空欄判定)+ 切り出し画像。"""

from __future__ import annotations

from collections.abc import Sequence
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


def validate_corners(corners: Sequence[Sequence[float]], src_shape: tuple[int, int]) -> np.ndarray:
    """表の 4 隅(元画像 px。左上・右上・右下・左下の順)の形を検査して (4, 2) の配列で返す。不正なら ValueError。
    画像(y が下向き)で時計回りの凸四角形であること(回転は可、鏡像は不可)、小さすぎない、画像から遠すぎない。"""
    pts = np.asarray(corners, dtype=np.float64)
    if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
        raise ValueError("隅は 4 点(x, y)で指定してください")
    h, w = src_shape
    if np.any(pts[:, 0] < -0.5 * w) or np.any(pts[:, 0] > 1.5 * w):
        raise ValueError("隅が画像から離れすぎています")
    if np.any(pts[:, 1] < -0.5 * h) or np.any(pts[:, 1] > 1.5 * h):
        raise ValueError("隅が画像から離れすぎています")
    cross = []
    for i in range(4):
        a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if not all(x > 0 for x in cross):
        raise ValueError("隅の順序か形が不正です(左上・右上・右下・左下の順の凸四角形)")
    area = 0.5 * abs(
        float(np.dot(pts[:, 0], np.roll(pts[:, 1], -1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], -1)))
    )
    if area < 0.05 * w * h:
        raise ValueError("指定された表が小さすぎます")
    return pts


def corners_to_homography(
    corners: Sequence[Sequence[float]],
    tpl: FormTemplate,
    work_scale: float,
    src_shape: tuple[int, int],
) -> np.ndarray:
    """利用者が指定した表の 4 隅(元画像 px。左上は見出し帯の下辺の左端)から、正準単位 → 作業画像 px の射影変換を作る。"""
    pts = validate_corners(corners, src_shape)
    th = tpl.table_height
    src_units = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, th], [0.0, th]], dtype=np.float32)
    dst_work = ((pts + 0.5) * work_scale - 0.5).astype(np.float32)  # 元画像 px → 作業画像 px
    return cv2.getPerspectiveTransform(src_units, dst_work).astype(np.float64)


def _straight_fit(H: np.ndarray, tpl: FormTemplate, notes: list[str]) -> GridFit:
    """罫線の個別の検出ができなかったとき: 指定された射影変換のまま、罫線は真っ直ぐとみなす。"""
    hl = [
        LineTrack(y, np.linspace(0.1, 0.9, 4), np.zeros(4), np.zeros(4), np.zeros(4, dtype=bool))
        for y in tpl.hline_y
    ]
    vl = [
        LineTrack(x, np.linspace(0.1, 1.2, 4), np.zeros(4), np.zeros(4), np.zeros(4, dtype=bool))
        for x in tpl.vline_x
    ]
    return GridFit(H, hl, vl, 0.0, 0.0, False, float("nan"), notes)


def _quality_notes(
    fit: GridFit, tpl: FormTemplate, work_shape: tuple[int, int]
) -> tuple[list[str], bool]:
    """位置合わせの質の注記と、要確認にすべきか。"""
    notes: list[str] = []
    if fit.good_h < MIN_GOOD_H:
        notes.append(f"横罫線の検出率が低い({fit.good_h:.0%})")
    if fit.good_v < MIN_GOOD_V:
        notes.append(f"縦罫線の検出率が低い({fit.good_v:.0%})")
    if not fit.outer_good:
        notes.append("表の外枠の検出に自信がない")
    if not np.isfinite(fit.rms_resid_px) or fit.rms_resid_px / 1200.0 > MAX_RESID_UNITS:
        notes.append("罫線の交点の整合性が低い")
    if _table_inside_fraction(fit, tpl, work_shape) < 0.85:
        notes.append("表が画像からはみ出している(撮影範囲が足りない可能性)")
    review = (
        fit.good_h < MIN_GOOD_H
        or fit.good_v < MIN_GOOD_V
        or not fit.outer_good
        or any(n.startswith(("罫線の交点", "表が画像")) for n in notes)
    )
    return notes, review


def analyze_image(
    bgr: np.ndarray,
    tpl: FormTemplate | None = None,
    *,
    source_page_index: int = 0,
    image_path: str | None = None,
    sha256: str | None = None,
    corners: Sequence[Sequence[float]] | None = None,
) -> LayoutResult:
    """画像(BGR)からレイアウトを解析する。検出に自信がなくても例外は出さず「要確認」で返す。

    `corners`(表の 4 隅。元画像 px、左上・右上・右下・左下の順)を渡すと、見出し帯の検出を行わずその位置を出発点にして
    罫線を微調整する(確認画面での手動補正の再切り出し。形が不正なら ValueError)。画像の向きは補正しない。
    """
    tpl = tpl or load_template()
    src_h, src_w = bgr.shape[:2]
    notes: list[str] = []
    flat, work_scale = preprocess_for_detection(bgr)

    band = None if corners is not None else find_header_band(flat)
    fit: GridFit | None = None
    status = PageStatus.OK
    if corners is not None:
        H0 = corners_to_homography(corners, tpl, work_scale, (src_h, src_w))
        notes.append("表の隅が手動で指定されました")
        try:
            fit = fit_grid(flat, None, tpl, H0=H0)
        except Exception as exc:
            notes.append(f"罫線の微調整に失敗: {type(exc).__name__}")
            fit = _straight_fit(H0, tpl, ["manual"])
            status = PageStatus.NEEDS_REVIEW
        else:
            qnotes, review = _quality_notes(fit, tpl, flat.shape)
            notes.extend(qnotes)
            if review:
                status = PageStatus.NEEDS_REVIEW
    elif band is None or band.confidence < MIN_BAND_CONFIDENCE:
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
            qnotes, review = _quality_notes(fit, tpl, flat.shape)
            notes.extend(qnotes)
            if review:
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


# ---------------------------------------------------------------- 確認画面の重ね表示(罫線の確認・補正)


def overlay_info(result: LayoutResult, n_points: int = 9) -> dict:
    """検出した表の 4 隅(左上・右上・右下・左下)と罫線の折れ線を、元画像の px で返す(確認画面が元画像に重ねて描く)。"""
    geom, tpl = result.geom, result.tpl
    th = tpl.table_height
    xs = np.linspace(0.0, 1.0, n_points)
    ys = np.linspace(0.0, th, n_points)

    def pts(a: np.ndarray) -> list[list[float]]:
        return [[round(float(x), 1), round(float(y), 1)] for x, y in a]

    src_h, src_w = result.source.shape[:2]
    corner_units = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, th], [0.0, th]])
    return {
        "width": int(src_w),
        "height": int(src_h),
        "corners": pts(geom.unit_points_to_source(corner_units)),
        # 罫線の位置(表の高さ・幅に対する 0〜1 の比)。隅を動かしたときの罫線の見込み表示に使う
        "hfrac": [round(float(y) / th, 5) for y in tpl.hline_y],
        "vfrac": [round(float(x), 5) for x in tpl.vline_x],
        "hlines": [
            pts(geom.unit_points_to_source(np.array([(x, y) for x in xs]))) for y in tpl.hline_y
        ],
        "vlines": [
            pts(geom.unit_points_to_source(np.array([(x, y) for y in ys]))) for x in tpl.vline_x
        ],
    }


def view_jpeg(
    result: LayoutResult, long_side: int = 1400, quality: int = 82
) -> tuple[bytes, float]:
    """確認画面に出す元画像(長辺を `long_side` に縮小した JPEG)と、その倍率(表示 px / 元画像 px)。"""
    img = result.source
    h, w = img.shape[:2]
    f = min(1.0, long_side / max(h, w))
    if f < 1.0:
        img = cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg_encode_failed")
    return buf.tobytes(), f


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
