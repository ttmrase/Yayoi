"""レイアウト解析結果の書き出し(開発・検証用)。実データ由来なので出力先はリポジトリの外のみ。"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from ..config import assert_safe_output_dir
from .pipeline import LayoutResult, draw_overlay, draw_rectified_overlay


def save_layout(result: LayoutResult, out_dir: str | Path, *, save_cells: bool = True) -> Path:
    """重ね描き・正準ページ・セル切り出し・layout.json を `out_dir` に書く。"""
    out = assert_safe_output_dir(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "overlay.png"), draw_overlay(result))
    cv2.imwrite(str(out / "rectified.png"), result.rect_color)
    cv2.imwrite(str(out / "rectified_overlay.png"), draw_rectified_overlay(result))
    if save_cells:
        (out / "cells" / "color").mkdir(parents=True, exist_ok=True)
        (out / "cells" / "binary").mkdir(parents=True, exist_ok=True)
        for (r, c), crop in result.crops.items():
            name = f"r{r + 1:02d}_c{c + 1}.png"
            cv2.imwrite(str(out / "cells" / "color" / name), crop.color)
            cv2.imwrite(str(out / "cells" / "binary" / name), crop.binary)
        for name, crop in (("title", result.title_crop), ("number", result.number_crop)):
            if crop is not None:
                cv2.imwrite(str(out / f"{name}_color.png"), crop.color)
                cv2.imwrite(str(out / f"{name}_binary.png"), crop.binary)
    diag = {
        "page": json.loads(result.page.model_dump_json()),
        "diagnostics": {
            "band_confidence": float(result.band.confidence) if result.band else None,
            "rotation_cw": result.band.rotation_cw if result.band else None,
            "good_h": float(result.fit.good_h) if result.fit else None,
            "good_v": float(result.fit.good_v) if result.fit else None,
            "outer_good": bool(result.fit.outer_good) if result.fit else None,
            "rms_resid_px": float(result.fit.rms_resid_px) if result.fit else None,
            "ink_threshold": float(result.ink.threshold),
        },
    }
    (out / "layout.json").write_text(
        json.dumps(diag, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return out
