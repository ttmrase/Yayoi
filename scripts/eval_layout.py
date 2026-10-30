#!/usr/bin/env python3
"""合成帳票でレイアウト検出を評価する(セル境界の IoU・検出失敗数)。

    python scripts/eval_layout.py --n 120 [--seed0 0] [--save-worst DIR]

「検出失敗」= 要確認(needs_review)が付かないまま平均 IoU が 0.8 未満のページ(黙った失敗)。
要確認として報告されたページは失敗に数えない(確認画面で手動補正できるため)。
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from yayoi_ocr.config import assert_safe_output_dir
from yayoi_ocr.layout.metrics import polygon_iou
from yayoi_ocr.layout.pipeline import analyze_image
from yayoi_ocr.models import PageStatus
from yayoi_ocr.synth.generator import SynthConfig, generate_sample


def _one(args: tuple[int, int | None]) -> dict:
    seed, rot = args
    s = generate_sample(SynthConfig(seed=seed, rotate90=rot))
    t = time.time()
    res = analyze_image(s.image)
    dt = time.time() - t
    ious = np.array(
        [
            polygon_iou(np.array(res.page.rows[r].cells[c].polygon), s.cell_polys[(r, c)])
            for r in range(29)
            for c in range(4)
        ]
    )
    return {
        "seed": seed,
        "dpi": s.meta["dpi"],
        "rot": rot or 0,
        "vis": s.visible_fraction,
        "status": res.page.status,
        "mean_iou": float(ious.mean()),
        "min_iou": float(ious.min()),
        "p10_iou": float(np.percentile(ious, 10)),
        "time": dt,
        "notes": res.page.notes,
    }


def evaluate(
    n: int, seed0: int = 0, rotate: bool = True, verbose: bool = True, jobs: int = 1
) -> dict:
    tasks = [
        (seed0 + i, [0, 90, 180, 270][i % 4] if rotate and i % 5 == 0 else None) for i in range(n)
    ]
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(jobs) as ex:
            rows = list(ex.map(_one, tasks))
    else:
        rows = [_one(t) for t in tasks]
    if verbose:
        for r in rows:
            print(
                f"seed={r['seed']:4d} dpi={r['dpi']:5.0f} rot={r['rot']:3d} vis={r['vis']:.2f} "
                f"{r['status'].value:12s} mean={r['mean_iou']:.3f} p10={r['p10_iou']:.3f} "
                f"min={r['min_iou']:.3f} t={r['time']:.1f}s {r['notes'] if r['notes'] else ''}",
                flush=True,
            )
    mean_all = float(np.mean([r["mean_iou"] for r in rows]))
    ok_pages = [r for r in rows if r["status"] == PageStatus.OK]
    silent = [r for r in ok_pages if r["mean_iou"] < 0.8]
    flagged = [r for r in rows if r["status"] != PageStatus.OK]
    return {
        "n": n,
        "mean_iou_all": mean_all,
        "mean_iou_ok_pages": float(np.mean([r["mean_iou"] for r in ok_pages]))
        if ok_pages
        else None,
        "n_flagged": len(flagged),
        "n_silent_failures": len(silent),
        "silent": [r["seed"] for r in silent],
        "flagged": [r["seed"] for r in flagged],
        "worst": sorted(rows, key=lambda r: r["mean_iou"])[:5],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="並列プロセス数")
    ap.add_argument("--save-worst", help="最悪のページの画像を保存するディレクトリ(リポジトリ外)")
    args = ap.parse_args()
    out = evaluate(args.n, args.seed0, verbose=not args.quiet, jobs=args.jobs)
    print("\n==== summary ====")
    for k, v in out.items():
        if k != "worst":
            print(f"{k}: {v}")
    print("worst:", [(w["seed"], round(w["mean_iou"], 3), w["status"].value) for w in out["worst"]])
    if args.save_worst:
        import cv2

        d = assert_safe_output_dir(args.save_worst)
        d.mkdir(parents=True, exist_ok=True)
        for w in out["worst"]:
            rot = w["rot"] or None
            s = generate_sample(SynthConfig(seed=w["seed"], rotate90=rot))
            cv2.imwrite(str(d / f"seed{w['seed']}.png"), s.image)
    return 0 if out["n_silent_failures"] == 0 and out["mean_iou_all"] >= 0.9 else 1


if __name__ == "__main__":
    sys.exit(main())
