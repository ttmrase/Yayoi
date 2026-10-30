#!/usr/bin/env python3
"""学習曲線のシミュレーション: 同じ書き手の連続するページを順に認識し、確定(正解を与える)するたびに
見覚えメモリ・摘要辞書・科目語彙・年を更新して、ページ数に対する要確認率・見逃し率を測る。

    python scripts/sim_learning.py --data DIR --classifier PATH [--confidence PATH]
        [--memory] [--dictionary] [--dump OUT.jsonl] [--mem-bonus 3.0] [--sim-threshold 0.85]

- データ: build_synth_cells.py --writers W --n W*P で作った、書き手ごとに P ページのデータセット(合成)。
- 設定の組み合わせ: 何も付けない = 静的(毎ページ初期状態)/ --dictionary = 摘要辞書を育てる / --memory = 見覚えメモリを育てる。
- 各ページは「確定前」の状態で認識して評価し、その後に正解(利用者が確認したもの)で状態を更新する。
- --dump: 較正用のセル記録(メモリありのとき、メモリ由来の特徴量も含む。fit_confidence.py の入力)。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from yayoi_ocr.config import assert_safe_output_dir
from yayoi_ocr.recognize import fields
from yayoi_ocr.recognize.celltype import CellTypeClassifier
from yayoi_ocr.recognize.confidence import ConfidenceModel
from yayoi_ocr.recognize.evaluate import dump_row, judge
from yayoi_ocr.recognize.inputs import from_page_data
from yayoi_ocr.recognize.learning import ConfirmedCell, confirm_page
from yayoi_ocr.recognize.memory import RecognitionMemory
from yayoi_ocr.recognize.page import JobContext, PageRecognizer
from yayoi_ocr.recognize.parseq.engine import ParseqEngine
from yayoi_ocr.synth.content import ACCOUNTS, TITLES
from yayoi_ocr.synth.dataset import list_pages, load_page


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--confidence")
    ap.add_argument("--memory", action="store_true")
    ap.add_argument("--dictionary", action="store_true")
    ap.add_argument("--mem-bonus", type=float, default=fields.MEMORY_BONUS)
    ap.add_argument("--sim-threshold", type=float, default=0.85)
    ap.add_argument("--temperature", type=float, default=0.04)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-writers", type=int, default=0)
    ap.add_argument("--refine-mode", choices=["onnx", "cloze", "none"], default="onnx")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--dump")
    ap.add_argument("--curve-out", help="ページ番号ごとの集計を JSON で保存(リポジトリの外)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    fields.MEMORY_BONUS = (
        args.mem_bonus
    )  # 既定引数は定義時に束縛されるので、各解決関数に渡す形で上書きする
    for fn in (
        fields.resolve_date,
        fields.resolve_amount,
        fields.resolve_closed,
        fields.resolve_description,
    ):
        d = list(fn.__defaults__ or ())
        # 末尾の引数 mem_bonus の既定値を差し替える
        d[-1] = args.mem_bonus
        fn.__defaults__ = tuple(d)
    engine = ParseqEngine(refine_mode=args.refine_mode)
    clf = CellTypeClassifier.load(args.classifier)
    cal = ConfidenceModel.load(args.confidence) if args.confidence else None
    rec = PageRecognizer(engine, clf, calibrator=cal)
    pages = [load_page(p) for p in list_pages(args.data)]
    by_writer: dict[int, list] = defaultdict(list)
    for pg in pages:
        by_writer[pg.meta["writer_id"]].append(pg)
    writers = sorted(by_writer)[: args.max_writers or None]
    dump = None
    if args.dump:
        assert_safe_output_dir(Path(args.dump).parent)
        dump = open(args.dump, "w", encoding="utf-8")  # noqa: SIM115
    stats: dict[int, list] = defaultdict(list)  # ページ番号 → Judged のリスト(全書き手)
    for w in writers:
        seq = sorted(by_writer[w], key=lambda p: p.meta["seed"])
        ctx = JobContext(
            accounts=list(ACCOUNTS),
            titles=list(TITLES),
            memory=RecognitionMemory(args.k, args.sim_threshold, args.temperature)
            if args.memory
            else None,
        )
        for idx, pg in enumerate(seq):
            inp = from_page_data(pg)
            res = rec.recognize_page(inp, ctx)
            page_review = pg.meta["status"] != "ok"
            for r, c in zip(res.cells, pg.cells, strict=True):
                j = judge(r, c, page_review)
                stats[idx].append(j)
                if dump:
                    dump.write(json.dumps(dump_row(pg.meta, c, r, j), ensure_ascii=False) + "\\n")
            confirm_page(
                ctx,
                res,
                [
                    ConfirmedCell(c.row, c.col, c.column, c.kind, c.raw, c.resolved)
                    for c in pg.cells
                ],
            )
            if not args.dictionary:
                ctx.dictionary.clear()
    if dump:
        dump.close()
    print(
        f"設定: memory={args.memory} dictionary={args.dictionary} bonus={args.mem_bonus} "
        f"sim>={args.sim_threshold} confidence={'較正済み' if cal else '暫定'} 書き手 {len(writers)} 人"
    )
    print(
        f"{'ページ':>6s} {'n頁':>4s} {'正解率(nb)':>10s} {'要確認率(nb)':>12s} {'見逃し%(全)':>11s} {'金額見逃し%':>11s} {'金額要確認':>10s}"
    )
    curve = []
    for idx in sorted(stats):
        js = stats[idx]
        nb = [j for j in js if j.gt_kind != "blank"]
        am = [j for j in js if j.column == "amount"]
        row = {
            "page": idx + 1,
            "n_pages": len(js) // 116,
            "acc_nb": float(np.mean([j.correct for j in nb])),
            "review_nb": float(np.mean([j.review for j in nb])),
            "miss_all": float(np.mean([j.miss for j in js])),
            "amount_miss": float(np.mean([j.miss for j in am])),
            "amount_review": float(np.mean([j.review for j in am if j.gt_kind != "blank"])),
        }
        curve.append(row)
        print(
            f"{row['page']:6d} {row['n_pages']:4d} {row['acc_nb']:10.3f} {row['review_nb']:12.3f} "
            f"{100 * row['miss_all']:11.2f} {100 * row['amount_miss']:11.2f} {row['amount_review']:10.3f}"
        )
    if args.curve_out:
        out = Path(args.curve_out)
        assert_safe_output_dir(out.parent)
        out.write_text(json.dumps(curve, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
