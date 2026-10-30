#!/usr/bin/env python3
"""合成データセットで認識(R1 + 後処理)を評価する。

    python scripts/eval_recognize.py --data DIR [DIR ...] [--classifier PATH | --oracle-kinds]
        [--vocab cold|warm] [--limit N] [--dump OUT.jsonl]

- --oracle-kinds: 種別(blank/ditto/arrow/wave/text)に正解を与え、R1 以降だけの精度を測る(切り分け用)。
- --classifier: 学習済みのセル種別分類器(train_celltype.py の出力)を使う(端から端までの評価)。
- --vocab: 摘要の辞書。cold = 空(初回)/ warm = 合成の語彙をすべて既知とした場合(学習が進んだ後の目安)。
- --dump: セルごとの特徴量・正誤を JSON Lines で保存(確信度の較正に使う。リポジトリの外に)。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from yayoi_ocr.config import assert_safe_output_dir
from yayoi_ocr.recognize.celltype import CellTypeClassifier
from yayoi_ocr.recognize.confidence import ConfidenceModel
from yayoi_ocr.recognize.evaluate import format_table, judge, summarize
from yayoi_ocr.recognize.inputs import from_page_data
from yayoi_ocr.recognize.page import JobContext, PageRecognizer
from yayoi_ocr.recognize.parseq.engine import ParseqEngine
from yayoi_ocr.synth.content import ACCOUNTS, DESCRIPTIONS, LONG_DESCRIPTIONS, TITLES
from yayoi_ocr.synth.dataset import list_pages, load_page


def make_context(vocab: str) -> JobContext:
    ctx = JobContext(accounts=list(ACCOUNTS), titles=list(TITLES))
    if vocab == "warm":
        # 合成の語彙(摘要)をすべて既知の確定値とみなす(月の数字を含む語は {m} を展開)
        d: dict[str, int] = {}
        for t in DESCRIPTIONS:
            for m in range(1, 13) if "{m}" in t else [0]:
                d[t.replace("{m}", str(m))] = 1
        for _, full, _ in LONG_DESCRIPTIONS:
            d[full] = 1
        ctx.dictionary = d
    return ctx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--classifier")
    ap.add_argument("--oracle-kinds", action="store_true")
    ap.add_argument("--vocab", choices=["cold", "warm"], default="cold")
    ap.add_argument("--limit", type=int, default=0, help="データセットごとの最大ページ数")
    ap.add_argument("--dump")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--confidence", help="fit_confidence.py の出力(較正した確信度モデルの JSON)")
    ap.add_argument(
        "--refine-mode",
        choices=["onnx", "cloze", "none"],
        default="onnx",
        help="PARSeq の推論モード: onnx(配布版)/ cloze(元実装)/ none(自己回帰のみ)",
    )
    args = ap.parse_args()

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    engine = ParseqEngine(refine_mode=args.refine_mode)
    clf = CellTypeClassifier.load(args.classifier) if args.classifier else None
    calibrator = ConfidenceModel.load(args.confidence) if args.confidence else None
    rec = PageRecognizer(engine, clf, calibrator=calibrator)
    ctx = make_context(args.vocab)
    dump = None
    if args.dump:
        assert_safe_output_dir(Path(args.dump).parent)
        dump = open(args.dump, "w", encoding="utf-8")  # noqa: SIM115
    for d in args.data:
        pages = list_pages(d)
        if args.limit:
            pages = pages[: args.limit]
        rows, t_rec, n_page_review = [], 0.0, 0
        by_font: dict[str, list] = defaultdict(list)
        for p in pages:
            page = load_page(p)
            inp = from_page_data(page)
            oracle = [c.kind for c in page.cells] if args.oracle_kinds else None
            t = time.time()
            res = rec.recognize_page(inp, ctx, oracle)
            t_rec += time.time() - t
            page_review = page.meta["status"] != "ok"
            n_page_review += int(page_review)
            for r, c in zip(res.cells, page.cells, strict=True):
                j = judge(r, c, page_review)
                rows.append(j)
                by_font[page.meta["font"]].append(j)
                if dump:
                    dump.write(
                        json.dumps(
                            {
                                "page": page.meta["seed"],
                                "writer": page.meta["writer_id"],
                                "row": c.row,
                                "col": c.col,
                                "column": c.column,
                                "gt_kind": j.gt_kind,
                                "pred_kind": j.pred_kind,
                                "correct": j.correct,
                                "page_review": page_review,
                                "features": r.features,
                                "cls_probs": r.cls_probs,
                                "reasons": r.reasons,
                                "iou": c.iou,
                                "faint": c.faint_only,
                                "overflow_gt": c.overflow_gt,
                                "stray": c.stray,
                                "annotation": bool(c.annotation),
                                "dpi": page.meta["dpi"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        print(
            f"\n===== {d}  pages={len(pages)} (レイアウト要確認ページ {n_page_review}) "
            f"認識 {t_rec / max(1, len(pages)):.1f}s/ページ  classifier="
            f"{'oracle' if args.oracle_kinds else 'trained' if clf else 'phase1'} vocab={args.vocab} ====="
        )
        print(format_table(summarize(rows, None), "all"))
        cols = {}
        for col in ("date", "account", "description", "amount"):
            cols[col] = summarize([j for j in rows if j.column == col])["all"]
        print(format_table(cols, "column"))
        print(format_table(summarize(rows, "gt_kind"), "gt_kind"))
        # 要確認でないセルのうち間違っているもの(見逃し)の内訳
        miss = [j for j in rows if j.miss]
        if miss:
            cnt: dict[str, int] = defaultdict(int)
            for j in miss:
                cnt[f"{j.column}/{j.gt_kind}->{j.pred_kind}"] += 1
            print("見逃しの内訳:", dict(sorted(cnt.items(), key=lambda kv: -kv[1])[:12]))
    if dump:
        dump.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
