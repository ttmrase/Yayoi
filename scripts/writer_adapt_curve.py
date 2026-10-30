#!/usr/bin/env python3
"""書き手への追加学習の「実現性の検証」: 確定ページ数 N を増やしたときの精度の曲線(Phase 4 の先行検証)。

    python scripts/writer_adapt_curve.py --data DIR --classifier CLF.pt --models-dir MODELS/adapted \\
        --confidence MODELS/adapted/confidence.json --out EVAL_DIR [--levels 1 3 5 10 20] [--shard 0/4]

DIR は書き手ごとの連続ページ(build_synth_cells.py --writers)。各書き手について、先頭の --train-pages ページを「確定済みページ」の候補、
続く --eval-pages ページを「同じ書き手の未使用ページ」として:
  N = 0(全体の適応済み重みのまま)と、N ∈ --levels のそれぞれで、先頭 N ページのその書き手のセルだけで追加学習(日付・科目・金額 = 幅 256、
  摘要 = 幅 384。L2-SP つき)→ 未使用ページを認識 → セルごとの特徴量・正誤・要確認を JSON Lines に保存する。
学習は N ごとに全体の適応済み重みからやり直す(履歴の累積ではなく「N ページで 1 回学習」した場合の曲線)。
--forget-data があれば、追加学習した書き手の重みで他の書き手・一般語彙のページも認識し、忘却(劣化)を測る。
出力はリポジトリの外に置くこと(実データではなく合成データだが、評価画像相当のため)。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from yayoi_ocr.config import assert_safe_output_dir
from yayoi_ocr.recognize.celltype import CellTypeClassifier
from yayoi_ocr.recognize.confidence import ConfidenceModel
from yayoi_ocr.recognize.evaluate import dump_row, judge
from yayoi_ocr.recognize.grammar import (
    amount_surface_priors,
    date_surfaces,
    norm_text,
    parse_date,
)
from yayoi_ocr.recognize.inputs import from_page_data
from yayoi_ocr.recognize.page import JobContext, PageRecognizer
from yayoi_ocr.recognize.parseq.engine import MODEL_FILES, ParseqEngine
from yayoi_ocr.recognize.parseq.finetune import FinetuneConfig, finetune
from yayoi_ocr.recognize.parseq.weights import load_converted
from yayoi_ocr.synth.content import ACCOUNTS, TITLES
from yayoi_ocr.synth.dataset import list_pages, load_page

WIDTH_COLUMNS = {256: ("date", "account", "amount"), 384: ("description",)}


def label_candidates(column: str, resolved: str) -> list[str]:
    """確定値(CSV に出す値)と整合する、紙への書き方の集合。確認画面は書かれた通りの文字列を聞かないので、学習側で書き方を推定する。"""
    if column == "date":
        v = parse_date(resolved)
        if v is None:
            return [resolved]
        return date_surfaces(v) + date_surfaces(v.with_year(None))  # 年が書かれていない場合も含む
    if column == "amount" and resolved.isdigit():
        return list(amount_surface_priors(int(resolved)))
    return [norm_text(resolved)]


def samples_from_pages(
    pages, columns, label_mode: str = "written", engine: ParseqEngine | None = None
) -> list[tuple[np.ndarray, str]]:
    """確定ページのセル → (認識用画像, 確定テキスト)。text セルだけ(〃・矢印・波線は除く)。

    label_mode = written: 書かれた通りの正解(理想。確認画面では得られない)。
    label_mode = resolved: 確定値だけが分かる想定。確定値と整合する書き方のうち、今の重みで最も尤度の高いものを正解とする(自己ラベル)。
    """
    out = []
    for pg in pages:
        for i, c in enumerate(pg.cells):
            if c.kind != "text" or c.column not in columns or i not in pg.rec or not c.raw:
                continue
            if label_mode == "written":
                out.append((pg.rec[i], norm_text(c.raw)))
                continue
            cands = label_candidates(c.column, c.resolved)
            if len(cands) > 1 and engine is not None:
                crop = np.stack([pg.rec[i]] * 3, axis=-1)
                scores = engine.score_many(crop, cands, c.column)
                cands = [max(zip(cands, scores, strict=True), key=lambda t: t[1].logprob)[0]]
            out.append((pg.rec[i], cands[0]))
    return out


def recognize_pages(rec: PageRecognizer, pages, tag: dict, sink, set_name: str = "own") -> None:
    for pg in pages:
        ctx = JobContext(accounts=list(ACCOUNTS), titles=list(TITLES))
        res = rec.recognize_page(from_page_data(pg), ctx)
        page_review = pg.meta["status"] != "ok"
        for r, c in zip(res.cells, pg.cells, strict=True):
            j = judge(r, c, page_review)
            sink.write(
                json.dumps(
                    {**dump_row(pg.meta, c, r, j), **tag, "set": set_name}, ensure_ascii=False
                )
                + "\n"
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--models-dir", required=True, help="全体の適応済み重み(出発点)")
    ap.add_argument("--confidence", required=True, help="全体の確信度モデル(要確認の判定に使う)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    ap.add_argument("--train-pages", type=int, default=20)
    ap.add_argument("--eval-pages", type=int, default=12)
    ap.add_argument("--shard", default="0/1", help="i/k: 書き手を k 分割したうちの i 番目")
    ap.add_argument("--max-writers", type=int, default=0)
    ap.add_argument(
        "--only-writers", type=int, nargs="*", default=[], help="この書き手 ID だけ処理する"
    )
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--l2sp", type=float, default=0.1)
    ap.add_argument("--frozen-blocks", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument(
        "--steps-samples",
        type=int,
        default=2000,
        help="エポック数 = clamp(この値 / 学習セル数, 3, 40)。少数のセルでも十分な更新回数にする",
    )
    ap.add_argument(
        "--label-mode",
        choices=["written", "resolved"],
        default="written",
        help="written = 書かれた通りの正解で学習(理想)/ resolved = 確定値だけから書き方を自己ラベル(確認画面で得られる情報だけ)",
    )
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--forget-data", nargs="*", default=[], help="忘却の測定に使うデータセット")
    ap.add_argument("--forget-limit", type=int, default=15)
    ap.add_argument("--forget-levels", type=int, nargs="*", default=[])
    ap.add_argument(
        "--forget-writers", type=int, default=2, help="シャードあたりの忘却を測る書き手の数"
    )
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out_dir = assert_safe_output_dir(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    i_shard, k_shard = (int(x) for x in args.shard.split("/"))

    clf = CellTypeClassifier.load(args.classifier)
    cal = ConfidenceModel.load(args.confidence)
    gdir = Path(args.models_dir)

    def global_models() -> tuple[dict[int, object], str]:
        ms, charset = {}, ""
        for w in (256, 384):
            m, charset, _ = load_converted(gdir / MODEL_FILES[w])
            ms[w] = m
        return ms, charset

    def writer_of(path: Path) -> int:
        with np.load(path, allow_pickle=False) as z:  # meta だけ読む(ページ全体は読み込まない)
            return int(json.loads(str(z["meta"]))["page"]["writer_id"])

    paths_by_writer: dict[int, list[Path]] = defaultdict(list)
    for path in list_pages(args.data):
        paths_by_writer[writer_of(path)].append(path)
    writers = sorted(paths_by_writer)[i_shard::k_shard]
    if args.only_writers:
        writers = [w for w in sorted(paths_by_writer) if w in set(args.only_writers)]
    if args.max_writers:
        writers = writers[: args.max_writers]
    by_writer = {
        w: [load_page(p) for p in paths_by_writer[w]] for w in writers
    }  # 担当の書き手のページだけ
    forget_pages = {
        Path(d).name: [load_page(p) for p in list_pages(d)[: args.forget_limit]]
        for d in args.forget_data
    }
    timing: list[dict] = []
    for wi, w in enumerate(writers):
        pages = sorted(by_writer[w], key=lambda p: p.meta["seed"])
        train_pool = pages[: args.train_pages]
        eval_pages = pages[args.train_pages : args.train_pages + args.eval_pages]
        if len(eval_pages) < args.eval_pages:
            print(f"書き手 {w}: ページ不足のためスキップ", flush=True)
            continue
        with open(out_dir / f"wa_{w}.jsonl", "w", encoding="utf-8") as sink:
            ms, charset = global_models()
            eng0 = ParseqEngine.from_models(ms, charset)
            rec0 = PageRecognizer(eng0, clf, calibrator=cal)
            recognize_pages(rec0, eval_pages, {"N": 0, "writer_set": w}, sink)
            if (
                wi == 0 and i_shard == 0 and forget_pages
            ):  # N=0 の忘却ベースラインは全体の重みで 1 回だけ
                for name, fp in forget_pages.items():
                    recognize_pages(rec0, fp, {"N": 0, "writer_set": w}, sink, f"forget:{name}")
            for n in args.levels:
                ms, charset = global_models()
                rec_t = {}
                for width, columns in WIDTH_COLUMNS.items():
                    samples = samples_from_pages(train_pool[:n], columns, args.label_mode, eng0)
                    if not samples:
                        continue
                    epochs = int(np.clip(round(args.steps_samples / len(samples)), 3, 40))
                    cfg = FinetuneConfig(
                        width=width,
                        epochs=epochs,
                        batch=args.batch,
                        lr=args.lr,
                        frozen_blocks=args.frozen_blocks,
                        l2sp=args.l2sp,
                        seed=w % 1000 + n,
                    )
                    t0 = time.time()
                    finetune(ms[width], charset, samples, cfg, log=lambda s: None)
                    rec_t[width] = {
                        "samples": len(samples),
                        "epochs": epochs,
                        "seconds": time.time() - t0,
                    }
                eng = ParseqEngine.from_models(ms, charset)
                rec = PageRecognizer(eng, clf, calibrator=cal)
                recognize_pages(rec, eval_pages, {"N": n, "writer_set": w}, sink)
                if wi < args.forget_writers and n in args.forget_levels:
                    for name, fp in forget_pages.items():
                        recognize_pages(rec, fp, {"N": n, "writer_set": w}, sink, f"forget:{name}")
                timing.append({"writer": w, "N": n, "threads": args.threads, "train": rec_t})
                print(f"書き手 {w} N={n}: {rec_t}", flush=True)
                sink.flush()
    (out_dir / f"timing_{i_shard}.json").write_text(json.dumps(timing, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
