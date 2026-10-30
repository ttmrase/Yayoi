#!/usr/bin/env python3
"""合成帳票を生成 → レイアウト解析 → セル単位のデータセット(.npz)を作る。

    python scripts/build_synth_cells.py --out DIR --n 200 --preset typical \
        [--seed0 0] [--writer0 0] [--jobs 4] [--font-exclude yusei klee]

- 出力先は **リポジトリの外**(例: $YAYOI_PRIVATE_DATA/datasets/train)。実データは使わない(合成のみ)。
- 学習用と評価用は書き手 ID の範囲(--writer0)とフォント(--font-include / --font-exclude)を分ける。
  例) 学習: --writer0 0 --font-exclude yusei klee / 評価: --writer0 100000 (別の書き手)、
      未知フォント評価: --writer0 200000 --font-include yusei klee
- --preset: clean / typical / default / 劣化名(rotation perspective curve lighting contrast blur noise jpeg lowres cut)
- 既にあるページはスキップする(中断・再開できる)。
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from yayoi_ocr.config import assert_safe_output_dir
from yayoi_ocr.synth.dataset import build_page, preset_config, save_page


def _one(task: tuple) -> tuple[int, float, str]:
    out, name, seed, writer, kw = task
    path = Path(out) / f"page_{seed:07d}.npz"
    if path.exists():
        return seed, 0.0, "skip"
    t = time.time()
    cfg = preset_config(name, seed=seed, writer_id=writer, **kw)
    page = build_page(cfg)
    save_page(path, page)
    return seed, time.time() - t, page.meta["status"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--preset", default="typical")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--writer0", type=int, default=0)
    ap.add_argument(
        "--writers", type=int, default=0, help="書き手の人数(0 = ページごとに別の書き手)"
    )
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--fill-rows", type=int, default=None)
    ap.add_argument("--vocab", default="base", choices=["base", "general-train", "general-eval"])
    ap.add_argument(
        "--strength-range",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        help="書き手の崩しの倍率をページごとに [LO, HI] の一様乱数で決める(学習用に 0.5〜1.0 を混ぜる)",
    )
    ap.add_argument(
        "--writer-strength", type=float, default=1.0, help="書き手の崩しの倍率(1.0 = 標準)"
    )
    ap.add_argument("--font-include", nargs="*", default=None)
    ap.add_argument("--font-exclude", nargs="*", default=[])
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    out = assert_safe_output_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kw: dict = {"degrade_strength": args.strength}
    if args.writer_strength != 1.0:
        kw["writer_strength"] = args.writer_strength
    if args.vocab != "base":
        kw["vocab"] = args.vocab
    if args.fill_rows:
        kw["fill_rows"] = args.fill_rows
    if args.font_include:
        kw["font_include"] = tuple(args.font_include)
    if args.font_exclude:
        kw["font_exclude"] = tuple(args.font_exclude)
    tasks = []
    for i in range(args.n):
        w = args.writer0 + (i % args.writers if args.writers else i)
        kwi = dict(kw)
        if args.strength_range:
            lo, hi = args.strength_range
            kwi["writer_strength"] = (
                lo + (hi - lo) * (((args.seed0 + i) * 2654435761) % 1000) / 1000.0
            )
        tasks.append((str(out), args.preset, args.seed0 + i, w, kwi))
    t0 = time.time()
    done = 0
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            it = ex.map(_one, tasks)
            for _seed, _dt, st in it:
                done += 1
                if done % 10 == 0 or done == len(tasks):
                    print(f"{done}/{len(tasks)} ({time.time() - t0:.0f}s) last={st}", flush=True)
    else:
        for t in tasks:
            seed, dt, st = _one(t)
            done += 1
            print(f"{done}/{len(tasks)} seed={seed} {dt:.1f}s {st}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
