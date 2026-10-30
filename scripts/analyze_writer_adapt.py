#!/usr/bin/env python3
"""書き手への追加学習の曲線(writer_adapt_curve.py の出力)を集計する。

    python scripts/analyze_writer_adapt.py EVAL_DIR [--calib-writers 6] [--margin 0.6] [--amount-margin 0.4]

書き手を較正用(A)と評価用(B)に分け(ID の昇順で先頭 --calib-writers 人が A)、N ごとに:
 ① 値の正解率(空欄を除く、列別)
 ② 要確認率(空欄を除く、列別)と見逃し率(全セルに対する、誤り ∧ 要確認なし):
    - 初期の較正のまま(writer_adapt_curve.py が記録した review)
    - 再較正(A の書き手の N ページ後の未使用ページで確信度モデルを作り直し、B に適用。Clopper-Pearson の上側限界で目標を守る)
    - 理想の再較正(同じ A のデータで、経験的な見逃し率が目標以下になる最小の要確認率。標本が少ないための保守性を除いた下限)
 ③ 忘却(--forget のデータがあれば)と、学習時間。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from yayoi_ocr.recognize.confidence import (
    fit_confidence,
    group_of,
    samples_from_dump,
)

COLUMNS = ("date", "account", "description", "amount")
TARGET = {"date": 0.005, "account": 0.005, "description": 0.005, "amount": 0.001}


def load(out_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(out_dir.glob("wa_*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh]
    return rows


def review_with(model, samples, rows) -> list[bool]:
    out = []
    for s in samples:
        g = model.groups.get(group_of(s.column, s.kind))
        out.append(bool(s.hard or g is None or g.prob(s.x) < g.threshold))
    return out


def empirical_threshold_review(p_a, c_a, n_cells_a, target, p_b) -> np.ndarray:
    """A の(自由なセルの)確率 p と正誤 c から、経験的な見逃し率 ≦ 目標になる最小のしきい値を選び、B の要確認(確率 < しきい値)を返す。"""
    order = np.argsort(-p_a)
    wrong_cum = np.cumsum(~c_a[order])
    ok = np.where(wrong_cum / max(n_cells_a, 1) <= target)[0]
    if len(ok) == 0:
        return np.ones(len(p_b), dtype=bool)
    # ok[-1] 番目までを要確認にしない(それより確率の低いものは要確認)
    thr = p_a[order][ok[-1]]
    return p_b < thr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--calib-writers", type=int, default=6)
    ap.add_argument("--margin", type=float, default=0.6)
    ap.add_argument("--amount-margin", type=float, default=0.4)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    rows = load(out_dir)
    own = [r for r in rows if r["set"] == "own"]
    writers = sorted({r["writer"] for r in own})
    wa = set(writers[: args.calib_writers])
    wb = set(writers[args.calib_writers :])
    levels = sorted({r["N"] for r in own})
    print(f"書き手 {len(writers)} 人(較正 A: {len(wa)} 人 / 評価 B: {len(wb)} 人)、N = {levels}")
    targets = {
        c: (args.amount_margin if c == "amount" else args.margin) * TARGET[c] for c in COLUMNS
    }

    def pick(n, ws):
        return [r for r in own if r["N"] == n and r["writer"] in ws]

    # 初期較正のまま(記録された review)/ A で再較正した CP 版 / 理想(経験的)版
    table: dict[int, dict] = {}
    for n in levels:
        a_rows, b_rows = pick(n, wa), pick(n, wb)
        a_samples, b_samples = samples_from_dump(a_rows), samples_from_dump(b_rows)
        n_cells_a = Counter(r["column"] for r in a_rows)
        model = fit_confidence(a_samples, dict(n_cells_a), targets, alpha=0.05)
        rev_cp = review_with(model, b_samples, b_rows)
        res: dict[str, dict] = {}
        for col in COLUMNS:
            idx = [i for i, r in enumerate(b_rows) if r["column"] == col]
            nb = [i for i in idx if b_rows[i]["gt_kind"] != "blank"]
            rb = [b_rows[i] for i in idx]
            acc = np.mean([b_rows[i]["correct"] for i in nb]) if nb else float("nan")
            asis_rev = np.mean([b_rows[i]["review"] for i in nb])
            asis_miss = sum((not b_rows[i]["correct"]) and (not b_rows[i]["review"]) for i in idx)
            cp_rev = np.mean([rev_cp[i] for i in nb])
            cp_miss = sum((not b_rows[i]["correct"]) and (not rev_cp[i]) for i in idx)
            # 理想: 列ごとに A のデータでしきい値を決める
            a_idx = [i for i, r in enumerate(a_rows) if r["column"] == col]
            p_a, c_a, p_b = [], [], []
            for i in a_idx:
                s = a_samples[i]
                if s.hard:
                    continue
                g = model.groups.get(group_of(s.column, s.kind))
                p_a.append(g.prob(s.x) if g and g.lr else (g.const if g else 0.0))
                c_a.append(s.correct)
            for i in idx:
                s = b_samples[i]
                g = model.groups.get(group_of(s.column, s.kind))
                p_b.append(
                    -1.0 if s.hard else (g.prob(s.x) if g and g.lr else (g.const if g else 0.0))
                )
            ideal = empirical_threshold_review(
                np.array(p_a), np.array(c_a, dtype=bool), len(a_idx), targets[col], np.array(p_b)
            )
            ideal_rev = np.mean([ideal[idx.index(i)] for i in nb])
            ideal_miss = sum((not b_rows[i]["correct"]) and (not ideal[idx.index(i)]) for i in idx)
            res[col] = {
                "acc": float(acc),
                "asis_rev": float(asis_rev),
                "asis_miss": asis_miss,
                "cp_rev": float(cp_rev),
                "cp_miss": cp_miss,
                "ideal_rev": float(ideal_rev),
                "ideal_miss": ideal_miss,
                "cells": len(rb),
            }
        table[n] = res
    for col in COLUMNS:
        print(
            f"\n=== {col}  (評価書き手 B の未使用ページ。見逃しは件数/全セル, 目標 {100 * TARGET[col]:.1f}%)"
        )
        print(
            f"{'N':>3s} {'正解率':>7s} | {'要確認率(初期較正)':>16s} {'見逃し':>11s} | {'要確認率(再較正CP)':>16s} {'見逃し':>11s} | {'要確認率(理想)':>14s} {'見逃し':>11s}"
        )
        for n in levels:
            r = table[n][col]
            print(
                f"{n:3d} {r['acc']:7.3f} | {r['asis_rev']:16.3f} {r['asis_miss']:4d}/{r['cells']:<5d} | "
                f"{r['cp_rev']:16.3f} {r['cp_miss']:4d}/{r['cells']:<5d} | {r['ideal_rev']:14.3f} {r['ideal_miss']:4d}/{r['cells']:<5d}"
            )
    # 全列
    print("\n=== 全列 (B)")
    for n in levels:
        b = pick(n, wb)
        nb = [r for r in b if r["gt_kind"] != "blank"]
        print(
            f"N={n:2d}: 値の正解率 {np.mean([r['correct'] for r in nb]):.3f}  要確認率(初期較正) {np.mean([r['review'] for r in nb]):.3f}  "
            f"見逃し(初期較正) {sum((not r['correct']) and (not r['review']) for r in b)}/{len(b)} = "
            f"{100 * sum((not r['correct']) and (not r['review']) for r in b) / len(b):.3f}%"
        )
    # 忘却
    forget = [r for r in rows if r["set"].startswith("forget:")]
    if forget:
        print("\n=== 忘却(他の書き手・一般語彙のページでの値の正解率、空欄を除く)")
        sets = sorted({r["set"] for r in forget})
        for s in sets:
            line = f"{s:28s}"
            for n in sorted({r["N"] for r in forget if r["set"] == s}):
                rs = [
                    r for r in forget if r["set"] == s and r["N"] == n and r["gt_kind"] != "blank"
                ]
                line += f"  N={n}: {np.mean([r['correct'] for r in rs]):.3f} (n={len(rs)})"
                for col in ("description",):
                    cs = [r for r in rs if r["column"] == col]
                    line += f" 摘要 {np.mean([r['correct'] for r in cs]):.3f}"
            print(line)
    # 学習時間
    tm: dict[int, list[float]] = defaultdict(list)
    for f in sorted(out_dir.glob("timing_*.json")):
        for t in json.loads(f.read_text(encoding="utf-8")):
            tm[t["N"]].append(sum(v["seconds"] for v in t["train"].values()))
    if tm:
        print(
            "\n=== 学習時間(幅 256 + 幅 384 の合計、1 スレッド換算ではなく実行時のスレッド数のまま)"
        )
        for n in sorted(tm):
            print(
                f"N={n:2d}: 平均 {np.mean(tm[n]):.0f} 秒 (最大 {np.max(tm[n]):.0f} 秒, {len(tm[n])} 回)"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
