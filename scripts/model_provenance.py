#!/usr/bin/env python3
"""同梱する重みの出所の検査(docs/model_provenance.md の根拠を機械的に確かめる。開発用)。

    python scripts/model_provenance.py [--models DIR] [--datasets DIR] [--reconvert] [--markdown]

検査:
  C1 実サンプルを読む経路: src/・scripts/・tests/ のうち、実サンプルの置き場(`samples_dir`)を参照するのは
     config.py の定義と tests/test_samples.py(レイアウトの構造チェック。内容は読まない)だけ。
  C2 学習・較正に使った合成データセットの全ページが、生成器の署名(seed・writer_id・OFL フォント名)を持つ。
  C3 データセット同士で seed と書き手 ID の範囲が重ならない(学習・較正・評価の分離)。
  C4 同梱する全ファイルのバイト列に、拒否リスト(実データの語・数値)が含まれない。語は出力しない(番号と件数のみ)。
  C5 --reconvert: NDLOCR-Lite の ONNX から基本の重みを作り直し、SHA256 が同梱のものと一致する(変換の再現性)。
出力: 同梱ファイルの SHA256、データセットの要約(--markdown で docs 用の表)。失敗があれば終了コード 1。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from yayoi_ocr.config import models_dir, private_data_dir  # noqa: E402

# 同梱する重み(make_trial_bundle.py の許可リストと同じ)。base 直下と adapted/
BASE_FILES = [
    "NDLmoji.yaml",
    "celltype.pt",
    "confidence.json",
    "parseq-ndl-24x256-30-tiny.pt",
    "parseq-ndl-24x384-50-tiny.pt",
]
ADAPTED_FILES = [
    "confidence.json",
    "parseq-ndl-24x256-30-tiny.pt",
    "parseq-ndl-24x384-50-tiny.pt",
]
# 学習・較正に使ったデータセット(役割は docs/model_provenance.md の表)
USED_DATASETS = ["train", "train_gen", "calib", "calib500", "calib_gen"]
# 実サンプルの置き場を参照してよいファイル
SAMPLES_ALLOWED = {"src/yayoi_ocr/config.py", "tests/test_samples.py"}
# この検査自身(パターンの文字列を含む)
SELF = {"scripts/model_provenance.py", "tests/test_provenance_paths.py"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bundled(models: Path) -> list[Path]:
    return [models / n for n in BASE_FILES] + [models / "adapted" / n for n in ADAPTED_FILES]


# ---- C1
def check_samples_paths() -> list[str]:
    """実サンプルの置き場(samples_dir / "…/samples")を参照するファイルが、許可リストだけであること。"""
    pat = re.compile(
        r"samples_dir|private[^\n]{0,40}['\"/]samples['\"/]|YAYOI_PRIVATE_DATA/samples"
    )
    bad: list[str] = []
    for top in ("src", "scripts", "tests"):
        for p in (ROOT / top).rglob("*.py"):
            rel = p.relative_to(ROOT).as_posix()
            if rel in SAMPLES_ALLOWED or rel in SELF:
                continue
            if pat.search(p.read_text(encoding="utf-8", errors="replace")):
                bad.append(rel)
    return bad


# ---- C2 / C3
def dataset_summary(root: Path, name: str, fonts: set[str]) -> dict:
    files = sorted((root / name).glob("*.npz"))
    seeds, writers, used_fonts, cells, bad = [], [], Counter(), 0, []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
        page = meta.get("page", {})
        ok = (
            isinstance(page.get("seed"), int)
            and isinstance(page.get("writer_id"), int)
            and page.get("font") in fonts
        )
        if not ok:
            bad.append(f.name)
            continue
        seeds.append(page["seed"])
        writers.append(page["writer_id"])
        used_fonts[page["font"]] += 1
        cells += len(meta.get("cells", []))
    return {
        "name": name,
        "pages": len(files),
        "bad": bad,
        "seeds": (min(seeds), max(seeds)) if seeds else None,
        "seed_set": set(seeds),
        "writers": (min(writers), max(writers)) if writers else None,
        "writer_set": set(writers),
        "n_writers": len(set(writers)),
        "fonts": dict(used_fonts),
        "cells": cells,
    }


def seed_runs(seeds: set[int]) -> list[tuple[int, int]]:
    s = sorted(seeds)
    runs, start, prev = [], s[0], s[0]
    for v in s[1:]:
        if v != prev + 1:
            runs.append((start, prev))
            start = v
        prev = v
    runs.append((start, prev))
    return runs


# ---- C4
def _load_denylist():
    spec = importlib.util.spec_from_file_location(
        "check_no_private_data", ROOT / "scripts" / "check_no_private_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, mod.load_denylist()


MIN_TENSOR_PATTERN = (
    6  # 重みのテンソル(生のバイト列)で照合する語の最短バイト数。短い語は偶然の一致が避けられない
)


def _members(path: Path) -> list[tuple[str, bytes, bool]]:
    """(名前, 中身, テンソルの生データか)。.pt は zip で、`…/data/<番号>` がテンソルの生データ。"""
    if path.suffix == ".pt":
        out = []
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                out.append((n, z.read(n), "/data/" in n))
        return out
    return [(path.name, path.read_bytes(), False)]


def scan_denylist(files: list[Path]) -> dict[str, list[str]]:
    """全ファイルを、拒否リストの語(UTF-8・UTF-16LE・cp932)で走査する。

    - テンソルの生データ(.pt の data/)は、`MIN_TENSOR_PATTERN` バイト以上のパターンだけ照合する(3 文字の ASCII の語などは、
      35 MB の浮動小数点の列に偶然一致する。NDL 配布の重みを変換しただけのファイルにも一致することを確認済み)。
    - それ以外(pickle のメタデータ・json・yaml)は全部の語で照合し、テキストは数値も検査する。
    """
    mod, deny = _load_denylist()
    if deny is None:
        raise SystemExit(
            "拒否リストがありません($YAYOI_PRIVATE_DATA/denylist.txt)。C4 を検査できません"
        )
    words, _numbers = deny
    pats: list[tuple[int, list[bytes]]] = []
    for i, w in enumerate(words):
        variants = {w.encode("utf-8"), w.encode("utf-16le")}
        with contextlib.suppress(UnicodeEncodeError):
            variants.add(w.encode("cp932"))
        pats.append((i + 1, [v for v in variants if v]))
    out: dict[str, list[str]] = {}
    for f in files:
        hits: set[str] = set()
        for _name, data, is_tensor in _members(f):
            low = data.lower()
            for no, vs in pats:
                for v in vs:
                    if is_tensor and len(v) < MIN_TENSOR_PATTERN:
                        continue
                    if v in data or v in low:
                        hits.add(f"語 #{no}")
            if f.suffix in {".json", ".yaml", ".yml"}:
                hits.update(mod.check_denylist(data.decode("utf-8", errors="replace"), deny))
        key = f.name if f.parent.name != "adapted" else f"adapted/{f.name}"
        out[key] = sorted(hits)
    return out


# ---- C6
def scan_dataset_text(root: Path, names: list[str]) -> dict[str, dict[str, int]]:
    """学習・較正データ(合成ページ)の文字列(正解の raw・resolved・タイトル)が、拒否リストの語・数値を含まないこと。

    語は強い証拠(1 件でも失敗)。数値(4 桁以上の金額など)は、乱数の金額と偶然一致し得るので件数だけ報告する。
    """
    _mod, deny = _load_denylist()
    words, numbers = deny
    out: dict[str, dict[str, int]] = {}
    for name in names:
        w_hits = n_hits = 0
        for f in sorted((root / name).glob("*.npz")):
            with np.load(f, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
            texts = [str(meta.get("page", {}).get("title", ""))]
            for c in meta.get("cells", []):
                texts += [str(c.get("raw", "")), str(c.get("resolved", ""))]
            blob = "\n".join(texts)
            low = blob.lower()
            w_hits += sum(1 for w in words if w in low)
            n_hits += sum(1 for rx in numbers if rx.search(blob))
        out[name] = {"word_hits": w_hits, "number_hits_coincidence": n_hits}
    return out


def null_number_rate(n_pages: int = 2000, seed0: int = 500_000_000) -> tuple[float, int]:
    """比較用: 学習に使っていない新しい seed で内容(論理データ)だけを作り、同じ照合での数値の一致率(ページあたり)を測る。
    学習データの一致率がこれと同程度なら、数値の一致は乱数の金額の偶然で説明できる。"""
    from yayoi_ocr.synth.content import plan_page

    _mod, deny = _load_denylist()
    _words, numbers = deny
    hits = 0
    for i in range(n_pages):
        plan = plan_page(np.random.default_rng(seed0 + i))
        texts = [plan.title]
        for row in plan.rows:
            for c in row.values():
                texts += [c.raw, c.resolved]
        blob = "\n".join(texts)
        hits += sum(1 for rx in numbers if rx.search(blob))
    return hits / n_pages, hits


# ---- C5
def reconvert(models: Path) -> dict[str, bool]:
    """ONNX → state_dict の変換をやり直し、同梱の基本の重みと SHA256 が一致するか。"""
    res: dict[str, bool] = {}
    tmp = Path(tempfile.mkdtemp(prefix="prov_"))
    try:
        for p in [*models.glob("*.onnx"), models / "NDLmoji.yaml"]:
            shutil.copy2(p, tmp / p.name)
        code = (
            f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); sys.argv = ['x', '--dest', {str(tmp)!r}]; "
            f"exec(open({str(ROOT / 'scripts' / 'fetch_models.py')!r}, encoding='utf-8').read())"
        )
        subprocess.run([sys.executable, "-I", "-c", code], check=True, capture_output=True)
        for n in BASE_FILES:
            if n.endswith(".pt") and n.startswith("parseq"):
                res[n] = sha256(tmp / n) == sha256(models / n)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--models", type=Path, default=None)
    ap.add_argument("--datasets", type=Path, default=None)
    ap.add_argument(
        "--reconvert", action="store_true", help="C5: ONNX からの変換を再現して SHA256 を比べる"
    )
    ap.add_argument("--markdown", action="store_true", help="docs 用の表で出力する")
    args = ap.parse_args()
    models = args.models or models_dir()
    datasets = args.datasets or private_data_dir() / "datasets"
    failures: list[str] = []

    files = bundled(models)
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise SystemExit("同梱するファイルが無い: " + ", ".join(missing))

    print("## 同梱ファイルの SHA256\n")
    print("| ファイル | バイト数 | SHA256 |\n|---|---:|---|")
    for f in files:
        rel = f.relative_to(models).as_posix()
        print(f"| `{rel}` | {f.stat().st_size:,} | `{sha256(f)}` |")

    bad_paths = check_samples_paths()
    print(f"\n## C1 実サンプルの置き場を参照するファイル: {bad_paths or 'なし(許可リストのみ)'}")
    if bad_paths:
        failures.append("C1")

    fonts = {p.stem for p in (private_data_dir() / "fonts").glob("*.ttf")}
    sums = [dataset_summary(datasets, n, fonts) for n in USED_DATASETS]
    print("\n## C2 合成データセット(学習・較正に使ったもの)\n")
    print(
        "| データセット | ページ | セル | seed の範囲 | 書き手 ID の範囲(人数) | フォント(ページ数) |\n|---|---:|---:|---|---|---|"
    )
    for s in sums:
        runs = ", ".join(f"{a}–{b}" for a, b in seed_runs(s["seed_set"]))
        fnt = ", ".join(f"{k} {v}" for k, v in sorted(s["fonts"].items()))
        print(
            f"| `{s['name']}` | {s['pages']} | {s['cells']:,} | {runs} |"
            f" {s['writers'][0]}–{s['writers'][1]}({s['n_writers']}) | {fnt} |"
        )
        if s["bad"]:
            failures.append(f"C2:{s['name']}")
            print(f"  署名のないページ: {s['bad'][:5]}")
    print(
        f"\nOFL フォント(置き場の{len(fonts)} 書体)以外のフォント・署名なしのページ: "
        f"{sum(len(s['bad']) for s in sums)} ページ"
    )

    overlaps = []
    for i, a in enumerate(sums):
        for b in sums[i + 1 :]:
            if a["seed_set"] & b["seed_set"] or a["writer_set"] & b["writer_set"]:
                overlaps.append((a["name"], b["name"]))
    print(f"\n## C3 データセット間の seed・書き手 ID の重なり: {overlaps or 'なし'}")
    if overlaps:
        failures.append("C3")

    scan = scan_denylist(files)
    n_hit = sum(len(v) for v in scan.values())
    print(f"\n## C4 拒否リストの語・数値の走査: {len(files)} ファイル、検出 {n_hit} 件")
    for k, v in scan.items():
        if v:
            print(f"  {k}: {v}")
            failures.append(f"C4:{k}")

    c6 = scan_dataset_text(datasets, USED_DATASETS)
    print(f"\n## C6 合成データの文字列と拒否リスト: {c6}")
    if any(v["word_hits"] for v in c6.values()):
        failures.append("C6")
    n_pages = sum(s["pages"] for s in sums)
    obs = sum(v["number_hits_coincidence"] for v in c6.values()) / n_pages
    null_rate, null_hits = null_number_rate()
    print(
        f"   数値の一致率(ページあたり): 学習・較正データ {obs:.4f} / 新しい seed の内容 {null_rate:.4f}"
        f"({null_hits} 件 / 2000 ページ) — 同程度なら偶然の一致(金額は乱数)"
    )

    if args.reconvert:
        rc = reconvert(models)
        print(f"\n## C5 ONNX からの再変換と SHA256 の一致: {rc}")
        if not all(rc.values()):
            failures.append("C5")

    print("\n結果:", "OK" if not failures else f"NG {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
