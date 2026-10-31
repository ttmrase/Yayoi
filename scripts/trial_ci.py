#!/usr/bin/env python3
"""試用版の配布物を GitHub Actions でビルドするときの部品(`.github/workflows/trial-build.yml` が呼ぶ。手元でも同じコマンドで動く)。

    python scripts/trial_ci.py plan [--mode full|mini] [--field shards|...]       ビルドの計画(合成データのシャードなど)を JSON で出す
    python scripts/trial_ci.py run STAGE --data D --models M --evals E [...]       1 つの段階(shard / celltype / adapt / rehearsal / dump / fit)を実行する
    python scripts/trial_ci.py summary --data D --out DIR --id SHARD               シャードの合成データの要約(ページ数・seed・書き手・フォント)
    python scripts/trial_ci.py provenance --models M --summaries DIR --info DIR ... 束に入れる出所の文書(model_provenance.md)を、実際の重みの SHA256 で作る
    python scripts/trial_ci.py augment ...                                         ビルドの記録に、zip・検査の結果・Windows の煙試験の結果を書き足す
    python scripts/trial_ci.py release-spec / release-check ...                    trial-publish.yml が使う: 公開の内容(JSON と本文)の検査、成果物との照合

- **full**: `docs/model_provenance.md` §3・`docs/trial_rc.md` §7.2 と同じ手順・seed・スレッド数(配布物)。合成データ 1,600 ページを 100 ページずつのシャードに分けて並列に作る
  (ページごとに seed と書き手 ID が決まるので、シャードに分けても、1 回で作ったときと同じページができる)。
- **mini**: 同じ経路を、ごく小さなデータ・1 エポックで通す(配布物のビルドの前に、段階のつなぎ・束の作成・検査・Windows の煙試験を、短い時間で確かめる用。重みは使い物にならない。**配布しない**)。
- 重みは、CPU・torch の版が違うと、開発のコンテナで作ったものとビット単位では一致しないことがある。束の出所の表は、実際に作った重みの SHA256 で作る。
- 合成データ・重み・ダンプは、`$YAYOI_PRIVATE_DATA` の下(リポジトリの外)に置く。実データは使わない(CI は実サンプルに触れない)。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = [sys.executable, "-I"]

FONT_EXCLUDE = ["--font-exclude", "yusei", "klee"]  # 学習・較正では評価専用の 2 書体を使わない
# (データセット, ラベル, preset, ページ数, seed0, writer0, 追加の引数) — docs/model_provenance.md §3 の生成コマンド
RECIPES = [
    ("train", "typical", "typical", 300, 1000000, 0, []),
    ("train", "clean", "clean", 100, 1100000, 300, []),
    ("train", "default", "default", 100, 1200000, 400, []),
    ("calib", "typical", "typical", 100, 2000000, 100000, []),
    ("calib500", "typical", "typical", 500, 7000000, 400000, []),
    (
        "train_gen",
        "typical",
        "typical",
        300,
        8000000,
        500000,
        ["--vocab", "general-train", "--strength-range", "0.5", "1.0"],
    ),
    ("calib_gen", "typical", "typical", 200, 9000000, 700000, ["--vocab", "general-eval"]),
]
MODES = {
    "full": {
        "chunk": 100,
        "pages": None,  # レシピのページ数のまま
        "celltype": ["--epochs", "8", "--threads", "4"],
        "adapt": ["--l2sp", "0.1", "--epochs", "3", "--threads", "3"],
        "dump": ["--threads", "3"],
    },
    "mini": {
        "chunk": 12,
        "pages": 12,
        "celltype": ["--epochs", "1", "--threads", "4"],
        "adapt": [
            "--l2sp",
            "0.1",
            "--epochs",
            "1",
            "--n256",
            "40",
            "--n384",
            "40",
            "--threads",
            "3",
        ],
        "dump": ["--threads", "3"],
    },
}
DUMPS = [
    ("calib500", "base"),
    ("calib500", "adapted"),
    ("calib_gen", "base"),
    ("calib_gen", "adapted"),
]
MODEL_FILES = [
    "NDLmoji.yaml",
    "celltype.pt",
    "confidence.json",
    "parseq-ndl-24x256-30-tiny.pt",
    "parseq-ndl-24x384-50-tiny.pt",
    "rehearsal.npz",
    "adapted/confidence.json",
    "adapted/parseq-ndl-24x256-30-tiny.pt",
    "adapted/parseq-ndl-24x384-50-tiny.pt",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 計画


def plan(mode: str) -> dict:
    cfg = MODES[mode]
    shards = []
    for dataset, label, preset, n, seed0, writer0, extra in RECIPES:
        total = cfg["pages"] or n
        # ページ i の seed = seed0 + i、書き手 ID = writer0 + i なので、同じオフセットでずらせば、1 回で作ったときと同じページになる
        for k, off in enumerate(range(0, total, cfg["chunk"])):
            shards.append(
                {
                    "id": f"{dataset}.{label}.{k}",
                    "dataset": dataset,
                    "preset": preset,
                    "n": min(cfg["chunk"], total - off),
                    "seed0": seed0 + off,
                    "writer0": writer0 + off,
                    "extra": FONT_EXCLUDE + extra,
                }
            )
    return {
        "mode": mode,
        "shards": shards,
        "celltype": cfg["celltype"],
        "adapt": cfg["adapt"],
        "dump": cfg["dump"],
        "dumps": [{"dataset": d, "weights": w} for d, w in DUMPS],
    }


# ---------------------------------------------------------------- 段階のコマンド


def stage_commands(
    stage: str, mode: str, data: Path, models: Path, evals: Path, **kw: str | int
) -> list[list[str]]:
    cfg = MODES[mode]
    if stage == "shard":
        s = next(x for x in plan(mode)["shards"] if x["id"] == kw["id"])
        return [
            [
                *PY,
                "scripts/build_synth_cells.py",
                "--jobs",
                str(kw.get("jobs", 3)),
                "--out",
                str(data / s["dataset"]),
                "--n",
                str(s["n"]),
                "--preset",
                s["preset"],
                "--seed0",
                str(s["seed0"]),
                "--writer0",
                str(s["writer0"]),
                *s["extra"],
            ]
        ]
    if stage == "celltype":
        return [
            [
                *PY,
                "scripts/train_celltype.py",
                "--train",
                str(data / "train"),
                "--calib",
                str(data / "calib"),
                "--out",
                str(models / "celltype.pt"),
                *cfg["celltype"],
            ]
        ]
    if stage == "adapt":
        return [
            [
                *PY,
                "scripts/train_adapt.py",
                "--train",
                str(data / "train"),
                str(data / "train_gen"),
                "--out-dir",
                str(models / "adapted"),
                *cfg["adapt"],
            ]
        ]
    if stage == "rehearsal":
        return [
            [
                *PY,
                "scripts/build_rehearsal.py",
                "--data",
                str(data / "train"),
                str(data / "train_gen"),
                "--out",
                str(models / "rehearsal.npz"),
            ]
        ]
    if stage == "dump":
        d, w = str(kw["dataset"]), str(kw["weights"])
        cmd = [
            *PY,
            "scripts/eval_recognize.py",
            "--data",
            str(data / d),
            "--classifier",
            str(models / "celltype.pt"),
        ]
        if w == "adapted":
            cmd += ["--models-dir", str(models / "adapted")]
        return [[*cmd, *cfg["dump"], "--dump", str(evals / f"{d}_{w}.jsonl")]]
    if stage == "fit":
        out = []
        for w, dest in (
            ("base", models / "confidence.json"),
            ("adapted", models / "adapted" / "confidence.json"),
        ):
            out.append(
                [
                    *PY,
                    "scripts/fit_confidence.py",
                    "--dump",
                    str(evals / f"calib500_{w}.jsonl"),
                    str(evals / f"calib_gen_{w}.jsonl"),
                    "--out",
                    str(dest),
                    "--margin",
                    "0.6",
                    "--margins",
                    "amount=0.4",
                    "--column-dump",
                    f"description={evals / f'calib_gen_{w}.jsonl'}",
                ]
            )
        return out
    raise SystemExit(f"不明な段階: {stage}")


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("torch", "numpy", "PIL", "cv2", "onnx", "onnxruntime", "yaml"):
        try:
            m = importlib.import_module(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return out


def run_stage(args: argparse.Namespace) -> int:
    data, models, evals = Path(args.data), Path(args.models), Path(args.evals)
    for p in (data, models, models / "adapted", evals):
        p.mkdir(parents=True, exist_ok=True)
    cmds = stage_commands(
        args.stage,
        args.mode,
        data,
        models,
        evals,
        id=args.id,
        dataset=args.dataset,
        weights=args.weights,
        jobs=args.jobs,
    )
    t0 = time.time()
    for cmd in cmds:
        print("+ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
    if args.info:
        info = {
            "stage": args.stage,
            "mode": args.mode,
            "id": args.id or f"{args.dataset or ''}_{args.weights or ''}".strip("_"),
            "commands": [" ".join(c) for c in cmds],
            "seconds": round(time.time() - t0, 1),
            "cpu": cpu_model(),
            "cpus": os.cpu_count(),
            "image": {k: os.environ.get(k, "") for k in ("ImageOS", "ImageVersion", "RUNNER_OS")},
            "versions": versions(),
        }
        Path(args.info).parent.mkdir(parents=True, exist_ok=True)
        Path(args.info).write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


# ---------------------------------------------------------------- 要約と出所の文書


def _model_provenance():
    spec = importlib.util.spec_from_file_location(
        "model_provenance", ROOT / "scripts" / "model_provenance.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmd_summary(args: argparse.Namespace) -> int:
    mp = _model_provenance()
    from yayoi_ocr.config import fonts_dir

    fonts = {p.stem for p in fonts_dir().glob("*.ttf")}
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for sub in sorted(p for p in data.iterdir() if p.is_dir()):
        s = mp.dataset_summary(data, sub.name, fonts)
        s["seed_set"] = sorted(s["seed_set"])
        s["writer_set"] = sorted(s["writer_set"])
        (out / f"summary.{args.id}.{sub.name}.json").write_text(
            json.dumps(s, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"{sub.name}: {s['pages']} ページ・{s['cells']:,} セル・署名なし {len(s['bad'])} ページ"
        )
        if s["bad"]:
            return 1
    return 0


_ROW = re.compile(r"^\|\s*`(models/[^`]+)`\s*\|.*`([0-9a-f]{64})`\s*\|\s*$")


def _merge_summaries(directory: Path) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for f in sorted(directory.glob("summary.*.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        m = merged.setdefault(
            s["name"],
            {"pages": 0, "cells": 0, "bad": 0, "seed_set": set(), "writer_set": set(), "fonts": {}},
        )
        m["pages"] += s["pages"]
        m["cells"] += s["cells"]
        m["bad"] += len(s["bad"])
        m["seed_set"] |= set(s["seed_set"])
        m["writer_set"] |= set(s["writer_set"])
        for k, v in s["fonts"].items():
            m["fonts"][k] = m["fonts"].get(k, 0) + v
    return merged


def _runs(values: set[int]) -> str:
    s = sorted(values)
    runs, start, prev = [], s[0], s[0]
    for v in s[1:]:
        if v != prev + 1:
            runs.append((start, prev))
            start = v
        prev = v
    runs.append((start, prev))
    return ", ".join(f"{a}–{b}" for a, b in runs)


def build_provenance(
    template: str,
    models: Path,
    summaries: Path,
    infos: list[dict],
    *,
    run_id: str,
    commit: str,
    mode: str,
    ref: str,
    now: str,
) -> str:
    """リポジトリの docs/model_provenance.md を、この束の重みの SHA256 と、ビルドの記録で書き直す。"""
    hashes = {f"models/{rel}": (models / rel) for rel in MODEL_FILES}
    for rel, p in hashes.items():
        if not p.is_file():
            raise SystemExit(f"重みがありません: {rel}")
    actual = {rel: sha256(p) for rel, p in hashes.items()}

    # §2 の表(同梱 9 ファイル)の SHA256 を、実際の値にする
    lines = template.splitlines()
    seen = set()
    for i, line in enumerate(lines):
        m = _ROW.match(line)
        if m:
            rel = m.group(1)
            if rel not in actual:
                raise SystemExit(f"出所の表に、同梱しないファイルがあります: {rel}")
            lines[i] = re.sub(r"`[0-9a-f]{64}`(\s*\|\s*)$", f"`{actual[rel]}`\\1", line)
            seen.add(rel)
    if seen != set(actual):
        raise SystemExit(f"出所の表に無い重みがあります: {sorted(set(actual) - seen)}")
    text = "\n".join(lines)

    # 「実行したスクリプトとコミット」(開発のコンテナの記録)は、このビルドの記録に置き換える
    text = re.sub(
        r"### 実行したスクリプトとコミット\n.*?(?=\n## 3\.)",
        "### 実行したスクリプトとコミット\n\nこの束は GitHub Actions で作ったので、上の §0 の記録(コミット・コマンド・seed・スレッド数)が、この束の実行記録である。\n",
        text,
        flags=re.S,
    )

    merged = _merge_summaries(summaries)
    names = sorted(merged)
    overlaps = [
        (a, b)
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        if merged[a]["seed_set"] & merged[b]["seed_set"]
        or merged[a]["writer_set"] & merged[b]["writer_set"]
    ]
    bad_pages = sum(m["bad"] for m in merged.values())
    if overlaps or bad_pages:
        raise SystemExit(f"合成データの検査に失敗: 重なり {overlaps}・署名のないページ {bad_pages}")
    mp = _model_provenance()
    c1 = mp.check_samples_paths()
    if c1:
        raise SystemExit(f"C1: 実サンプルの置き場を参照するファイルがあります: {c1}")

    def cell(rows: list[dict], key: str) -> str:
        vals = sorted({str(r.get(key, "")) for r in rows if r.get(key, "")})
        return ", ".join(vals) if vals else "—"

    cpus = cell(infos, "cpu")
    vers = {}
    for r in infos:
        for k, v in r.get("versions", {}).items():
            vers.setdefault(k, set()).add(v)
    ver_text = "・".join(f"{k} {', '.join(sorted(v))}" for k, v in sorted(vers.items()))
    stage_rows = "\n".join(
        f"| `{r['stage']}` | {('`' + r['id'] + '`') if r['id'] else '—'} | {r['seconds']:.0f} 秒 | `{r['commands'][0][:230]}` |"
        for r in sorted(infos, key=lambda r: (r["stage"], r["id"]))
        if r["stage"] != "shard"
    )
    n_shards = sum(1 for r in infos if r["stage"] == "shard")
    ds_rows = "\n".join(
        f"| `{n}` | {merged[n]['pages']} | {merged[n]['cells']:,} | {_runs(merged[n]['seed_set'])} |"
        f" {min(merged[n]['writer_set'])}–{max(merged[n]['writer_set'])}({len(merged[n]['writer_set'])}) |"
        f" {', '.join(f'{k} {v}' for k, v in sorted(merged[n]['fonts'].items()))} |"
        for n in names
    )
    head = f"""# 同梱する重みの出所(この束は GitHub Actions でビルドした)

> **この束の重みは、GitHub Actions(run {run_id}、コミット `{commit}`)で、合成データだけから作り直したものです。**
> §0 がこのビルドの記録です。§1 以降は、開発のコンテナでビルドした重みについての記録(リポジトリの `docs/model_provenance.md`)で、
> **§2 の SHA256 の表だけを、この束の重みの実際の値に置き換えてあります**(CPU・ライブラリの違いで、コンテナで作った重みとは値が一致しないことがあります)。
> §4 の検査(C4・C6: 拒否リストの走査)は、コンテナでビルドした重みについての記録です。**この束の重みには行っていません**(拒否リストを CI に置けないため)。
> 代わりに、入力が合成データだけであること(C1〜C3。下の §0)と、束に画像・PDF・CSV・DB の類が入っていないこと(`scripts/verify_trial_bundle.py` の B7)を確かめています。

## 0. このビルドの記録

| 項目 | 内容 |
|---|---|
| ビルド | GitHub Actions の run {run_id}(モード `{mode}`)、コミット `{commit}`、{now} |
| 入力の固定 | フォント: google/fonts の `{ref}`(全 14 ファイルの SHA256 を `requirements/fonts.sha256` と照合)。基本の重み: NDLOCR-Lite の `e57f7c6bd9cfb3de5f12f008e71ad2122bd44bfd`(ONNX の SHA256 を検証)。同梱の Python・wheel: lock とハッシュで固定 |
| 実行環境 | CPU: {cpus}。{ver_text} |
| 合成データ | {n_shards} 個のシャードに分けて並列に生成(ページごとに seed・書き手 ID が決まるので、1 回で作ったときと同じページになる) |
| 手順 | `docs/model_provenance.md` §3・`docs/trial_rc.md` §7.2 と同じ(seed 0。セル種別分類器 `--threads 4`、適応済みの重み `--threads 3`、ダンプ `--threads 3`) |

| 段階 | 対象 | 所要時間 | コマンド(1 つ目) |
|---|---|---:|---|
{stage_rows}

**合成データの要約**(シャードを合わせたもの。検査 C2: 全ページが生成器の署名を持つ = 署名のないページ {bad_pages}。C3: データセット間の seed・書き手 ID の重なり = {overlaps or "なし"}。C1: 実サンプルの置き場を参照するコード = なし)

| データセット | ページ | セル | seed の範囲 | 書き手 ID の範囲(人数) | フォント(ページ数) |
|---|---:|---:|---|---|---|
{ds_rows}

---

"""
    body = text
    body = body.replace(
        "# 同梱する重みの出所(試用版の配布前の記録)",
        "# 付録: 開発のコンテナでビルドした重みの記録(リポジトリの docs/model_provenance.md。§2 の SHA256 はこの束の値)",
        1,
    )
    return head + body + ("\n" if not body.endswith("\n") else "")


def cmd_provenance(args: argparse.Namespace) -> int:
    template = Path(args.template).read_text(encoding="utf-8")
    infos = [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(args.info).glob("*.json"))
    ]
    pin = Path(ROOT / "requirements" / "fonts-pin.txt").read_text(encoding="utf-8")
    ref = re.search(r"^ref=(\w+)", pin, re.M).group(1)
    text = build_provenance(
        template,
        Path(args.models),
        Path(args.summaries),
        infos,
        run_id=args.run_id,
        commit=args.commit,
        mode=args.mode,
        ref=ref,
        now=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
    )
    Path(args.out).write_text(text, encoding="utf-8")
    info = {
        "run_id": args.run_id,
        "commit": args.commit,
        "mode": args.mode,
        "weights": {f"models/{rel}": sha256(Path(args.models) / rel) for rel in MODEL_FILES},
        "stages": infos,
    }
    Path(args.build_info).write_text(
        json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"出所の文書: {args.out}({len(text):,} 文字)")
    return 0


def cmd_augment(args: argparse.Namespace) -> int:
    """ビルドの記録(build-info.json)に、zip・検査の結果・Windows の煙試験の結果・GitHub の run を書き足す。"""
    info = json.loads(Path(args.info).read_text(encoding="utf-8"))
    zip_path = Path(args.zip)
    info["zip"] = {
        "name": zip_path.name,
        "sha256": sha256(zip_path),
        "size": zip_path.stat().st_size,
    }
    if args.verify_output:
        text = Path(args.verify_output).read_text(encoding="utf-8")
        info["verify_bundle"] = {"ok": text.rstrip().endswith("結果: OK"), "output": text}
    if args.smoke:
        smoke = json.loads(Path(args.smoke).read_text(encoding="utf-8"))
        # Windows の煙試験が、**この zip そのもの**(SHA256 が同じ)に対して通ったことを確かめる
        ok = smoke.get("passed") is True and smoke.get("zip_sha256") == info["zip"]["sha256"]
        info["smoke_windows"] = {"ok": ok, "detail": smoke}
    info["github"] = {
        k: os.environ.get(v, "")
        for k, v in (
            ("repository", "GITHUB_REPOSITORY"),
            ("run_id", "GITHUB_RUN_ID"),
            ("run_attempt", "GITHUB_RUN_ATTEMPT"),
            ("sha", "GITHUB_SHA"),
            ("workflow_ref", "GITHUB_WORKFLOW_REF"),
        )
    }
    Path(args.out).write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"ビルドの記録: {args.out}(zip {info['zip']['name']} {info['zip']['sha256']})")
    return 0


_HEX64 = re.compile(r"[0-9a-f]{64}")


def load_release_spec(spec_path: Path, notes_path: Path) -> dict:
    """公開の内容(docs/release/*.json)を検査して返す。不正なら SystemExit。"""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    problems = []
    if not re.fullmatch(r"trial-v\d+\.\d+\.\d+", str(spec.get("tag", ""))):
        problems.append("tag は trial-vX.Y.Z の形")
    if not re.fullmatch(r"[0-9a-f]{40}", str(spec.get("commit", ""))):
        problems.append("commit は 40 桁の 16 進(完全なコミット ID)")
    if not re.fullmatch(r"\d+", str(spec.get("run_id", ""))):
        problems.append("run_id は数字")
    if not _HEX64.fullmatch(str(spec.get("zip_sha256", ""))):
        problems.append("zip_sha256 は 64 桁の小文字の 16 進")
    if not re.fullmatch(r"yayoi-ocr-trial-[0-9.]+\.zip", str(spec.get("zip_name", ""))):
        problems.append("zip_name は yayoi-ocr-trial-X.Y.Z.zip(MINI や SMOKE は不可)")
    if not str(spec.get("title", "")).strip():
        problems.append("title が空")
    if problems:
        raise SystemExit("公開の内容(JSON)が不正: " + " / ".join(problems))
    notes = notes_path.read_text(encoding="utf-8")
    found = set(_HEX64.findall(notes))
    if found != {spec["zip_sha256"]}:
        raise SystemExit(
            f"Release の本文に書かれた SHA256 が、JSON の zip_sha256 と一致しません(本文に {len(found)} 種類の 64 桁の 16 進)"
        )
    return spec


def cmd_release_spec(args: argparse.Namespace) -> int:
    spec = load_release_spec(Path(args.json), Path(args.notes))
    for k in ("tag", "commit", "run_id", "zip_name", "title"):
        print(f"{k}={spec[k]}")
    print(f"sha256={spec['zip_sha256']}")
    return 0


def cmd_release_check(args: argparse.Namespace) -> int:
    """ダウンロードしたビルドの成果物(artifact)が、公開の内容(JSON)と一致し、検査に通っていることを確かめる。"""
    spec = load_release_spec(Path(args.json), Path(args.notes))
    d = Path(args.dir)
    problems = []
    zip_path = d / spec["zip_name"]
    if not zip_path.is_file():
        raise SystemExit(f"zip がありません: {spec['zip_name']}")
    actual = sha256(zip_path)
    if actual != spec["zip_sha256"]:
        problems.append(f"zip の SHA256 が JSON と違う({actual})")
    side = d / (spec["zip_name"] + ".sha256")
    if not side.is_file() or side.read_text(encoding="utf-8").split()[:1] != [spec["zip_sha256"]]:
        problems.append(".sha256 のファイルが JSON と違う、またはない")
    if not (d / "model_provenance.generated.md").is_file():
        problems.append("model_provenance.generated.md がない")
    try:
        info = json.loads((d / "build-info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise SystemExit("build-info.json を読めません") from None
    gh = info.get("github", {})
    checks = {
        "mode が full": info.get("mode") == "full",
        "ビルドの run_id が JSON と同じ": str(info.get("run_id"))
        == spec["run_id"]
        == str(gh.get("run_id")),
        "ビルドのコミットが JSON と同じ": info.get("commit") == spec["commit"] == gh.get("sha"),
        "build-info の zip の SHA256 が JSON と同じ": info.get("zip", {}).get("sha256")
        == spec["zip_sha256"],
        "束の検査(B1〜B7)が OK": info.get("verify_bundle", {}).get("ok") is True,
        "Windows の煙試験が OK": info.get("smoke_windows", {}).get("ok") is True,
    }
    problems += [f"{k}: 不一致" for k, ok in checks.items() if not ok]
    for k, ok in checks.items():
        print(f"[{'OK' if ok else 'NG'}] {k}")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print("成果物は、公開の内容と一致し、検査に通っています")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--mode", choices=sorted(MODES), default="full")
    p.add_argument("--field", help="計画の一部だけを、JSON 1 行で出す(例: shards)")
    r = sub.add_parser("run")
    r.add_argument("stage", choices=["shard", "celltype", "adapt", "rehearsal", "dump", "fit"])
    r.add_argument("--mode", choices=sorted(MODES), default="full")
    r.add_argument("--data", required=True)
    r.add_argument("--models", required=True)
    r.add_argument("--evals", required=True)
    r.add_argument("--id", help="shard: シャードの ID")
    r.add_argument("--dataset", help="dump: calib500 / calib_gen")
    r.add_argument("--weights", help="dump: base / adapted")
    r.add_argument("--jobs", type=int, default=3)
    r.add_argument("--info", help="段階の記録(JSON)の出力先")
    s = sub.add_parser("summary")
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--id", required=True)
    v = sub.add_parser("provenance")
    v.add_argument("--template", default=str(ROOT / "docs" / "model_provenance.md"))
    v.add_argument("--models", required=True)
    v.add_argument("--summaries", required=True)
    v.add_argument("--info", required=True)
    v.add_argument("--run-id", required=True)
    v.add_argument("--commit", required=True)
    v.add_argument("--mode", choices=sorted(MODES), default="full")
    v.add_argument("--out", required=True)
    v.add_argument("--build-info", required=True)
    a = sub.add_parser("augment")
    a.add_argument("--info", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--zip", required=True)
    a.add_argument("--verify-output")
    a.add_argument("--smoke")
    rs = sub.add_parser("release-spec")
    rs.add_argument("--json", required=True)
    rs.add_argument("--notes", required=True)
    rc = sub.add_parser("release-check")
    rc.add_argument("--json", required=True)
    rc.add_argument("--notes", required=True)
    rc.add_argument("--dir", required=True)
    args = ap.parse_args()
    if args.cmd == "release-spec":
        return cmd_release_spec(args)
    if args.cmd == "release-check":
        return cmd_release_check(args)
    if args.cmd == "augment":
        return cmd_augment(args)
    if args.cmd == "plan":
        pl = plan(args.mode)
        print(
            json.dumps(
                pl[args.field] if args.field else pl, ensure_ascii=False, separators=(",", ":")
            )
        )
        return 0
    if args.cmd == "run":
        return run_stage(args)
    if args.cmd == "summary":
        return cmd_summary(args)
    return cmd_provenance(args)


if __name__ == "__main__":
    raise SystemExit(main())
