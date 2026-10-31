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
# 最終ホールドアウト(docs/recognition.md「確信度の較正」: 学習・較正に一度も使っていない新規 200 ページ。
# 標準語彙・一般語彙 x 未使用の書き手・未知フォント(評価専用の yusei・klee)各 50 ページ)の生成レシピ。メタデータと正解の文字列が、コンテナで作ったものと一致することを確認済み。
EVAL_RECIPES = [
    ("final_std", "typical", 50, 9900000, 900000, FONT_EXCLUDE),
    ("final_gen", "typical", 50, 9910000, 910000, [*FONT_EXCLUDE, "--vocab", "general-eval"]),
    ("final_uf_std", "typical", 50, 9920000, 920000, ["--font-include", "yusei", "klee"]),
    (
        "final_uf_gen",
        "typical",
        50,
        9930000,
        930000,
        ["--font-include", "yusei", "klee", "--vocab", "general-eval"],
    ),
]
# 最終ホールドアウト v2(director 指示): 科目の見逃し率が基準を超えたあと、確信度の較正を直した束の**最終の確認**に使う、重ならない新しい 200 ページ。
# 学習・較正・v1 のどの seed・書き手 ID とも重ならない(試験で確かめる)。構成は v1 と同じ(標準語彙・一般語彙 x 未使用の書き手・未知フォント 各 50 ページ)。
EVAL_RECIPES_V2 = [
    ("final2_std", "typical", 50, 9940000, 940000, FONT_EXCLUDE),
    ("final2_gen", "typical", 50, 9950000, 950000, [*FONT_EXCLUDE, "--vocab", "general-eval"]),
    ("final2_uf_std", "typical", 50, 9960000, 960000, ["--font-include", "yusei", "klee"]),
    (
        "final2_uf_gen",
        "typical",
        50,
        9970000,
        970000,
        ["--font-include", "yusei", "klee", "--vocab", "general-eval"],
    ),
]
HOLDOUT_RECIPES = {"v1": EVAL_RECIPES, "v2": EVAL_RECIPES_V2}
FINAL_HOLDOUT = "v2"  # trial-eval が使い、trial-publish が要求するホールドアウト
# 公開の合格基準(director 指示): 点推定で、全体の見逃し率 <= 0.5%、金額 <= 0.1%。列ごとの値の正解率(空欄を除く)が、コンテナでの出荷時の評価
# から 3 ポイントを超えて下がっていない。基本の重みと適応済みの重みの両方に適用する。
MISS_LIMIT_ALL = 0.005
# 列ごとの見逃し率の上限(Phase 2 の指示 e: 日付・科目・摘要はそれぞれ <= 0.5%、金額は <= 0.1%)。全体の上限とは別に、列ごとに判定する。
MISS_LIMIT_COLUMN = {"date": 0.005, "account": 0.005, "description": 0.005, "amount": 0.001}
MISS_LIMIT_AMOUNT = MISS_LIMIT_COLUMN["amount"]
ACC_DROP_LIMIT = 0.03
# 確信度の較正の余裕(margin。目標の見逃し率 0.5% に掛ける係数。小さいほど、要確認が増えて見逃しが減る)。director の判断で、事前に固定した値:
# 科目 0.3(最終ホールドアウト v1 で、科目の見逃し率が 0.62% と基準 0.5% を超えたため。新しいホールドアウト v2 を見る前に決めた)、金額 0.4、日付・摘要 0.6。基本と適応済みの両方に適用する。
FIT_MARGIN = 0.6
FIT_MARGINS = {"account": 0.3, "amount": 0.4}
# 評価の結果(JSON)の、基準の版。2 = 列ごとの見逃し率の基準を含む。trial-publish は、この版以上の結果だけを受け付ける。
EVAL_CRITERIA_VERSION = 2
COLUMN_LABELS = {"date": "日付", "account": "科目", "description": "摘要", "amount": "金額"}
# コンテナで出荷した重み(RC の束。SHA256 5d69f666…)を、元の最終ホールドアウト(200 ページ)に、この評価と同じコード(`evaluate_weights`)で
# 当てて測った値(列ごとの accuracy_nonblank。小数第 3 位に丸めた)。見逃し数・要確認率は docs/recognition.md の表と一致することを確かめてある
# (適応済み: 見逃し 12/21/8/2 件)。docs の表の「値の正解率」は定義が少し違う(空欄の扱い)ため、基準には、測り直した値を使う。
BASELINES = {
    "v1": {
        "adapted": {"date": 0.586, "account": 0.853, "description": 0.443, "amount": 0.642},
        "base": {"date": 0.409, "account": 0.726, "description": 0.203, "amount": 0.540},
    },
    # v2: コンテナで出荷した重みを、ホールドアウト v2 に当てて測り直した値
    "v2": {  # TEMP: 測定中(測り直した値で置き換える)
        "adapted": {"date": 0.586, "account": 0.853, "description": 0.443, "amount": 0.642},
        "base": {"date": 0.409, "account": 0.726, "description": 0.203, "amount": 0.540},
    },
}
BASELINE_ACCURACY = BASELINES[FINAL_HOLDOUT]
# 基準値が仮のホールドアウト。含まれている間は、評価は必ず不合格になる(仮の値のまま合格しないため。checker 指摘 V-2)。測り直した値を入れたら外す。
BASELINES_PROVISIONAL = {"v2"}
MODES = {
    "full": {
        "chunk": 100,
        "pages": None,  # レシピのページ数のまま
        "eval_pages": None,
        "celltype": ["--epochs", "8", "--threads", "4"],
        "adapt": ["--l2sp", "0.1", "--epochs", "3", "--threads", "3"],
        "dump": ["--threads", "3"],
    },
    "mini": {
        "chunk": 12,
        "pages": 12,
        "eval_pages": 6,
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


def eval_shards(mode: str, holdout: str) -> list[dict]:
    """最終ホールドアウトのシャード(1 データセット = 1 シャード)。mini は eval_pages ページ。"""
    cfg = MODES[mode]
    return [
        {
            "id": f"{name}.0",
            "dataset": name,
            "preset": preset,
            "n": cfg["eval_pages"] or n,
            "seed0": seed0,
            "writer0": writer0,
            "extra": extra,
        }
        for name, preset, n, seed0, writer0, extra in HOLDOUT_RECIPES[holdout]
    ]


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
        "eval_shards": eval_shards(mode, FINAL_HOLDOUT),  # trial-eval が使う最終ホールドアウト(v2)
        "eval_shards_v1": eval_shards(mode, "v1"),  # 最初の評価に使ったホールドアウト(履歴)
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
        pl = plan(mode)
        s = next(
            x
            for x in [*pl["shards"], *pl["eval_shards"], *pl["eval_shards_v1"]]
            if x["id"] == kw["id"]
        )
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
                    str(FIT_MARGIN),
                    "--margins",
                    ",".join(f"{k}={v}" for k, v in FIT_MARGINS.items()),
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


# ---------------------------------------------------------------- 重みの再利用(確信度の較正だけをやり直すビルド)

# 前の run で作った重み・較正のダンプを再利用してよいのは、その run を作った処理が、いまのコミットで実行する処理と同じとき。
# ① 学習・ダンプの段階のコマンドが、前の run の記録と一致する(レシピ・seed・スレッド数・引数)。② 前の run のコミットが、いまのコミットの祖先。
# ③ 学習・認識・合成に関わるコード(下の一覧)が、前の run のコミットから変わっていない。確信度の較正(fit)は、いまのコミットの処理でやり直す。
REUSED_STAGES = ("shard", "celltype", "adapt", "rehearsal", "dump")
REUSE_PROTECTED = (
    "src/",
    "requirements/",
    "pyproject.toml",
    "scripts/build_synth_cells.py",
    "scripts/train_celltype.py",
    "scripts/train_adapt.py",
    "scripts/build_rehearsal.py",
    "scripts/eval_recognize.py",
    "scripts/fetch_models.py",
    "scripts/fetch_fonts.py",
)
_PY_HEAD = re.compile(r"^\S*python[0-9.]* -I ")


def _normalize_command(cmd: str) -> str:
    """インタプリタのパス(ランナーの Python の版で変わる)を無視して、コマンドを比べる。"""
    return _PY_HEAD.sub("<python> -I ", cmd)


def expected_reused_commands(
    mode: str, data: Path, models: Path, evals: Path
) -> dict[tuple[str, str], list[str]]:
    """再利用する段階(shard・celltype・adapt・rehearsal・dump)について、いまのコミットが実行するコマンド。キーは (段階, 記録の id)。"""
    pl = plan(mode)

    def norm(cmds: list[list[str]]) -> list[str]:
        return [_normalize_command(" ".join(c)) for c in cmds]

    out: dict[tuple[str, str], list[str]] = {}
    for sh in pl["shards"]:
        out[("shard", sh["id"])] = norm(
            stage_commands("shard", mode, data, models, evals, id=sh["id"], jobs=3)
        )
    for st in ("celltype", "adapt", "rehearsal"):
        out[(st, "")] = norm(stage_commands(st, mode, data, models, evals))
    for d in pl["dumps"]:
        out[("dump", f"{d['dataset']}_{d['weights']}")] = norm(
            stage_commands(
                "dump", mode, data, models, evals, dataset=d["dataset"], weights=d["weights"]
            )
        )
    return out


def reuse_problems(
    mode: str,
    infos: list[dict],
    changed_files: list[str],
    data: Path,
    models: Path,
    evals: Path,
) -> list[str]:
    """前の run の段階の記録(infos)とコミットの差が、再利用してよいものか。問題の一覧(空なら再利用してよい)。"""
    problems = []
    expected = expected_reused_commands(mode, data, models, evals)
    got: dict[tuple[str, str], dict] = {}
    for r in infos:
        if r.get("stage") in REUSED_STAGES:
            got[(r["stage"], r.get("id", ""))] = r
    for key in sorted(set(expected) - set(got)):
        problems.append(f"前の run に、段階の記録がない: {key[0]} {key[1]}".rstrip())
    for key in sorted(set(got) - set(expected)):
        problems.append(f"前の run に、いまの計画に無い段階がある: {key[0]} {key[1]}".rstrip())
    for key in sorted(set(expected) & set(got)):
        rec = got[key]
        if rec.get("mode") != mode:
            problems.append(
                f"モードが違う: {key[0]} {key[1]}(前の run: {rec.get('mode')})".rstrip()
            )
        recorded = [_normalize_command(c) for c in rec.get("commands", [])]
        if recorded != expected[key]:
            problems.append(f"コマンドが違う: {key[0]} {key[1]}".rstrip())
    for f in changed_files:
        if any(f == p or (p.endswith("/") and f.startswith(p)) for p in REUSE_PROTECTED):
            problems.append(
                f"学習・認識・合成に関わるファイルが、前の run のコミットから変わっている: {f}"
            )
    return problems


def cmd_reuse_check(args: argparse.Namespace) -> int:
    base = args.base_commit
    if not re.fullmatch(r"[0-9a-f]{40}", base):
        raise SystemExit("--base-commit は 40 桁の 16 進(前の run のコミット)")
    infos = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(Path(args.infos).glob("*.json"))
        if not p.name.startswith("info.fit")
    ]
    problems = []
    anc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, "HEAD"], cwd=ROOT, capture_output=True
    )
    if anc.returncode != 0:
        problems.append("前の run のコミットが、いまのコミットの祖先ではない")
        changed: list[str] = []
    else:
        diff = subprocess.run(
            ["git", "diff", "--name-only", base, "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        changed = [x for x in diff.stdout.splitlines() if x]
    problems += reuse_problems(
        args.mode, infos, changed, Path(args.data), Path(args.models), Path(args.evals)
    )
    print(
        f"前の run のコミット {base[:12]} → いまのコミット: 変更 {len(changed)} ファイル・段階の記録 {len(infos)} 件"
    )
    if problems:
        print("\n".join(f"[NG] {p}" for p in problems), file=sys.stderr)
        return 1
    print("再利用してよい: 学習・ダンプの段階のコマンドが一致し、関わるコードの変更もない")
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
    reuse: dict | None = None,
) -> str:
    """リポジトリの docs/model_provenance.md を、この束の重みの SHA256 と、ビルドの記録で書き直す。
    reuse = {"run_id", "commit"} のとき、重みと較正のダンプは前の run で作ったものの再利用(確信度の較正だけをこの run でやり直した)と書く。"""
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
        f"| `{r['stage']}` | {('`' + r['id'] + '`') if r['id'] else '—'} | {r.get('run_id') or run_id} | {r['seconds']:.0f} 秒 | `{r['commands'][0][:230]}` |"
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
    margins = "・".join(
        [f"{COLUMN_LABELS[c]} {m}" for c, m in FIT_MARGINS.items()]
        + [
            f"{'・'.join(COLUMN_LABELS[c] for c in COLUMN_LABELS if c not in FIT_MARGINS)} {FIT_MARGIN}"
        ]
    )
    if reuse:
        built = (
            f"GitHub Actions で、合成データだけから作り直したものです(重みの学習と較正のダンプは run {reuse['run_id']}"
            f"〔コミット `{reuse['commit']}`〕、確信度の較正は run {run_id}〔コミット `{commit}`〕)"
        )
        reuse_row = (
            f"\n| 再利用 | シャード(合成データ)・セル種別分類器・基本/適応済みの重み・リハーサルの束・較正のダンプは、**run {reuse['run_id']}(コミット `{reuse['commit']}`)で作ったものを再利用**した。"
            f"再利用の前に、学習・ダンプの段階のコマンドが、その run の記録と一致すること、学習・認識・合成に関わるコードが変わっていないことを照合している。"
            f"**この run でやり直したのは、確信度の較正(`fit`)だけ**で、余裕(margin)を変えた(科目 0.6 → 0.3。ほかは同じ) |"
        )
    else:
        built = f"GitHub Actions(run {run_id}、コミット `{commit}`)で、合成データだけから作り直したものです"
        reuse_row = ""
    head = f"""# 同梱する重みの出所(この束は GitHub Actions でビルドした)

> **この束の重みは、{built}。**
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
| 手順 | `docs/model_provenance.md` §3・`docs/trial_rc.md` §7.2 と同じ(seed 0。セル種別分類器 `--threads 4`、適応済みの重み `--threads 3`、ダンプ `--threads 3`)。**ただし、確信度の較正の余裕(margin)は {margins}**(コンテナで作った重みは、科目も 0.6。§3 の記録はコンテナの値) |{reuse_row}

| 段階 | 対象 | run | 所要時間 | コマンド(1 つ目) |
|---|---|---|---:|---|
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
        {**json.loads(p.read_text(encoding="utf-8")), "run_id": args.run_id}
        for p in sorted(Path(args.info).glob("*.json"))
    ]
    reuse = None
    if args.reuse_run_id:
        if not (args.reuse_commit and args.reuse_info):
            raise SystemExit("--reuse-run-id には --reuse-commit と --reuse-info が要る")
        reuse = {"run_id": args.reuse_run_id, "commit": args.reuse_commit}
        infos += [
            {**json.loads(p.read_text(encoding="utf-8")), "run_id": args.reuse_run_id}
            for p in sorted(Path(args.reuse_info).glob("*.json"))
            if not p.name.startswith("info.fit")
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
        reuse=reuse,
    )
    Path(args.out).write_text(text, encoding="utf-8")
    info = {
        "run_id": args.run_id,
        "commit": args.commit,
        "mode": args.mode,
        "reuse": reuse,
        "fit_margins": {"default": FIT_MARGIN, **FIT_MARGINS},
        "weights": {f"models/{rel}": sha256(Path(args.models) / rel) for rel in MODEL_FILES},
        "stages": infos,
    }
    Path(args.build_info).write_text(
        json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"出所の文書: {args.out}({len(text):,} 文字)")
    return 0


# ---------------------------------------------------------------- 重みの質の評価(trial-eval.yml)


def _nan_to_none(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in d.items()}


def evaluate_weights(
    models: Path,
    data: Path,
    names: list[str],
    weights: str,
    threads: int = 3,
    holdout: str = FINAL_HOLDOUT,
) -> dict:
    """束の重み(base = 素 / adapted = 適応済み)で、最終ホールドアウトのページを認識して、列ごとの正解率・要確認率・見逃し率を数える。
    `scripts/eval_recognize.py` と同じ処理(出荷設定の確信度・種別分類器・辞書なし = cold)。"""
    import torch

    from yayoi_ocr.recognize.celltype import CellTypeClassifier
    from yayoi_ocr.recognize.confidence import ConfidenceModel
    from yayoi_ocr.recognize.evaluate import judge, summarize
    from yayoi_ocr.recognize.inputs import from_page_data
    from yayoi_ocr.recognize.page import JobContext, PageRecognizer
    from yayoi_ocr.recognize.parseq.engine import ParseqEngine
    from yayoi_ocr.synth.content import ACCOUNTS, TITLES
    from yayoi_ocr.synth.dataset import list_pages, load_page

    torch.set_num_threads(threads)
    adapted = weights == "adapted"
    engine = ParseqEngine(models_dir=models / "adapted" if adapted else models)
    clf = CellTypeClassifier.load(models / "celltype.pt")
    cal = ConfidenceModel.load(
        models / ("adapted/confidence.json" if adapted else "confidence.json")
    )
    rec = PageRecognizer(engine, clf, calibrator=cal)
    ctx = JobContext(accounts=list(ACCOUNTS), titles=list(TITLES))
    columns = ("date", "account", "description", "amount")
    rows_all: list = []
    per_dataset: dict[str, dict] = {}
    for name in names:
        rows: list = []
        pages = list_pages(str(data / name))
        for p in pages:
            page = load_page(p)
            res = rec.recognize_page(from_page_data(page), ctx, None)
            page_review = page.meta["status"] != "ok"
            rows += [judge(r, c, page_review) for r, c in zip(res.cells, page.cells, strict=True)]
        rows_all += rows
        per_dataset[name] = {
            "pages": len(pages),
            "overall": _nan_to_none(summarize(rows)["all"]),
            "columns": {
                c: _nan_to_none(summarize([j for j in rows if j.column == c])["all"])
                for c in columns
            },
        }
        print(f"{weights} {name}: {len(pages)} ページ", flush=True)
    return {
        "weights": weights,
        "holdout": holdout,
        "pages": sum(d["pages"] for d in per_dataset.values()),
        "overall": _nan_to_none(summarize(rows_all)["all"]),
        "columns": {
            c: _nan_to_none(summarize([j for j in rows_all if j.column == c])["all"])
            for c in columns
        },
        "per_dataset": per_dataset,
    }


def required_eval_check_ids() -> list[str]:
    """合格に必要な判定の ID(基本と適応済みの両方 x 全体・列ごとの見逃し率・列ごとの値の正解率、と評価ページ数)。"""
    ids = []
    for w in ("base", "adapted"):
        ids.append(f"{w}:miss:overall")
        ids += [f"{w}:miss:{c}" for c in MISS_LIMIT_COLUMN]
        ids += [f"{w}:acc:{c}" for c in BASELINE_ACCURACY[w]]
    ids += ["baseline_measured", "holdout", "pages"]
    return ids


def eval_checks(result: dict, weights: str) -> list[dict]:
    """合格の基準(director 指示)。点推定で判定する。"""
    out = [
        {
            "id": f"{weights}:miss:overall",
            "name": f"{weights}: 全体の見逃し率 <= {MISS_LIMIT_ALL:.1%}",
            "value": result["overall"]["miss_rate"],
            "limit": MISS_LIMIT_ALL,
            "ok": result["overall"]["miss_rate"] <= MISS_LIMIT_ALL,
        }
    ]
    for col, limit in MISS_LIMIT_COLUMN.items():
        rate = result["columns"][col]["miss_rate"]
        out.append(
            {
                "id": f"{weights}:miss:{col}",
                "name": f"{weights}: {COLUMN_LABELS[col]}の見逃し率 <= {limit:.1%}",
                "value": rate,
                "limit": limit,
                "ok": rate is not None and rate <= limit,
            }
        )
    for col, base in BASELINES[result.get("holdout", FINAL_HOLDOUT)][weights].items():
        acc = result["columns"][col]["accuracy_nonblank"]
        floor = round(base - ACC_DROP_LIMIT, 4)
        out.append(
            {
                "id": f"{weights}:acc:{col}",
                "name": f"{weights}: {col} の値の正解率 >= コンテナの値 {base} - {ACC_DROP_LIMIT}",
                "value": acc,
                "limit": floor,
                "ok": acc is not None and acc >= floor,
            }
        )
    return out


def _fmt(v, spec: str = ".3f") -> str:
    return "-" if v is None else format(v, spec)


def _miss_count(stats: dict) -> str:
    """見逃しの件数 / セル数(割合)。"""
    return f"{int(stats['miss'])}/{int(stats['n'])}({_fmt(stats['miss_rate'], '.3%')})"


def format_eval_table(result: dict) -> list[str]:
    """列ごとの、セル数・正解率・要確認率・見逃しの件数を、ログと要約に出す表にする。値は統計だけ(帳票の内容は含まない)。"""
    lines = [f"■ {result['weights']}: {result['pages']} ページ"]
    o = result["overall"]
    lines.append(
        f"  全体  セル {int(o['n'])}(空欄でない {int(o['n_nonblank'])})  値の正解率(空欄を除く) {_fmt(o['accuracy_nonblank'])}"
        f"  要確認率(空欄を除く) {_fmt(o['review_rate_nonblank'])}  見逃し {_miss_count(o)}"
    )
    for col in MISS_LIMIT_COLUMN:
        c = result["columns"][col]
        lines.append(
            f"  {COLUMN_LABELS[col]}  セル {int(c['n'])}(空欄でない {int(c['n_nonblank'])})  値の正解率(空欄を除く) {_fmt(c['accuracy_nonblank'])}"
            f"  要確認率(空欄を除く) {_fmt(c['review_rate_nonblank'])}  見逃し {_miss_count(c)}"
        )
    for name, d in result.get("per_dataset", {}).items():
        cols = "・".join(
            f"{COLUMN_LABELS[col]} {int(d['columns'][col]['miss'])}" for col in MISS_LIMIT_COLUMN
        )
        lines.append(
            f"  [{name}] {d['pages']} ページ  見逃し {_miss_count(d['overall'])}  内訳: {cols}"
        )
    return lines


def merge_eval(base: dict, adapted: dict, meta: dict) -> dict:
    checks = eval_checks(base, "base") + eval_checks(adapted, "adapted")
    pages_ok = base["pages"] >= 200 and adapted["pages"] >= 200 if meta["mode"] == "full" else True
    holdout = base.get("holdout")
    checks.append(
        {
            "id": "baseline_measured",
            "name": "コンテナの重みの基準値が、測り直した値(仮の値でない)",
            "value": holdout,
            "limit": "測定済み",
            "ok": holdout not in BASELINES_PROVISIONAL,
        }
    )
    checks.append(
        {
            "id": "holdout",
            "name": f"最終ホールドアウトは {FINAL_HOLDOUT}(重ならない新しい seed)",
            "value": holdout,
            "limit": FINAL_HOLDOUT,
            "ok": holdout == adapted.get("holdout") == FINAL_HOLDOUT,
        }
    )
    checks.append(
        {
            "id": "pages",
            "name": "評価ページ数 >= 200(full)",
            "value": base["pages"],
            "limit": 200,
            "ok": pages_ok,
        }
    )
    return {
        **meta,
        "criteria_version": EVAL_CRITERIA_VERSION,
        "holdout": holdout,
        "criteria": {
            "miss_limit_all": MISS_LIMIT_ALL,
            "miss_limit_column": MISS_LIMIT_COLUMN,
            "miss_limit_amount": MISS_LIMIT_AMOUNT,
            "accuracy_drop_limit": ACC_DROP_LIMIT,
            "baseline_accuracy": BASELINES.get(holdout, BASELINE_ACCURACY),
        },
        "results": {"base": base, "adapted": adapted},
        "checks": checks,
        "passed": all(c["ok"] for c in checks),
    }


def cmd_eval(args: argparse.Namespace) -> int:
    names = (
        args.datasets.split(",") if args.datasets else [r[0] for r in HOLDOUT_RECIPES[args.holdout]]
    )
    result = evaluate_weights(
        Path(args.models), Path(args.data), names, args.weights, args.threads, args.holdout
    )
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    o = result["overall"]
    print(
        f"{args.weights}: {result['pages']} ページ 正解率 {o['accuracy']:.3f} 要確認率 {o['review_rate']:.3f} 見逃し率 {o['miss_rate']:.4%}"
    )
    print("\n".join(format_eval_table(result)))
    return 0


def cmd_eval_merge(args: argparse.Namespace) -> int:
    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    adapted = json.loads(Path(args.adapted).read_text(encoding="utf-8"))
    meta = {
        "zip_name": args.zip_name,
        "zip_sha256": args.zip_sha256,
        "build_run_id": str(args.build_run_id),
        "build_commit": args.build_commit,
        "mode": args.mode,
        "eval_commit": os.environ.get("GITHUB_SHA", ""),
        "eval_run_id": os.environ.get("GITHUB_RUN_ID", ""),
    }
    final = merge_eval(base, adapted, meta)
    Path(args.out).write_text(json.dumps(final, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in (base, adapted):
        print("\n".join(format_eval_table(r)))
    print()
    for c in final["checks"]:
        v = c["value"]
        shown = f"{v:.4f}" if isinstance(v, int | float) and not isinstance(v, bool) else str(v)
        print(f"[{'OK' if c['ok'] else 'NG'}] {c['name']}: {shown}(基準 {c['limit']})")
    print("結果:", "合格" if final["passed"] else "不合格")
    return 0


def cmd_run_check(args: argparse.Namespace) -> int:
    """`gh api repos/.../actions/runs/ID` の結果が、期待したワークフロー・ブランチ・コミットの、成功した実行であることを確かめる(W-2)。"""
    r = json.loads(Path(args.run).read_text(encoding="utf-8"))
    checks = [
        ("ワークフローのパス", r.get("path"), r.get("path") == args.workflow_path),
        ("ブランチ", r.get("head_branch"), r.get("head_branch") == args.branch),
        (
            "イベント(push か workflow_dispatch)",
            r.get("event"),
            r.get("event") in ("push", "workflow_dispatch"),
        ),
        (
            "完了して成功",
            f"{r.get('status')}/{r.get('conclusion')}",
            r.get("status") == "completed" and r.get("conclusion") == "success",
        ),
    ]
    if args.sha:
        checks.append(("コミット", r.get("head_sha"), r.get("head_sha") == args.sha))
    for label, value, ok in checks:
        print(f"[{'OK' if ok else 'NG'}] {label}: {value}")
    if not all(ok for _l, _v, ok in checks):
        print(f"run {r.get('id')} は、期待した実行ではありません", file=sys.stderr)
        return 1
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
    if not re.fullmatch(r"\d+", str(spec.get("eval_run_id", ""))):
        problems.append("eval_run_id は数字(trial-eval の run)")
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
    for k in ("tag", "commit", "run_id", "eval_run_id", "zip_name", "title"):
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
        "確信度の較正の余裕が、固定した値(科目 0.3・金額 0.4・ほか 0.6)": info.get("fit_margins")
        == {"default": FIT_MARGIN, **FIT_MARGINS},
    }
    if args.eval_dir:
        try:
            ev = json.loads((Path(args.eval_dir) / "eval-result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise SystemExit("eval-result.json を読めません") from None
        got = {c.get("id"): c.get("ok") for c in ev.get("checks", []) if isinstance(c, dict)}
        checks.update(
            {
                "評価の結果が、この zip の SHA256 に対するもの": ev.get("zip_sha256")
                == spec["zip_sha256"],
                "評価したビルドの run が JSON と同じ": str(ev.get("build_run_id"))
                == spec["run_id"],
                "評価したビルドのコミットが JSON と同じ": ev.get("build_commit") == spec["commit"],
                "評価の run が JSON の eval_run_id と同じ": str(ev.get("eval_run_id"))
                == spec["eval_run_id"],
                "評価の mode が full": ev.get("mode") == "full",
                "評価のホールドアウトが、最終のもの(v2。重ならない新しい seed)": ev.get("holdout")
                == FINAL_HOLDOUT,
                "重みの質の評価が合格": ev.get("passed") is True,
                "評価の基準の版が、列ごとの見逃し率を含む版以上": isinstance(
                    ev.get("criteria_version"), int
                )
                and ev["criteria_version"] >= EVAL_CRITERIA_VERSION,
                "必要な判定(全体・列ごとの見逃し率・列ごとの値の正解率・ページ数。基本と適応済みの両方)がすべて入っていて合格": all(
                    got.get(i) is True for i in required_eval_check_ids()
                ),
            }
        )
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
    v.add_argument(
        "--reuse-run-id",
        help="重みと較正のダンプを再利用した run(確信度の較正だけをこの run でやり直した)",
    )
    v.add_argument("--reuse-commit", help="再利用した run のコミット")
    v.add_argument("--reuse-info", help="再利用した run の段階の記録(meta.* の infos)のフォルダ")
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
    e = sub.add_parser("eval")
    e.add_argument("--models", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--weights", choices=["base", "adapted"], required=True)
    e.add_argument("--holdout", choices=sorted(HOLDOUT_RECIPES), default=FINAL_HOLDOUT)
    e.add_argument(
        "--datasets", help="評価するデータセット名(カンマ区切り)。既定は --holdout の 4 つ"
    )
    e.add_argument("--threads", type=int, default=3)
    e.add_argument("--out", required=True)
    em = sub.add_parser("eval-merge")
    em.add_argument("--base", required=True)
    em.add_argument("--adapted", required=True)
    em.add_argument("--zip-name", required=True)
    em.add_argument("--zip-sha256", required=True)
    em.add_argument("--build-run-id", required=True)
    em.add_argument("--build-commit", required=True)
    em.add_argument("--mode", choices=sorted(MODES), required=True)
    em.add_argument("--out", required=True)
    ru = sub.add_parser("reuse-check")
    ru.add_argument("--mode", choices=sorted(MODES), required=True)
    ru.add_argument(
        "--infos", required=True, help="前の run の段階の記録(meta.* の infos)のフォルダ"
    )
    ru.add_argument("--base-commit", required=True, help="前の run のコミット(40 桁)")
    ru.add_argument("--data", required=True)
    ru.add_argument("--models", required=True)
    ru.add_argument("--evals", required=True)
    rk = sub.add_parser("run-check")
    rk.add_argument("--run", required=True)
    rk.add_argument("--workflow-path", required=True)
    rk.add_argument("--branch", required=True)
    rk.add_argument("--sha")
    rs = sub.add_parser("release-spec")
    rs.add_argument("--json", required=True)
    rs.add_argument("--notes", required=True)
    rc = sub.add_parser("release-check")
    rc.add_argument("--json", required=True)
    rc.add_argument("--notes", required=True)
    rc.add_argument("--dir", required=True)
    rc.add_argument("--eval-dir", help="trial-eval の artifact(eval-result.json)。公開では必須")
    args = ap.parse_args()
    if args.cmd == "eval":
        return cmd_eval(args)
    if args.cmd == "eval-merge":
        return cmd_eval_merge(args)
    if args.cmd == "reuse-check":
        return cmd_reuse_check(args)
    if args.cmd == "run-check":
        return cmd_run_check(args)
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
