#!/usr/bin/env python3
"""試用版の配布物(zip)を作る(開発用。リポジトリの外に出力する)。

    python scripts/make_trial_bundle.py [--out DIR] [--models DIR] [--with-wheels] [--python-version 3.11]

zip の中身:
  wheels/yayoi_ocr-*.whl      このリポジトリのビルド(フロントエンドのビルド成果物 app/static を含む。先に `cd web && npm run build`)
  wheels/<依存>.whl           --with-wheels のとき: Windows 向けの依存(torch の CPU 版など)。無ければセットアップ時に pip がネットワークから取得する
  models/                     認識の重み(許可リストの名前だけ。実データは含まない: NDLOCR-Lite 由来 + 合成データでの追加学習)
  setup.ps1, start-yayoi-ocr.cmd, README.txt, NOTICE.txt, windows_setup.md, windows_checklist.md
  MANIFEST.sha256             配布物のハッシュ一覧

重みと zip は **コミットしない**(出力先がリポジトリの内側なら拒否する)。
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from yayoi_ocr import __version__
from yayoi_ocr.config import assert_safe_output_dir, models_dir, private_data_dir
from yayoi_ocr.recognize.parseq.engine import MODEL_FILES

ROOT = Path(__file__).resolve().parent.parent

# 同梱してよい重み(許可リスト)。これ以外のファイルは入れない(評価用の画像・正解・学習データが紛れ込まないように)
MODEL_NAMES = [
    "NDLmoji.yaml",
    "celltype.pt",
    "confidence.json",
    *MODEL_FILES.values(),
]
ADAPTED_NAMES = ["confidence.json", *MODEL_FILES.values()]

NOTICE = """\
yayoi-ocr 試用版 — 利用しているもの

- 文字列認識モデル: 国立国会図書館「NDLOCR-Lite」の学習済みモデル(CC BY 4.0)。
  https://github.com/ndl-lab/ndlocr-lite  形式変換(ONNX → PyTorch のテンソル)と、合成データでの追加学習を行った派生物を含む。
- モデルのコード: baudm/parseq (Apache License 2.0)。
- 追加学習に使ったのは合成データ(実際の帳票は含まない)。
"""


def write_crlf(src: Path, dest: Path) -> None:
    """Windows のスクリプト・テキストは CRLF で入れる(Git は LF で保持するので、コピーのときに揃える)。"""
    data = src.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    dest.write_bytes(data)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_models(src: Path, dest: Path) -> list[Path]:
    """許可リストにある重みだけを dest にコピーする。足りないものは例外(中途半端な配布物を作らない)。"""
    copied: list[Path] = []
    missing: list[str] = []
    for sub, names in (("", MODEL_NAMES), ("adapted", ADAPTED_NAMES)):
        for n in names:
            s = src / sub / n if sub else src / n
            if not s.is_file():
                # 適応済みの重みと、その確信度モデルは無くてもよい(素の重みだけで動く)
                if sub == "adapted":
                    continue
                missing.append(str(s))
                continue
            d = dest / sub / n if sub else dest / n
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
            copied.append(d)
    if missing:
        raise SystemExit("重みが足りません:\n  " + "\n  ".join(missing))
    return copied


def build_wheel(out: Path) -> Path:
    if not (ROOT / "src/yayoi_ocr/app/static/index.html").is_file():
        raise SystemExit("フロントエンドがビルドされていません: cd web && npm ci && npm run build")
    out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(
        ROOT / "build", ignore_errors=True
    )  # 前のビルドの残り(古い static のアセット等)を持ち込まない
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "-w",
            str(out),
            str(ROOT),
        ],
        check=True,
    )
    wheels = sorted(out.glob("yayoi_ocr-*.whl"))
    if not wheels:
        raise SystemExit("wheel を作れませんでした")
    return wheels[-1]


def download_wheels(out: Path, python_version: str) -> None:
    """Windows 向けの依存を取得する(開発機のネットワークを使う。配布先はオフラインでよい)。torch は CPU 版。"""
    base = [
        sys.executable, "-m", "pip", "download", "--dest", str(out),
        "--platform", "win_amd64", "--python-version", python_version,
        "--implementation", "cp", "--only-binary=:all:",
    ]  # fmt: skip
    # torch は CPU 版の索引から、それ以外は PyPI から
    subprocess.run(
        [*base, "--index-url", "https://download.pytorch.org/whl/cpu", "torch>=2.4"], check=True
    )
    subprocess.run(
        [*base, "numpy>=1.26", "opencv-python-headless>=4.9", "Pillow>=10.0", "pypdfium2>=4.30",
         "pydantic>=2.5", "PyYAML>=6", "fastapi>=0.115", "uvicorn>=0.30", "pywebview>=5.0"],
        check=True,
    )  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, help="出力先(既定: $YAYOI_PRIVATE_DATA/trial)")
    ap.add_argument("--models", type=Path, help="重みの置き場(既定: models_dir())")
    ap.add_argument(
        "--with-wheels",
        action="store_true",
        help="Windows 向けの依存の wheel も同梱(オフラインでセットアップできる)",
    )
    ap.add_argument(
        "--python-version", default="3.11", help="--with-wheels の対象の Python(既定 3.11)"
    )
    args = ap.parse_args()

    out = assert_safe_output_dir(args.out or private_data_dir() / "trial")
    stage = out / "stage"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "wheels").mkdir(parents=True)

    build_wheel(stage / "wheels")
    if args.with_wheels:
        download_wheels(stage / "wheels", args.python_version)
    collect_models(args.models or models_dir(), stage / "models")

    for src, name in (
        (ROOT / "scripts/windows/setup.ps1", "setup.ps1"),
        (ROOT / "scripts/windows/start-yayoi-ocr.cmd", "start-yayoi-ocr.cmd"),
        (ROOT / "scripts/windows/README.txt", "README.txt"),
    ):
        write_crlf(src, stage / name)
    for src, name in (
        (ROOT / "docs/windows_setup.md", "windows_setup.md"),
        (ROOT / "docs/windows_checklist.md", "windows_checklist.md"),
        (ROOT / "docs/yayoi_export.md", "yayoi_export.md"),
    ):
        shutil.copy2(src, stage / name)
    (stage / "NOTICE.txt").write_bytes(NOTICE.replace("\n", "\r\n").encode("utf-8-sig"))

    files = sorted(p for p in stage.rglob("*") if p.is_file())
    (stage / "MANIFEST.sha256").write_text(
        "".join(f"{sha256(p)}  {p.relative_to(stage).as_posix()}\n" for p in files),
        encoding="utf-8",
    )
    zip_path = out / f"yayoi-ocr-trial-{__version__}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage).as_posix())
    print(f"作成: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB, {len(files) + 1} ファイル)")
    print(f"SHA256: {sha256(zip_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
