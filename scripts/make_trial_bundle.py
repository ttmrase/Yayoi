#!/usr/bin/env python3
"""試用版の配布物(zip)を作る(開発用。リポジトリの外に出力する)。**オフライン版だけ**を作る。

    python scripts/make_trial_bundle.py [--out DIR] [--models DIR] [--python-installer FILE]

zip の中身:
  wheels/yayoi_ocr-*.whl      このリポジトリのビルド(フロントエンドのビルド成果物 app/static を含む。先に `cd web && npm run build`)
  wheels/<依存>.whl           Windows 64 ビット・Python 3.11 向け。バージョン固定 + ハッシュつきの lock(requirements/trial-win-cp311.txt)から
                              `pip download --require-hashes` で取得して検証したもの(torch は PyPI の Windows 向け = CPU 版)。
                              sdist しかない純 Python のパッケージは、lock のハッシュを検証してから wheel にする
  requirements-lock.txt       上の lock(setup.ps1 が `pip install --no-index --require-hashes` に使う)
  models/                     認識の重み(許可リストの名前だけ。docs/model_provenance.md)
  python-installer.exe        --python-installer を指定したときだけ(python.org の 3.11 の公式インストーラ。setup.ps1 が Python 3.11 が無いときに使う)
  setup.ps1, start-yayoi-ocr.cmd, README.txt, NOTICE.txt, windows_setup.md, windows_checklist.md, yayoi_export.md, model_provenance.md
  MANIFEST.sha256             配布物のハッシュ一覧

重みと zip は **コミットしない**(出力先がリポジトリの内側なら拒否する)。zip の SHA256 は最後に表示し、`<zip>.sha256` に書く
(配布経路とは別の経路で利用者に伝える)。
"""

from __future__ import annotations

import argparse
import hashlib
import re
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
  https://github.com/ndl-lab/ndlocr-lite
  models\\*.pt は、その ONNX を PyTorch のテンソルに形式変換したもの(値は変更していない)。
  models\\adapted\\*.pt は、それを合成データで追加学習した派生物。
- モデルのコード: baudm/parseq (Apache License 2.0)。
- セル種別分類器・確信度モデル・追加学習の重みは、合成データ(架空の帳票。実際の帳票は含まない)だけで学習・較正した。
  重みの出所(由来・データ・スクリプト・SHA256)は model_provenance.md に記録している。
- 依存ライブラリ(PyTorch・NumPy・OpenCV・FastAPI など)は requirements-lock.txt のとおり。ライセンスは各配布元の表示に従う。
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


def download_wheels(out: Path) -> None:
    """lock(バージョン固定 + ハッシュ)から、Windows 64 ビット・Python 3.11 向けの wheel を取得して検証する。
    ハッシュが合わなければ pip が失敗する。torch は lock が CPU 版であることを確認済み(make_trial_lock.py)。"""
    lock = ROOT / "requirements/trial-win-cp311.txt"
    sdist = ROOT / "requirements/trial-win-cp311-sdist.txt"
    if not lock.is_file():
        raise SystemExit("lock がありません: python scripts/make_trial_lock.py")
    subprocess.run(
        [
            sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "--require-hashes",
            "--platform", "win_amd64", "--python-version", "3.11", "--implementation", "cp",
            "--dest", str(out), "-r", str(lock),
        ],
        check=True,
    )  # fmt: skip
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--require-hashes",
            "-w",
            str(out),
            "-r",
            str(sdist),
        ],
        check=True,
    )
    expected = {
        m.group(1).lower().replace("-", "_").replace(".", "_")
        for m in re.finditer(
            r"^([A-Za-z0-9_.-]+)==",
            lock.read_text(encoding="utf-8") + sdist.read_text(encoding="utf-8"),
            re.M,
        )
    }
    got = {
        w.name.split("-")[0].lower().replace("-", "_").replace(".", "_") for w in out.glob("*.whl")
    }
    got.discard("yayoi_ocr")
    if expected - got:
        raise SystemExit(f"lock にあるのに wheel が無い: {sorted(expected - got)}")
    shutil.copy2(lock, out.parent / "requirements-lock.txt")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, help="出力先(既定: $YAYOI_PRIVATE_DATA/trial)")
    ap.add_argument("--models", type=Path, help="重みの置き場(既定: models_dir())")
    ap.add_argument(
        "--python-installer",
        type=Path,
        help="python.org の 3.11 の公式インストーラ(python-3.11.x-amd64.exe)。同梱すると、Python が無い PC で"
        " setup.ps1 が静かに導入する。配布元(python.org の公開チェックサム)で検証してから渡すこと",
    )
    args = ap.parse_args()

    out = assert_safe_output_dir(args.out or private_data_dir() / "trial")
    stage = out / "stage"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "wheels").mkdir(parents=True)

    build_wheel(stage / "wheels")
    download_wheels(stage / "wheels")
    if args.python_installer:
        if not re.fullmatch(r"python-3\.11\.\d+-amd64\.exe", args.python_installer.name):
            raise SystemExit("python-3.11.x-amd64.exe を指定してください")
        shutil.copy2(args.python_installer, stage / "python-installer.exe")
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
        (ROOT / "docs/model_provenance.md", "model_provenance.md"),
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
    digest = sha256(zip_path)
    (out / (zip_path.name + ".sha256")).write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    print(f"作成: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB, {len(files) + 1} ファイル)")
    print(f"SHA256: {digest}")
    print(
        "→ この SHA256 は、zip の配布経路とは別の経路で利用者に伝えてください(docs/windows_setup.md)。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
