#!/usr/bin/env python3
"""試用版(Windows・64 ビット・Python 3.11)の依存の lock(バージョン固定 + SHA256)を作る(開発用。PyPI に接続する)。

    pip install uv                                   # 開発用。lock の解決に使う
    python scripts/make_trial_lock.py [--in requirements/trial-win-cp311.in] [--out requirements/trial-win-cp311.txt]

手順:
  1. `uv pip compile --python-platform windows --python-version 3.11 --generate-hashes` で、Windows 向けの依存の閉包をバージョン固定 + ハッシュつきで解決する
     (pip の `--platform` は環境マーカーをホスト(Linux)で評価するので、Windows 向けの解決には使えない)。
  2. 固定した wheel を `pip download --require-hashes` で取り直し、ハッシュが通ること、METADATA の依存(Windows・3.11 で有効なもの)が全部揃っていることを確認する。
  3. torch は PyPI の Windows 向けの wheel(CPU 版)であること — バージョンに `+cu` などのローカル版数がなく、CUDA の DLL(torch_cuda・cudart など)を含まず、
     nvidia-* / cuda-* の依存がないこと — を確認する。
  4. wheel が無い(sdist だけの)パッケージは、別ファイル(`…-sdist.txt`)に分けて書く。配布物を作るときに、ハッシュ検証つきで sdist から wheel を作る。

配布物を作るときは `make_trial_bundle.py` が、この lock から同じ wheel を取り直してハッシュを検証する。
"""

from __future__ import annotations

import argparse
import email.parser
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parent.parent
PY = "3.11"
PLATFORM = "win_amd64"
WIN_ENV = {
    **default_environment(),
    "sys_platform": "win32",
    "platform_system": "Windows",
    "os_name": "nt",
    "platform_machine": "AMD64",
    "python_version": PY,
    "python_full_version": f"{PY}.9",
    "implementation_name": "cpython",
    "platform_python_implementation": "CPython",
}
PIP_TARGET = ["--platform", PLATFORM, "--python-version", PY, "--implementation", "cp"]
# CUDA 版の torch を含む名前(wheel の中身・依存名)
CUDA_MEMBER = re.compile(r"(torch_cuda|cudart64|cublas64|cudnn64|nvrtc64)", re.I)
CUDA_DEP = re.compile(r"(^|[-_])(cuda|nvidia)[-_]", re.I)


def metadata(wheel: Path):
    with zipfile.ZipFile(wheel) as z:
        name = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        return email.parser.Parser().parsestr(z.read(name).decode("utf-8", errors="replace"))


def parse_lock(text: str) -> dict[str, tuple[str, list[str]]]:
    """uv の出力 → {正規化名: (バージョン, [sha256…])}。"""
    out: dict[str, tuple[str, list[str]]] = {}
    for m in re.finditer(
        r"^([A-Za-z0-9_.-]+)==([^\s\\]+)[ \\]*\n((?:\s+--hash=sha256:[0-9a-f]{64}[ \\]*\n?)+)",
        text,
        re.M,
    ):
        out[canonicalize_name(m.group(1))] = (m.group(2), re.findall(r"[0-9a-f]{64}", m.group(3)))
    return out


def render(pins: dict[str, tuple[str, list[str]]], header: list[str]) -> str:
    lines = list(header)
    for name in sorted(pins):
        ver, hashes = pins[name]
        lines.append(f"{name}=={ver} \\")
        lines.append(" \\\n".join(f"    --hash=sha256:{h}" for h in hashes))
    return "\n".join(lines) + "\n"


def closure_missing(wheels: list[Path]) -> list[str]:
    """各 wheel の必須の依存(Windows・3.11 で有効なもの)のうち、集合に無いもの。"""
    have = {canonicalize_name(metadata(w)["Name"]) for w in wheels}
    missing: set[str] = set()
    for w in wheels:
        for line in metadata(w).get_all("Requires-Dist") or []:
            req = Requirement(line)
            if req.marker is not None and not req.marker.evaluate({**WIN_ENV, "extra": ""}):
                continue
            if canonicalize_name(req.name) not in have:
                missing.add(f"{req.name}(要求元: {metadata(w)['Name']})")
    return sorted(missing)


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in", dest="inp", type=Path, default=ROOT / "requirements/trial-win-cp311.in")
    ap.add_argument("--out", type=Path, default=ROOT / "requirements/trial-win-cp311.txt")
    args = ap.parse_args()
    sdist_out = args.out.with_name(args.out.stem + "-sdist.txt")
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv がありません: pip install uv")
    with tempfile.TemporaryDirectory(prefix="triallock_") as tmp:
        tmp_path = Path(tmp)
        compiled = tmp_path / "compiled.txt"
        run([
            uv, "pip", "compile", str(args.inp), "--python-platform", "windows", "--python-version", PY,
            "--generate-hashes", "-o", str(compiled),
        ])  # fmt: skip
        pins = parse_lock(compiled.read_text(encoding="utf-8"))
        if "torch" not in pins:
            raise SystemExit("torch が含まれていません")

        # 2) 固定した wheel を取り直す(ハッシュ検証つき)。wheel が無いものは sdist だけのパッケージ
        wheel_dir = tmp_path / "wheels"
        wheel_dir.mkdir()
        wheel_pins: dict[str, tuple[str, list[str]]] = {}
        sdist_pins: dict[str, tuple[str, list[str]]] = {}
        for name, (ver, hashes) in pins.items():
            req_file = tmp_path / f"{name}.txt"
            req_file.write_text(
                f"{name}=={ver} " + " ".join(f"--hash=sha256:{h}" for h in hashes) + "\n"
            )
            r = subprocess.run(
                [sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "--require-hashes",
                 *PIP_TARGET, "--dest", str(wheel_dir), "-r", str(req_file)],
                capture_output=True, text=True,
            )  # fmt: skip
            if r.returncode == 0:
                wheel_pins[name] = (ver, hashes)
            elif "No matching distribution" in r.stderr or "Could not find a version" in r.stderr:
                sdist_pins[name] = (ver, hashes)
            else:
                raise SystemExit(f"{name}=={ver} を取得できません:\n{r.stderr[-800:]}")
        wheels = sorted(wheel_dir.glob("*.whl"))

        # sdist だけのものを、検証用に wheel にする(純 Python のはず)
        built = tmp_path / "built"
        built.mkdir()
        for name, (ver, hashes) in sdist_pins.items():
            req_file = tmp_path / f"{name}-sdist.txt"
            req_file.write_text(
                f"{name}=={ver} " + " ".join(f"--hash=sha256:{h}" for h in hashes) + "\n"
            )
            run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-deps",
                    "--require-hashes",
                    "-w",
                    str(built),
                    "-r",
                    str(req_file),
                ]
            )
        built_wheels = sorted(built.glob("*.whl"))
        for w in built_wheels:
            if not w.name.endswith("-none-any.whl"):
                raise SystemExit(f"sdist から作った wheel が純 Python ではありません: {w.name}")

        missing = closure_missing([*wheels, *built_wheels])
        if missing:
            raise SystemExit("依存が足りません(.in に足してください):\n  " + "\n  ".join(missing))

        # 3) torch は CPU 版
        torch_ver, _ = pins["torch"]
        if "+" in torch_ver:
            raise SystemExit(f"torch のバージョンにローカル版数があります: {torch_ver}")
        torch_wheel = next(w for w in wheels if canonicalize_name(metadata(w)["Name"]) == "torch")
        with zipfile.ZipFile(torch_wheel) as z:
            cuda_members = [n for n in z.namelist() if CUDA_MEMBER.search(n)]
        cuda_deps = [n for n in pins if CUDA_DEP.search(n + "-")]
        if cuda_members or cuda_deps:
            raise SystemExit(f"torch が CUDA 版の疑い: members={cuda_members[:3]} deps={cuda_deps}")

        base = [
            "# 自動生成: scripts/make_trial_lock.py(編集しない。入力は trial-win-cp311.in)。",
            f"# 対象: Windows 64 ビット・CPython {PY}。torch は PyPI の Windows 向け(CPU 版。CUDA の DLL・nvidia-*/cuda-* の依存なしを確認済み)。",
            "# 使い方: pip install --no-index --find-links wheels --no-deps --require-hashes -r この-ファイル",
        ]
        args.out.write_text(render(wheel_pins, base), encoding="utf-8")
        sdist_out.write_text(
            render(
                sdist_pins,
                [
                    "# 自動生成: wheel が無い(sdist だけの)純 Python パッケージ。配布物を作るとき、ハッシュ検証つきで wheel を作る:",
                    "#   pip wheel --no-deps --require-hashes -w wheels -r この-ファイル",
                ],
            ),
            encoding="utf-8",
        )
        total = sum(w.stat().st_size for w in wheels) + sum(w.stat().st_size for w in built_wheels)
        print(
            f"書き出し: {args.out}({len(wheel_pins)} パッケージ)、{sdist_out}({len(sdist_pins)} パッケージ)"
        )
        print(f"torch=={torch_ver}(CPU 版を確認)、wheel の合計 {total / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
