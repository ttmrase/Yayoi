#!/usr/bin/env python3
"""試用版に同梱する Python(python-build-standalone の Windows x86_64 / CPython 3.11)を取得して検証する(開発用。ネットワークを使う)。

    python scripts/fetch_python_standalone.py [--dest DIR]

- 取得元: https://github.com/astral-sh/python-build-standalone のリリース(`requirements/python-standalone.txt` の tag)。
- 検証: ① リポジトリに固定した SHA256 ② 同じリリースの公式 `SHA256SUMS` の該当行、の**両方**に一致しなければ失敗する。
- 保存先は既定で $YAYOI_PRIVATE_DATA/cache/python(リポジトリの外。コミットしない)。既に同じ SHA256 のファイルがあれば再取得しない。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from yayoi_ocr.config import assert_safe_output_dir, private_data_dir

ROOT = Path(__file__).resolve().parent.parent
PIN = ROOT / "requirements" / "python-standalone.txt"
BASE = "https://github.com/astral-sh/python-build-standalone/releases/download"


def read_pin() -> dict[str, str]:
    pin = {}
    for line in PIN.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            pin[k.strip()] = v.strip()
    for k in ("tag", "version", "file", "sha256"):
        if k not in pin:
            raise SystemExit(f"{PIN} に {k} がありません")
    return pin


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _get(url: str, timeout: int = 300) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def official_sha(pin: dict[str, str]) -> str:
    """同じリリースの公式 SHA256SUMS から、該当ファイルの値を取り出す。"""
    text = _get(f"{BASE}/{pin['tag']}/SHA256SUMS").decode("utf-8")
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == pin["file"]:
            return parts[0]
    raise SystemExit("公式の SHA256SUMS に該当のファイルがありません")


def ensure(dest: Path | None = None) -> Path:
    """固定した Python のアーカイブを dest に用意して、そのパスを返す(検証つき)。"""
    pin = read_pin()
    out = assert_safe_output_dir(dest or private_data_dir() / "cache" / "python")
    out.mkdir(parents=True, exist_ok=True)
    path = out / pin["file"]
    if not (path.is_file() and sha256(path) == pin["sha256"]):
        url = f"{BASE}/{pin['tag']}/{urllib.parse.quote(pin['file'])}"
        print(f"取得: {url}")
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(_get(url, timeout=900))
        tmp.replace(path)
    got = sha256(path)
    if got != pin["sha256"]:
        path.unlink(missing_ok=True)
        raise SystemExit(f"SHA256 がリポジトリの固定値と違います: {got}")
    official = official_sha(pin)
    if official != got:
        path.unlink(missing_ok=True)
        raise SystemExit(f"SHA256 が公式の SHA256SUMS と違います: {got} / {official}")
    print(f"検証 OK(固定値と公式 SHA256SUMS の両方に一致): {path.name}\n  SHA256 {got}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dest", type=Path)
    args = ap.parse_args()
    ensure(args.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
