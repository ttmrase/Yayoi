#!/usr/bin/env python3
"""配布物に入れる依存ライブラリのライセンス一覧(アプリの「ライセンス」画面が読む JSON)と、ライセンスの本文を、lock から自動で作る。

    python scripts/gen_licenses.py --wheels DIR --out JSON [--texts DIR] [--npm-modules web/node_modules]

- pip: `requirements/trial-win-cp311.txt`(と sdist 由来の lock)の名前とバージョンが正。ライセンスは、**配布物に入れる wheel の METADATA**
  (`License-Expression` > 短い `License` > 分類子)から読む。wheel の `*.dist-info` にあるライセンス・NOTICE のファイルは、`--texts` に
  `pip/<名前>-<版>/` として写す(MIT・BSD・Apache-2.0 などは、再配布のときにライセンス本文を同梱する条件があるため)。
- npm: `web/package-lock.json` の本番の依存(フロントエンドのビルドに入るもの。開発用は含めない)。ライセンス本文は `--npm-modules` があれば写す。
- `make_trial_bundle.py` が呼ぶ。出力 JSON は `src/yayoi_ocr/app/data/licenses.json`(コミットしない)に置くと、wheel に入ってアプリが読む。
ネットワークは使わない(wheel と lock だけを読む)。
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCKS = ("requirements/trial-win-cp311.txt", "requirements/trial-win-cp311-sdist.txt")
NPM_LOCK = "web/package-lock.json"
MAX_TEXT_BYTES = 1 << 20
UNKNOWN = "(要確認)"
_LICENSE_FILE = re.compile(r"(^|/)(LICEN[CS]E|COPYING|NOTICE|AUTHORS)[^/]*$", re.I)


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock(path: Path) -> dict[str, str]:
    """`name==version` の行から、正規化した名前 → バージョン。"""
    out: dict[str, str] = {}
    for m in re.finditer(r"^([A-Za-z0-9_.-]+)==([^\s\;]+)", path.read_text(encoding="utf-8"), re.M):
        out[norm(m.group(1))] = m.group(2)
    return out


def license_from_files(files: dict[str, bytes]) -> str | None:
    """METADATA に表記が無いとき、ライセンスの本文から判定する(MIT と Apache-2.0 の定型の本文だけ。ほかは判定しない)。"""
    for data in files.values():
        text = data.decode("utf-8", errors="replace")
        head = text.strip().splitlines()[0].strip() if text.strip() else ""
        if (
            head.upper().startswith("MIT LICENSE")
            and "Permission is hereby granted, free of charge" in text
        ):
            return "MIT(LICENSE の本文から判定)"
        if head.upper().startswith("APACHE LICENSE") and "Version 2.0" in text[:400]:
            return "Apache-2.0(LICENSE の本文から判定)"
    return None


def license_of(meta, files: dict[str, bytes] | None = None) -> str:
    """METADATA の License-Expression > 短い License > 分類子 > (ライセンスの本文)の順で、短い表記にする。"""
    expr = (meta.get("License-Expression") or "").strip()
    if expr:
        return expr[:200]
    lic = (meta.get("License") or "").strip()
    if lic and "\n" not in lic and len(lic) <= 80 and lic.upper() != "UNKNOWN":
        return lic
    classifiers = [
        c.split("::")[-1].strip()
        for c in meta.get_all("Classifier") or []
        if c.startswith("License ::")
    ]
    classifiers = [c for c in classifiers if c and c != "OSI Approved"]
    return "; ".join(dict.fromkeys(classifiers))[:200] or license_from_files(files or {}) or UNKNOWN


def read_wheel(whl: Path) -> tuple[dict, dict[str, bytes]]:
    """wheel の METADATA と、dist-info にあるライセンス・NOTICE のファイル(名前 → 内容)。"""
    with zipfile.ZipFile(whl) as zf:
        names = zf.namelist()
        meta_name = next((n for n in names if n.endswith(".dist-info/METADATA")), None)
        if meta_name is None:
            return {}, {}
        meta = Parser().parsestr(zf.read(meta_name).decode("utf-8", errors="replace"))
        base = meta_name.rsplit("/", 1)[0] + "/"
        files: dict[str, bytes] = {}
        for n in names:
            if n.startswith(base) and _LICENSE_FILE.search(n) and not n.endswith("/"):
                info = zf.getinfo(n)
                if info.file_size <= MAX_TEXT_BYTES:
                    files[n[len(base) :]] = zf.read(n)
    return meta, files


def pip_components(wheels: Path | None, locks: list[Path], texts: Path | None) -> list[dict]:
    pins: dict[str, str] = {}
    for lock in locks:
        pins.update(parse_lock(lock))
    metas: dict[str, tuple[dict, dict[str, bytes]]] = {}
    if wheels is not None:
        for whl in sorted(wheels.glob("*.whl")):
            meta, files = read_wheel(whl)
            if meta:
                metas[norm(meta.get("Name", ""))] = (meta, files)
    out = []
    for name, version in sorted(pins.items()):
        meta, files = metas.get(name, ({}, {}))
        out.append(
            {
                "name": meta.get("Name") or name,
                "version": version,
                "license": license_of(meta, files) if meta else UNKNOWN,
                "kind": "python",
                "texts": sorted(files),
            }
        )
        if texts is not None and files:
            d = texts / "pip" / f"{name}-{version}"
            d.mkdir(parents=True, exist_ok=True)
            for rel, data in files.items():
                (d / Path(rel).name).write_bytes(data)
    return out


def npm_components(lock: Path, modules: Path | None, texts: Path | None) -> list[dict]:
    pk = json.loads(lock.read_text(encoding="utf-8")).get("packages", {})
    out = []
    for key, v in sorted(pk.items()):
        if not key or v.get("dev") or not key.startswith("node_modules/"):
            continue
        name = key.rsplit("node_modules/", 1)[1]
        files: list[str] = []
        if modules is not None:
            src = modules / name
            for f in sorted(src.glob("*")) if src.is_dir() else []:
                if (
                    f.is_file()
                    and _LICENSE_FILE.search(f.name)
                    and f.stat().st_size <= MAX_TEXT_BYTES
                ):
                    files.append(f.name)
                    if texts is not None:
                        d = texts / "npm" / f"{name.replace('/', '__')}-{v.get('version', '')}"
                        d.mkdir(parents=True, exist_ok=True)
                        (d / f.name).write_bytes(f.read_bytes())
        out.append(
            {
                "name": name,
                "version": v.get("version", ""),
                "license": str(v.get("license") or UNKNOWN)[:200],
                "kind": "npm",
                "texts": files,
            }
        )
    return out


def generate(
    wheels: Path | None,
    out_json: Path,
    texts: Path | None = None,
    npm_modules: Path | None = None,
    locks: list[Path] | None = None,
    npm_lock: Path | None = None,
) -> dict:
    comps = pip_components(wheels, locks or [ROOT / p for p in LOCKS], texts)
    comps += npm_components(npm_lock or ROOT / NPM_LOCK, npm_modules, texts)
    data = {"generated": True, "components": comps}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--wheels", type=Path, help="配布物に入れる wheel のディレクトリ(ライセンスの読み取り元)"
    )
    ap.add_argument("--out", type=Path, required=True, help="出力する JSON")
    ap.add_argument("--texts", type=Path, help="ライセンス本文の写し先")
    ap.add_argument("--npm-modules", type=Path, help="web/node_modules(あればライセンス本文を写す)")
    args = ap.parse_args()
    data = generate(args.wheels, args.out, args.texts, args.npm_modules)
    unknown = [c["name"] for c in data["components"] if c["license"] == UNKNOWN]
    print(
        f"{len(data['components'])} 件(ライセンス不明 {len(unknown)} 件: {', '.join(unknown[:10])})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
