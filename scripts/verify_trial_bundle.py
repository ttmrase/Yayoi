#!/usr/bin/env python3
"""試用版の束(zip)の検査(開発用)。`make_trial_bundle.py` が作った束を、配布の前に**別の経路**で確かめる。zip を読むだけで、展開しない。

    python scripts/verify_trial_bundle.py ZIP [--sha256 HEX] [--provenance docs/model_provenance.md] [--markdown]

検査:
  B1 zip の SHA256 が `<zip>.sha256`(または --sha256)と一致する
  B2 zip の構造: 壊れていない・名前が安全(絶対パス・`..`・`\\`・重複・ディレクトリ項目がない)
  B3 `MANIFEST.sha256` の全行が zip の中身と一致し、zip に MANIFEST に無いファイルがない
  B4 `models/` が `docs/model_provenance.md` の表のファイルと過不足なく一致し、SHA256 が一致する
  B5 `wheels/`: アプリの wheel が 1 つで版が `yayoi_ocr.__version__` と一致する。ほかの wheel は lock(`requirements/trial-win-cp311*.txt`)の
     `名前==版` にあり、lock が wheel のハッシュを持つものはそのハッシュと一致する。同梱の `requirements-lock.txt` は、リポジトリの lock と同じ
  B6 同梱の Python のアーカイブが `python-standalone.pin` の SHA256 と一致する
  B7 画像・PDF・CSV・DB・zip の類が、wheel の外に無い。拒否リスト($YAYOI_PRIVATE_DATA/denylist.txt)の語が、wheel の外のファイルと
     アプリの wheel の中身に無い(語は出さない。番号と件数だけ)。重みのテンソルは B4 のハッシュ(出所の検査 C4 で走査済み)で担保する
出力の最後に、束に含まれる全ファイルの一覧(パス・大きさ・SHA256)を `--markdown` で出す。失敗があれば終了コード 1。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from yayoi_ocr import __version__  # noqa: E402

LOCKS = ("requirements/trial-win-cp311.txt", "requirements/trial-win-cp311-sdist.txt")
FORBIDDEN_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
    ".webp",
    ".pdf",
    ".csv",
    ".sqlite3",
    ".db",
    ".zip",
}
_PROVENANCE_ROW = re.compile(r"^\|\s*`(models/[^`]+)`\s*\|.*`([0-9a-f]{64})`\s*\|\s*$")
_WHEEL = re.compile(r"^(?P<name>[A-Za-z0-9_.]+)-(?P<ver>[0-9][^-]*)-.*\.whl$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def provenance_table(text: str) -> dict[str, str]:
    """model_provenance.md の「同梱するファイル」の表 → {zip 内のパス: SHA256}。"""
    return {
        m.group(1): m.group(2) for line in text.splitlines() if (m := _PROVENANCE_ROW.match(line))
    }


def parse_lock(text: str) -> dict[str, tuple[str, set[str]]]:
    """pip の lock → {正規化した名前: (版, ハッシュの集合)}。"""
    out: dict[str, tuple[str, set[str]]] = {}
    current: str | None = None
    for line in text.splitlines():
        if m := re.match(r"^([A-Za-z0-9_.\-]+)==([^\s\\]+)", line):
            current = _norm(m.group(1))
            out[current] = (m.group(2), set())
        elif current and (h := re.search(r"--hash=sha256:([0-9a-f]{64})", line)):
            out[current][1].add(h.group(1))
    return out


def _load_denylist():
    spec = importlib.util.spec_from_file_location(
        "check_no_private_data", ROOT / "scripts" / "check_no_private_data.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_denylist()


def _deny_patterns(deny) -> list[tuple[int, list[bytes]]]:
    pats = []
    for i, w in enumerate(deny[0]):
        variants = {w.encode("utf-8"), w.encode("utf-16le")}
        with contextlib.suppress(UnicodeEncodeError):
            variants.add(w.encode("cp932"))
        pats.append((i + 1, [v for v in variants if v]))
    return pats


def _hits(data: bytes, pats: list[tuple[int, list[bytes]]]) -> set[int]:
    low = data.lower()
    return {no for no, vs in pats for v in vs if v in data or v in low}


def verify(
    zip_path: Path,
    *,
    expected_sha256: str | None = None,
    provenance: Path = ROOT / "docs" / "model_provenance.md",
    locks: tuple[Path, ...] | None = None,
    deny=None,
    version: str = __version__,
) -> tuple[list[tuple[str, bool, str]], list[tuple[str, int, str]]]:
    """(検査の結果 [(ID, 成否, 説明)], 全ファイルの一覧 [(パス, 大きさ, SHA256)])。"""
    results: list[tuple[str, bool, str]] = []
    listing: list[tuple[str, int, str]] = []

    def add(check: str, ok: bool, text: str) -> None:
        results.append((check, ok, text))

    # B1
    digest = _sha256_file(zip_path)
    if expected_sha256 is None:
        side = zip_path.with_name(zip_path.name + ".sha256")
        if side.is_file():
            expected_sha256 = side.read_text(encoding="utf-8").split()[0]
    add(
        "B1",
        expected_sha256 is not None and digest == expected_sha256.lower(),
        f"zip の SHA256 {digest}"
        + (
            " = 控え"
            if expected_sha256 and digest == expected_sha256.lower()
            else " ≠ 控え(または控えなし)"
        ),
    )

    with zipfile.ZipFile(zip_path) as z:
        # B2
        bad_zip = z.testzip()
        infos = z.infolist()
        names = [i.filename for i in infos]
        unsafe = [
            n
            for n in names
            if n.startswith("/") or n.endswith("/") or "\\" in n or ".." in PurePosixPath(n).parts
        ]
        dup = len(names) - len(set(names))
        add(
            "B2",
            bad_zip is None and not unsafe and dup == 0,
            f"{len(names)} ファイル。壊れ {'なし' if bad_zip is None else '*あり*'}・"
            f"危険な名前 {len(unsafe)} 件・重複 {dup} 件",
        )
        data = {
            n: z.read(n)
            for n in names
            if not n.startswith("wheels/") and n != "python-standalone.tar.gz"
        }
        sizes = {i.filename: i.file_size for i in infos}
        hashes: dict[str, str] = {}
        for n in names:
            if n in data:
                hashes[n] = _sha256_bytes(data[n])
            else:
                h = hashlib.sha256()
                with z.open(n) as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                hashes[n] = h.hexdigest()
        listing = [(n, sizes[n], hashes[n]) for n in sorted(names)]

        # B3
        manifest: dict[str, str] = {}
        if "MANIFEST.sha256" in data:
            for line in data["MANIFEST.sha256"].decode("utf-8").splitlines():
                h, _sp, p = line.partition("  ")
                if p:
                    manifest[p] = h
        expect_files = {n for n in names if n != "MANIFEST.sha256"}
        wrong = [p for p, h in manifest.items() if hashes.get(p) != h]
        missing = sorted(set(manifest) - expect_files)
        extra = sorted(expect_files - set(manifest))
        add(
            "B3",
            bool(manifest) and not wrong and not missing and not extra,
            f"MANIFEST {len(manifest)} 行。ハッシュ不一致 {len(wrong)}・zip に無い {len(missing)}・MANIFEST に無い {len(extra)}",
        )

        # B4
        table = provenance_table(provenance.read_text(encoding="utf-8"))
        models = {n: hashes[n] for n in names if n.startswith("models/")}
        diff_hash = sorted(n for n in models if n in table and models[n] != table[n])
        only_zip = sorted(set(models) - set(table))
        only_doc = sorted(set(table) - set(models))
        add(
            "B4",
            bool(table) and not diff_hash and not only_zip and not only_doc,
            f"models/ {len(models)} ファイル、出所の表 {len(table)} ファイル。SHA256 不一致 {len(diff_hash)}"
            f"・表に無い {len(only_zip)}{only_zip}・束に無い {len(only_doc)}{only_doc}",
        )

        # B5
        wheels = sorted(n for n in names if n.startswith("wheels/"))
        app = [w for w in wheels if _norm(PurePosixPath(w).name.split("-")[0]) == "yayoi-ocr"]
        app_ok = len(app) == 1 and f"-{version}-" in app[0]
        lock_all: dict[str, tuple[str, set[str]]] = {}
        lock_main: dict[str, tuple[str, set[str]]] = {}
        for i, lk in enumerate(locks if locks is not None else tuple(ROOT / p for p in LOCKS)):
            parsed = parse_lock(lk.read_text(encoding="utf-8"))
            lock_all.update(parsed)
            if i == 0:
                lock_main = parsed
        not_in_lock: list[str] = []
        bad_hash: list[str] = []
        n_hash_checked = 0
        for w in wheels:
            if w in app:
                continue
            m = _WHEEL.match(PurePosixPath(w).name)
            if not m:
                not_in_lock.append(PurePosixPath(w).name)
                continue
            nm = _norm(m.group("name"))
            if nm not in lock_all or lock_all[nm][0] != m.group("ver"):
                not_in_lock.append(PurePosixPath(w).name)
            elif nm in lock_main:
                n_hash_checked += 1
                if hashes[w] not in lock_main[nm][1]:
                    bad_hash.append(PurePosixPath(w).name)
        bundled_lock = data.get("requirements-lock.txt", b"").decode("utf-8")
        main_text = (ROOT / LOCKS[0]).read_text(encoding="utf-8") if locks is None else None
        lock_same = (
            main_text is None
            or bundled_lock == main_text
            or (bundled_lock.replace("\r\n", "\n") == main_text.replace("\r\n", "\n"))
        )
        add(
            "B5",
            app_ok and not not_in_lock and not bad_hash and lock_same,
            f"wheel {len(wheels)} 個(アプリの wheel {len(app)} 個・版 {version} {'一致' if app_ok else '*不一致*'})。"
            f"lock に無い {len(not_in_lock)}{not_in_lock}・ハッシュ照合 {n_hash_checked} 個中の不一致 {len(bad_hash)}{bad_hash}"
            f"・同梱の lock = リポジトリの lock: {'はい' if lock_same else '*いいえ*'}",
        )

        # B6
        pin = {
            k.strip(): v.strip()
            for k, _eq, v in (
                ln.partition("=")
                for ln in data.get("python-standalone.pin", b"").decode("utf-8").splitlines()
            )
            if _eq and not k.startswith("#")
        }
        py_ok = (
            pin.get("sha256") is not None
            and hashes.get("python-standalone.tar.gz") == pin.get("sha256")
            and pin.get("file", "").startswith("cpython-3.11.")
        )
        add(
            "B6",
            py_ok,
            f"同梱の Python {pin.get('file', '?')}(SHA256 {'一致' if py_ok else '*不一致*'})",
        )

        # B7
        forbidden = sorted(
            n
            for n in names
            if not n.startswith("wheels/") and PurePosixPath(n).suffix.lower() in FORBIDDEN_SUFFIXES
        )
        deny_result = "拒否リストなし(未検査)"
        deny_ok = False
        if deny is None:
            with contextlib.suppress(Exception):
                deny = _load_denylist()
        if deny is not None:
            pats = _deny_patterns(deny)
            hit_files = 0
            hit_words: set[int] = set()
            targets: list[bytes] = [
                d for n, d in data.items() if not n.startswith("models/") and n != "MANIFEST.sha256"
            ]
            for w in app:
                with zipfile.ZipFile(io.BytesIO(z.read(w))) as wz:
                    targets.extend(wz.read(m) for m in wz.namelist() if not m.endswith("/"))
            for t in targets:
                found = _hits(t, pats)
                if found:
                    hit_files += 1
                    hit_words |= found
            deny_ok = hit_files == 0
            deny_result = f"拒否リストの語: {hit_files} ファイルで検出(語 {sorted(hit_words)})"
        add(
            "B7",
            not forbidden and deny_ok,
            f"画像・PDF・CSV・DB・zip の類 {len(forbidden)}{forbidden}。{deny_result}",
        )
    return results, listing


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("zip", type=Path)
    ap.add_argument("--sha256", help="zip の SHA256 の控え(既定: <zip>.sha256)")
    ap.add_argument("--provenance", type=Path, default=ROOT / "docs" / "model_provenance.md")
    ap.add_argument("--markdown", action="store_true", help="全ファイルの一覧を表で出す")
    args = ap.parse_args()
    results, listing = verify(args.zip, expected_sha256=args.sha256, provenance=args.provenance)
    for check, ok, text in results:
        print(f"[{'OK' if ok else 'NG'}] {check} {text}")
    if args.markdown:
        print("\n| パス | バイト数 | SHA256 |\n|---|---:|---|")
        for p, size, h in listing:
            print(f"| `{p}` | {size:,} | `{h}` |")
    ok_all = all(ok for _c, ok, _t in results)
    print("結果:", "OK" if ok_all else "NG")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
