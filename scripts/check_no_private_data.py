#!/usr/bin/env python3
"""実データ(画像・PDF・大きなバイナリ等)がコミットに混入していないか検査する。

リポジトリは public のため、サンプル画像・切り出し画像・正解データ・学習データ・
モデルをコミットしてはならない。pre-commit / pre-push フックとテストから呼ばれる。

追加の検査(テキストファイル):
  - 拒否リスト: $YAYOI_PRIVATE_DATA/denylist.txt(リポジトリ外。1 行 1 語/数値、# でコメント)の
    語・数値(4 桁以上)が含まれていたら失敗する。ファイルが無ければ警告だけ出して通す。
    拒否リストの内容はリポジトリに入れない(ハッシュも書かない)。
  - 合成器・テストの整数 3 個以上のリスト(金額らしい固定値)。意図的なら行末に `# allow-int-list`。
  - --unpushed では、未 push コミットのメッセージも拒否リストで検査する。

モード:
  (既定)       ステージ済みファイル(git diff --cached)を検査
  --all        git ls-files の全追跡ファイルを検査
  --unpushed   まだどのリモートにも無いコミットに含まれる全 blob を検査(履歴の混入対策)

終了コード: 0=問題なし / 1=違反あり / 2=使い方・git エラー
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable

BLOCKED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".heic",
    ".heif", ".ico", ".psd", ".pdf",
    ".sqlite", ".sqlite3", ".db", ".npy", ".npz", ".pt", ".pth", ".onnx", ".ckpt",
    ".safetensors", ".pkl", ".pickle", ".zip", ".7z", ".tar", ".gz", ".csv",
    ".ttf", ".otf", ".ttc",
}  # fmt: skip

# 先頭バイトで判別するバイナリ形式(拡張子を偽装しても検出する)
MAGIC_SIGNATURES: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "JPEG"),
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"%PDF", "PDF"),
    (b"GIF87a", "GIF"),
    (b"GIF89a", "GIF"),
    (b"II*\x00", "TIFF"),
    (b"MM\x00*", "TIFF"),
    (b"BM", "BMP?"),
    (b"PK\x03\x04", "ZIP"),
    (b"\x1f\x8b", "GZIP"),
    (b"SQLite format 3\x00", "SQLite"),
    (b"\x93NUMPY", "NumPy"),
    (b"\x00\x01\x00\x00\x00", "TrueType font"),
    (b"OTTO", "OpenType font"),
]

BLOCKED_DIR_NAMES = {"ground_truth", "samples", "private", "yayoi_private", "models", "outputs"}
# コミットを許すファイル(正解データの JSON スキーマ等。実データではない)
ALLOWED_PATHS = {"docs/ground_truth_schema.json"}

MAX_BINARY_BYTES = 64 * 1024  # 64KB 超のバイナリは不可
MAX_TEXT_BYTES = 1024 * 1024  # 1MB 超のテキストも不可(データ流出の疑い)
MAX_JSON_BYTES = 256 * 1024
MAX_LINE_LEN = 20000  # base64 埋め込み等の疑い
BASE64_RE = re.compile(rb"[A-Za-z0-9+/=]{4000,}")


def _git(*args: str, input_: bytes | None = None) -> bytes:
    proc = subprocess.run(["git", *args], input=input_, capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", "replace"))
        raise SystemExit(2)
    return proc.stdout


INT_LIST_DIRS = ("src/yayoi_ocr/synth/", "tests/")
INT_LIST_RE = re.compile(r"[\[(]\s*(?:\d[\d_]{2,}\s*,\s*){2,}\d[\d_]{2,}")
INT_LIST_ALLOW = "allow-int-list"


def denylist_path() -> str:
    root = os.environ.get("YAYOI_PRIVATE_DATA", "~/yayoi_private")
    return os.path.join(os.path.expanduser(root), "denylist.txt")


def load_denylist() -> tuple[list[str], list[re.Pattern[str]]] | None:
    """(語のリスト(小文字化), 数値の正規表現のリスト)。ファイルが無ければ None。"""
    path = denylist_path()
    if not os.path.isfile(path):
        return None
    words: list[str] = []
    numbers: list[re.Pattern[str]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            digits = entry.replace(",", "")
            if re.fullmatch(r"[0-9,]+", entry):
                if len(digits) >= 4:  # 3 桁以下の数値は誤検出が多すぎるので対象外
                    body = r"[,]?".join(re.escape(c) for c in digits)
                    # カンマ区切り・なしの両方に一致。前後が数字・カンマの途中なら除く
                    numbers.append(re.compile(rf"(?<![0-9,.]){body}(?![0-9]|,\d)"))
                    comma = f"{int(digits):,}"
                    numbers.append(re.compile(rf"(?<![0-9,.]){re.escape(comma)}(?![0-9]|,\d)"))
            else:
                words.append(entry.lower())
    return words, numbers


# ロックファイルの完全性ハッシュ(sha512-… の base64)。短い語が偶然含まれる誤検出が多いので、検査の前に取り除く
INTEGRITY_RE = re.compile(r"sha(?:1|256|384|512)-[A-Za-z0-9+/=]{20,}")


def check_denylist(text: str, deny: tuple[list[str], list[re.Pattern[str]]]) -> list[str]:
    words, numbers = deny
    text = INTEGRITY_RE.sub("", text)
    low = text.lower()
    hits = [f"拒否リストの語を含む(#{i + 1})" for i, w in enumerate(words) if w in low]
    hits += [f"拒否リストの数値を含む(#{i + 1})" for i, rx in enumerate(numbers) if rx.search(text)]
    return hits[:3]  # 値そのものは表示しない(番号のみ)


def check_int_lists(path: str, text: str) -> list[str]:
    if not path.startswith(INT_LIST_DIRS):
        return []
    out = []
    for no, line in enumerate(text.splitlines(), 1):
        if INT_LIST_RE.search(line) and INT_LIST_ALLOW not in line:
            out.append(f"整数 3 個以上のリスト({no} 行目): 金額などの固定値は乱数・規則で生成する")
    return out


def check_text_rules(path: str, data: bytes, deny) -> list[str]:
    if b"\x00" in data[:8192] or path in ALLOWED_PATHS:
        return []
    text = data.decode("utf-8", "replace")
    problems = check_int_lists(path, text)
    if deny is not None:
        problems += check_denylist(text, deny)
    return problems


def check_blob(path: str, head: bytes, size: int, is_text_hint: bool | None = None) -> list[str]:
    """1 ファイル分の検査。`head` は先頭の数 KB 以上、`size` は全体サイズ。違反理由のリスト。"""
    problems: list[str] = []
    if path in ALLOWED_PATHS:
        return problems
    low = path.lower()
    ext = "." + low.rsplit(".", 1)[-1] if "." in low.rsplit("/", 1)[-1] else ""
    if ext in BLOCKED_EXTENSIONS:
        problems.append(f"禁止された拡張子 {ext}")
    parts = low.split("/")
    if any(p in BLOCKED_DIR_NAMES for p in parts[:-1]):
        problems.append("実データ用ディレクトリ配下のファイル")
    for sig, name in MAGIC_SIGNATURES:
        if head.startswith(sig):
            # "BM" は短いので、BMP ヘッダ(14 バイト以上のバイナリ)らしい場合のみ
            if sig == b"BM" and not (len(head) > 14 and b"\x00" in head[:14]):
                continue
            problems.append(f"{name} 形式のバイナリ(拡張子を偽装していても検出)")
            break
    is_binary = b"\x00" in head[:8192]
    if is_binary and size > MAX_BINARY_BYTES:
        problems.append(f"大きなバイナリ ({size} bytes > {MAX_BINARY_BYTES})")
    elif is_binary and not problems:
        problems.append("バイナリファイル(コミット対象は原則テキストのみ)")
    if not is_binary:
        if size > MAX_TEXT_BYTES:
            problems.append(f"大きなテキスト ({size} bytes)")
        if low.endswith(".json") and size > MAX_JSON_BYTES:
            problems.append(f"大きな JSON ({size} bytes): 正解・学習データの疑い")
        if BASE64_RE.search(head):
            problems.append("base64 らしい長い文字列(画像の埋め込みの疑い)")
        elif head and max((len(line) for line in head.split(b"\n")), default=0) > MAX_LINE_LEN:
            problems.append("極端に長い行(データ埋め込みの疑い)")
    return problems


def _read_index(path: str) -> tuple[bytes, int]:
    size = int(_git("cat-file", "-s", f":{path}").strip() or 0)
    data = _git("show", f":{path}")
    return data[:MAX_TEXT_BYTES], size


def _read_worktree(path: str) -> tuple[bytes, int]:
    try:
        with open(path, "rb") as fh:
            data = fh.read(MAX_TEXT_BYTES)
            fh.seek(0, 2)
            return data, fh.tell()
    except OSError:
        return b"", 0


def staged_files() -> list[str]:
    out = _git("diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR")
    return [p for p in out.decode("utf-8", "surrogateescape").split("\0") if p]


def tracked_files() -> list[str]:
    out = _git("ls-files", "-z")
    return [p for p in out.decode("utf-8", "surrogateescape").split("\0") if p]


def unpushed_objects() -> list[tuple[str, str]]:
    """(blob sha, path) のリスト。どのリモート参照にも含まれないコミット由来の blob。"""
    out = _git("rev-list", "--objects", "HEAD", "--not", "--remotes").decode("utf-8", "replace")
    blobs: list[tuple[str, str]] = []
    for line in out.splitlines():
        if " " not in line:
            continue
        sha, path = line.split(" ", 1)
        blobs.append((sha, path))
    if not blobs:
        return blobs
    batch = "\n".join(s for s, _ in blobs).encode() + b"\n"
    types = _git("cat-file", "--batch-check=%(objecttype)", input_=batch).decode().splitlines()
    return [b for b, t in zip(blobs, types, strict=True) if t == "blob"]


def run(mode: str) -> list[tuple[str, list[str]]]:
    violations: list[tuple[str, list[str]]] = []
    deny = load_denylist()
    if deny is None:
        print(
            f"警告: 拒否リスト({denylist_path()})が無いので、語・数値の検査は行いません。",
            file=sys.stderr,
        )
    if mode == "unpushed":
        for sha, path in unpushed_objects():
            size = int(_git("cat-file", "-s", sha).strip() or 0)
            data = _git("cat-file", "blob", sha)[:MAX_TEXT_BYTES]
            probs = check_blob(path, data[:65536], size) + check_text_rules(path, data, deny)
            if probs:
                violations.append((f"{path} (blob {sha[:10]})", probs))
        if deny is not None:
            msgs = _git("log", "--format=%H%x00%B%x01", "HEAD", "--not", "--remotes").decode(
                "utf-8", "replace"
            )
            for rec in msgs.split("\x01"):
                if "\x00" in rec:
                    sha, body = rec.strip().split("\x00", 1)
                    probs = check_denylist(body, deny)
                    if probs:
                        violations.append((f"コミットメッセージ {sha[:10]}", probs))
        return violations
    paths: Iterable[str] = tracked_files() if mode == "all" else staged_files()
    for path in paths:
        data, size = _read_worktree(path) if mode == "all" else _read_index(path)
        probs = check_blob(path, data[:65536], size) + check_text_rules(path, data, deny)
        if probs:
            violations.append((path, probs))
    return violations


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows のコンソールでも日本語メッセージで落ちない
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="追跡中の全ファイルを検査")
    group.add_argument("--unpushed", action="store_true", help="未 push コミットの全 blob を検査")
    args = parser.parse_args(argv)
    mode = "all" if args.all else "unpushed" if args.unpushed else "staged"
    violations = run(mode)
    if violations:
        print("【中止】実データの混入の恐れがあるファイルが見つかりました:", file=sys.stderr)
        for path, probs in violations:
            print(f"  - {path}: {' / '.join(probs)}", file=sys.stderr)
        print(
            "実データは YAYOI_PRIVATE_DATA(既定 ~/yayoi_private)に置き、"
            "リポジトリには入れないでください。",
            file=sys.stderr,
        )
        return 1
    print(f"check_no_private_data: OK ({mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
