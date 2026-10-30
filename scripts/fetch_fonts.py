#!/usr/bin/env python3
"""合成帳票用の手書き風日本語フォントを google/fonts リポジトリから取得する。

    python scripts/fetch_fonts.py [--dest DIR] [--write-docs]

- 対象は SIL Open Font License 1.1(OFL)のフォントだけ。各フォントの METADATA.pb が
  `license: "OFL"` であること、同じディレクトリに OFL.txt があることを確認してから取得する。
- フォントファイルは **コミットしない**(既定の保存先は $YAYOI_PRIVATE_DATA/fonts)。
  コミットするのは、このスクリプトと docs/fonts.md(取得できたフォントとライセンスの一覧)だけ。
- 取得先: https://github.com/google/fonts (raw.githubusercontent.com)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from yayoi_ocr.config import fonts_dir

BASE = "https://raw.githubusercontent.com/google/fonts/main/ofl"
# 手書き風(ディレクター指示の候補)。ディレクトリ名は google/fonts の ofl/ 配下。
CANDIDATES = [
    "yomogi",
    "zenkurenaido",
    "kleeone",
    "hachimarupop",
    "yuseimagic",
    "kiwimaru",
    "zenmarugothic",
]
OFL_MARKER = "SIL OPEN FONT LICENSE Version 1.1"


def fetch(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def parse_metadata(text: str) -> dict:
    def first(pat: str) -> str | None:
        m = re.search(pat, text, re.MULTILINE)
        return m.group(1) if m else None

    return {
        "name": first(r'^name: "([^"]+)"'),
        "designer": first(r'^designer: "([^"]+)"'),
        "license": first(r'^license: "([^"]+)"'),
        "category": first(r'^category: "([^"]+)"'),
        "files": re.findall(r'filename: "([^"]+)"', text),
        "copyright": first(r'copyright: "([^"]+)"'),
    }


def can_render_japanese(path: Path) -> bool:
    from PIL import ImageFont

    from yayoi_ocr.synth.fonts import render_glyph

    try:
        font = ImageFont.truetype(str(path), 40)
    except OSError:
        return False
    a, b = render_glyph(font, "日"), render_glyph(font, "鬱")
    return a is not None and b is not None and (a.shape != b.shape or bool((a != b).any()))


def fetch_one(dirname: str, dest: Path) -> dict | None:
    meta_url = f"{BASE}/{dirname}/METADATA.pb"
    try:
        meta = parse_metadata(fetch(meta_url).decode("utf-8"))
        ofl_text = fetch(f"{BASE}/{dirname}/OFL.txt").decode("utf-8", "replace")
    except urllib.error.URLError as exc:
        print(f"[skip] {dirname}: 取得できません ({exc})", file=sys.stderr)
        return None
    if meta["license"] != "OFL" or OFL_MARKER.lower() not in ofl_text.lower():
        print(f"[skip] {dirname}: OFL 1.1 であることを確認できません", file=sys.stderr)
        return None
    files = []
    for fn in meta["files"]:
        url = f"{BASE}/{dirname}/{fn}"
        try:
            data = fetch(url)
        except urllib.error.URLError as exc:
            print(f"[skip] {fn}: {exc}", file=sys.stderr)
            continue
        path = dest / fn
        path.write_bytes(data)
        if not can_render_japanese(path):
            print(f"[skip] {fn}: 日本語字形を描けません", file=sys.stderr)
            path.unlink()
            continue
        files.append({"file": fn, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    if not files:
        return None
    (dest / f"{dirname}-OFL.txt").write_text(ofl_text, encoding="utf-8")
    return {
        "dir": dirname,
        "name": meta["name"],
        "designer": meta["designer"],
        "license": "SIL Open Font License 1.1",
        "category": meta["category"],
        "copyright": meta["copyright"],
        "source": f"https://github.com/google/fonts/tree/main/ofl/{dirname}",
        "files": files,
    }


def write_docs(entries: list[dict], path: Path) -> None:
    lines = [
        "# 合成帳票用フォント(手書き風・SIL OFL 1.1)",
        "",
        "フォントファイルはリポジトリに含めない。`python scripts/fetch_fonts.py` で",
        "`$YAYOI_PRIVATE_DATA/fonts/`(環境変数 `YAYOI_FONT_DIR` で変更可)に取得する。",
        "取得元は google/fonts リポジトリ(`ofl/` 配下)。取得時に METADATA.pb のライセンス表記が",
        "OFL であること、OFL.txt の本文が SIL Open Font License 1.1 であることを確認する。",
        "",
        "| フォント | 作者 | ライセンス | 取得ファイル | 出所 |",
        "|---|---|---|---|---|",
    ]
    for e in entries:
        files = ", ".join(f["file"] for f in e["files"])
        lines.append(
            f"| {e['name']} | {e['designer']} | {e['license']} | {files} | {e['source']} |"
        )
    lines += [
        "",
        "各フォントの著作権表示:",
        "",
    ]
    for e in entries:
        lines.append(f"- {e['name']}: {e['copyright']}")
    lines += [
        "",
        "合成帳票は学習・評価用の架空データであり、フォントそのものを再配布しない。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dest", default=None, help="保存先(既定: $YAYOI_PRIVATE_DATA/fonts)")
    ap.add_argument("--write-docs", action="store_true", help="docs/fonts.md を更新する")
    args = ap.parse_args()
    dest = Path(args.dest).expanduser() if args.dest else fonts_dir()
    dest.mkdir(parents=True, exist_ok=True)
    entries = [e for d in CANDIDATES if (e := fetch_one(d, dest))]
    (dest / "manifest.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    for e in entries:
        print(f"{e['name']:18s} {e['license']}  {', '.join(f['file'] for f in e['files'])}")
    print(f"{len(entries)} フォントを {dest} に取得しました")
    if args.write_docs:
        root = Path(__file__).resolve().parents[1]
        write_docs(entries, root / "docs" / "fonts.md")
        print("docs/fonts.md を更新しました")
    return 0 if entries else 1


if __name__ == "__main__":
    sys.exit(main())
