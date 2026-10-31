#!/usr/bin/env python3
"""利用者向けの手引き(docs/user_manual.md)を、1 ファイルの HTML(外部の資源なし)にする。配布物の `manual.html`。

    python scripts/build_manual.py OUT.html

必要な書き方だけを変換する小さな変換器(見出し・段落・箇条書き・番号つきの一覧・表・引用・コードの塊・太字・インラインコード)。外部のライブラリは使わない。
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSS = """
body { font-family: "Yu Gothic UI", "Meiryo", sans-serif; line-height: 1.7; max-width: 52rem; margin: 1.5rem auto; padding: 0 1rem; color: #1a1a1a; background: #fff; }
h1 { font-size: 1.6rem; border-bottom: 2px solid #345; padding-bottom: .3rem; }
h2 { font-size: 1.3rem; margin-top: 2rem; border-bottom: 1px solid #99a; }
h3 { font-size: 1.1rem; margin-top: 1.4rem; }
table { border-collapse: collapse; margin: .8rem 0; }
th, td { border: 1px solid #99a; padding: .3rem .6rem; text-align: left; vertical-align: top; }
th { background: #eef; }
blockquote { margin: .8rem 0; padding: .4rem 1rem; border-left: 4px solid #c96; background: #fff8ee; }
code { background: #f1f1f4; padding: 0 .25em; border-radius: 3px; }
pre { background: #f1f1f4; padding: .6rem 1rem; overflow-x: auto; }
pre code { background: none; padding: 0; }
@media (prefers-color-scheme: dark) {
  body { color: #e8e8ee; background: #16161c; }
  th { background: #2a2a3a; } code, pre { background: #26262e; } blockquote { background: #2c2418; }
}
"""


def inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)


def dedent_fences(lines: list[str]) -> list[str]:
    """リストの項目の中にインデントして書いたコードの塊を、左端の塊にする(リストはそこで一度終わり、番号は `start` で続ける)。"""
    out: list[str] = []
    indent = None
    for ln in lines:
        m = re.match(r"^(\s+)```", ln)
        if indent is None and m:
            indent = len(m.group(1))
            out.append("")
            out.append(ln[indent:])
        elif indent is not None:
            out.append(ln[indent:] if ln.startswith(" " * indent) else ln)
            if ln.strip() == "```":
                indent = None
                out.append("")
        else:
            out.append(ln)
    return out


def convert(md: str) -> str:
    out: list[str] = []
    lines = dedent_fences(md.splitlines())
    i = 0
    title = "ご利用の手引き"
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append("<pre><code>" + html.escape("\n".join(lines[i + 1 : j])) + "</code></pre>")
            i = j + 1
            continue
        m = re.match(r"^(#{1,3}) (.*)$", line)
        if m:
            n = len(m.group(1))
            if n == 1 and title == "ご利用の手引き":
                title = m.group(2)
            out.append(f"<h{n}>{inline(m.group(2))}</h{n}>")
            i += 1
            continue
        if line.startswith("> "):
            buf = []
            while i < len(lines) and lines[i].startswith("> "):
                buf.append(lines[i][2:])
                i += 1
            out.append("<blockquote>" + inline(" ".join(buf)) + "</blockquote>")
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, body = rows[0], [r for r in rows[2:]]
            out.append(
                "<table><thead><tr>"
                + "".join(f"<th>{inline(c)}</th>" for c in head)
                + "</tr></thead><tbody>"
                + "".join(
                    "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body
                )
                + "</tbody></table>"
            )
            continue
        if re.match(r"^\s*([-*]|\d+\.) ", line):
            ordered = bool(re.match(r"^\s*\d+\. ", line))
            tag = "ol" if ordered else "ul"
            start = int(re.match(r"^\s*(\d+)\. ", line).group(1)) if ordered else 1
            items: list[str] = []
            while i < len(lines) and (
                re.match(r"^\s*([-*]|\d+\.) ", lines[i]) or (lines[i].startswith("   ") and items)
            ):
                if not items or (
                    re.match(r"^\s*([-*]|\d+\.) ", lines[i]) and not lines[i].startswith("   ")
                ):
                    items.append(re.sub(r"^\s*([-*]|\d+\.) ", "", lines[i]))
                else:  # 入れ子や続きの行は、直前の項目にまとめる
                    items[-1] += " " + lines[i].strip().lstrip("-*").strip()
                i += 1
            attr = f' start="{start}"' if ordered and start != 1 else ""
            out.append(
                f"<{tag}{attr}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>"
            )
            continue
        if not line.strip():
            i += 1
            continue
        buf = [line]
        i += 1
        while (
            i < len(lines)
            and lines[i].strip()
            and not re.match(r"^(#{1,3} |> |\||```|\s*([-*]|\d+\.) )", lines[i])
        ):
            buf.append(lines[i])
            i += 1
        out.append("<p>" + inline(" ".join(buf)) + "</p>")
    body = "\n".join(out)
    return (
        '<!doctype html>\n<html lang="ja"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>\n{body}\n</body></html>\n"
    )


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        convert((ROOT / "docs/user_manual.md").read_text(encoding="utf-8")), encoding="utf-8"
    )
    print(f"作成: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
