"""バックアップの作成・検査・取り出しを行う、**使い捨ての別プロセス**(`backup_mgr.BackupManager` が起動する)。

利用者が持ち込んだ zip と、その中の DB は、トークンと DB を持つサーバのプロセスでは解かない・開かない(アップロードの検査 `probe.py` と同じ方針)。

    python -I -m yayoi_ocr.app.backup_worker create  --root DIR --snapshot FILE --out ZIP --version V [--originals] [--progress FILE]
    python -I -m yayoi_ocr.app.backup_worker inspect --zip ZIP [--progress FILE]
    python -I -m yayoi_ocr.app.backup_worker extract --zip ZIP --staging DIR [--progress FILE]
    python -I -m yayoi_ocr.app.backup_worker checkdb --db FILE        (入れ替えの直前の、取り出し済みの DB の再検査)

標準出力の最後の 1 行が結果の JSON: 成功 `{"ok": true, "result": {...}}`、失敗 `{"ok": false, "code": コード}`(終了コード 2)。
進捗は `--progress` のファイルに JSON `{"stage", "done", "total"}` を上書きで書く(親が読む)。メッセージ・パス・値は出力しない(コードと件数だけ)。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .backup import BackupError, check_db, create_archive, extract_archive, inspect_archive


def _progress_writer(path: Path | None):
    last = [0.0]

    def write(stage: str, done: int, total: int) -> None:
        if path is None:
            return
        now = time.monotonic()
        if now - last[0] < 0.2 and done < total:  # 書きすぎない
            return
        last[0] = now
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps({"stage": stage, "done": done, "total": total}), encoding="utf-8")
        os.replace(tmp, path)

    return write


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="backup_worker")
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("create")
    c.add_argument("--root", type=Path, required=True)
    c.add_argument("--snapshot", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--version", default="")
    c.add_argument("--originals", action="store_true")
    i = sub.add_parser("inspect")
    i.add_argument("--zip", type=Path, required=True)
    e = sub.add_parser("extract")
    e.add_argument("--zip", type=Path, required=True)
    e.add_argument("--staging", type=Path, required=True)
    k = sub.add_parser("checkdb")
    k.add_argument("--db", type=Path, required=True)
    for p in (c, i, e):
        p.add_argument("--progress", type=Path)
    args = ap.parse_args(argv)
    progress = _progress_writer(getattr(args, "progress", None))
    try:
        if args.mode == "create":
            result = create_archive(
                args.root,
                args.snapshot,
                args.out,
                include_originals=args.originals,
                app_version=args.version,
                progress=progress,
            )
        elif args.mode == "checkdb":
            result = {"schema": check_db(args.db)}
        elif args.mode == "inspect":
            result = inspect_archive(args.zip, progress=progress)
        else:
            result = extract_archive(args.zip, args.staging, progress=progress)
    except BackupError as exc:
        print(json.dumps({"ok": False, "code": exc.code, **exc.extra}), flush=True)
        return 2
    except MemoryError:
        print(json.dumps({"ok": False, "code": "memory"}), flush=True)
        return 2
    except OSError as exc:
        code = "disk_full" if getattr(exc, "errno", None) == 28 else "io_error"
        print(json.dumps({"ok": False, "code": code}), flush=True)
        return 2
    except Exception as exc:  # 型名だけ(メッセージにパスが入りうる)
        print(json.dumps({"ok": False, "code": f"error_{type(exc).__name__}"}), flush=True)
        return 2
    print(json.dumps({"ok": True, "result": result}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
