"""`python -m yayoi_ocr.app`(または `yayoi-ocr app`)で起動する。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .launcher import run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="yayoi-ocr app")
    ap.add_argument(
        "--data-dir", type=Path, help="DB・画像の置き場(既定: %%LOCALAPPDATA%%\\yayoi-ocr など)"
    )
    ap.add_argument(
        "--no-window", action="store_true", help="ウィンドウを開かず URL を表示する(開発・E2E 用)"
    )
    ap.add_argument("--port", type=int, default=0, help="開発・E2E 専用(既定: ランダム)")
    ap.add_argument("--token", help="開発・E2E 専用(既定: 起動ごとにランダム)")
    args = ap.parse_args(argv)
    return run(args.data_dir, window=not args.no_window, port=args.port, token=args.token)


if __name__ == "__main__":
    raise SystemExit(main())
