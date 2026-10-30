"""`python -m yayoi_ocr.app`(または `yayoi-ocr app`)で起動する。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .launcher import run


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--data-dir", type=Path, help="DB・画像の置き場(既定: %%LOCALAPPDATA%%\\yayoi-ocr など)"
    )
    ap.add_argument(
        "--no-window", action="store_true", help="ウィンドウを開かず URL を表示する(開発・E2E 用)"
    )
    ap.add_argument("--port", type=int, default=0, help="開発・E2E 専用(既定: ランダム)")
    ap.add_argument("--token", help="開発・E2E 専用(既定: 起動ごとにランダム)")
    ap.add_argument(
        "--dev", action="store_true", help="開発・E2E 専用: 合成ジョブを作る API を有効にする"
    )
    ap.add_argument(
        "--backend",
        choices=["real", "fake"],
        default="real",
        help="認識の実装。fake は合成の結果を返す(開発・E2E 専用。画像は読まない)",
    )


def run_args(args: argparse.Namespace) -> int:
    return run(
        args.data_dir,
        window=not args.no_window,
        port=args.port,
        token=args.token,
        dev=args.dev,
        backend=args.backend,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="yayoi-ocr app")
    add_arguments(ap)
    return run_args(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
