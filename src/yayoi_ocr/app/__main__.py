"""`python -m yayoi_ocr.app`(または `yayoi-ocr app`)で起動する。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .launcher import run


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--data-dir", type=Path, help="DB・画像の置き場(既定: %%LOCALAPPDATA%%\\yayoi-ocr など)"
    )
    ap.add_argument(
        "--no-window",
        action="store_true",
        help="[開発専用 / --dev が必要] ウィンドウを開かず URL(トークンを含む)を表示する。共有マシンでは使わない",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=0,
        help="[開発専用 / --dev が必要] ポートの固定(既定: ランダム)。共有マシンでは使わない",
    )
    ap.add_argument(
        "--token",
        help="[開発専用 / --dev が必要] トークンの固定(既定: 起動ごとにランダム)。"
        "プロセス一覧からトークンが見えるので、共有マシンでは使わない",
    )
    ap.add_argument(
        "--dev",
        action="store_true",
        help="[開発専用] 合成ジョブを作る API を有効にし、--token/--port/--no-window/--backend fake を許す",
    )
    ap.add_argument(
        "--backend",
        choices=["real", "fake"],
        default="real",
        help="認識の実装。fake は合成の結果を返す([開発専用 / --dev が必要]。画像は読まない)",
    )


DEV_ONLY = ("token", "port", "no_window")


def dev_only_options(args: argparse.Namespace) -> list[str]:
    """`--dev` なしで指定された開発専用の引数(本番の起動経路では受け付けない)。"""
    if args.dev:
        return []
    bad = [f"--{k.replace('_', '-')}" for k in DEV_ONLY if getattr(args, k)]
    if args.backend != "real":
        bad.append("--backend")
    return bad


def run_args(args: argparse.Namespace) -> int:
    bad = dev_only_options(args)
    if bad:
        print(
            f"エラー: {', '.join(bad)} は開発専用です(--dev と一緒にだけ使えます。共有マシンでは使わないでください)。",
            file=sys.stderr,
        )
        return 2
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
