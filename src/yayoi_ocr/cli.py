"""開発用 CLI。

python -m yayoi_ocr.cli layout <画像> --out <出力先>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import assert_safe_output_dir


def _cmd_layout(args: argparse.Namespace) -> int:
    from .ingest import load_pages
    from .layout.output import save_layout
    from .layout.pipeline import analyze_image

    out = assert_safe_output_dir(args.out)
    pages = load_pages(args.image)
    status = 0
    for i, pg in enumerate(pages):
        res = analyze_image(
            pg.image,
            source_page_index=pg.source_page_index,
            image_path=Path(pg.source_path).name,
            sha256=pg.sha256,
        )
        dest = out if len(pages) == 1 else out / f"page{i + 1:02d}"
        save_layout(res, dest, save_cells=not args.no_cells)
        n_blank = sum(1 for r in res.page.rows for c in r.cells if "blank" in c.flags)
        print(
            f"[{dest}] status={res.page.status.value} rows={len(res.page.rows)} "
            f"blank_cells={n_blank} notes={res.page.notes}"
        )
        if res.needs_review:
            status = 3  # 要確認(エラーではない)
    return status


def _cmd_synth(args: argparse.Namespace) -> int:
    import cv2

    from .groundtruth import save_ground_truth
    from .synth.generator import SynthConfig, generate_sample

    out = assert_safe_output_dir(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        seed = args.seed0 + i
        sample = generate_sample(
            SynthConfig(seed=seed, writer_id=args.writer, degrade=not args.clean, dpi=args.dpi)
        )
        name = f"synth_{seed:05d}"
        cv2.imwrite(str(out / f"{name}.png"), sample.image)
        save_ground_truth(sample.ground_truth(f"{name}.png"), out / f"{name}.json")
    print(f"{args.n} 枚の合成帳票(画像 + 正解 JSON)を {out} に書きました")
    return 0


def _cmd_app(args: argparse.Namespace) -> int:
    from .app.launcher import run

    return run(
        args.data_dir, window=not args.no_window, port=args.port, token=args.token, dev=args.dev
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="yayoi_ocr", description="手書き帳票 OCR → CSV(開発用 CLI)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("layout", help="表の位置合わせとセル切り出しを行い、検証用画像を出力する")
    p.add_argument("image", help="入力画像 / PDF / TIFF")
    p.add_argument("--out", required=True, help="出力先ディレクトリ(リポジトリの外を指定)")
    p.add_argument("--no-cells", action="store_true", help="セルごとの切り出し画像を書かない")
    p.set_defaults(func=_cmd_layout)
    q = sub.add_parser("synth", help="合成帳票(画像 + 正解 JSON)を生成する")
    q.add_argument("--out", required=True, help="出力先ディレクトリ(リポジトリの外を指定)")
    q.add_argument("-n", type=int, default=10, help="枚数")
    q.add_argument("--seed0", type=int, default=0)
    q.add_argument("--writer", type=int, default=None, help="書き手 ID(省略時は seed ごとに変わる)")
    q.add_argument("--dpi", type=float, default=None)
    q.add_argument("--clean", action="store_true", help="撮影による劣化を与えない")
    q.set_defaults(func=_cmd_synth)
    a = sub.add_parser("app", help="確認 UI(ローカルサーバ + ウィンドウ)を起動する")
    a.add_argument("--data-dir", type=Path, help="DB・画像の置き場")
    a.add_argument(
        "--no-window", action="store_true", help="ウィンドウを開かず URL を表示する(開発・E2E 用)"
    )
    a.add_argument("--port", type=int, default=0, help="開発・E2E 専用")
    a.add_argument("--token", help="開発・E2E 専用")
    a.add_argument(
        "--dev", action="store_true", help="開発・E2E 専用: 合成ジョブを作る API を有効にする"
    )
    a.set_defaults(func=_cmd_app)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except PermissionError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
