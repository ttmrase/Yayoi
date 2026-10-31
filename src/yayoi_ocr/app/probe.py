"""アップロードされたファイルの検査(ページ数・全ページの寸法)。**使い捨ての別プロセスで実行する**(`uploads.inspect_file`)。

利用者が用意した PDF・画像のデコーダ(PIL・pdfium)は、トークンと DB を持つサーバのプロセスでは動かさない。壊れた/悪意あるファイルで
デコーダが暴走・クラッシュしても、サーバ本体は影響を受けない(親は終了コードとタイムアウトだけを見る)。

    python -I -m yayoi_ocr.app.probe FILE      → 標準出力に 1 行の JSON {"n_pages": n, "sizes": [[幅, 高さ], ...]}(描画したときの画素数)
                                                  失敗: 終了コード 2 と {"error": コード}

ページ数・各ページ(全フレーム)の寸法だけを見て、画素の展開はしない。メモリの上限は POSIX では setrlimit で付ける(できる環境のみ)。
ここでは実データの値を出さない(寸法とページ数だけ)。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MEMORY_LIMIT_BYTES = 2 * 1024**3


def _limit_memory() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except (ImportError, ValueError, OSError):  # Windows など。親のタイムアウトが最後の砦
        pass


def inspect(path: Path) -> dict:
    from ..ingest import PDF_RENDER_DPI, SUPPORTED_SUFFIXES, _check_size

    suffix = path.suffix.lower()
    sizes: list[list[int]] = []
    if suffix == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        try:
            scale = PDF_RENDER_DPI / 72.0
            for i in range(len(pdf)):
                w_pt, h_pt = pdf[i].get_size()
                w, h = int(w_pt * scale), int(h_pt * scale)
                _check_size(w, h)  # 描画したときの画素数(解像度から見積もる)
                sizes.append([w, h])
        finally:
            pdf.close()
    elif suffix in SUPPORTED_SUFFIXES:
        from PIL import Image

        with Image.open(path) as im:
            n = int(getattr(im, "n_frames", 1))
            for i in range(n):  # 全フレーム(マルチページ TIFF の 2 枚目以降も)
                im.seek(i)
                _check_size(*im.size)
                sizes.append([int(im.size[0]), int(im.size[1])])
    else:
        raise ValueError("unsupported_type")
    return {"n_pages": len(sizes), "sizes": sizes}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(json.dumps({"error": "usage"}))
        return 2
    _limit_memory()
    try:
        print(json.dumps(inspect(Path(args[0]))))
    except MemoryError:
        print(json.dumps({"error": "memory"}))
        return 2
    except Exception as exc:  # 例外の型名だけ(ファイルの内容・名前は出さない)
        print(json.dumps({"error": type(exc).__name__}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
