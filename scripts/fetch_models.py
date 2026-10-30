#!/usr/bin/env python3
"""NDLOCR-Lite の文字列認識モデル(PARSeq tiny, 高さ 24)を取得して、アプリが読む重みに変換する(開発用)。

    python scripts/fetch_models.py [--dest DIR] [--with-ckpt]

- 取得元: https://github.com/ndl-lab/ndlocr-lite の **固定コミット**(CC BY 4.0, 国立国会図書館)の
  `src/model/parseq-ndl-24x{256,384}-….onnx` と `src/config/NDLmoji.yaml`。SHA256 を検証する。
- ONNX の重みを PyTorch のモデル(`yayoi_ocr.recognize.parseq`)の state_dict に復元して、
  `parseq-ndl-24x{256,384}-….pt`(テンソルと基本型のみ。`torch.load(weights_only=True)` で読める)に保存する。
  ONNX 由来の重みと自作モデルの数値一致は tests/test_parseq_equivalence.py で検証している。
- `--with-ckpt`: NDL が提供する PyTorch チェックポイント(ファインチューニング用)も取得する(参照・比較用。
  256 の ckpt は同梱 ONNX と重みが異なる。docs/dependencies.md)。
- 保存先は既定で $YAYOI_PRIVATE_DATA/models。重みは **コミットしない**。アプリは実行時にダウンロードしない。
- 依存: `pip install -e ".[ml,ml-dev]"`(torch, onnx, PyYAML)。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from yayoi_ocr.config import models_dir

NDLOCR_LITE_COMMIT = "e57f7c6bd9cfb3de5f12f008e71ad2122bd44bfd"
RAW = f"https://raw.githubusercontent.com/ndl-lab/ndlocr-lite/{NDLOCR_LITE_COMMIT}"
CKPT_BASE = "https://lab.ndl.go.jp/dataset/ndlocr-lite/ver1.2_models"

# 幅 → (ONNX/ckpt のファイル名の幹, ONNX の SHA256, ckpt の SHA256)
MODELS = {
    256: (
        "parseq-ndl-24x256-30-tiny-189epoch-tegaki3-r8data-202604",
        "9e651bae4c1a4d5254da1127e86e82e21ef62d5339b37e62d4a3d3d30831772d",
        "7bd3b5d552e24500a0f94d702904a3beb82368c9612df2f0751f4aa137a8c7ea",
    ),
    384: (
        "parseq-ndl-24x384-50-tiny-300epoch-tegaki3-r8data-202604",
        "49cea9db4552f19eb05c8ee202fcf74714977749b2f4c9376b127fde41b07a99",
        "fbdd5a165d9505caea264f07f56309429fdd6a638d1bc51d74341f20a9dc3bfb",
    ),
}
CHARSET_FILE = ("NDLmoji.yaml", "f6ad5a2de444b495155866af811cf1a98309dcae3225db802767ea531a2dc529")
CONVERTED = {256: "parseq-ndl-24x256-30-tiny.pt", 384: "parseq-ndl-24x384-50-tiny.pt"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, expected: str) -> None:
    if dest.is_file() and sha256(dest) == expected:
        print(f"  既に取得済み(SHA256 一致): {dest.name}")
        return
    print(f"  取得: {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    got = sha256(tmp)
    if got != expected:
        tmp.unlink()
        raise SystemExit(f"SHA256 が一致しません: {dest.name}\n  期待 {expected}\n  実際 {got}")
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", help="保存先(既定: $YAYOI_PRIVATE_DATA/models)")
    ap.add_argument("--with-ckpt", action="store_true", help="NDL の PyTorch ckpt も取得する")
    args = ap.parse_args()
    dest = Path(args.dest).expanduser() if args.dest else models_dir()
    dest.mkdir(parents=True, exist_ok=True)

    import yaml

    from yayoi_ocr.recognize.parseq.weights import save_converted, state_dict_from_onnx

    name, digest = CHARSET_FILE
    download(f"{RAW}/src/config/{name}", dest / name, digest)
    with open(dest / name, encoding="utf-8") as fh:
        charset = yaml.safe_load(fh)["model"]["charset_train"]
    print(f"文字集合: {len(charset)} 字")
    for width, (stem, onnx_sha, ckpt_sha) in MODELS.items():
        print(f"[幅 {width}]")
        download(f"{RAW}/src/model/{stem}.onnx", dest / f"{stem}.onnx", onnx_sha)
        if args.with_ckpt:
            download(f"{CKPT_BASE}/{stem}.ckpt", dest / f"{stem}.ckpt", ckpt_sha)
        sd = state_dict_from_onnx(dest / f"{stem}.onnx")
        save_converted(
            dest / CONVERTED[width],
            sd,
            charset,
            width,
            {
                "origin": "ndl-lab/ndlocr-lite (CC BY 4.0)",
                "commit": NDLOCR_LITE_COMMIT,
                "onnx": f"{stem}.onnx",
                "onnx_sha256": onnx_sha,
            },
        )
        print(f"  変換: {CONVERTED[width]}")
    print(f"完了: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
