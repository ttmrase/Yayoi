"""NDLOCR-Lite の学習済み PARSeq(PyTorch Lightning の .ckpt)を安全に読み込む。

ckpt は pickle 形式なので、`torch.load(weights_only=False)` は使わない(任意コードが実行されうる)。
`weights_only=True`(テンソルと基本型だけを許可)で読み、ckpt に含まれる学習用のスケジューラ状態が
参照する `getattr` と `SWALR` だけを許可する(内容は ckpt の pickle を実行せずに調べて確認済み:
OrderedDict / torch の再構築関数 / FloatStorage / getattr / SWALR のみ)。これ以外のグローバルが含まれる
ファイルは読み込みを拒否する。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim.swa_utils import SWALR

from .model import PARSeq

# ハイパーパラメータのうちモデル構成に使うもの(ckpt の hyper_parameters と同名)
_ARCH_KEYS = (
    "img_size",
    "patch_size",
    "embed_dim",
    "enc_num_heads",
    "enc_mlp_ratio",
    "enc_depth",
    "dec_num_heads",
    "dec_mlp_ratio",
    "dec_depth",
    "max_label_length",
)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    with torch.serialization.safe_globals([getattr, SWALR]):
        return torch.load(Path(path), map_location="cpu", weights_only=True)


def build_model(hparams: dict[str, Any], charset_size: int) -> PARSeq:
    """hyper_parameters から構成を取り出してモデルを作る(トークン数 = 文字数 + EOS/BOS/PAD)。"""
    missing = [k for k in _ARCH_KEYS if k not in hparams]
    if missing:
        raise ValueError(f"ckpt の hyper_parameters に構成が足りません: {missing}")
    return PARSeq(
        num_tokens=charset_size + 3,
        max_label_length=int(hparams["max_label_length"]),
        img_size=tuple(hparams["img_size"]),
        patch_size=tuple(hparams["patch_size"]),
        embed_dim=int(hparams["embed_dim"]),
        enc_num_heads=int(hparams["enc_num_heads"]),
        enc_mlp_ratio=int(hparams["enc_mlp_ratio"]),
        enc_depth=int(hparams["enc_depth"]),
        dec_num_heads=int(hparams["dec_num_heads"]),
        dec_mlp_ratio=int(hparams["dec_mlp_ratio"]),
        dec_depth=int(hparams["dec_depth"]),
        refine_iters=int(hparams.get("refine_iters", 1)),
    )


def state_dict_from_onnx(path: str | Path) -> dict[str, torch.Tensor]:
    """NDLOCR-Lite 同梱の ONNX(parseq-ndl-…)から PARSeq の state_dict を復元する(`onnx` パッケージが必要)。

    ONNX では Linear の重みが転置されて `MatMul` の定数になり、デコーダの MultiheadAttention は
    Q(192x192)と KV(192x384)に分割されている。それをモデルの名前・形状へ戻す。
    """
    import onnx
    from onnx import numpy_helper

    g = onnx.load(str(path)).graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    sd: dict[str, np.ndarray] = {
        k.removeprefix("model."): v for k, v in init.items() if k.startswith("model.")
    }
    nodes = {n.name: n for n in g.node}

    def w(node_name: str) -> np.ndarray:  # MatMul/Add の定数入力
        return next(init[i] for i in nodes[node_name].input if i in init)

    depth = 1 + max(
        int(m.group(1))
        for n in nodes
        if (m := re.match(r"/encoder/blocks/blocks\.(\d+)/norm1/", n))
    )
    for i in range(depth):
        for mod in ("attn/qkv", "attn/proj", "mlp/fc1", "mlp/fc2"):
            key = f"encoder.blocks.{i}.{mod.replace('/', '.')}.weight"
            sd[key] = w(f"/encoder/blocks/blocks.{i}/{mod}/MatMul").T
    dec = sorted(
        {int(m.group(1)) for n in nodes if (m := re.match(r"/decoder/layers\.(\d+)/norm_q/", n))}
    )
    for i in dec:
        base = f"/decoder/layers.{i}"
        for att in ("self_attn", "cross_attn"):
            sd[f"decoder.layers.{i}.{att}.in_proj_weight"] = np.concatenate(
                [w(f"{base}/{att}/MatMul").T, w(f"{base}/{att}/MatMul_1").T], axis=0
            )
            sd[f"decoder.layers.{i}.{att}.in_proj_bias"] = np.concatenate(
                [w(f"{base}/{att}/Add"), w(f"{base}/{att}/Add_1")]
            )
        for mod in ("linear1", "linear2"):
            sd[f"decoder.layers.{i}.{mod}.weight"] = w(f"{base}/{mod}/MatMul").T
    sd["head.weight"] = w("/head/MatMul").T
    # pos_queries は匿名の Expand の入力(形状 1 x (最大文字数 + 1) x 埋め込み次元)として保存されている
    expand = w("/Expand")
    sd["pos_queries"] = expand
    return {k: torch.from_numpy(np.array(v, dtype=np.float32)) for k, v in sd.items()}


def hparams_from_name(width: int) -> dict[str, Any]:
    """ONNX にはハイパーパラメータが無いので、NDLOCR-Lite の parseq-tiny の構成(ckpt の hyper_parameters と同じ)。"""
    return {
        "img_size": (24, width),
        "patch_size": (4, 8),
        "embed_dim": 192,
        "enc_num_heads": 3,
        "enc_mlp_ratio": 4,
        "enc_depth": 12,
        "dec_num_heads": 6,
        "dec_mlp_ratio": 4,
        "dec_depth": 1,
        "max_label_length": {256: 30, 384: 50, 768: 100}[width],
        "refine_iters": 1,
    }


def load_parseq_onnx(path: str | Path, width: int, charset: str) -> PARSeq:
    """ONNX の重みで PARSeq を作る(ckpt が無い・ckpt と ONNX の重みが違うときの経路)。"""
    model = build_model(hparams_from_name(width), len(charset))
    model.load_state_dict(state_dict_from_onnx(path), strict=True)
    return model.eval()


CONVERTED_FORMAT = 1


def save_converted(
    path: str | Path,
    state_dict: dict[str, torch.Tensor],
    charset: str,
    width: int,
    source: dict[str, str],
) -> None:
    """ONNX から復元した重みを、テンソルと基本型だけの .pt として保存する(weights_only で読める)。"""
    torch.save(
        {
            "format": CONVERTED_FORMAT,
            "width": width,
            "charset": charset,
            "source": source,
            "state_dict": {k: v.contiguous() for k, v in state_dict.items()},
        },
        Path(path),
    )


def load_converted(path: str | Path) -> tuple[PARSeq, str, dict[str, str]]:
    """`save_converted` の .pt → (モデル(eval)、charset、出所)。実行時にネットワークは使わない。"""
    d = torch.load(Path(path), map_location="cpu", weights_only=True)
    if d.get("format") != CONVERTED_FORMAT:
        raise ValueError(f"{path} は未対応の形式です")
    model = build_model(hparams_from_name(int(d["width"])), len(d["charset"]))
    model.load_state_dict(d["state_dict"], strict=True)
    return model.eval(), d["charset"], dict(d.get("source", {}))


def load_parseq(path: str | Path) -> tuple[PARSeq, str]:
    """ckpt → (モデル(eval)、charset_train)。state_dict は厳密に(キーも形状も)一致しなければならない。"""
    ck = read_checkpoint(path)
    hp = dict(ck["hyper_parameters"])
    charset: str = hp["charset_train"]
    model = build_model(hp, len(charset))
    sd = {
        k.removeprefix("model."): v for k, v in ck["state_dict"].items() if k.startswith("model.")
    }
    model.load_state_dict(sd, strict=True)
    return model.eval(), charset
